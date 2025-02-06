import React from "react";
import * as d3 from "d3";
import { connect, shallowEqual } from "react-redux";
import { mat3, vec2 } from "gl-matrix";
import _regl from "regl";
import memoize from "memoize-one";
import Async from "react-async";
import { Button } from "@blueprintjs/core";

import debounce from "lodash/debounce";
import setupSVGandBrushElements from "./setupSVGandBrush";
import _camera from "../../util/camera";
import _drawPoints from "./drawPointsRegl";
import {
  createColorTable,
  createColorQuery,
} from "../../util/stateManager/colorHelpers";
import * as globals from "../../globals";

import GraphOverlayLayer from "./overlays/graphOverlayLayer";
import CentroidLabels from "./overlays/centroidLabels";
import HoverProteinLabels from "./overlays/hoverProteinLabels";
import actions from "../../actions";
import renderThrottle from "../../util/renderThrottle";

import {
  flagBackground,
  flagSelected,
  flagHighlight,
} from "../../util/glHelpers";

/*
Simple 2D transforms control all point painting. There are three:
  * model - convert from underlying per-point coordinate to a layout.
    Currently used to move from data to webgl coordinate system.
  * camera - apply a 2D camera transformation (pan, zoom)
  * projection - apply any transformation required for screen size and layout
*/
function createProjectionTF(viewportWidth, viewportHeight) {
  /*
  The projection transform accounts for the screen size & other layout.
  */
  const fractionToUse = 0.95; // fraction of min dimension to use
  const topGutterSizePx = 32; // top gutter for tools
  const bottomGutterSizePx = 32; // bottom gutter for tools
  const heightMinusGutter =
    viewportHeight - topGutterSizePx - bottomGutterSizePx;
  const minDim = Math.min(viewportWidth, heightMinusGutter);
  const aspectScale = [
    (fractionToUse * minDim) / viewportWidth,
    (fractionToUse * minDim) / viewportHeight,
  ];
  const m = mat3.create();
  mat3.fromTranslation(m, [
    0,
    (bottomGutterSizePx - topGutterSizePx) / viewportHeight / aspectScale[1],
  ]);
  mat3.scale(m, m, aspectScale);
  return m;
}

function createModelTF() {
  /*
  Preallocate coordinate system transformation between data and GL.
  Data arrives in a [0,1] range, and we operate elsewhere in [-1,1].
  */
  const m = mat3.fromScaling(mat3.create(), [2, 2]);
  mat3.translate(m, m, [-0.5, -0.5]);
  return m;
}

@connect((state) => ({
  annoMatrix: state.annoMatrix,
  crossfilter: state.obsCrossfilter,
  selectionTool: state.graphSelection.tool,
  currentSelection: state.graphSelection.selection,
  layoutChoice: state.layoutChoice,
  graphInteractionMode: state.controls.graphInteractionMode,
  colors: state.colors,
  pointDilation: state.pointDilation,
  genesets: state.genesets.genesets,
  proteinHover: state.proteinHover,
}))
class Graph extends React.Component {
  /**
   * Create the regl state for the canvas.
   * @param {HTMLCanvasElement} canvas
   * @returns {Object} regl state including camera, regl, drawPoints and buffers.
   */
  static createReglState(canvas) {
    // Setup canvas, WebGL draw function and camera.
    const camera = _camera(canvas);
    const regl = _regl(canvas);
    const drawPoints = _drawPoints(regl);

    // Preallocate WebGL buffers.
    const pointBuffer = regl.buffer();
    const colorBuffer = regl.buffer();
    const flagBuffer = regl.buffer();

    return {
      camera,
      regl,
      drawPoints,
      pointBuffer,
      colorBuffer,
      flagBuffer,
    };
  }

  static watchAsync(props, prevProps) {
    return !shallowEqual(props.watchProps, prevProps.watchProps);
  }

  computePointPositions = memoize((X, Y, modelTF) => {
    /*
    Compute the model coordinate for each point.
    */
    const positions = new Float32Array(2 * X.length);
    for (let i = 0, len = X.length; i < len; i += 1) {
      const p = vec2.fromValues(X[i], Y[i]);
      vec2.transformMat3(p, p, modelTF);
      positions[2 * i] = p[0];
      positions[2 * i + 1] = p[1];
    }
    return positions;
  });

  computePointColors = memoize((rgb) => {
    /*
    Compute WebGL colors for each point.
    */
    const colors = new Float32Array(3 * rgb.length);
    for (let i = 0, len = rgb.length; i < len; i += 1) {
      colors.set(rgb[i], 3 * i);
    }
    return colors;
  });

  computeSelectedFlags = memoize(
    (crossfilter, _flagSelected, _flagUnselected) => {
      const x = crossfilter.fillByIsSelected(
        new Float32Array(crossfilter.size()),
        _flagSelected,
        _flagUnselected
      );
      return x;
    }
  );

  computeHighlightFlags = memoize(
    (nObs, pointDilationData, pointDilationLabel) => {
      const flags = new Float32Array(nObs);
      if (pointDilationData) {
        for (let i = 0, len = flags.length; i < len; i += 1) {
          if (pointDilationData[i] === pointDilationLabel) {
            flags[i] = flagHighlight;
          }
        }
      }
      return flags;
    }
  );

  computeColorByFlags = memoize((nObs, colorByData) => {
    const flags = new Float32Array(nObs);
    if (colorByData) {
      for (let i = 0, len = flags.length; i < len; i += 1) {
        const val = colorByData[i];
        if (typeof val === "number" && !Number.isFinite(val)) {
          flags[i] = flagBackground;
        }
      }
    }
    return flags;
  });

  computePointFlags = memoize(
    (crossfilter, colorByData, pointDilationData, pointDilationLabel) => {
      /*
      We communicate with the shader using three flags:
      - isNaN -- the value is a NaN. Only makes sense when we have a colorAccessor.
      - isSelected -- the value is selected.
      - isHighlighted -- the value is highlighted in the UI (orthogonal from selection highlighting).

      Due to constraints in WebGL vertex shader attributes, these are encoded in a float, "kinda"
      like bitmasks.
      */
      const nObs = crossfilter.size();
      const flags = new Float32Array(nObs);

      const selectedFlags = this.computeSelectedFlags(
        crossfilter,
        flagSelected,
        0
      );
      const highlightFlags = this.computeHighlightFlags(
        nObs,
        pointDilationData,
        pointDilationLabel
      );
      const colorByFlags = this.computeColorByFlags(nObs, colorByData);

      for (let i = 0; i < nObs; i += 1) {
        flags[i] = selectedFlags[i] + highlightFlags[i] + colorByFlags[i];
      }
      return flags;
    }
  );

  constructor(props) {
    super(props);
    this.hoverQuadtree = null;
    this.lastHoveredProtein = null;
    const viewport = this.getViewportDimensions();
    this.reglCanvas = null;
    this.cachedAsyncProps = null;
    const modelTF = createModelTF();
    this.state = {
      toolSVG: null,
      tool: null,
      container: null,
      viewport,

      // Projection transforms.
      camera: null,
      modelTF,
      modelInvTF: mat3.invert([], modelTF),
      projectionTF: createProjectionTF(viewport.width, viewport.height),

      // Regl state.
      regl: null,
      drawPoints: null,
      pointBuffer: null,
      colorBuffer: null,
      flagBuffer: null,

      // Component rendering derived state.
      layoutState: {
        layoutDf: null,
        layoutChoice: null,
      },
      colorState: {
        colors: null,
        colorDf: null,
        colorTable: null,
      },
      pointDilationState: {
        pointDilation: null,
        pointDilationDf: null,
      },
    };
  }

  componentDidMount() {
    window.addEventListener("resize", this.handleResize);
  }

  componentDidUpdate(prevProps, prevState) {
    const {
      selectionTool,
      currentSelection,
      graphInteractionMode,
      proteinHover,
    } = this.props;
    const { toolSVG, viewport } = this.state;
    const hasResized =
      prevState.viewport.height !== viewport.height ||
      prevState.viewport.width !== viewport.width;
    let stateChanges = {};

    if (
      (viewport.height && viewport.width && !toolSVG) || // first time init
      hasResized || // window size changed; recreate SVG tools
      selectionTool !== prevProps.selectionTool || // selection tool changed
      prevProps.graphInteractionMode !== graphInteractionMode // zoom mode switched
    ) {
      stateChanges = { ...stateChanges, ...this.createToolSVG() };
    }

    /*
    If the selection tool or state has changed, ensure that the tool correctly reflects
    the underlying selection.
    */
    if (
      currentSelection !== prevProps.currentSelection ||
      graphInteractionMode !== prevProps.graphInteractionMode ||
      stateChanges.toolSVG
    ) {
      const { tool, container } = this.state;
      this.selectionToolUpdate(
        stateChanges.tool ? stateChanges.tool : tool,
        stateChanges.container ? stateChanges.container : container
      );
    }

    if (proteinHover?.isEnabled && proteinHover !== prevProps.proteinHover) {
      this.enableProteinHover();
    } else if (
      !proteinHover?.isEnabled &&
      proteinHover !== prevProps.proteinHover
    ) {
      this.disableProteinHover();
    }

    if (Object.keys(stateChanges).length > 0) {
      this.setState(stateChanges);
    }
  }

  componentWillUnmount() {
    window.removeEventListener("resize", this.handleResize);
  }

  /**
   * Handler for mouse out events on the canvas.
   * Dispatches an action to end protein hover.
   */
  handleMouseOut = () => {
    const { dispatch } = this.props;
    dispatch({ type: "protein hover end" });
  };

  /**
   * Handles the hover event for proteins on the graph.
   * Converts mouse coordinates from screen space to data space, finds the nearest point
   * using the quadtree, and dispatches an action if a protein is hovered.
   *
   * @param {MouseEvent} event - The mouse event triggered by hovering over the graph.
   */
  handleProteinHover = debounce(
    async (event) => {
      const { dispatch, colors } = this.props;
      const { layoutState, colorState } = this.state;

      // Verify that required data is available.
      if (
        !layoutState?.layoutDf ||
        !colorState?.colorDf ||
        !colors?.colorAccessor ||
        !this.reglCanvas
      ) {
        return;
      }

      try {
        // 1. Get mouse coordinates.
        const rect = this.reglCanvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        // 2. Convert screen coordinates to data space.
        const dataXY = this.mapScreenToPoint([mouseX, mouseY]);
        const maxRadius = 0.002; // Maximum radius for the search (data space units).

        // 3. Find the nearest point using the quadtree.
        const nearest = this.hoverQuadtree
          ? this.hoverQuadtree.find(dataXY[0], dataXY[1], maxRadius)
          : null;

        // 4. If the same protein is already hovered, do nothing.
        if (
          this.lastHoveredProtein &&
          nearest &&
          this.lastHoveredProtein.index === nearest.index
        ) {
          return;
        }

        if (nearest) {
          this.lastHoveredProtein = nearest;
          dispatch({
            type: "protein hover start",
            payload: {
              protein: `${nearest.label} | ${nearest.index}`, // Label with actual observation index
              coordinates: [nearest.x, nearest.y],
            },
          });
        } else if (this.lastHoveredProtein !== null) {
          this.lastHoveredProtein = null;
          dispatch({ type: "protein hover end" });
        }
      } catch (error) {
        console.error("❌ Error in protein hover:", error);
        dispatch({ type: "protein hover end" });
      }
    },
    25 // Debounce of 25ms.
  );

  handleResize = () => {
    const viewport = this.getViewportDimensions();
    const projectionTF = createProjectionTF(viewport.width, viewport.height);
    this.setState({ viewport, projectionTF });
  };

  /**
   * Handler for various canvas events.
   * Passes events to the camera and renders the canvas if needed.
   */
  handleCanvasEvent = (e) => {
    const { camera, projectionTF } = this.state;
    if (e.type !== "wheel") e.preventDefault();
    if (camera.handleEvent(e, projectionTF)) {
      this.renderCanvas();
      this.setState((state) => ({
        ...state,
        updateOverlay: !state.updateOverlay,
      }));
    }
  };

  handleBrushDragAction() {
    /*
      Event describing brush position:
      @-------|
      |       |
      |       |
      |-------@
    */
    // Ignore programmatically generated events.
    if (d3.event.sourceEvent === null || !d3.event.selection) return;

    const { dispatch, layoutChoice } = this.props;
    const s = d3.event.selection;
    const northwest = this.mapScreenToPoint(s[0]);
    const southeast = this.mapScreenToPoint(s[1]);
    const [minX, maxY] = northwest;
    const [maxX, minY] = southeast;
    dispatch(
      actions.graphBrushChangeAction(layoutChoice.current, {
        minX,
        minY,
        maxX,
        maxY,
        northwest,
        southeast,
      })
    );
  }

  handleBrushStartAction() {
    // Ignore programmatically generated events.
    if (!d3.event.sourceEvent) return;
    const { dispatch } = this.props;
    dispatch(actions.graphBrushStartAction());
  }

  handleBrushEndAction() {
    // Ignore programmatically generated events.
    if (!d3.event.sourceEvent) return;
    const { dispatch, layoutChoice } = this.props;
    const s = d3.event.selection;
    if (s) {
      const northwest = this.mapScreenToPoint(s[0]);
      const southeast = this.mapScreenToPoint(s[1]);
      const [minX, maxY] = northwest;
      const [maxX, minY] = southeast;
      dispatch(
        actions.graphBrushEndAction(layoutChoice.current, {
          minX,
          minY,
          maxX,
          maxY,
          northwest,
          southeast,
        })
      );
    } else {
      dispatch(actions.graphBrushDeselectAction(layoutChoice.current));
    }
  }

  handleBrushDeselectAction() {
    const { dispatch, layoutChoice } = this.props;
    dispatch(actions.graphBrushDeselectAction(layoutChoice.current));
  }

  handleLassoStart() {
    const { dispatch, layoutChoice } = this.props;
    dispatch(actions.graphLassoStartAction(layoutChoice.current));
  }

  // When a lasso is completed, filter to the points within the lasso polygon.
  handleLassoEnd(polygon) {
    const minimumPolygonArea = 10;
    const { dispatch, layoutChoice } = this.props;
    if (
      polygon.length < 3 ||
      Math.abs(d3.polygonArea(polygon)) < minimumPolygonArea
    ) {
      // If less than three points or super small area, treat as a clear selection.
      dispatch(actions.graphLassoDeselectAction(layoutChoice.current));
    } else {
      dispatch(
        actions.graphLassoEndAction(
          layoutChoice.current,
          polygon.map((xy) => this.mapScreenToPoint(xy))
        )
      );
    }
  }

  handleLassoCancel() {
    const { dispatch, layoutChoice } = this.props;
    dispatch(actions.graphLassoCancelAction(layoutChoice.current));
  }

  handleLassoDeselectAction() {
    const { dispatch, layoutChoice } = this.props;
    dispatch(actions.graphLassoDeselectAction(layoutChoice.current));
  }

  handleDeselectAction() {
    const { selectionTool } = this.props;
    if (selectionTool === "brush") this.handleBrushDeselectAction();
    if (selectionTool === "lasso") this.handleLassoDeselectAction();
  }

  handleOpacityRangeChange(e) {
    const { dispatch } = this.props;
    dispatch({
      type: "change opacity deselected proteins in 2d graph background",
      data: e.target.value,
    });
  }

  /**
   * Updates the hover quadtree using the provided data.
   *
   * Modified: Uses the observation index dataframe fetched via annomatrix,
   * using the column name defined in the schema (schema.annotations.obs.index).
   *
   * @param {DataFrame} layoutDf - The layout dataframe.
   * @param {DataFrame} colorDf - The color dataframe.
   * @param {Object} colors - Colors configuration.
   * @param {Object} layoutChoice - Layout choice configuration.
   * @param {DataFrame} indexDf - The dataframe containing the observation index.
   */
  updateHoverQuadtreeFromData = (
    layoutDf,
    colorDf,
    colors,
    layoutChoice,
    indexDf
  ) => {
    if (!layoutDf || !colorDf || !colors?.colorAccessor) return;

    const { currentDimNames } = layoutChoice;
    const X = layoutDf.col(currentDimNames[0]).asArray();
    const Y = layoutDf.col(currentDimNames[1]).asArray();
    const labels = colorDf.col(colors.colorAccessor).asArray();

    let indices = [];
    if (indexDf) {
      // Use destructuring to get the obs index column name from the annomatrix schema.
      const { annoMatrix } = this.props;
      const { schema } = annoMatrix;
      const obsIndexName = schema.annotations.obs.index;
      indices = indexDf.col(obsIndexName).asArray();
    } else {
      indices = Array.from({ length: X.length }, (_, i) => i);
    }

    const points = new Array(X.length);
    for (let i = 0; i < X.length; i += 1) {
      points[i] = { x: X[i], y: Y[i], index: indices[i], label: labels[i] };
    }

    this.hoverQuadtree = d3
      .quadtree()
      .x((d) => d.x)
      .y((d) => d.y)
      .addAll(points);
  };

  enableProteinHover = () => {
    if (this.reglCanvas) {
      this.reglCanvas.addEventListener("mousemove", this.handleProteinHover);
      this.reglCanvas.addEventListener("mouseout", this.handleMouseOut);
    }
  };

  disableProteinHover = () => {
    if (this.reglCanvas) {
      this.reglCanvas.removeEventListener("mousemove", this.handleProteinHover);
      this.reglCanvas.removeEventListener("mouseout", this.handleMouseOut);
    }
  };

  setReglCanvas = (canvas) => {
    this.reglCanvas = canvas;
    this.setState({
      ...Graph.createReglState(canvas),
    });
  };

  getViewportDimensions = () => {
    const { viewportRef } = this.props;
    return {
      height: viewportRef.clientHeight,
      width: viewportRef.clientWidth,
    };
  };

  createToolSVG = () => {
    /*
    Called from componentDidUpdate. Create the tool SVG, and return any state changes.
    */
    const { selectionTool, graphInteractionMode } = this.props;
    const { viewport } = this.state;
    const lasso = d3.select("#lasso-layer");
    if (lasso.empty()) return {}; // still initializing
    lasso.selectAll(".lasso-group").remove();

    // Do not render or recreate toolSVG if currently in zoom mode.
    if (graphInteractionMode !== "select") {
      const { toolSVG } = this.state;
      if (toolSVG === undefined) return {};
      return { toolSVG: undefined };
    }

    let handleStart;
    let handleDrag;
    let handleEnd;
    let handleCancel;
    if (selectionTool === "brush") {
      handleStart = this.handleBrushStartAction.bind(this);
      handleDrag = this.handleBrushDragAction.bind(this);
      handleEnd = this.handleBrushEndAction.bind(this);
    } else {
      handleStart = this.handleLassoStart.bind(this);
      handleEnd = this.handleLassoEnd.bind(this);
      handleCancel = this.handleLassoCancel.bind(this);
    }

    const {
      svg: newToolSVG,
      tool,
      container,
    } = setupSVGandBrushElements(
      selectionTool,
      handleStart,
      handleDrag,
      handleEnd,
      handleCancel,
      viewport
    );

    return { toolSVG: newToolSVG, tool, container };
  };

  fetchAsyncProps = async (props) => {
    const {
      annoMatrix,
      colors: colorsProp,
      layoutChoice,
      crossfilter,
      pointDilation,
      viewport,
    } = props.watchProps;
    const { modelTF } = this.state;

    const [layoutDf, colorDf, pointDilationDf, indexDf] = await this.fetchData(
      annoMatrix,
      layoutChoice,
      colorsProp,
      pointDilation
    );

    const { currentDimNames } = layoutChoice;
    const X = layoutDf.col(currentDimNames[0]).asArray();
    const Y = layoutDf.col(currentDimNames[1]).asArray();
    const positions = this.computePointPositions(X, Y, modelTF);

    const colorTable = this.updateColorTable(colorsProp, colorDf);
    const colors = this.computePointColors(colorTable.rgb);

    const { colorAccessor } = colorsProp;
    const colorByData = colorDf?.col(colorAccessor)?.asArray();
    const {
      metadataField: pointDilationCategory,
      categoryField: pointDilationLabel,
    } = pointDilation;
    const pointDilationData = pointDilationDf
      ?.col(pointDilationCategory)
      ?.asArray();
    const flags = this.computePointFlags(
      crossfilter,
      colorByData,
      pointDilationData,
      pointDilationLabel
    );

    const { width, height } = viewport;
    return {
      positions,
      colors,
      flags,
      width,
      height,
      layoutDf, // Store layout dataframe.
      colorDf, // Store color dataframe.
      pointDilationDf, // Store point dilation dataframe.
      indexDf, // Store observation index dataframe.
    };
  };

  renderCanvas = renderThrottle(() => {
    const {
      regl,
      drawPoints,
      colorBuffer,
      pointBuffer,
      flagBuffer,
      camera,
      projectionTF,
    } = this.state;
    this.renderPoints(
      regl,
      drawPoints,
      colorBuffer,
      pointBuffer,
      flagBuffer,
      camera,
      projectionTF
    );
  });

  updateReglAndRender = (asyncProps, prevAsyncProps) => {
    // Destructure the needed props from this.props.
    const { layoutChoice, colors: propColors } = this.props;
    // Destructure asyncProps.
    const {
      positions,
      colors,
      flags,
      width,
      height,
      layoutDf,
      colorDf,
      pointDilationDf,
      indexDf,
    } = asyncProps;

    // Update state with new data.
    this.setState((prevState) => ({
      layoutState: {
        ...prevState.layoutState,
        layoutDf: layoutDf || prevState.layoutState.layoutDf,
        layoutChoice,
      },
      colorState: {
        ...prevState.colorState,
        colorDf: colorDf || prevState.colorState.colorDf,
        colorTable: this.updateColorTable(propColors, colorDf),
      },
      pointDilationState: {
        ...prevState.pointDilationState,
        pointDilationDf:
          pointDilationDf || prevState.pointDilationState.pointDilationDf,
      },
    }));

    const { pointBuffer, colorBuffer, flagBuffer } = this.state;
    let needToRenderCanvas = false;

    // If layout or color data changed, update the hover quadtree using the new data.
    if (
      !prevAsyncProps ||
      asyncProps.layoutDf !== prevAsyncProps.layoutDf ||
      asyncProps.colorDf !== prevAsyncProps.colorDf
    ) {
      this.updateHoverQuadtreeFromData(
        layoutDf,
        colorDf,
        propColors,
        layoutChoice,
        indexDf
      );
      this.lastHoveredProtein = null;
    }

    if (height !== prevAsyncProps?.height || width !== prevAsyncProps?.width) {
      needToRenderCanvas = true;
    }

    if (positions !== prevAsyncProps?.positions) {
      pointBuffer({ data: positions, dimension: 2 });
      needToRenderCanvas = true;
    }

    if (colors !== prevAsyncProps?.colors) {
      colorBuffer({ data: colors, dimension: 3 });
      needToRenderCanvas = true;
    }

    if (flags !== prevAsyncProps?.flags) {
      flagBuffer({ data: flags, dimension: 1 });
      needToRenderCanvas = true;
    }

    // Update cache and render if necessary.
    this.cachedAsyncProps = asyncProps;
    if (needToRenderCanvas) {
      this.renderCanvas();
    }
  };

  colorByQuery() {
    const {
      annoMatrix: { schema },
      colors: { colorMode, colorAccessor },
      genesets,
    } = this.props;
    return createColorQuery(colorMode, colorAccessor, schema, genesets);
  }

  // ----------------- End of custom methods -----------------
  // Only one fetchData definition is maintained.
  /**
   * Modified fetchData to also fetch the observation index using the column name defined in the schema.
   *
   * @param {Object} annoMatrix - The annotation matrix.
   * @param {Object} layoutChoice - Layout choice configuration.
   * @param {Object} colors - Colors configuration.
   * @param {Object} pointDilation - Point dilation configuration.
   * @returns {Promise<Array>} A promise that resolves to an array containing layout, color, pointDilation, and index data.
   */
  async fetchData(annoMatrix, layoutChoice, colors, pointDilation) {
    const { metadataField: pointDilationAccessor } = pointDilation;
    const promises = [];
    // Fetch layout ("emb")
    promises.push(annoMatrix.fetch("emb", layoutChoice.current));

    // Fetch color data.
    const query = this.createColorByQuery(colors);
    if (query) {
      promises.push(annoMatrix.fetch(...query));
    } else {
      promises.push(Promise.resolve(null));
    }

    // Fetch point dilation data if available.
    if (pointDilationAccessor) {
      promises.push(annoMatrix.fetch("obs", pointDilationAccessor));
    } else {
      promises.push(Promise.resolve(null));
    }

    // Fetch observation index using the column name from the schema.
    const obsIndexName = annoMatrix.schema.annotations.obs.index;
    promises.push(annoMatrix.fetch("obs", obsIndexName));

    return Promise.all(promises);
  }

  brushToolUpdate(tool, container) {
    /*
    Called from componentDidUpdate. Update the brush tool to reflect the current selection.
    */
    const { currentSelection } = this.props;
    if (container) {
      const toolCurrentSelection = d3.brushSelection(container.node());

      if (currentSelection.mode === "within-rect") {
        const screenCoords = [
          this.mapScreenToPoint(currentSelection.brushCoords.northwest),
          this.mapScreenToPoint(currentSelection.brushCoords.southeast),
        ];
        if (!toolCurrentSelection) {
          container.call(tool.move, screenCoords);
        } else {
          let delta = 0;
          for (let x = 0; x < 2; x += 1) {
            for (let y = 0; y < 2; y += 1) {
              delta += Math.abs(
                screenCoords[x][y] - toolCurrentSelection[x][y]
              );
            }
          }
          if (delta > 0) {
            container.call(tool.move, screenCoords);
          }
        }
      } else if (toolCurrentSelection) {
        container.call(tool.move, null);
      }
    }
  }

  lassoToolUpdate(tool) {
    /*
    Called from componentDidUpdate. Update the lasso tool to reflect the current polygon selection.
    */
    const { currentSelection } = this.props;
    if (currentSelection.mode === "within-polygon") {
      const polygon = currentSelection.polygon.map((p) =>
        this.mapPointToScreen(p)
      );
      tool.move(polygon);
    } else {
      tool.reset();
    }
  }

  selectionToolUpdate(tool, container) {
    /*
    Called from componentDidUpdate. Update the selection tool (brush or lasso) based on the current selection.
    */
    const { selectionTool } = this.props;
    switch (selectionTool) {
      case "brush":
        this.brushToolUpdate(tool, container);
        break;
      case "lasso":
        this.lassoToolUpdate(tool);
        break;
      default:
        break;
    }
  }

  mapScreenToPoint(pin) {
    /*
    Map an (x,y) coordinate from screen space to data space,
    accounting for the current pan/zoom camera.
    */
    const { camera, projectionTF, modelInvTF, viewport } = this.state;
    const cameraInvTF = camera.invView();

    // Screen -> GL.
    const x = (2 * pin[0]) / viewport.width - 1;
    const y = 2 * (1 - pin[1] / viewport.height) - 1;

    const xy = vec2.fromValues(x, y);
    const projectionInvTF = mat3.invert(mat3.create(), projectionTF);
    vec2.transformMat3(xy, xy, projectionInvTF);
    vec2.transformMat3(xy, xy, cameraInvTF);
    vec2.transformMat3(xy, xy, modelInvTF);
    return xy;
  }

  mapPointToScreen(xyCell) {
    /*
    Map an (x,y) coordinate from data space to screen space.
    (Inverse of mapScreenToPoint)
    */
    const { camera, projectionTF, modelTF, viewport } = this.state;
    const cameraTF = camera.view();

    const xy = vec2.transformMat3(vec2.create(), xyCell, modelTF);
    vec2.transformMat3(xy, xy, cameraTF);
    vec2.transformMat3(xy, xy, projectionTF);

    return [
      Math.round(((xy[0] + 1) * viewport.width) / 2),
      Math.round(-((xy[1] + 1) / 2 - 1) * viewport.height),
    ];
  }

  updateColorTable(colors, colorDf) {
    const { annoMatrix } = this.props;
    const { schema } = annoMatrix;

    /* Update color table state */
    if (!colors || !colorDf) {
      return createColorTable(null, null, null, schema, null);
    }

    const { colorAccessor, userColors, colorMode } = colors;
    return createColorTable(
      colorMode,
      colorAccessor,
      colorDf,
      schema,
      userColors
    );
  }

  createColorByQuery(colors) {
    const { annoMatrix, genesets } = this.props;
    const { schema } = annoMatrix;
    const { colorMode, colorAccessor } = colors;
    return createColorQuery(colorMode, colorAccessor, schema, genesets);
  }

  renderPoints(
    regl,
    drawPoints,
    colorBuffer,
    pointBuffer,
    flagBuffer,
    camera,
    projectionTF
  ) {
    const { annoMatrix } = this.props;
    if (!this.reglCanvas || !annoMatrix) return;

    const { schema } = annoMatrix;
    const cameraTF = camera.view();
    const projView = mat3.multiply(mat3.create(), projectionTF, cameraTF);
    const { width, height } = this.reglCanvas;
    regl.poll();
    regl.clear({ depth: 1, color: [1, 1, 1, 1] });
    drawPoints({
      distance: camera.distance(),
      color: colorBuffer,
      position: pointBuffer,
      flag: flagBuffer,
      count: annoMatrix.nObs,
      projView,
      nPoints: schema.dataframe.nObs,
      minViewportDimension: Math.min(width, height),
    });
    regl._gl.flush();
  }

  render() {
    const {
      graphInteractionMode,
      annoMatrix,
      colors,
      layoutChoice,
      pointDilation,
      crossfilter,
    } = this.props;
    const { modelTF, projectionTF, camera, viewport, regl } = this.state;
    const cameraTF = camera?.view()?.slice();

    return (
      <div id="graph-wrapper" style={{ position: "relative", top: 0, left: 0 }}>
        <GraphOverlayLayer
          width={viewport.width}
          height={viewport.height}
          cameraTF={cameraTF}
          modelTF={modelTF}
          projectionTF={projectionTF}
          handleCanvasEvent={
            graphInteractionMode === "zoom" ? this.handleCanvasEvent : undefined
          }
        >
          <CentroidLabels />
          <HoverProteinLabels />
        </GraphOverlayLayer>

        <svg
          id="lasso-layer"
          data-testid="layout-overlay"
          className="graph-svg"
          style={{ position: "absolute", top: 0, left: 0, zIndex: 1 }}
          width={viewport.width}
          height={viewport.height}
          pointerEvents={graphInteractionMode === "select" ? "auto" : "none"}
        />
        <canvas
          width={viewport.width}
          height={viewport.height}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            padding: 0,
            margin: 0,
            shapeRendering: "crispEdges",
          }}
          className="graph-canvas"
          data-testid="layout-graph"
          ref={this.setReglCanvas}
          onMouseDown={this.handleCanvasEvent}
          onMouseUp={this.handleCanvasEvent}
          onMouseMove={this.handleCanvasEvent}
          onDoubleClick={this.handleCanvasEvent}
          onWheel={this.handleCanvasEvent}
        />

        <Async
          watchFn={Graph.watchAsync}
          promiseFn={this.fetchAsyncProps}
          watchProps={{
            annoMatrix,
            colors,
            layoutChoice,
            pointDilation,
            crossfilter,
            viewport,
          }}
        >
          <Async.Pending initial>
            <StillLoading
              displayName={layoutChoice.current}
              width={viewport.width}
              height={viewport.height}
            />
          </Async.Pending>
          <Async.Rejected>
            {(error) => (
              <ErrorLoading
                displayName={layoutChoice.current}
                error={error}
                width={viewport.width}
                height={viewport.height}
              />
            )}
          </Async.Rejected>
          <Async.Fulfilled>
            {(asyncProps) => {
              if (regl && !shallowEqual(asyncProps, this.cachedAsyncProps)) {
                this.updateReglAndRender(asyncProps, this.cachedAsyncProps);
              }
              return null;
            }}
          </Async.Fulfilled>
        </Async>
      </div>
    );
  }
}

const ErrorLoading = ({ displayName, error, width, height }) => {
  console.log(error); // Log error to console as this is an unexpected error.
  return (
    <div
      style={{
        position: "fixed",
        fontWeight: 500,
        top: height / 2,
        left: globals.leftSidebarWidth + width / 2 - 50,
      }}
    >
      <span>{`Failure loading ${displayName}`}</span>
    </div>
  );
};

const StillLoading = ({ displayName, width, height }) => (
  // Render a busy/loading indicator.
  <div
    style={{
      position: "fixed",
      fontWeight: 500,
      top: height / 2,
      width,
    }}
  >
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Button minimal loading intent="primary" />
      <span style={{ fontStyle: "italic" }}>Loading {displayName}</span>
    </div>
  </div>
);

export default Graph;
