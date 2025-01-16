import React from "react";
import { connect } from "react-redux";
import { vec2, mat3 } from "gl-matrix";

/**
 * HoverProteinLabels Component
 *
 * Displays a label near the hovered protein's position on the graph.
 * It transforms the protein coordinates from data space to screen space
 * and applies the appropriate transformations for correct positioning.
 *
 * @param {Object} hoveredProtein - The currently hovered protein data.
 * @param {mat3} projectionTF - The transformation matrix for projection.
 * @param {string} inverseTransform - The inverse transformation for text positioning.
 */
const HoverProteinLabels = ({
  hoveredProtein,
  projectionTF,
  inverseTransform,
}) => {
  if (!hoveredProtein || !hoveredProtein.coordinates || !projectionTF) {
    return <g id="hover-protein-label-placeholder" />;
  }

  const { name, coordinates } = hoveredProtein;

  // Validate if `coordinates` is a valid array
  if (!Array.isArray(coordinates) || coordinates.length !== 2) {
    console.error("🔴 Invalid coordinates for hover:", coordinates);
    return <g id="hover-protein-label-placeholder" />;
  }

  // Transform coordinates to screen space
  const screenCoords = vec2.create();
  vec2.transformMat3(screenCoords, coordinates, projectionTF);
  const [x, y] = screenCoords;

  return (
    <g id="hover-protein-label" transform={`translate(${x}, ${y})`}>
      <text
        transform={`scale(0.005, -0.005) ${inverseTransform}`}
        textAnchor="middle"
        dy="-1" // Moves the text slightly above the point
        style={{
          fontSize: "2px",
          fontWeight: "bold",
          fill: "black",
          userSelect: "none",
        }}
        pointerEvents="none"
      >
        {name}
      </text>
    </g>
  );
};

const mapStateToProps = (state) => ({
  hoveredProtein: state.proteinHover?.hoveredProtein || null,
  projectionTF: state.graph?.projectionTF || mat3.create(),
  inverseTransform: state.graph?.inverseTransform || "",
});

export default connect(mapStateToProps)(HoverProteinLabels);
