import React from "react";
import { connect } from "react-redux";
import { vec2 } from "gl-matrix";

const ProteinLabels = ({ hoveredProtein, points, projectionTF }) => {
  console.log("hoveredProtein =>> ", hoveredProtein);
  if (!hoveredProtein || !points || !projectionTF) return null;

  const labelPositions = points
    .filter((point) => point.protein === hoveredProtein)
    .map((point) => {
      const [x, y] = vec2.transformMat3([], [point.x, point.y], projectionTF);
      return { x, y, name: point.protein };
    });

  return (
    <g>
      {labelPositions.map(({ x, y, name }) => (
        <text
          key={name} // Usando o nome da proteína como chave, assumindo que é único
          x={x}
          y={y}
          textAnchor="middle"
          style={{
            fontSize: "12px",
            fontWeight: "bold",
            fill: "black",
            userSelect: "none",
          }}
        >
          {name}
        </text>
      ))}
    </g>
  );
};

export default connect((state) => ({
  hoveredProtein: state.proteinHover.hoveredProtein,
  projectionTF: state.layoutChoice.projectionTF, // Assumindo que esta transformação existe no estado
}))(ProteinLabels);
