const initialState = {
  isEnabled: false, // Controle se o hover está habilitado
  hoveredProtein: null, // Informações da proteína atualmente em hover
};

const proteinHover = (state = initialState, action) => {
  switch (action.type) {
    case "toggle protein hover":
      // Alterna o estado de habilitação do hover
      return {
        ...state,
        isEnabled: action.payload.enableProteinHover,
      };

    case "protein hover start":
      // Atualiza o estado com a proteína sendo hover
      return {
        ...state,
        hoveredProtein: action.payload.protein,
      };

    case "protein hover end":
      // Limpa o estado da proteína em hover
      return {
        ...state,
        hoveredProtein: null,
      };

    default:
      return state;
  }
};

export default proteinHover;
