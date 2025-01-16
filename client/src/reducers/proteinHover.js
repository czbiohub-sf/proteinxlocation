/**
 * Redux reducer for managing protein hover state.
 *
 * This reducer handles the hover interactions with proteins on the graph,
 * enabling hover mode, tracking the currently hovered protein, and resetting state when necessary.
 *
 * @param {Object} state - The current state of the hover interaction.
 * @param {boolean} state.isEnabled - Indicates whether hover detection is enabled.
 * @param {Object|null} state.hoveredProtein - The currently hovered protein, containing name and coordinates.
 * @param {Object} action - The dispatched Redux action.
 * @param {string} action.type - The type of action being handled.
 * @param {Object} [action.payload] - Additional data associated with the action.
 * @returns {Object} The updated state based on the dispatched action.
 */

const initialState = {
  isEnabled: false, // Indicates whether hover detection is enabled
  hoveredProtein: null, // Stores the currently hovered protein's name and coordinates
};

const proteinHover = (state = initialState, action) => {
  switch (action.type) {
    /**
     * Toggles the hover detection feature.
     * @param {boolean} action.payload.enableProteinHover - Enables or disables protein hover detection.
     */
    case "toggle protein hover":
      return {
        ...state,
        isEnabled: action.payload.enableProteinHover,
      };

    /**
     * Sets the currently hovered protein.
     * @param {string} action.payload.protein - Name of the hovered protein.
     * @param {Array<number>} action.payload.coordinates - The (x, y) coordinates of the hovered protein.
     */
    case "protein hover start":
      return {
        ...state,
        hoveredProtein: {
          name: action.payload.protein,
          coordinates: action.payload.coordinates,
        },
      };

    /**
     * Clears the hovered protein when the hover interaction ends.
     */
    case "protein hover end":
      return {
        ...state,
        hoveredProtein: null,
      };

    default:
      return state;
  }
};

export default proteinHover;
