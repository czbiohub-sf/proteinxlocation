import React from "react";
import fuzzysort from "fuzzysort";
import { connect } from "react-redux";
import { Suggest } from "@blueprintjs/select";
import {
  Button,
  ControlGroup,
  FormGroup,
  InputGroup,
  Intent,
  MenuItem,
} from "@blueprintjs/core";
import * as globals from "../../../globals";
import actions from "../../../actions";
import {
  postUserErrorToast,
  keepAroundErrorToast,
} from "../../framework/toasters";

import { memoize } from "../../../util/dataframe/util";
import parseBulkGeneString from "../../../util/parseBulkGeneString";

const renderGene = (fuzzySortResult, { handleClick, modifiers }) => {
  if (!modifiers.matchesPredicate) {
    return null;
  }
  /* the fuzzysort wraps the object with other properties, like a score */
  const geneName = fuzzySortResult.target;

  return (
    <MenuItem
      active={modifiers.active}
      disabled={modifiers.disabled}
      data-testid={`suggest-menu-item-${geneName}`}
      key={geneName}
      onClick={(g) =>
        /* this fires when user clicks a menu item */
        handleClick(g)
      }
      text={geneName}
    />
  );
};

const filterGenes = (query, genes) =>
  /* fires on load, once, and then for each character typed into the input */
  fuzzysort.go(query, genes, {
    limit: 5,
    threshold: -10000, // don't return bad results
  });

@connect((state) => ({
  annoMatrix: state.annoMatrix,
  userDefinedGenes: state.controls.userDefinedGenes,
  userDefinedGenesLoading: state.controls.userDefinedGenesLoading,
}))
class AddGenes extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      bulkAdd: "",
      tab: "autosuggest",
      activeItem: null,
      geneNames: [],
      status: "pending",
    };
  }

  componentDidMount() {
    this.updateState();
  }

  componentDidUpdate(prevProps) {
    this.updateState(prevProps);
  }

  handleClick(g) {
    const { dispatch, userDefinedGenes } = this.props;
    const { geneNames } = this.state;
    if (!g) return;
    const gene = g.target;
    if (userDefinedGenes.indexOf(gene) !== -1) {
      postUserErrorToast("That sample already exists");
    } else if (userDefinedGenes.length > globals.maxUserDefinedGenes) {
      postUserErrorToast(
        `That's too many samples, you can have at most ${globals.maxUserDefinedGenes} user defined samples`
      );
    } else if (geneNames.indexOf(gene) === undefined) {
      postUserErrorToast("That doesn't appear to be a valid sample name.");
    } else {
      dispatch({ type: "single user defined sample start" });
      dispatch(actions.requestUserDefinedGene(gene));
      dispatch({ type: "single user defined sample complete" });
    }
  }

  _genesToUpper = (listGenes) => {
    // Has to be a Map to preserve index
    const upperGenes = new Map();
    for (let i = 0, { length } = listGenes; i < length; i += 1) {
      upperGenes.set(listGenes[i].toUpperCase(), i);
    }

    return upperGenes;
  };

  // eslint-disable-next-line react/sort-comp -- memo requires a defined _genesToUpper
  _memoGenesToUpper = memoize(this._genesToUpper, (arr) => arr);

  handleBulkAddClick = () => {
    const { dispatch, userDefinedGenes } = this.props;
    const { bulkAdd, geneNames } = this.state;

    /*
      test:
      Apod,,, Cd74,,    ,,,    Foo,    Bar-2,,
    */
    if (bulkAdd !== "") {
      const genes = parseBulkGeneString(bulkAdd);
      if (genes.length === 0) {
        return keepAroundErrorToast("Must enter a sample name.");
      }
      // These gene lists are unique enough where memoization is useless
      const upperGenes = this._genesToUpper(genes);
      const upperUserDefinedGenes = this._genesToUpper(userDefinedGenes);

      const upperGeneNames = this._memoGenesToUpper(geneNames);

      dispatch({ type: "bulk user defined sample start" });

      Promise.all(
        [...upperGenes.keys()].map((upperGene) => {
          if (upperUserDefinedGenes.get(upperGene) !== undefined) {
            return keepAroundErrorToast("That sample already exists");
          }

          const indexOfGene = upperGeneNames.get(upperGene);

          if (indexOfGene === undefined) {
            return keepAroundErrorToast(
              `${
                genes[upperGenes.get(upperGene)]
              } doesn't appear to be a valid sample name.`
            );
          }
          return dispatch(
            actions.requestUserDefinedGene(geneNames[indexOfGene])
          );
        })
      ).then(
        () => dispatch({ type: "bulk user defined sample complete" }),
        () => dispatch({ type: "bulk user defined sample error" })
      );
    }

    this.setState({ bulkAdd: "" });
    return undefined;
  };

  async updateState(prevProps) {
    const { annoMatrix } = this.props;
    if (!annoMatrix) return;
    if (annoMatrix !== prevProps?.annoMatrix) {
      const { schema } = annoMatrix;
      const varIndex = schema.annotations.var.index;

      this.setState({ status: "pending" });
      try {
        const df = await annoMatrix.fetch("var", varIndex);
        this.setState({
          status: "success",
          geneNames: df.col(varIndex).asArray(),
        });
      } catch (error) {
        this.setState({ status: "error" });
        throw error;
      }
    }
  }

  placeholderGeneNames() {
    /*
    return a string containing sample name suggestions for use as a user hint.
    Eg.,    Apod, Cd74, ...
    Will return a max of 3 samples, totalling 15 characters in length.
    Randomly selects sample names.

    NOTE: the random selection means it will re-render constantly.
    */
    const { geneNames } = this.state;
    if (geneNames.length > 0) {
      const placeholder = [];
      let len = geneNames.length;
      const maxGeneNameCount = 3;
      const maxStrLength = 15;
      len = len < maxGeneNameCount ? len : maxGeneNameCount;
      for (let i = 0, strLen = 0; i < len && strLen < maxStrLength; i += 1) {
        const deal = Math.floor(Math.random() * geneNames.length);
        const geneName = geneNames[deal];
        placeholder.push(geneName);
        strLen += geneName.length + 2; // '2' is the length of a comma and space
      }
      placeholder.push("...");
      return placeholder.join(", ");
    }
    // default - should never happen.
    return "Apod, Cd74, ...";
  }

  render() {
    const { userDefinedGenesLoading } = this.props;
    const { tab, bulkAdd, activeItem, status, geneNames } = this.state;

    // may still be loading!
    if (status !== "success") return null;

    return (
      <div>
        <div
          style={{
            padding: globals.leftSidebarSectionPadding,
          }}
        >
          <Button
            active={tab === "autosuggest"}
            style={{ marginRight: 5 }}
            minimal
            small
            data-testid="tab-autosuggest"
            onClick={() => {
              this.setState({ tab: "autosuggest" });
            }}
          >
            Autosuggest samples
          </Button>
          <Button
            active={tab === "bulkadd"}
            minimal
            small
            data-testid="section-bulk-add"
            onClick={() => {
              this.setState({ tab: "bulkadd" });
            }}
          >
            Bulk add samples
          </Button>
        </div>
        {tab === "autosuggest" ? (
          <ControlGroup
            style={{
              paddingLeft: globals.leftSidebarSectionPadding,
              paddingBottom: globals.leftSidebarSectionPadding,
            }}
          >
            <Suggest
              resetOnSelect
              closeOnSelect
              resetOnClose
              itemDisabled={userDefinedGenesLoading ? () => true : () => false}
              noResults={<MenuItem disabled text="No matching samples." />}
              onItemSelect={(g) => {
                /* this happens on 'enter' */
                this.handleClick(g);
              }}
              initialContent={<MenuItem disabled text="Enter a sample…" />}
              inputProps={{ "data-testid": "gene-search" }}
              inputValueRenderer={() => ""}
              itemListPredicate={filterGenes}
              onActiveItemChange={(item) => this.setState({ activeItem: item })}
              itemRenderer={renderGene}
              items={geneNames || ["No samples"]}
              popoverProps={{ minimal: true }}
            />
            <Button
              intent={Intent.PRIMARY}
              data-testid="add-gene"
              loading={userDefinedGenesLoading}
              onClick={() => this.handleClick(activeItem)}
            >
              Add sample
            </Button>
          </ControlGroup>
        ) : null}
        {tab === "bulkadd" ? (
          <div style={{ paddingLeft: globals.leftSidebarSectionPadding }}>
            <form
              onSubmit={(e) => {
                e.preventDefault();
                this.handleBulkAddClick();
              }}
            >
              <FormGroup
                helperText="Add a list of samples (comma delimited)"
                labelFor="text-input-bulk-add"
              >
                <ControlGroup>
                  <InputGroup
                    onChange={(e) => {
                      this.setState({ bulkAdd: e.target.value });
                    }}
                    id="text-input-bulk-add"
                    data-testid="input-bulk-add"
                    placeholder={this.placeholderGeneNames()}
                    value={bulkAdd}
                  />
                  <Button
                    intent="primary"
                    onClick={this.handleBulkAddClick}
                    loading={userDefinedGenesLoading}
                  >
                    Add samples
                  </Button>
                </ControlGroup>
              </FormGroup>
            </form>
          </div>
        ) : null}
      </div>
    );
  }
}

export default AddGenes;
