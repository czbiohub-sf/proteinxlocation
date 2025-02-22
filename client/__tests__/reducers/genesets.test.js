import genesetsReducer from "../../src/reducers/genesets";

describe("initial reducer state", () => {
  test("some other action", () => {
    expect(genesetsReducer(undefined, { type: "foo" })).toMatchObject({
      initialized: false,
      lastTid: undefined,
      genesets: new Map(),
    });
  });
});

describe("sampleset: initial load", () => {
  test("missing JSON response", () => {
    expect(() =>
      genesetsReducer(undefined, {
        type: "sampleset: initial load",
      })
    ).toThrow("missing or malformed JSON response");
  });

  test("empty geneset", () => {
    expect(
      genesetsReducer(undefined, {
        type: "sampleset: initial load",
        data: {
          tid: 0,
          genesets: [],
        },
      })
    ).toMatchObject({
      initialized: true,
      lastTid: 0,
      genesets: new Map(),
    });
  });

  test("non-empty geneset", () => {
    expect(
      genesetsReducer(undefined, {
        type: "sampleset: initial load",
        data: {
          tid: 99,
          genesets: [
            {
              geneset_name: "G1",
              genes: [{ gene_symbol: "F5" }],
            },
            {
              geneset_name: "G2",
              geneset_description: "G2 desc",
              genes: [{ gene_symbol: "F6" }],
            },
            {
              geneset_name: "G3",
              geneset_description: "G3 desc",
              genes: [{ gene_symbol: "F7", gene_description: "gene desc" }],
            },
          ],
        },
      })
    ).toMatchObject({
      initialized: true,
      lastTid: 99,
      genesets: new Map([
        [
          "G1",
          {
            genesetName: "G1",
            genesetDescription: "",
            genes: new Map([["F5", { geneSymbol: "F5", geneDescription: "" }]]),
          },
        ],
        [
          "G2",
          {
            genesetName: "G2",
            genesetDescription: "G2 desc",
            genes: new Map([["F6", { geneSymbol: "F6", geneDescription: "" }]]),
          },
        ],
        [
          "G3",
          {
            genesetName: "G3",
            genesetDescription: "G3 desc",
            genes: new Map([
              ["F7", { geneSymbol: "F7", geneDescription: "gene desc" }],
            ]),
          },
        ],
      ]),
    });
  });
});

describe("sampleset: create", () => {
  const initialState = genesetsReducer(undefined, {
    type: "sampleset: initial load",
    data: {
      tid: 0,
      genesets: [],
    },
  });

  test("simple create", () => {
    expect(
      genesetsReducer(initialState, {
        type: "sampleset: create",
        genesetName: "a geneset",
        genesetDescription: "",
      })
    ).toMatchObject({
      ...initialState,
      genesets: new Map([
        [
          "a geneset",
          {
            genesetName: "a geneset",
            genesetDescription: "",
            genes: new Map(),
          },
        ],
      ]),
    });
  });

  test("error - duplicate name", () => {
    expect(() => {
      genesetsReducer(
        genesetsReducer(initialState, {
          type: "sampleset: create",
          genesetName: "foo",
          genesetDescription: "foo",
        }),
        {
          type: "sampleset: create",
          genesetName: "foo",
          genesetDescription: "bar",
        }
      );
    }).toThrow("name already defined");
  });

  test("error - missing required action values", () => {
    expect(() => {
      genesetsReducer(initialState, {
        type: "sampleset: create",
        genesetDescription: "foo",
      });
    }).toThrow();
    expect(() => {
      genesetsReducer(initialState, {
        type: "sampleset: create",
        genesetName: "foo",
      });
    }).toThrow("name or description unspecified");
  });
});

describe("sampleset: delete", () => {
  const initialState = genesetsReducer(undefined, {
    type: "sampleset: initial load",
    data: {
      tid: 0,
      genesets: [],
    },
  });

  test("simple delete", () => {
    expect(
      genesetsReducer(
        genesetsReducer(initialState, {
          type: "sampleset: create",
          genesetName: "foo",
          genesetDescription: "foo",
        }),
        {
          type: "sampleset: delete",
          genesetName: "foo",
        }
      )
    ).toMatchObject({
      initialized: true,
      lastTid: 0,
      genesets: new Map(),
    });
  });

  test("error - missing name", () => {
    expect(() => {
      genesetsReducer(initialState, {
        type: "sampleset: delete",
        genesetName: "foo",
      });
    }).toThrow("name does not exist");
  });
});

describe("sampleset: update", () => {
  const initialState = genesetsReducer(undefined, {
    type: "sampleset: initial load",
    data: {
      tid: 0,
      genesets: [],
    },
  });

  test("simple update", () => {
    expect(
      genesetsReducer(
        genesetsReducer(
          genesetsReducer(initialState, {
            type: "sampleset: create",
            genesetName: "foo1",
            genesetDescription: "foo1",
          }),
          {
            type: "sampleset: create",
            genesetName: "foo2",
            genesetDescription: "foo2",
          }
        ),
        {
          type: "sampleset: update",
          genesetName: "foo1",
          update: {
            genesetName: "bar",
            genesetDescription: "bar",
          },
        }
      )
    ).toMatchObject({
      initialized: true,
      lastTid: 0,
      genesets: new Map([
        [
          "bar",
          { genesetName: "bar", genesetDescription: "bar", genes: new Map() },
        ],
        [
          "foo2",
          { genesetName: "foo2", genesetDescription: "foo2", genes: new Map() },
        ],
      ]),
    });
  });

  test("error - unknown name", () => {
    expect(() => {
      genesetsReducer(initialState, {
        type: "sampleset: update",
        genesetName: "foo",
        update: {
          genesetName: "foo",
          genesetDescription: "bar",
        },
      });
    }).toThrow("name unspecified or does not exist");
  });

  test("error - duplicate name", () => {
    expect(() => {
      genesetsReducer(
        genesetsReducer(initialState, {
          type: "sampleset: create",
          genesetName: "foo",
          genesetDescription: "foo",
        }),
        {
          type: "sampleset: update",
          genesetName: "foo",
          update: {
            genesetName: "foo",
            genesetDescription: "foo",
          },
        }
      );
    }).toThrow("update specified existing name and description");
  });
});

describe("sampleset: add samples", () => {
  const initialState = genesetsReducer(
    genesetsReducer(undefined, {
      type: "sampleset: initial load",
      data: {
        tid: 0,
        genesets: [],
      },
    }),
    {
      type: "sampleset: create",
      genesetName: "test",
      genesetDescription: "",
    }
  );

  test("add a gene", () => {
    expect(
      genesetsReducer(initialState, {
        type: "sampleset: add samples",
        genesetName: "test",
        genes: [{ geneSymbol: "F5" }],
      })
    ).toMatchObject({
      ...initialState,
      genesets: new Map([
        [
          "test",
          {
            genesetName: "test",
            genesetDescription: "",
            genes: new Map([["F5", { geneSymbol: "F5", geneDescription: "" }]]),
          },
        ],
      ]),
    });

    expect(
      genesetsReducer(initialState, {
        type: "sampleset: add samples",
        genesetName: "test",
        genes: [
          { geneSymbol: "F5", geneDescription: "desc" },
          { geneSymbol: "SET1", geneDescription: "" },
        ],
      })
    ).toMatchObject({
      ...initialState,
      genesets: new Map([
        [
          "test",
          {
            genesetName: "test",
            genesetDescription: "",
            genes: new Map([
              ["F5", { geneSymbol: "F5", geneDescription: "desc" }],
              ["SET1", { geneSymbol: "SET1", geneDescription: "" }],
            ]),
          },
        ],
      ]),
    });
  });

  test("no such geneset error", () => {
    expect(() => {
      genesetsReducer(initialState, {
        type: "sampleset: add samples",
        genesetName: "mumble",
        genes: [],
      });
    }).toThrow("geneset name does not exist");
  });
});

describe("sampleset: delete samples", () => {
  const initialState = genesetsReducer(
    genesetsReducer(
      genesetsReducer(undefined, {
        type: "sampleset: initial load",
        data: {
          tid: 0,
          genesets: [],
        },
      }),
      {
        type: "sampleset: create",
        genesetName: "test",
        genesetDescription: "",
      }
    ),
    {
      type: "sampleset: add samples",
      genesetName: "test",
      genes: [{ geneSymbol: "F5" }],
    }
  );

  test("simple", () => {
    expect(
      genesetsReducer(initialState, {
        type: "sampleset: delete samples",
        genesetName: "test",
        geneSymbols: ["F5"],
      })
    ).toMatchObject({
      ...initialState,
      genesets: new Map([
        [
          "test",
          {
            genesetName: "test",
            genesetDescription: "",
            genes: new Map(),
          },
        ],
      ]),
    });
  });

  test("no such geneset error", () => {
    expect(() => {
      genesetsReducer(initialState, {
        type: "sampleset: delete samples",
        genesetName: "mumble",
        geneSymbols: [],
      });
    }).toThrow("name does not exist");
  });
});

describe("sampleset: set sample description", () => {
  const initialState = genesetsReducer(
    genesetsReducer(
      genesetsReducer(undefined, {
        type: "sampleset: initial load",
        data: {
          tid: 0,
          genesets: [],
        },
      }),
      {
        type: "sampleset: create",
        genesetName: "test",
        genesetDescription: "",
      }
    ),
    {
      type: "sampleset: add samples",
      genesetName: "test",
      genes: [{ geneSymbol: "F5" }],
    }
  );

  test("simple set", () => {
    expect(
      genesetsReducer(initialState, {
        type: "sampleset: set sample description",
        genesetName: "test",
        update: {
          geneSymbol: "F5",
          geneDescription: "mumble",
        },
      })
    ).toMatchObject({
      ...initialState,
      genesets: new Map([
        [
          "test",
          {
            genesetName: "test",
            genesetDescription: "",
            genes: new Map([
              ["F5", { geneSymbol: "F5", geneDescription: "mumble" }],
            ]),
          },
        ],
      ]),
    });
  });

  test("no such geneset error", () => {
    expect(() => {
      genesetsReducer(initialState, {
        type: "sampleset: set sample description",
        genesetName: "does not exist",
        update: {
          geneSymbol: "F5",
          geneDescription: "mumble",
        },
      });
    }).toThrow("geneset name does not exist");
  });

  test("no such gene error", () => {
    expect(() => {
      genesetsReducer(initialState, {
        type: "sampleset: set sample description",
        genesetName: "test",
        update: {
          geneSymbol: "NO SUCH GENE",
          geneDescription: "mumble",
        },
      });
    }).toThrow("no such gene");
  });
});

describe("sampleset: set tid", () => {
  test("simple set", () => {
    expect(
      genesetsReducer(undefined, {
        type: "sampleset: set tid",
        tid: 1,
      })
    ).toMatchObject({ lastTid: 1 });
  });

  test("not a number error", () => {
    expect(() => {
      genesetsReducer(
        { lastTid: 1 },
        {
          type: "sampleset: set tid",
          tid: "0",
        }
      );
    }).toThrow("must be a positive integer");
  });

  test("decrement error", () => {
    expect(() => {
      genesetsReducer(
        { lastTid: 1 },
        {
          type: "sampleset: set tid",
          tid: 0,
        }
      );
    }).toThrow("may not be decremented");
  });
});
