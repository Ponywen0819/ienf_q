import { beforeEach, describe, expect, it, vi } from "vitest";

import { AnnotationGrowLinker } from "../src/linker.js";
import { DEFAULT_OPTIONS, SEGMENT_LENGTH } from "../src/types.js";
import type {
  Connections,
  CrossingAnalysisResult,
  Graph,
  LinkerDependencies,
  MultiSourceDijkstraResult,
  NdArray,
  PreprocessingPipeline,
  PreprocessingPipelineConfig,
  PreprocessingResult,
} from "../src/types.js";

// ── Helpers to build stub objects ───────────────────────────────────────────

const arr = (shape: number[], data: number[]): NdArray => ({ shape, data });

function mkGraph(id: string, nodes = 0, edges = 0): Graph {
  return {
    number_of_nodes: () => nodes,
    number_of_edges: () => edges,
    // helper field for identity assertions in tests
    __id: id,
  } as Graph & { __id: string };
}

function makePreprocessingResult(): PreprocessingResult {
  return {
    cost_map: arr([2, 2], [0.1, 0.2, 0.3, 0.4]),
    roi_mask: arr([2, 2], [200, 0, 200, 0]),
    roi_annotation: arr([2, 2], [0, 200, 200, 50]),
    roi_image: arr([2, 2], [10, 20, 30, 40]),
  };
}

interface CallLog {
  order: string[];
  preprocessingConfig?: PreprocessingPipelineConfig;
  preprocessing?: { image: NdArray; mask: NdArray; annotation: NdArray };
  getComponents?: { annotationBin: NdArray };
  dijkstra?: {
    costMap: NdArray;
    annotationLabeled: NdArray;
    connectivity: number;
    roiMask: NdArray;
  };
  meetingPoints?: { owner: NdArray; dist: NdArray; py: NdArray; px: NdArray };
  buildComponentGraph?: { connections: Connections; nComponents: number };
  pruneEdges?: { graph: Graph; threshold: number };
  mst?: { graph: Graph };
  buildResultGraph?: { mst: Graph; annotationBin: NdArray; segment_length: number };
  crossingAnalysis?: {
    resultGraph: Graph;
    mask: NdArray;
    annotationLabeled: NdArray;
    min_tree_components: number;
    stub_length_threshold: number;
  };
}

function makeDeps(): { deps: LinkerDependencies; log: CallLog; pre: PreprocessingResult; labeled: NdArray; dijkstra: MultiSourceDijkstraResult; connections: Connections; graphs: Record<string, Graph>; crossing: CrossingAnalysisResult; } {
  const log: CallLog = { order: [] };
  const pre = makePreprocessingResult();
  const labeled = arr([2, 2], [1, 2, 0, 3]); // max=3 → nComponents=3
  const dijkstra: MultiSourceDijkstraResult = {
    owner_map: arr([2, 2], [1, 1, 2, 2]),
    dist_map: arr([2, 2], [0, 1, 1, 2]),
    prev_y: arr([2, 2], [0, 0, 1, 1]),
    prev_x: arr([2, 2], [0, 0, 1, 1]),
  };
  const connections: Connections = { kind: "connections" };
  const graphs = {
    component: mkGraph("component-graph", 3, 5),
    pruned: mkGraph("pruned-graph", 3, 4),
    mst: mkGraph("mst-graph", 3, 2),
    result: mkGraph("result-graph", 10, 9),
    labeled: mkGraph("labeled-graph", 10, 9),
  };
  const crossing: CrossingAnalysisResult = {
    valid_count: 7,
    labeled_graph: graphs.labeled,
  };

  const pipeline: PreprocessingPipeline = {
    run: vi.fn((image, mask, annotation) => {
      log.order.push("preprocessing.run");
      log.preprocessing = { image, mask, annotation };
      return pre;
    }),
  };

  const deps: LinkerDependencies = {
    createPreprocessingPipeline: vi.fn((config) => {
      log.order.push("createPreprocessingPipeline");
      log.preprocessingConfig = config;
      return pipeline;
    }),
    getComponents: vi.fn((annotationBin) => {
      log.order.push("getComponents");
      log.getComponents = { annotationBin };
      return labeled;
    }),
    multiSourceDijkstra: vi.fn((args) => {
      log.order.push("multiSourceDijkstra");
      log.dijkstra = args;
      return dijkstra;
    }),
    findMeetingPoints: vi.fn((owner, dist, py, px) => {
      log.order.push("findMeetingPoints");
      log.meetingPoints = { owner, dist, py, px };
      return connections;
    }),
    buildComponentGraph: vi.fn((connections, nComponents) => {
      log.order.push("buildComponentGraph");
      log.buildComponentGraph = { connections, nComponents };
      return graphs.component;
    }),
    pruneEdges: vi.fn((graph, threshold) => {
      log.order.push("pruneEdges");
      log.pruneEdges = { graph, threshold };
      return graphs.pruned;
    }),
    minimumSpanningForest: vi.fn((graph) => {
      log.order.push("minimumSpanningForest");
      log.mst = { graph };
      return graphs.mst;
    }),
    buildResultGraph: vi.fn((args) => {
      log.order.push("buildResultGraph");
      log.buildResultGraph = args;
      return graphs.result;
    }),
    runCrossingAnalysis: vi.fn((args) => {
      log.order.push("runCrossingAnalysis");
      log.crossingAnalysis = args;
      return crossing;
    }),
  };

  return { deps, log, pre, labeled, dijkstra, connections, graphs, crossing };
}

// ── Tests ───────────────────────────────────────────────────────────────────

describe("AnnotationGrowLinker constructor", () => {
  it("stores all 10 defaults matching the Python __init__", () => {
    const { deps } = makeDeps();
    const linker = new AnnotationGrowLinker(deps);
    expect(linker.options).toEqual(DEFAULT_OPTIONS);
    expect(linker.options.offset_px).toBe(50);
    expect(linker.options.bg_kernel_size).toBe(51);
    expect(linker.options.clahe_clip).toBe(20.0);
    expect(linker.options.clahe_grid).toEqual([16, 16]);
    expect(linker.options.sato_sigmas_start).toBe(3);
    expect(linker.options.sato_sigmas_stop).toBe(8);
    expect(linker.options.connectivity).toBe(8);
    expect(linker.options.prune_threshold).toBe(20.0);
    expect(linker.options.min_tree_components).toBe(5);
    expect(linker.options.stub_length_threshold).toBe(5);
  });

  it("overrides defaults with provided options", () => {
    const { deps } = makeDeps();
    const linker = new AnnotationGrowLinker(deps, {
      offset_px: 100,
      prune_threshold: 5.5,
      clahe_grid: [8, 8],
    });
    expect(linker.options.offset_px).toBe(100);
    expect(linker.options.prune_threshold).toBe(5.5);
    expect(linker.options.clahe_grid).toEqual([8, 8]);
    // untouched defaults still apply
    expect(linker.options.bg_kernel_size).toBe(51);
  });
});

describe("AnnotationGrowLinker.run — orchestration", () => {
  let fixture: ReturnType<typeof makeDeps>;
  let linker: AnnotationGrowLinker;

  beforeEach(() => {
    fixture = makeDeps();
    linker = new AnnotationGrowLinker(fixture.deps);
  });

  it("calls dependencies in the exact order of linker.py", () => {
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 200, 0, 200]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);
    linker.run(image, mask, annotation);

    expect(fixture.log.order).toEqual([
      "createPreprocessingPipeline",
      "preprocessing.run",
      "getComponents",
      "multiSourceDijkstra",
      "findMeetingPoints",
      "buildComponentGraph",
      "pruneEdges",
      "minimumSpanningForest",
      "buildResultGraph",
      "runCrossingAnalysis",
    ]);
  });

  it("squeezes a 3D mask (H, W, 3) to (H, W) before preprocessing", () => {
    const image = arr([2, 2, 3], [1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12]);
    // 2x2x3 mask, channel 0 = [100, 200, 50, 150]
    const mask3d = arr([2, 2, 3], [100, 9, 9,  200, 9, 9,  50, 9, 9,  150, 9, 9]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    linker.run(image, mask3d, annotation);

    expect(fixture.log.preprocessing).toBeDefined();
    const passedMask = fixture.log.preprocessing!.mask;
    expect(Array.from(passedMask.shape)).toEqual([2, 2]);
    expect(Array.from(passedMask.data)).toEqual([100, 200, 50, 150]);
  });

  it("passes the SAME squeezed 2D mask to runCrossingAnalysis (not the roi_mask)", () => {
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask3d = arr([2, 2, 3], [100, 9, 9,  200, 9, 9,  50, 9, 9,  150, 9, 9]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    linker.run(image, mask3d, annotation);

    const passedToPreprocessing = fixture.log.preprocessing!.mask;
    const passedToCrossing = fixture.log.crossingAnalysis!.mask;
    expect(passedToCrossing).toBe(passedToPreprocessing);
    // confirm it's NOT pre.roi_mask
    expect(passedToCrossing).not.toBe(fixture.pre.roi_mask);
  });

  it("passes 2D mask unchanged to runCrossingAnalysis when input is already 2D", () => {
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask2d = arr([2, 2], [10, 20, 30, 40]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    linker.run(image, mask2d, annotation);

    expect(fixture.log.crossingAnalysis!.mask).toBe(mask2d);
  });

  it("builds annotation_bin via `> 127`, producing 0/1 values", () => {
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    // values around the 127 boundary
    const annotation = arr([2, 2], [0, 127, 128, 255]);

    // override the preprocessing result so roi_annotation echoes our input
    (fixture.pre as { roi_annotation: NdArray }).roi_annotation = annotation;

    linker.run(image, mask, annotation);

    const annotationBin = fixture.log.getComponents!.annotationBin;
    expect(Array.from(annotationBin.data)).toEqual([0, 0, 1, 1]);
  });

  it("passes a boolean (`> 127`) roi_mask to multiSourceDijkstra", () => {
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);
    (fixture.pre as { roi_mask: NdArray }).roi_mask = arr(
      [2, 2],
      [0, 127, 128, 255],
    );

    linker.run(image, mask, annotation);

    const roiMask = fixture.log.dijkstra!.roiMask;
    expect(Array.from(roiMask.data)).toEqual([0, 0, 1, 1]);
  });

  it("forwards n_components = int(annot_labeled.max())", () => {
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    linker.run(image, mask, annotation);

    expect(fixture.log.buildComponentGraph!.nComponents).toBe(3);
  });

  it("forwards prune_threshold from options to pruneEdges", () => {
    const customLinker = new AnnotationGrowLinker(fixture.deps, {
      prune_threshold: 12.5,
    });
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    customLinker.run(image, mask, annotation);

    expect(fixture.log.pruneEdges!.threshold).toBe(12.5);
  });

  it("forwards connectivity from options to multiSourceDijkstra", () => {
    const customLinker = new AnnotationGrowLinker(fixture.deps, {
      connectivity: 4,
    });
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    customLinker.run(image, mask, annotation);

    expect(fixture.log.dijkstra!.connectivity).toBe(4);
  });

  it("passes segment_length=500 (hardcoded) and the binarized annotation to buildResultGraph", () => {
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    linker.run(image, mask, annotation);

    expect(fixture.log.buildResultGraph!.segment_length).toBe(500);
    expect(SEGMENT_LENGTH).toBe(500);
    // annotation_bin (same identity as the one handed to getComponents)
    expect(fixture.log.buildResultGraph!.annotationBin).toBe(
      fixture.log.getComponents!.annotationBin,
    );
  });

  it("forwards min_tree_components and stub_length_threshold to runCrossingAnalysis", () => {
    const customLinker = new AnnotationGrowLinker(fixture.deps, {
      min_tree_components: 9,
      stub_length_threshold: 11,
    });
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    customLinker.run(image, mask, annotation);

    expect(fixture.log.crossingAnalysis!.min_tree_components).toBe(9);
    expect(fixture.log.crossingAnalysis!.stub_length_threshold).toBe(11);
  });

  it("forwards the preprocessing config from options to the pipeline factory", () => {
    const customLinker = new AnnotationGrowLinker(fixture.deps, {
      offset_px: 80,
      bg_kernel_size: 31,
      clahe_clip: 5.5,
      clahe_grid: [4, 4],
      sato_sigmas_start: 1,
      sato_sigmas_stop: 6,
    });
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    customLinker.run(image, mask, annotation);

    expect(fixture.log.preprocessingConfig).toEqual({
      offset_px: 80,
      bg_kernel_size: 31,
      clahe_clip: 5.5,
      clahe_grid: [4, 4],
      sato_sigmas_start: 1,
      sato_sigmas_stop: 6,
    });
  });

  it("assembles LinkerResult from the correct sources", () => {
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    const result = linker.run(image, mask, annotation);

    expect(result.annotation).toBe(fixture.pre.roi_annotation);
    expect(result.image).toBe(fixture.pre.roi_image);
    expect(result.mask).toBe(fixture.pre.roi_mask);
    expect(result.graph).toBe(fixture.graphs.labeled);
    expect(result.valid_count).toBe(7);
  });

  it("routes the pruned graph (not the original) into the MST stage", () => {
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    linker.run(image, mask, annotation);

    expect(fixture.log.pruneEdges!.graph).toBe(fixture.graphs.component);
    expect(fixture.log.mst!.graph).toBe(fixture.graphs.pruned);
    expect(fixture.log.buildResultGraph!.mst).toBe(fixture.graphs.mst);
    expect(fixture.log.crossingAnalysis!.resultGraph).toBe(fixture.graphs.result);
  });

  it("passes the Dijkstra outputs to findMeetingPoints in (owner, dist, prev_y, prev_x) order", () => {
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    linker.run(image, mask, annotation);

    expect(fixture.log.meetingPoints!.owner).toBe(fixture.dijkstra.owner_map);
    expect(fixture.log.meetingPoints!.dist).toBe(fixture.dijkstra.dist_map);
    expect(fixture.log.meetingPoints!.py).toBe(fixture.dijkstra.prev_y);
    expect(fixture.log.meetingPoints!.px).toBe(fixture.dijkstra.prev_x);
  });

  it("hands the labeled-component map (not the binary annotation) to dijkstra and crossing analysis", () => {
    const image = arr([2, 2], [1, 2, 3, 4]);
    const mask = arr([2, 2], [0, 0, 0, 0]);
    const annotation = arr([2, 2], [0, 130, 0, 130]);

    linker.run(image, mask, annotation);

    expect(fixture.log.dijkstra!.annotationLabeled).toBe(fixture.labeled);
    expect(fixture.log.crossingAnalysis!.annotationLabeled).toBe(fixture.labeled);
  });
});
