/**
 * Minimal n-dimensional array shape used by the linker orchestrator.
 *
 * Real callers will plug in a backing buffer (Uint8Array, Float32Array, ...);
 * tests can pass plain arrays. Only `shape` and `data` are required to express
 * the operations this orchestrator performs (squeeze 3D→2D, element-wise
 * threshold, scalar max).
 */
export interface NdArray<T extends ArrayLike<number> = ArrayLike<number>> {
  readonly shape: readonly number[];
  readonly data: T;
}

/** Output of the shared preprocessing stage. Mirrors `pre.*` in linker.py. */
export interface PreprocessingResult {
  readonly cost_map: NdArray;
  readonly roi_mask: NdArray;
  readonly roi_annotation: NdArray;
  readonly roi_image: NdArray;
}

/** Result of multi-source Dijkstra expansion. */
export interface MultiSourceDijkstraResult {
  readonly owner_map: NdArray;
  readonly dist_map: NdArray;
  readonly prev_y: NdArray;
  readonly prev_x: NdArray;
}

/**
 * Opaque graph handle. The orchestrator only needs to forward graph objects
 * between stages; structural details belong to the dependency implementations.
 */
export interface Graph {
  number_of_nodes(): number;
  number_of_edges(): number;
}

/** Inter-component meeting points returned by `findMeetingPoints`. */
export type Connections = unknown;

/** Constructor options for `AnnotationGrowLinker`. Mirrors linker.py defaults. */
export interface AnnotationGrowLinkerOptions {
  // Preprocessing
  offset_px?: number;
  bg_kernel_size?: number;
  clahe_clip?: number;
  clahe_grid?: readonly [number, number];
  sato_sigmas_start?: number;
  sato_sigmas_stop?: number;
  // Dijkstra
  connectivity?: number;
  // Edge pruning
  prune_threshold?: number;
  // Subtree filtering
  min_tree_components?: number;
  stub_length_threshold?: number;
}

/** Resolved (defaulted) options used internally. */
export type ResolvedOptions = Required<AnnotationGrowLinkerOptions>;

/** Final pipeline output. Mirrors `common.data_types.LinkerResult`. */
export interface LinkerResult {
  readonly annotation: NdArray;
  readonly image: NdArray;
  readonly mask: NdArray;
  readonly graph: Graph;
  readonly valid_count: number;
}

// ── Dependency injection interfaces ─────────────────────────────────────────

export interface PreprocessingPipelineConfig {
  offset_px: number;
  bg_kernel_size: number;
  clahe_clip: number;
  clahe_grid: readonly [number, number];
  sato_sigmas_start: number;
  sato_sigmas_stop: number;
}

export interface PreprocessingPipeline {
  run(image: NdArray, mask: NdArray, annotation: NdArray): PreprocessingResult;
}

export type PreprocessingPipelineFactory = (
  config: PreprocessingPipelineConfig,
) => PreprocessingPipeline;

export type GetComponentsFn = (annotationBin: NdArray) => NdArray;

export interface MultiSourceDijkstraArgs {
  costMap: NdArray;
  annotationLabeled: NdArray;
  connectivity: number;
  roiMask: NdArray; // boolean-valued (pre.roi_mask > 127)
}

export type MultiSourceDijkstraFn = (
  args: MultiSourceDijkstraArgs,
) => MultiSourceDijkstraResult;

export type FindMeetingPointsFn = (
  ownerMap: NdArray,
  distMap: NdArray,
  prevY: NdArray,
  prevX: NdArray,
) => Connections;

export type BuildComponentGraphFn = (
  connections: Connections,
  nComponents: number,
) => Graph;

export type PruneEdgesFn = (graph: Graph, threshold: number) => Graph;

export type MinimumSpanningForestFn = (graph: Graph) => Graph;

export interface BuildResultGraphArgs {
  mst: Graph;
  annotationBin: NdArray;
  segment_length: number;
}

export type BuildResultGraphFn = (args: BuildResultGraphArgs) => Graph;

export interface RunCrossingAnalysisArgs {
  resultGraph: Graph;
  mask: NdArray; // the 2D-squeezed raw mask
  annotationLabeled: NdArray;
  min_tree_components: number;
  stub_length_threshold: number;
}

export interface CrossingAnalysisResult {
  readonly valid_count: number;
  readonly labeled_graph: Graph;
}

export type RunCrossingAnalysisFn = (
  args: RunCrossingAnalysisArgs,
) => CrossingAnalysisResult;

/** Bundle of dependencies the linker calls. */
export interface LinkerDependencies {
  createPreprocessingPipeline: PreprocessingPipelineFactory;
  getComponents: GetComponentsFn;
  multiSourceDijkstra: MultiSourceDijkstraFn;
  findMeetingPoints: FindMeetingPointsFn;
  buildComponentGraph: BuildComponentGraphFn;
  pruneEdges: PruneEdgesFn;
  minimumSpanningForest: MinimumSpanningForestFn;
  buildResultGraph: BuildResultGraphFn;
  runCrossingAnalysis: RunCrossingAnalysisFn;
}

/** Hardcoded in linker.py at the build_result_graph call site. */
export const SEGMENT_LENGTH = 500;

/** Default values mirroring the Python __init__ signature. */
export const DEFAULT_OPTIONS: ResolvedOptions = {
  offset_px: 50,
  bg_kernel_size: 51,
  clahe_clip: 20.0,
  clahe_grid: [16, 16],
  sato_sigmas_start: 3,
  sato_sigmas_stop: 8,
  connectivity: 8,
  prune_threshold: 20.0,
  min_tree_components: 5,
  stub_length_threshold: 5,
};
