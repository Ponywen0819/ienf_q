/**
 * AnnotationGrowLinker — TypeScript port of the orchestrator at
 * `src/neural_reconstruction/algorithms/annotation_grow/linker.py`.
 *
 * Scope: this port mirrors the orchestration only. The heavy dependencies
 * (preprocessing pipeline, multi-source Dijkstra, MST, skeletonization,
 * crossing analysis) are typed as injected interfaces and not reimplemented
 * here — see `types.ts`.
 *
 * Algorithm (unchanged from the Python source):
 *   1. Preprocess: green channel → background removal → CLAHE → Sato → cost map
 *   2. Dijkstra:   multi-source expansion with per-component adaptive stopping
 *   3. Meeting:    find minimum-cost touching point for each component pair
 *   4. Graph:      build component graph → prune high-cost edges → MST
 *   5. Skeleton:   per-CC dist-threshold corridor → skeletonize → pixel graph
 *   6. Crossing:   subtree validation against the epidermis mask
 */

import {
  binarizeToBool,
  binarizeToUint8,
  maxValue,
  squeezeFirstChannel,
} from "./array_utils.js";
import {
  DEFAULT_OPTIONS,
  SEGMENT_LENGTH,
  type AnnotationGrowLinkerOptions,
  type LinkerDependencies,
  type LinkerResult,
  type NdArray,
  type ResolvedOptions,
} from "./types.js";

export class AnnotationGrowLinker {
  readonly options: ResolvedOptions;
  private readonly deps: LinkerDependencies;

  constructor(deps: LinkerDependencies, options: AnnotationGrowLinkerOptions = {}) {
    this.deps = deps;
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  run(image: NdArray, mask: NdArray, annotation: NdArray): LinkerResult {
    // ── 1. Shared preprocessing ────────────────────────────────────────────
    // Keep a 2-D copy of the raw epidermis mask for crossing analysis.
    const mask2d = squeezeFirstChannel(mask);

    const pre = this.deps
      .createPreprocessingPipeline({
        offset_px: this.options.offset_px,
        bg_kernel_size: this.options.bg_kernel_size,
        clahe_clip: this.options.clahe_clip,
        clahe_grid: this.options.clahe_grid,
        sato_sigmas_start: this.options.sato_sigmas_start,
        sato_sigmas_stop: this.options.sato_sigmas_stop,
      })
      .run(image, mask2d, annotation);

    // ── 2. Connected components ────────────────────────────────────────────
    const annotationBin = binarizeToUint8(pre.roi_annotation, 127);
    const annotLabeled = this.deps.getComponents(annotationBin);
    const nComponents = Math.trunc(maxValue(annotLabeled));

    // ── 3. Multi-source Dijkstra ───────────────────────────────────────────
    const { owner_map, dist_map, prev_y, prev_x } = this.deps.multiSourceDijkstra({
      costMap: pre.cost_map,
      annotationLabeled: annotLabeled,
      connectivity: this.options.connectivity,
      roiMask: binarizeToBool(pre.roi_mask, 127),
    });

    // ── 4. Meeting points → component graph ────────────────────────────────
    const connections = this.deps.findMeetingPoints(
      owner_map,
      dist_map,
      prev_y,
      prev_x,
    );
    const G = this.deps.buildComponentGraph(connections, nComponents);

    // ── 5. Prune + MST ─────────────────────────────────────────────────────
    const gPruned = this.deps.pruneEdges(G, this.options.prune_threshold);
    const mst = this.deps.minimumSpanningForest(gPruned);

    // ── 6. Per-CC skeleton → pixel-level graph ────────────────────────────
    const resultGraph = this.deps.buildResultGraph({
      mst,
      annotationBin,
      segment_length: SEGMENT_LENGTH,
    });

    const { valid_count, labeled_graph } = this.deps.runCrossingAnalysis({
      resultGraph,
      mask: mask2d,
      annotationLabeled: annotLabeled,
      min_tree_components: this.options.min_tree_components,
      stub_length_threshold: this.options.stub_length_threshold,
    });

    return {
      annotation: pre.roi_annotation,
      image: pre.roi_image,
      mask: pre.roi_mask,
      graph: labeled_graph,
      valid_count,
    };
  }
}

export { DEFAULT_OPTIONS, SEGMENT_LENGTH } from "./types.js";
export type {
  AnnotationGrowLinkerOptions,
  LinkerDependencies,
  LinkerResult,
  NdArray,
} from "./types.js";
