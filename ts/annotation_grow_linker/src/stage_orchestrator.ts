/**
 * StageOrchestrator — DAG + memo on top of the Python worker.
 *
 * Mirrors the stage dependency table from `tools/grid_search/staged_grid_search.py`:
 *
 *   roi_mask       ← offset_px
 *   annot_comp     ← offset_px            (uses roi_mask)
 *   bg_removed     ← + bg_kernel_size
 *   clahe_applied  ← + clahe_clip, clahe_grid
 *   sato_per_sigma ← + sigma              (per-sigma cache, fan-out parent)
 *   roi_image      ← + sato_sigmas_{start,stop}
 *   cost_map       ← (same)
 *   dijkstra       ← + connectivity
 *   comp_graph     ← (same)
 *   pruned_graph   ← + prune_threshold
 *   mst            ← (same)
 *   result_graph   ← (same)
 *   labeled_graph  ← + min_tree_components, stub_length_threshold
 *
 * Cache key = (stage name, own governing params, upstream handle IDs). Because
 * upstream handle IDs deterministically reflect their own params, this single
 * rule gives correct invalidation for the whole DAG without re-deriving param
 * tuples.
 */

// PythonWorker is the canonical implementation of WorkerLike; the orchestrator
// only depends on the minimal interface so tests can substitute a stub.

export interface StageParams {
  offset_px: number;
  bg_kernel_size: number;
  clahe_clip: number;
  clahe_grid: readonly [number, number];
  sato_sigmas_start: number;
  sato_sigmas_stop: number;
  connectivity: number;
  prune_threshold: number;
  min_tree_components: number;
  stub_length_threshold: number;
}

export interface SampleHandles {
  green: string;
  mask: string;
  annotation: string;
}

export interface AnnotCompResult {
  annot_labeled: string;
  annotation_bin: string;
  n_components: number;
}

export interface SubtreeLength {
  tree_id: number;
  num_nodes: number;
  num_edges: number;
  total_length: number;
}

export interface LabeledGraphResult {
  labeled_graph: string;
  pred_count: number;
  subtree_lengths: SubtreeLength[];
}

export type Handle = string;

/** Minimal subset of PythonWorker that the orchestrator actually depends on. */
export interface WorkerLike {
  call<T = unknown>(method: string, params?: Record<string, unknown>): Promise<T>;
}

export class StageOrchestrator {
  private readonly cache = new Map<string, Promise<unknown>>();

  constructor(
    private readonly worker: WorkerLike,
    private readonly sample: SampleHandles,
  ) {}

  /** Number of memoized stages currently live (for tests / introspection). */
  cacheSize(): number {
    return this.cache.size;
  }

  /** Drop everything. Caller is responsible for `free()`-ing worker handles. */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Memo wrapper. The key fully captures what this stage's output depends on:
   * the stage name, its OWN governing params, and the upstream handle IDs.
   *
   * Storing the promise (not the awaited value) collapses concurrent callers
   * onto a single underlying RPC.
   */
  private memo<T>(
    stage: string,
    ownParams: Record<string, unknown>,
    upstream: readonly string[],
    compute: () => Promise<T>,
  ): Promise<T> {
    const sortedParams = Object.keys(ownParams)
      .sort()
      .map((k) => [k, ownParams[k]] as const);
    const key = JSON.stringify([stage, sortedParams, upstream]);
    const hit = this.cache.get(key);
    if (hit) return hit as Promise<T>;
    const promise = compute();
    this.cache.set(key, promise);
    // If the compute throws, evict so the next attempt isn't a permanent error.
    promise.catch(() => this.cache.delete(key));
    return promise;
  }

  // ── Stage resolvers ────────────────────────────────────────────────────

  roiMask(p: StageParams): Promise<Handle> {
    return this.memo(
      "roi_mask",
      { offset_px: p.offset_px },
      [this.sample.mask],
      () =>
        this.worker.call<string>("stage_roi_mask", {
          mask: this.sample.mask,
          offset_px: p.offset_px,
        }),
    );
  }

  async annotComp(p: StageParams): Promise<AnnotCompResult> {
    const roi = await this.roiMask(p);
    return this.memo(
      "annot_comp",
      {},
      [this.sample.annotation, roi],
      () =>
        this.worker.call<AnnotCompResult>("stage_annot_comp", {
          annotation: this.sample.annotation,
          roi_mask: roi,
        }),
    );
  }

  async bgRemoved(p: StageParams): Promise<Handle> {
    const roi = await this.roiMask(p);
    return this.memo(
      "bg_removed",
      { bg_kernel_size: p.bg_kernel_size },
      [this.sample.green, roi],
      () =>
        this.worker.call<string>("stage_bg_removed", {
          green: this.sample.green,
          roi_mask: roi,
          bg_kernel_size: p.bg_kernel_size,
        }),
    );
  }

  async claheApplied(p: StageParams): Promise<Handle> {
    const bg = await this.bgRemoved(p);
    return this.memo(
      "clahe_applied",
      {
        clahe_clip: p.clahe_clip,
        clahe_grid: [p.clahe_grid[0], p.clahe_grid[1]],
      },
      [bg],
      () =>
        this.worker.call<string>("stage_clahe_applied", {
          bg_removed: bg,
          clahe_clip: p.clahe_clip,
          clahe_grid: [p.clahe_grid[0], p.clahe_grid[1]],
        }),
    );
  }

  /**
   * Per-sigma vesselness. Cached individually so overlapping (start, stop)
   * ranges share their per-sigma work — matching the Python implementation.
   */
  async satoPerSigma(
    p: StageParams,
    sigma: number,
    cached?: { clahe?: Handle; roi?: Handle },
  ): Promise<Handle> {
    const clahe = cached?.clahe ?? (await this.claheApplied(p));
    const roi = cached?.roi ?? (await this.roiMask(p));
    return this.memo(
      "sato_per_sigma",
      { sigma },
      [clahe, roi],
      () =>
        this.worker.call<string>("stage_sato_per_sigma", {
          clahe_applied: clahe,
          roi_mask: roi,
          sigma,
        }),
    );
  }

  /**
   * Element-wise max over `range(start, stop)` Sato responses, normalised to
   * uint8. Fan-out parent for `satoPerSigma`.
   */
  async roiImage(p: StageParams): Promise<Handle> {
    const clahe = await this.claheApplied(p);
    const roi = await this.roiMask(p);
    const sigmas: number[] = [];
    for (let s = p.sato_sigmas_start; s < p.sato_sigmas_stop; s++) sigmas.push(s);
    const perSigma = await Promise.all(
      sigmas.map((s) => this.satoPerSigma(p, s, { clahe, roi })),
    );
    return this.memo(
      "roi_image",
      {},
      perSigma,
      () => this.worker.call<string>("stage_roi_image", { per_sigma: perSigma }),
    );
  }

  async costMap(p: StageParams): Promise<Handle> {
    const roiImg = await this.roiImage(p);
    return this.memo(
      "cost_map",
      {},
      [roiImg],
      () => this.worker.call<string>("stage_cost_map", { roi_image: roiImg }),
    );
  }

  async dijkstra(p: StageParams): Promise<Handle> {
    const [cost, comp, roi] = await Promise.all([
      this.costMap(p),
      this.annotComp(p),
      this.roiMask(p),
    ]);
    return this.memo(
      "dijkstra",
      { connectivity: p.connectivity },
      [cost, comp.annot_labeled, roi],
      () =>
        this.worker.call<string>("stage_dijkstra", {
          cost_map: cost,
          annot_labeled: comp.annot_labeled,
          roi_mask: roi,
          connectivity: p.connectivity,
        }),
    );
  }

  async compGraph(p: StageParams): Promise<Handle> {
    const [dij, comp] = await Promise.all([this.dijkstra(p), this.annotComp(p)]);
    return this.memo(
      "comp_graph",
      { n_components: comp.n_components },
      [dij],
      () =>
        this.worker.call<string>("stage_comp_graph", {
          dijkstra: dij,
          n_components: comp.n_components,
        }),
    );
  }

  async prunedGraph(p: StageParams): Promise<Handle> {
    const g = await this.compGraph(p);
    return this.memo(
      "pruned_graph",
      { prune_threshold: p.prune_threshold },
      [g],
      () =>
        this.worker.call<string>("stage_pruned_graph", {
          comp_graph: g,
          prune_threshold: p.prune_threshold,
        }),
    );
  }

  async mst(p: StageParams): Promise<Handle> {
    const g = await this.prunedGraph(p);
    return this.memo(
      "mst",
      {},
      [g],
      () => this.worker.call<string>("stage_mst", { pruned_graph: g }),
    );
  }

  async resultGraph(p: StageParams): Promise<Handle> {
    const [m, comp] = await Promise.all([this.mst(p), this.annotComp(p)]);
    return this.memo(
      "result_graph",
      {},
      [m, comp.annotation_bin],
      () =>
        this.worker.call<string>("stage_result_graph", {
          mst: m,
          annotation_bin: comp.annotation_bin,
        }),
    );
  }

  /**
   * Reconstruction-side post-processing: segment detect → stub trim → re-segment.
   * The graph returned here is what the UI displays for editing; counting is
   * a separate stage so it can be re-run against a user-edited graph.
   */
  async reconstructedGraph(p: StageParams): Promise<Handle> {
    const rg = await this.resultGraph(p);
    return this.memo(
      "reconstructed_graph",
      { stub_length_threshold: p.stub_length_threshold },
      [rg],
      () =>
        this.worker.call<string>("stage_reconstructed_graph", {
          result_graph: rg,
          stub_length_threshold: p.stub_length_threshold,
        }),
    );
  }

  /**
   * Counting stage on top of an arbitrary graph handle. Caller passes the
   * graph explicitly so this can be invoked against either the reconstructed
   * graph or a user-edited graph (imported via `import_graph`).
   *
   * Not memoised: edits invalidate any prior count, and the caller already
   * decides when to invoke this. annotComp (and its upstream roi_mask) IS
   * memoised via the standard cache, so the same `offset_px` reuses work.
   */
  async count(graphHandle: Handle, p: StageParams): Promise<LabeledGraphResult> {
    const comp = await this.annotComp(p);
    return this.worker.call<LabeledGraphResult>("stage_count", {
      reconstructed_graph: graphHandle,
      mask: this.sample.mask,
      annot_labeled: comp.annot_labeled,
      min_tree_components: p.min_tree_components,
    });
  }

  async labeledGraph(p: StageParams): Promise<LabeledGraphResult> {
    const [rg, comp] = await Promise.all([this.resultGraph(p), this.annotComp(p)]);
    return this.memo(
      "labeled_graph",
      {
        min_tree_components: p.min_tree_components,
        stub_length_threshold: p.stub_length_threshold,
      },
      // raw sample mask, not roi_mask — matches staged_grid_search.py line 384
      [rg, this.sample.mask, comp.annot_labeled],
      () =>
        this.worker.call<LabeledGraphResult>("stage_labeled_graph", {
          result_graph: rg,
          mask: this.sample.mask,
          annot_labeled: comp.annot_labeled,
          min_tree_components: p.min_tree_components,
          stub_length_threshold: p.stub_length_threshold,
        }),
    );
  }
}
