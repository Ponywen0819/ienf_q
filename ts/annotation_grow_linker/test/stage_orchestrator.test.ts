/**
 * StageOrchestrator unit tests — uses a hand-rolled worker stub so the DAG
 * and cache logic can be exercised without spawning Python.
 *
 * The stub returns deterministic handles (`<method>:<call#>`) and records
 * every call. That lets us assert: (a) call counts per stage across multiple
 * `resolve` calls with overlapping params; (b) the actual params the worker
 * receives; (c) the order of upstream resolution.
 */

import { describe, expect, it } from "vitest";

import {
  StageOrchestrator,
  type SampleHandles,
  type StageParams,
  type WorkerLike,
} from "../src/stage_orchestrator.js";

// ── Stub worker ─────────────────────────────────────────────────────────────

interface CallRecord {
  method: string;
  params: Record<string, unknown>;
  result: unknown;
}

interface StubWorker extends WorkerLike {
  calls: CallRecord[];
  countsByMethod(): Record<string, number>;
}

function makeStubWorker(): StubWorker {
  const calls: CallRecord[] = [];
  let counter = 0;

  const responder = (method: string): unknown => {
    counter++;
    const h = `${method}:${counter}`;
    switch (method) {
      case "stage_annot_comp":
        return {
          annot_labeled: `annot_labeled:${counter}`,
          annotation_bin: `annotation_bin:${counter}`,
          n_components: 7,
        };
      case "stage_labeled_graph":
        return {
          labeled_graph: `labeled_graph:${counter}`,
          pred_count: 42,
        };
      default:
        return h;
    }
  };

  return {
    calls,
    countsByMethod() {
      const out: Record<string, number> = {};
      for (const c of calls) out[c.method] = (out[c.method] ?? 0) + 1;
      return out;
    },
    async call<T = unknown>(
      method: string,
      params: Record<string, unknown> = {},
    ): Promise<T> {
      const result = responder(method);
      calls.push({ method, params, result });
      return result as T;
    },
  };
}

const SAMPLE: SampleHandles = {
  green: "green-h",
  mask: "mask-h",
  annotation: "annotation-h",
};

const BASE: StageParams = {
  offset_px: 50,
  bg_kernel_size: 51,
  clahe_clip: 20.0,
  clahe_grid: [16, 16],
  sato_sigmas_start: 3,
  sato_sigmas_stop: 8, // sigmas: 3, 4, 5, 6, 7  → 5 per-sigma calls
  connectivity: 8,
  prune_threshold: 20.0,
  segment_length: 500,
  min_tree_components: 5,
  stub_length_threshold: 5,
};

// ── Tests ───────────────────────────────────────────────────────────────────

describe("StageOrchestrator — single resolve fans out the whole DAG once", () => {
  it("runs every stage exactly once when labeledGraph is requested fresh", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    await o.labeledGraph(BASE);

    const counts = w.countsByMethod();
    expect(counts.stage_roi_mask).toBe(1);
    expect(counts.stage_annot_comp).toBe(1);
    expect(counts.stage_bg_removed).toBe(1);
    expect(counts.stage_clahe_applied).toBe(1);
    expect(counts.stage_sato_per_sigma).toBe(5); // sigmas 3..7
    expect(counts.stage_roi_image).toBe(1);
    expect(counts.stage_cost_map).toBe(1);
    expect(counts.stage_dijkstra).toBe(1);
    expect(counts.stage_comp_graph).toBe(1);
    expect(counts.stage_pruned_graph).toBe(1);
    expect(counts.stage_mst).toBe(1);
    expect(counts.stage_result_graph).toBe(1);
    expect(counts.stage_labeled_graph).toBe(1);
  });
});

describe("StageOrchestrator — preprocessing cache survives downstream param tweaks", () => {
  it("changing prune_threshold 3× only re-runs pruned_graph and below", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    await o.labeledGraph(BASE);
    await o.labeledGraph({ ...BASE, prune_threshold: 10 });
    await o.labeledGraph({ ...BASE, prune_threshold: 30 });

    const c = w.countsByMethod();
    // upstream of prune — should NOT re-run
    expect(c.stage_roi_mask).toBe(1);
    expect(c.stage_annot_comp).toBe(1);
    expect(c.stage_bg_removed).toBe(1);
    expect(c.stage_clahe_applied).toBe(1);
    expect(c.stage_sato_per_sigma).toBe(5);
    expect(c.stage_roi_image).toBe(1);
    expect(c.stage_cost_map).toBe(1);
    expect(c.stage_dijkstra).toBe(1);
    expect(c.stage_comp_graph).toBe(1);
    // prune-and-below — should re-run each time
    expect(c.stage_pruned_graph).toBe(3);
    expect(c.stage_mst).toBe(3);
    expect(c.stage_result_graph).toBe(3);
    expect(c.stage_labeled_graph).toBe(3);
  });

  it("changing only stub_length_threshold re-runs ONLY labeled_graph", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    await o.labeledGraph(BASE);
    await o.labeledGraph({ ...BASE, stub_length_threshold: 9 });
    await o.labeledGraph({ ...BASE, min_tree_components: 9 });

    const c = w.countsByMethod();
    expect(c.stage_pruned_graph).toBe(1);
    expect(c.stage_mst).toBe(1);
    expect(c.stage_result_graph).toBe(1);
    expect(c.stage_labeled_graph).toBe(3);
  });

  it("changing connectivity invalidates dijkstra-and-below but keeps preprocessing", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    await o.labeledGraph(BASE);
    await o.labeledGraph({ ...BASE, connectivity: 4 });

    const c = w.countsByMethod();
    expect(c.stage_roi_mask).toBe(1);
    expect(c.stage_bg_removed).toBe(1);
    expect(c.stage_clahe_applied).toBe(1);
    expect(c.stage_sato_per_sigma).toBe(5);
    expect(c.stage_roi_image).toBe(1);
    expect(c.stage_cost_map).toBe(1);
    expect(c.stage_dijkstra).toBe(2);
    expect(c.stage_comp_graph).toBe(2);
    expect(c.stage_pruned_graph).toBe(2);
    expect(c.stage_labeled_graph).toBe(2);
  });

  it("changing bg_kernel_size invalidates bg_removed-and-below but keeps roi_mask + annot_comp", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    await o.labeledGraph(BASE);
    await o.labeledGraph({ ...BASE, bg_kernel_size: 31 });

    const c = w.countsByMethod();
    expect(c.stage_roi_mask).toBe(1);
    expect(c.stage_annot_comp).toBe(1);
    expect(c.stage_bg_removed).toBe(2);
    expect(c.stage_clahe_applied).toBe(2);
    // sato re-fires for each of 5 sigmas under the new clahe_applied
    expect(c.stage_sato_per_sigma).toBe(10);
  });
});

describe("StageOrchestrator — sato per-sigma overlap sharing", () => {
  it("overlapping sigma ranges share per-sigma cache entries", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    // range(3, 8) = {3,4,5,6,7}     — 5 per-sigma calls
    await o.roiImage(BASE);
    // range(4, 9) = {4,5,6,7,8}     — only sigma=8 is new
    await o.roiImage({ ...BASE, sato_sigmas_start: 4, sato_sigmas_stop: 9 });

    const c = w.countsByMethod();
    expect(c.stage_sato_per_sigma).toBe(6); // 5 + 1 new
    expect(c.stage_roi_image).toBe(2); // 2 distinct ranges → 2 combinations
  });
});

describe("StageOrchestrator — concurrent identical resolves dedupe", () => {
  it("two concurrent labeledGraph(BASE) calls cause one underlying RPC per stage", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    await Promise.all([o.labeledGraph(BASE), o.labeledGraph(BASE)]);

    const c = w.countsByMethod();
    expect(c.stage_roi_mask).toBe(1);
    expect(c.stage_labeled_graph).toBe(1);
  });
});

describe("StageOrchestrator — argument plumbing", () => {
  it("passes the raw sample mask (not roi_mask) to stage_labeled_graph", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    await o.labeledGraph(BASE);

    const call = w.calls.find((c) => c.method === "stage_labeled_graph");
    expect(call).toBeDefined();
    expect(call!.params.mask).toBe(SAMPLE.mask);
    // sanity: the param should NOT be the roi_mask handle from stage_roi_mask
    const roiCall = w.calls.find((c) => c.method === "stage_roi_mask");
    expect(call!.params.mask).not.toBe(roiCall!.result);
  });

  it("forwards prune_threshold to stage_pruned_graph", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    await o.labeledGraph({ ...BASE, prune_threshold: 12.5 });
    const call = w.calls.find((c) => c.method === "stage_pruned_graph")!;
    expect(call.params.prune_threshold).toBe(12.5);
  });

  it("forwards segment_length to stage_result_graph", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    await o.labeledGraph({ ...BASE, segment_length: 250 });
    const call = w.calls.find((c) => c.method === "stage_result_graph")!;
    expect(call.params.segment_length).toBe(250);
  });

  it("forwards clahe_grid as a [w, h] tuple to stage_clahe_applied", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    await o.labeledGraph({ ...BASE, clahe_grid: [8, 8] });
    const call = w.calls.find((c) => c.method === "stage_clahe_applied")!;
    expect(call.params.clahe_grid).toEqual([8, 8]);
  });

  it("expands sato_sigmas_{start,stop} into individual sigma calls", async () => {
    const w = makeStubWorker();
    const o = new StageOrchestrator(w, SAMPLE);

    await o.roiImage({ ...BASE, sato_sigmas_start: 2, sato_sigmas_stop: 5 });

    const sigmas = w.calls
      .filter((c) => c.method === "stage_sato_per_sigma")
      .map((c) => c.params.sigma);
    expect(sigmas).toEqual([2, 3, 4]);
  });
});

describe("StageOrchestrator — failed compute does not poison the cache", () => {
  it("retrying a failed stage re-attempts instead of returning the rejected promise", async () => {
    let throwOnce = true;
    const w: StubWorker = {
      calls: [],
      countsByMethod() {
        const out: Record<string, number> = {};
        for (const c of this.calls) out[c.method] = (out[c.method] ?? 0) + 1;
        return out;
      },
      async call<T = unknown>(method: string, params: Record<string, unknown> = {}) {
        if (method === "stage_roi_mask" && throwOnce) {
          throwOnce = false;
          this.calls.push({ method, params, result: null });
          throw new Error("transient");
        }
        const h = `${method}:ok`;
        this.calls.push({ method, params, result: h });
        return h as T;
      },
    };
    const o = new StageOrchestrator(w, SAMPLE);
    await expect(o.roiMask(BASE)).rejects.toThrow("transient");
    await expect(o.roiMask(BASE)).resolves.toBe("stage_roi_mask:ok");
    expect(w.countsByMethod().stage_roi_mask).toBe(2);
  });
});
