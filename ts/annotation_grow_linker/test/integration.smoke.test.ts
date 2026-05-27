/**
 * Integration smoke test — spawns the real Python worker and runs the full
 * pipeline on a real sample, verifying that the cache survives multiple
 * downstream parameter tweaks.
 *
 * Heavy: Sato + Dijkstra on the real image takes several seconds. This test
 * is intentionally slow; tag it as integration if you need to gate it.
 */

import { describe, expect, it } from "vitest";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

import { PythonWorker } from "../src/python_worker.js";
import {
  StageOrchestrator,
  type StageParams,
} from "../src/stage_orchestrator.js";

const REPO_ROOT = resolve(__dirname, "../../..");
const SAMPLE_DIR = resolve(REPO_ROOT, "data_0510/S1140-2_a");

const BASE: StageParams = {
  offset_px: 50,
  bg_kernel_size: 51,
  clahe_clip: 20.0,
  clahe_grid: [16, 16],
  sato_sigmas_start: 3,
  sato_sigmas_stop: 8,
  connectivity: 8,
  prune_threshold: 20.0,
  segment_length: 500,
  min_tree_components: 5,
  stub_length_threshold: 5,
};

interface WorkerStats {
  calls: Record<string, number>;
  handles: number;
}

const SAMPLE_FILES_EXIST = [
  resolve(SAMPLE_DIR, "image.png"),
  resolve(SAMPLE_DIR, "mask.png"),
  resolve(SAMPLE_DIR, "weka.png"),
].every(existsSync);

describe.skipIf(!SAMPLE_FILES_EXIST)("integration: Python worker + orchestrator", () => {
  it(
    "preprocessing runs once across three prune_threshold trials on a real sample",
    async () => {
      const worker = new PythonWorker({ cwd: REPO_ROOT });
      try {
        await worker.ready();

        const sample = await worker.call<{
          green: string;
          mask: string;
          annotation: string;
          shape: number[];
        }>("load_sample", {
          image_path: resolve(SAMPLE_DIR, "image.png"),
          mask_path: resolve(SAMPLE_DIR, "mask.png"),
          annotation_path: resolve(SAMPLE_DIR, "weka.png"),
        });
        expect(sample.green).toMatch(/^[0-9a-f]+$/);
        expect(sample.mask).toMatch(/^[0-9a-f]+$/);
        expect(sample.annotation).toMatch(/^[0-9a-f]+$/);
        expect(sample.shape.length).toBe(2);

        const orchestrator = new StageOrchestrator(worker, {
          green: sample.green,
          mask: sample.mask,
          annotation: sample.annotation,
        });

        // First full run
        const r1 = await orchestrator.labeledGraph(BASE);
        expect(r1.labeled_graph).toMatch(/^[0-9a-f]+$/);
        expect(typeof r1.pred_count).toBe("number");

        // Tweak prune_threshold twice more
        await orchestrator.labeledGraph({ ...BASE, prune_threshold: 10 });
        await orchestrator.labeledGraph({ ...BASE, prune_threshold: 30 });

        const stats = await worker.call<WorkerStats>("stats");

        // Preprocessing chain ran exactly once
        expect(stats.calls.roi_mask).toBe(1);
        expect(stats.calls.annot_comp).toBe(1);
        expect(stats.calls.bg_removed).toBe(1);
        expect(stats.calls.clahe_applied).toBe(1);
        // sigmas: 3,4,5,6,7 → 5 per-sigma calls total
        expect(stats.calls.sato_per_sigma).toBe(5);
        expect(stats.calls.roi_image).toBe(1);
        expect(stats.calls.cost_map).toBe(1);
        expect(stats.calls.dijkstra).toBe(1);
        expect(stats.calls.comp_graph).toBe(1);

        // Pruning + downstream re-ran for each prune_threshold value
        expect(stats.calls.pruned_graph).toBe(3);
        expect(stats.calls.mst).toBe(3);
        expect(stats.calls.result_graph).toBe(3);
        expect(stats.calls.labeled_graph).toBe(3);

        // Sanity: graph summary for the labeled_graph is a real nx.Graph
        const summary = await worker.call<{ kind: string; nodes: number; edges: number }>(
          "summary",
          { handle: r1.labeled_graph },
        );
        expect(summary.kind).toBe("graph");
        expect(summary.nodes).toBeGreaterThan(0);
      } finally {
        await worker.close();
      }
    },
    180_000, // 3-minute timeout — Sato + Dijkstra on a real image is slow
  );
});

describe.skipIf(SAMPLE_FILES_EXIST)("integration (skipped: sample files missing)", () => {
  it("placeholder", () => {
    // Vitest requires at least one test in a non-empty file
    expect(true).toBe(true);
  });
});
