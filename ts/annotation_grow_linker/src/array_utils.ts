import type { NdArray } from "./types.js";

/**
 * If `mask.shape` is 3D (H, W, C) return a view with shape (H, W) taking the
 * first channel — mirrors `mask = mask[:, :, 0]`. Otherwise return as-is.
 */
export function squeezeFirstChannel(mask: NdArray): NdArray {
  if (mask.shape.length !== 3) {
    return mask;
  }
  const [h, w, c] = mask.shape as [number, number, number];
  const out = new Array<number>(h * w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[y * w + x] = mask.data[(y * w + x) * c]!;
    }
  }
  return { shape: [h, w], data: out };
}

/**
 * Element-wise `(arr > threshold).astype(uint8)` — values become 0 or 1.
 * Preserves the 2-D shape; returns a fresh array backed by a plain number[].
 */
export function binarizeToUint8(arr: NdArray, threshold: number): NdArray {
  const n = arr.data.length;
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    out[i] = arr.data[i]! > threshold ? 1 : 0;
  }
  return { shape: arr.shape, data: out };
}

/**
 * Element-wise `(arr > threshold)` producing a boolean-valued array.
 * Kept as 0/1 numbers because `NdArray.data` is numeric; downstream code
 * should treat it as a mask, not a count.
 */
export function binarizeToBool(arr: NdArray, threshold: number): NdArray {
  // Same payload as binarizeToUint8 for now, but the *role* (boolean mask vs.
  // uint8 label) is preserved by separate call sites.
  return binarizeToUint8(arr, threshold);
}

/** Returns the maximum scalar in `arr.data`. Empty input → 0 (Python int cast). */
export function maxValue(arr: NdArray): number {
  const n = arr.data.length;
  if (n === 0) return 0;
  let m = arr.data[0]!;
  for (let i = 1; i < n; i++) {
    const v = arr.data[i]!;
    if (v > m) m = v;
  }
  return m;
}
