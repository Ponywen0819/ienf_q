import { describe, expect, it } from "vitest";

import {
  binarizeToBool,
  binarizeToUint8,
  maxValue,
  squeezeFirstChannel,
} from "../src/array_utils.js";
import type { NdArray } from "../src/types.js";

const arr = (shape: number[], data: number[]): NdArray => ({ shape, data });

describe("squeezeFirstChannel", () => {
  it("returns 2D input unchanged (identity)", () => {
    const input = arr([2, 3], [1, 2, 3, 4, 5, 6]);
    const out = squeezeFirstChannel(input);
    expect(out).toBe(input);
  });

  it("extracts channel 0 from a 3D (H, W, C) array", () => {
    // 2x2x3 with channel-0 values 10/20/30/40, channel-1/2 noise
    const data = [
      10, 99, 99,  20, 88, 88,
      30, 77, 77,  40, 66, 66,
    ];
    const out = squeezeFirstChannel(arr([2, 2, 3], data));
    expect(Array.from(out.shape)).toEqual([2, 2]);
    expect(Array.from(out.data)).toEqual([10, 20, 30, 40]);
  });
});

describe("binarizeToUint8", () => {
  it("threshold > 127 produces 0/1 values, preserving shape", () => {
    const out = binarizeToUint8(arr([2, 3], [0, 127, 128, 200, 50, 255]), 127);
    expect(Array.from(out.shape)).toEqual([2, 3]);
    expect(Array.from(out.data)).toEqual([0, 0, 1, 1, 0, 1]);
  });

  it("is strict greater-than (127 is NOT > 127)", () => {
    const out = binarizeToUint8(arr([1, 1], [127]), 127);
    expect(out.data[0]).toBe(0);
  });
});

describe("binarizeToBool", () => {
  it("yields the same 0/1 payload as uint8 binarization", () => {
    const input = arr([2, 2], [0, 130, 127, 200]);
    const u = binarizeToUint8(input, 127);
    const b = binarizeToBool(input, 127);
    expect(Array.from(b.data)).toEqual(Array.from(u.data));
  });
});

describe("maxValue", () => {
  it("returns the max scalar", () => {
    expect(maxValue(arr([3], [1, 5, 3]))).toBe(5);
  });

  it("returns 0 for empty input", () => {
    expect(maxValue(arr([0], []))).toBe(0);
  });

  it("handles single element", () => {
    expect(maxValue(arr([1], [42]))).toBe(42);
  });
});
