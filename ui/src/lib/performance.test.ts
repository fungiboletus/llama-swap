import { describe, expect, it } from "vitest";
import { formatDelta, gpuDatasets, gpuTimestamps, WINDOWS } from "./performance";
import type { GpuStat } from "./types";

const base = Date.parse("2026-09-15T12:00:00Z");
const gpu = (id: number, minutes: number, value: number): GpuStat => ({
  timestamp: new Date(base + minutes * 60_000).toISOString(),
  id, name: `GPU ${id}`, uuid: `uuid-${id}`, gpu_util_pct: value,
  temp_c: 40, vram_temp_c: 0, mem_util_pct: 0, mem_used_mb: 0,
  mem_total_mb: 0, fan_speed_pct: 0, power_draw_w: 0,
});

describe("performance history charts", () => {
  it("preserves saved short-window indices and adds day/week ranges", () => {
    expect(WINDOWS.map((w) => w.value)).toEqual(["5m", "15m", "1h", "24h", "7d"]);
    expect(WINDOWS[4].ms).toBe(7 * 24 * 60 * 60 * 1000);
  });

  it("keeps half-hour labels distinct across all seven days", () => {
    const labels = Array.from({ length: 337 }, (_, i) => formatDelta(new Date(base - i * 30 * 60_000).toISOString(), base));
    expect(new Set(labels).size).toBe(labels.length);
    expect(labels[49]).toBe("-1d 0h 30m");
    expect(labels[2]).toBe("-1h 0m");
    expect(formatDelta(new Date(base - 5000).toISOString(), base)).toBe("-5s");
  });

  it("aligns missing GPU samples with gaps instead of shifting their data", () => {
    const stats = [gpu(0, 0, 20), gpu(1, 0, 30), gpu(0, 30, 40), gpu(0, 60, 60), gpu(1, 60, 80)];
    expect(gpuTimestamps(stats)).toEqual([base, base + 30 * 60_000, base + 60 * 60_000]);
    const series = gpuDatasets(stats, "gpu_util_pct", ["red", "blue"]);
    expect(series[0].data).toEqual([20, 40, 60]);
    expect(series[1].data).toEqual([30, null, 80]);
  });

  it("keeps replacement GPUs with the same ID separate", () => {
    const stats = [gpu(0, 0, 20), { ...gpu(0, 30, 40), uuid: "replacement" }];
    const series = gpuDatasets(stats, "gpu_util_pct", ["red", "blue"]);
    expect(series.map((s) => s.data)).toEqual([[20, null], [null, 40]]);
  });
});
