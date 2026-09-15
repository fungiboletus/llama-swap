import type { GpuStat } from "./types";

export const WINDOWS = [
  { label: "5 min", ms: 5 * 60 * 1000, value: "5m" },
  { label: "15 min", ms: 15 * 60 * 1000, value: "15m" },
  { label: "1 hr", ms: 60 * 60 * 1000, value: "1h" },
  { label: "1 day", ms: 24 * 60 * 60 * 1000, value: "24h" },
  { label: "7 days", ms: 7 * 24 * 60 * 60 * 1000, value: "7d" },
] as const;

export function formatDelta(ts: string, refTime: number): string {
  const diffSec = Math.round((new Date(ts).getTime() - refTime) / 1000);
  const absSec = Math.abs(diffSec);
  const sign = diffSec <= 0 ? "-" : "+";
  const minutes = Math.floor((absSec % 3600) / 60);
  if (absSec >= 86400) return `${sign}${Math.floor(absSec / 86400)}d ${Math.floor((absSec % 86400) / 3600)}h ${minutes}m`;
  if (absSec >= 3600) return `${sign}${Math.floor(absSec / 3600)}h ${minutes}m`;
  if (absSec < 60) return `${sign}${absSec}s`;
  const sec = absSec % 60;
  if (sec === 0) return `${sign}${minutes}m`;
  return `${sign}${minutes}:${sec.toString().padStart(2, "0")}`;
}

export function gpuTimestamps(stats: GpuStat[]): number[] {
  return [...new Set(stats.map((g) => Math.round(new Date(g.timestamp).getTime() / 1000) * 1000))].sort((a, b) => a - b);
}

export function gpuDatasets(
  stats: GpuStat[],
  field: keyof Pick<GpuStat, "gpu_util_pct" | "mem_util_pct" | "temp_c" | "vram_temp_c" | "power_draw_w">,
  colors: string[],
) {
  const timestamps = gpuTimestamps(stats);
  const devices = new Map<string, { name: string; values: Map<number, number> }>();
  for (const g of stats) {
    const key = JSON.stringify([g.id, g.uuid, g.name]);
    if (!devices.has(key)) devices.set(key, { name: g.name || `GPU ${g.id}`, values: new Map() });
    devices.get(key)!.values.set(Math.round(new Date(g.timestamp).getTime() / 1000) * 1000, g[field]);
  }
  return [...devices.values()].map((device, i) => ({
    label: device.name,
    data: timestamps.map((ts) => device.values.get(ts) ?? null),
    borderColor: colors[i % colors.length],
  }));
}
