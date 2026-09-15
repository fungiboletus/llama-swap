<script lang="ts">
  import { onMount } from "svelte";
  import { WINDOWS, formatDelta, gpuTimestamps, gpuDatasets } from "../lib/performance";
  import { fetchPerformance } from "../stores/api";
  import { persistentStore } from "../stores/persistent";
  import type { SysStat, GpuStat } from "../lib/types";
  import PerformanceChart from "../components/PerformanceChart.svelte";
  import SegmentedControl from "../components/SegmentedControl.svelte";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Button } from "$lib/components/ui/button/index.js";
  import { RefreshCw } from "@lucide/svelte";

  const COLORS = [
    "#3b82f6",
    "#ef4444",
    "#10b981",
    "#f59e0b",
    "#8b5cf6",
    "#ec4899",
    "#06b6d4",
    "#84cc16",
    "#f97316",
    "#14b8a6",
    "#a855f7",
    "#e11d48",
    "#0ea5e9",
    "#eab308",
    "#d946ef",
    "#22d3ee",
  ];

  const INTERVALS = [
    { label: "Off", ms: 0 },
    { label: "5s", ms: 5000 },
    { label: "10s", ms: 10000 },
    { label: "30s", ms: 30000 },
    { label: "60s", ms: 60000 },
  ] as const;

  let selectedWindow = persistentStore("perf-window", 0);
  let selectedInterval = persistentStore("perf-refresh-interval", 0);
  let sysData = $state<SysStat[]>([]);
  let gpuData = $state<GpuStat[]>([]);
  let refreshing = $state(false);
  let loadError = $state(false);
  let requestID = 0;
  const currentWindow = $derived(WINDOWS[$selectedWindow] ?? WINDOWS[0]);

  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let visible = $state(true);
  let mounted = $state(false);

  function cutoffTime(): number {
    return Date.now() - currentWindow.ms;
  }

  const sysLabels = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length === 0) return [];
    const refTime = new Date(stats[stats.length - 1].timestamp).getTime();
    return stats.map((s) => formatDelta(s.timestamp, refTime));
  });

  async function loadAll() {
    const id = ++requestID;
    refreshing = true;
    const resp = await fetchPerformance(undefined, currentWindow.value);
    if (id !== requestID || !mounted) return;
    refreshing = false;
    loadError = !resp;
    if (resp) {
      // Replace the snapshot: the unfinished average bucket changes on each poll.
      sysData = resp.sys_stats ?? [];
      gpuData = resp.gpu_stats ?? [];
    }
  }

  function handleWindowChange(i: number) {
    $selectedWindow = i;
    sysData = [];
    gpuData = [];
    void loadAll();
  }

  function startPolling() {
    stopPolling();
    if (!mounted || !visible) return;
    const ms = (INTERVALS[$selectedInterval] ?? INTERVALS[0]).ms;
    if (ms <= 0) return;
    pollTimer = setInterval(() => {
      if (visible && !refreshing) {
        void loadAll();
      }
    }, ms);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function handleVisibility() {
    visible = !document.hidden;
    if (visible && mounted) {
      loadAll().then(() => startPolling());
    } else {
      stopPolling();
    }
  }

  function handleIntervalChange(i: number) {
    $selectedInterval = i;
    if (visible && mounted) {
      startPolling();
    }
  }

  async function manualRefresh() {
    await loadAll();
  }

  $effect(() => {
    return () => {
      stopPolling();
    };
  });

  onMount(() => {
    mounted = true;
    document.addEventListener("visibilitychange", handleVisibility);
    loadAll().then(() => startPolling());

    return () => {
      mounted = false;
      stopPolling();
      document.removeEventListener("visibilitychange", handleVisibility);
    };
  });

  // --- System charts (filtered by time window) ---

  const filteredSysStats = $derived(sysData.filter((s) => new Date(s.timestamp).getTime() >= cutoffTime()));

  const cpuDatasets = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length === 0) return [];
    const coreCount = stats[0].cpu_util_per_core.length;
    const datasets = [];
    for (let i = 0; i < coreCount; i++) {
      datasets.push({
        label: `Core ${i}`,
        data: stats.map((s) => s.cpu_util_per_core[i]),
        borderColor: COLORS[i % COLORS.length],
      });
    }
    return datasets;
  });

  const memSwapDatasets = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length === 0) return [];
    return [
      {
        label: "Memory Used %",
        data: stats.map((s) => (s.mem_used_mb / s.mem_total_mb) * 100),
        borderColor: "#3b82f6",
      },
      {
        label: "Swap Used %",
        data: stats.map((s) => (s.swap_total_mb > 0 ? (s.swap_used_mb / s.swap_total_mb) * 100 : 0)),
        borderColor: "#8b5cf6",
      },
    ];
  });

  const latestMemSwap = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length === 0) return null;
    const s = stats[stats.length - 1];
    return {
      mem_total_mb: s.mem_total_mb,
      mem_used_mb: s.mem_used_mb,
      mem_used_pct: ((s.mem_used_mb / s.mem_total_mb) * 100).toFixed(1),
      swap_total_mb: s.swap_total_mb,
      swap_used_mb: s.swap_used_mb,
      swap_used_pct: s.swap_total_mb > 0 ? ((s.swap_used_mb / s.swap_total_mb) * 100).toFixed(1) : null,
    };
  });

  const loadDatasets = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length === 0) return [];
    return [
      {
        label: "1 min",
        data: stats.map((s) => s.load_avg_1),
        borderColor: "#10b981",
      },
      {
        label: "5 min",
        data: stats.map((s) => s.load_avg_5),
        borderColor: "#f59e0b",
      },
      {
        label: "15 min",
        data: stats.map((s) => s.load_avg_15),
        borderColor: "#ef4444",
      },
    ];
  });

  const netBandwidthDatasets = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length < 2) return [];

    const ifaceNames = new Set<string>();
    for (const s of stats) {
      for (const n of s.net_io ?? []) {
        ifaceNames.add(n.name);
      }
    }

    const interfaces = [...ifaceNames].sort();
    if (interfaces.length === 0) return [];

    const datasets: { label: string; data: number[]; borderColor: string }[] = [];
    let colorIdx = 0;

    for (const iface of interfaces) {
      const recvData: number[] = [];
      const sentData: number[] = [];

      for (let i = 1; i < stats.length; i++) {
        const prev = stats[i - 1];
        const curr = stats[i];
        const prevIO = (prev.net_io ?? []).find((n) => n.name === iface);
        const currIO = (curr.net_io ?? []).find((n) => n.name === iface);

        if (!prevIO || !currIO) {
          recvData.push(0);
          sentData.push(0);
          continue;
        }

        const dtMs = new Date(curr.timestamp).getTime() - new Date(prev.timestamp).getTime();
        if (dtMs <= 0) {
          recvData.push(0);
          sentData.push(0);
          continue;
        }

        const dtSec = dtMs / 1000;
        recvData.push(((Math.max(0, currIO.bytes_recv - prevIO.bytes_recv) / dtSec) * 8) / 1_000_000);
        sentData.push(((Math.max(0, currIO.bytes_sent - prevIO.bytes_sent) / dtSec) * 8) / 1_000_000);
      }

      datasets.push({
        label: `${iface} in`,
        data: recvData,
        borderColor: COLORS[colorIdx % COLORS.length],
      });
      colorIdx++;
      datasets.push({
        label: `${iface} out`,
        data: sentData,
        borderColor: COLORS[colorIdx % COLORS.length],
      });
      colorIdx++;
    }

    return datasets;
  });

  const netBandwidthLabels = $derived.by(() => {
    const stats = filteredSysStats;
    if (stats.length < 2) return [];
    const refTime = new Date(stats[stats.length - 1].timestamp).getTime();
    return stats.slice(1).map((s) => formatDelta(s.timestamp, refTime));
  });

  // --- GPU charts (filtered by time window) ---

  const filteredGpuStats = $derived(gpuData.filter((g) => new Date(g.timestamp).getTime() >= cutoffTime()));

  const hasGpuData = $derived(gpuData.length > 0);

  const gpuLabels = $derived.by(() => {
    const timestamps = gpuTimestamps(filteredGpuStats);
    const refTime = timestamps[timestamps.length - 1];
    return timestamps.map((ts) => formatDelta(new Date(ts).toISOString(), refTime));
  });

  const gpuUtilDatasets = $derived(gpuDatasets(filteredGpuStats, "gpu_util_pct", COLORS));
  const gpuMemDatasets = $derived(gpuDatasets(filteredGpuStats, "mem_util_pct", COLORS));
  const gpuTempDatasets = $derived(gpuDatasets(filteredGpuStats, "temp_c", COLORS));
  const gpuVramTempDatasets = $derived(gpuDatasets(filteredGpuStats, "vram_temp_c", COLORS));
  const gpuPowerDatasets = $derived(gpuDatasets(filteredGpuStats, "power_draw_w", COLORS));
  const hasVramTemp = $derived(filteredGpuStats.some((g) => g.vram_temp_c > 0));
</script>

<div class="space-y-6">
  <div class="flex flex-wrap items-center justify-between gap-3">
    <h2 class="text-xl font-semibold text-foreground">Performance (Experimental)</h2>
    <div class="flex flex-wrap items-center gap-4">
      <SegmentedControl items={WINDOWS} selected={$selectedWindow} onSelect={handleWindowChange} />
      <SegmentedControl
        items={INTERVALS}
        selected={$selectedInterval}
        onSelect={handleIntervalChange}
        label="Refresh:"
      />
      <Button variant="outline" size="icon-sm" title="Refresh" onclick={manualRefresh} disabled={refreshing}>
        <RefreshCw class={refreshing ? "animate-spin" : ""} />
      </Button>
    </div>
  </div>
  <p class="text-sm text-muted-foreground">
    This is an experimental feature. Please use <a
      class="underline hover:text-foreground"
      href="https://github.com/mostlygeek/llama-swap/discussions/771">discussion #771</a
    > for instructions and to share feedback.
  </p>

  <p class="text-sm text-muted-foreground">
    {#if currentWindow.value === "24h"}Five-minute averages.
    {:else if currentWindow.value === "7d"}Thirty-minute averages.
    {:else}Individual samples.{/if}
    History accumulates while the server runs and resets on restart or performance configuration changes.
    The newest average is still accumulating; network charts show average bandwidth between samples.
  </p>
  {#if loadError}
    <p role="alert" class="text-sm text-destructive">Could not refresh performance data. Please try again.</p>
  {/if}

  <!-- GPU Section -->
  <section class="space-y-4">
    <h3 class="text-lg font-medium text-foreground">GPU</h3>
    {#if !hasGpuData}
      <Card.Root class="py-0">
        <Card.Content class="p-4">
          <p class="text-muted-foreground">No GPU data available</p>
        </Card.Content>
      </Card.Root>
    {:else}
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <PerformanceChart
          title="GPU Utilization (%)"
          labels={gpuLabels}
          datasets={gpuUtilDatasets}
          yMin={0}
          yMax={100}
          yLabel="%"
        />
        <PerformanceChart
          title="GPU Memory Utilization (%)"
          labels={gpuLabels}
          datasets={gpuMemDatasets}
          yMin={0}
          yMax={100}
          yLabel="%"
        />
        <PerformanceChart
          title="GPU Temperature (°C)"
          labels={gpuLabels}
          datasets={gpuTempDatasets}
          yMin={0}
          yLabel="°C"
        />
        {#if hasVramTemp}
          <PerformanceChart
            title="GPU VRAM Temperature (°C)"
            labels={gpuLabels}
            datasets={gpuVramTempDatasets}
            yMin={0}
            yLabel="°C"
          />
        {/if}
        <PerformanceChart
          title="GPU Power Draw (W)"
          labels={gpuLabels}
          datasets={gpuPowerDatasets}
          yMin={0}
          yLabel="W"
        />
      </div>
    {/if}
  </section>

  <!-- System Section -->
  <section class="space-y-4">
    <h3 class="text-lg font-medium text-foreground">System</h3>
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <PerformanceChart
        title="CPU Utilization (%)"
        labels={sysLabels}
        datasets={cpuDatasets}
        yMin={0}
        yMax={100}
        yLabel="%"
        showLegend={false}
      />
      <div>
        <PerformanceChart
          title="Memory & Swap Usage (%)"
          labels={sysLabels}
          datasets={memSwapDatasets}
          yMin={0}
          yMax={100}
          yLabel="%"
        />
        {#if latestMemSwap}
          <div class="flex items-center justify-center gap-4 text-xs text-muted-foreground mt-1 px-4">
            <span
              >Mem: <span class="text-foreground font-medium"
                >{latestMemSwap.mem_used_mb.toLocaleString()} / {latestMemSwap.mem_total_mb.toLocaleString()} MB ({latestMemSwap.mem_used_pct}%)</span
              ></span
            >
            {#if latestMemSwap.swap_used_pct !== null}
              <span
                >Swap: <span class="text-foreground font-medium"
                  >{latestMemSwap.swap_used_mb.toLocaleString()} / {latestMemSwap.swap_total_mb.toLocaleString()} MB ({latestMemSwap.swap_used_pct}%)</span
                ></span
              >
            {/if}
          </div>
        {/if}
      </div>
      <PerformanceChart title="Load Average" labels={sysLabels} datasets={loadDatasets} yMin={0} />
      {#if netBandwidthDatasets.length > 0}
        <PerformanceChart
          title="Network Bandwidth (Mbit/s)"
          labels={netBandwidthLabels}
          datasets={netBandwidthDatasets}
          yMin={0}
          yLabel="Mbit/s"
          showLegend={false}
        />
      {/if}
    </div>
  </section>
</div>
