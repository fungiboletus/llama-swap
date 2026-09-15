package perf

import (
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMonitor_HistoryRetentionAndResolution(t *testing.T) {
	m, err := New(config.PerformanceConfig{Every: time.Minute}, newTestLogger())
	require.NoError(t, err)
	now := time.Date(2026, 9, 15, 12, 2, 0, 0, time.UTC)
	for ts := now.Add(-8 * 24 * time.Hour); !ts.After(now); ts = ts.Add(time.Minute) {
		m.recordSys(SysStat{Timestamp: ts, CpuUtilPerCore: []float64{20, 80}, MemUsedMB: 120})
		m.recordGPU([]GpuStat{{Timestamp: ts, ID: 0, GpuUtilPct: 40}})
	}
	for _, tc := range []struct {
		window time.Duration
		max    int
		step   time.Duration
	}{
		{time.Hour, 60, time.Minute},
		{24 * time.Hour, 289, 5 * time.Minute},
		{7 * 24 * time.Hour, 337, 30 * time.Minute},
	} {
		sys, gpu := m.History(tc.window, now)
		require.NotEmpty(t, sys)
		assert.LessOrEqual(t, len(sys), tc.max)
		assert.Len(t, gpu, len(sys))
		assert.False(t, sys[0].Timestamp.Before(now.Add(-tc.window)))
		assert.WithinDuration(t, now.Add(-tc.window), sys[0].Timestamp, tc.step)
		assert.Equal(t, now, sys[len(sys)-1].Timestamp)
		assert.Equal(t, []float64{20, 80}, sys[0].CpuUtilPerCore)
	}
	sys, gpu := m.History(7*24*time.Hour, now.Add(8*24*time.Hour))
	assert.Empty(t, sys)
	assert.Empty(t, gpu)
}

func TestMonitor_HistoryAveragesAndPartialBucket(t *testing.T) {
	m, err := New(config.PerformanceConfig{Every: 5 * time.Second}, newTestLogger())
	require.NoError(t, err)
	ts := time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)
	first := SysStat{Timestamp: ts, CpuUtilPerCore: []float64{10, 30}, MemTotalMB: 100, MemUsedMB: 20, MemFreeMB: 80, SwapTotalMB: 10, SwapUsedMB: 2, LoadAvg1: 2, LoadAvg5: 4, LoadAvg15: 6, NetIO: []NetIOStat{{Name: "en0", BytesRecv: 100, BytesSent: 200}}}
	second := SysStat{Timestamp: ts.Add(time.Minute), CpuUtilPerCore: []float64{30, 50}, MemTotalMB: 100, MemUsedMB: 40, MemFreeMB: 60, SwapTotalMB: 10, SwapUsedMB: 4, LoadAvg1: 4, LoadAvg5: 6, LoadAvg15: 8, NetIO: []NetIOStat{{Name: "en0", BytesRecv: 500, BytesSent: 600}}}
	m.recordSys(first)
	sys, _ := m.History(24*time.Hour, second.Timestamp)
	require.Len(t, sys, 1)
	assert.Equal(t, 20, sys[0].MemUsedMB)
	m.recordSys(second)
	sys, _ = m.History(24*time.Hour, second.Timestamp)
	require.Len(t, sys, 1)
	assert.Equal(t, []float64{20, 40}, sys[0].CpuUtilPerCore)
	assert.Equal(t, 30, sys[0].MemUsedMB)
	assert.Equal(t, 70, sys[0].MemFreeMB)
	assert.Equal(t, 3, sys[0].SwapUsedMB)
	assert.Equal(t, 3.0, sys[0].LoadAvg1)
	assert.Equal(t, 5.0, sys[0].LoadAvg5)
	assert.Equal(t, 7.0, sys[0].LoadAvg15)
	assert.Equal(t, second.NetIO, sys[0].NetIO)
	assert.Equal(t, second.Timestamp, sys[0].Timestamp)
	assert.Equal(t, []float64{10, 30}, first.CpuUtilPerCore, "averaging must not mutate raw samples")
	m.recordSys(SysStat{Timestamp: ts.Add(20 * time.Minute), MemUsedMB: 90})
	sys, _ = m.History(24*time.Hour, ts.Add(20*time.Minute))
	require.Len(t, sys, 2, "missing buckets must not be invented as zero usage")
	assert.Equal(t, 30, sys[0].MemUsedMB)
	assert.Equal(t, 90, sys[1].MemUsedMB)
	raw, _ := m.Current()
	require.Len(t, raw, 3)
	assert.Equal(t, 20, raw[0].MemUsedMB)
}

func TestMonitor_HistoryGPUDeviceAverages(t *testing.T) {
	m, err := New(config.PerformanceConfig{Every: 5 * time.Second}, newTestLogger())
	require.NoError(t, err)
	ts := time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)
	m.recordGPU([]GpuStat{
		{Timestamp: ts, ID: 0, UUID: "a", Name: "A", GpuUtilPct: 20, MemUtilPct: 10, TempC: 40, VramTempC: 50, MemUsedMB: 100, MemTotalMB: 1000, FanSpeedPct: 20, PowerDrawW: 100},
		{Timestamp: ts, ID: 1, UUID: "b", Name: "B", GpuUtilPct: 80},
	})
	m.recordGPU([]GpuStat{{Timestamp: ts.Add(time.Minute), ID: 0, UUID: "a", Name: "A", GpuUtilPct: 60, MemUtilPct: 30, TempC: 60, VramTempC: 70, MemUsedMB: 300, MemTotalMB: 1000, FanSpeedPct: 40, PowerDrawW: 200}})
	_, gpu := m.History(7*24*time.Hour, ts.Add(time.Minute))
	require.Len(t, gpu, 2)
	assert.Equal(t, 40.0, gpu[0].GpuUtilPct)
	assert.Equal(t, 20.0, gpu[0].MemUtilPct)
	assert.Equal(t, 50, gpu[0].TempC)
	assert.Equal(t, 60, gpu[0].VramTempC)
	assert.Equal(t, 200, gpu[0].MemUsedMB)
	assert.Equal(t, 1000, gpu[0].MemTotalMB)
	assert.Equal(t, 30.0, gpu[0].FanSpeedPct)
	assert.Equal(t, 150.0, gpu[0].PowerDrawW)
	assert.Equal(t, 80.0, gpu[1].GpuUtilPct, "absent devices must not count as zero samples")
}

func TestMonitor_HistoryConfigReset(t *testing.T) {
	m, err := New(config.PerformanceConfig{Every: time.Minute}, newTestLogger())
	require.NoError(t, err)
	ts := time.Now()
	m.recordSys(SysStat{Timestamp: ts})
	m.recordGPU([]GpuStat{{Timestamp: ts}})
	m.UpdateConfig(config.PerformanceConfig{Every: time.Minute, Disabled: true})
	sys, gpu := m.History(7*24*time.Hour, ts)
	assert.Empty(t, sys)
	assert.Empty(t, gpu)
}

func TestMonitor_HistoryWeekWeightsIndividualSamples(t *testing.T) {
	m, err := New(config.PerformanceConfig{Every: time.Minute}, newTestLogger())
	require.NoError(t, err)
	ts := time.Date(2026, 9, 15, 12, 0, 0, 0, time.UTC)
	for _, s := range []SysStat{
		{Timestamp: ts, LoadAvg1: 0},
		{Timestamp: ts.Add(time.Minute), LoadAvg1: 0},
		{Timestamp: ts.Add(5 * time.Minute), LoadAvg1: 90},
	} {
		m.recordSys(s)
	}
	day, _ := m.History(24*time.Hour, ts.Add(5*time.Minute))
	week, _ := m.History(7*24*time.Hour, ts.Add(5*time.Minute))
	require.Len(t, day, 2)
	require.Len(t, week, 1)
	assert.Equal(t, 30.0, week[0].LoadAvg1, "do not average unequal-size five-minute buckets with equal weights")
}
