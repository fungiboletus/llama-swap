package perf

import (
	"sort"
	"time"

	"github.com/mostlygeek/llama-swap/internal/ring"
)

// rollup keeps completed averages plus the current bucket's raw samples.
// Each tier is bounded independently of how long the monitor has been running.
// Callers hold Monitor.mutex when reading or updating a rollup.
type rollup[T any] struct {
	interval  time.Duration
	start     time.Time
	active    []T
	completed ring.Buffer[T]
	average   func([]T) T
}

func newRollup[T any](window, interval time.Duration, average func([]T) T) rollup[T] {
	return rollup[T]{interval: interval, completed: ring.NewBuffer[T](int(window / interval)), average: average}
}

func (r *rollup[T]) push(ts time.Time, sample T) {
	start := ts.Truncate(r.interval)
	if !r.start.Equal(start) {
		if len(r.active) > 0 {
			r.completed.Push(r.average(r.active))
		}
		r.start = start
		r.active = nil
	}
	r.active = append(r.active, sample)
}

func (r *rollup[T]) current() []T {
	samples := r.completed.Slice()
	if len(r.active) > 0 {
		samples = append(samples, r.average(r.active))
	}
	return samples
}

// Gauges are arithmetic sample means. Network counters and timestamps use the
// last sample so adjacent points still give valid average bandwidth deltas.
func averageSys(samples []SysStat) SysStat {
	out := samples[len(samples)-1]
	out.CpuUtilPerCore = make([]float64, len(out.CpuUtilPerCore))
	counts := make([]int, len(out.CpuUtilPerCore))
	out.MemTotalMB, out.MemUsedMB, out.MemFreeMB = 0, 0, 0
	out.SwapTotalMB, out.SwapUsedMB = 0, 0
	out.LoadAvg1, out.LoadAvg5, out.LoadAvg15 = 0, 0, 0
	for _, s := range samples {
		for i, value := range s.CpuUtilPerCore {
			if i < len(out.CpuUtilPerCore) {
				out.CpuUtilPerCore[i] += value
				counts[i]++
			}
		}
		out.MemTotalMB += s.MemTotalMB
		out.MemUsedMB += s.MemUsedMB
		out.MemFreeMB += s.MemFreeMB
		out.SwapTotalMB += s.SwapTotalMB
		out.SwapUsedMB += s.SwapUsedMB
		out.LoadAvg1 += s.LoadAvg1
		out.LoadAvg5 += s.LoadAvg5
		out.LoadAvg15 += s.LoadAvg15
	}
	for i := range out.CpuUtilPerCore {
		if counts[i] > 0 {
			out.CpuUtilPerCore[i] /= float64(counts[i])
		}
	}
	n := len(samples)
	out.MemTotalMB /= n
	out.MemUsedMB /= n
	out.MemFreeMB /= n
	out.SwapTotalMB /= n
	out.SwapUsedMB /= n
	out.LoadAvg1 /= float64(n)
	out.LoadAvg5 /= float64(n)
	out.LoadAvg15 /= float64(n)
	out.NetIO = append([]NetIOStat(nil), out.NetIO...)
	return out
}

func averageGPU(snapshots [][]GpuStat) []GpuStat {
	type device struct {
		id         int
		uuid, name string
	}
	byDevice := make(map[device][]GpuStat)
	for _, snapshot := range snapshots {
		for _, s := range snapshot {
			key := device{s.ID, s.UUID, s.Name}
			byDevice[key] = append(byDevice[key], s)
		}
	}
	out := make([]GpuStat, 0, len(byDevice))
	for _, samples := range byDevice {
		s := samples[len(samples)-1]
		// Give every device in a bucket the same chart position, including
		// devices absent from the last snapshot.
		s.Timestamp = snapshots[len(snapshots)-1][0].Timestamp
		s.TempC, s.VramTempC, s.MemUsedMB, s.MemTotalMB = 0, 0, 0, 0
		s.GpuUtilPct, s.MemUtilPct, s.FanSpeedPct, s.PowerDrawW = 0, 0, 0, 0
		for _, g := range samples {
			s.TempC += g.TempC
			s.VramTempC += g.VramTempC
			s.MemUsedMB += g.MemUsedMB
			s.MemTotalMB += g.MemTotalMB
			s.GpuUtilPct += g.GpuUtilPct
			s.MemUtilPct += g.MemUtilPct
			s.FanSpeedPct += g.FanSpeedPct
			s.PowerDrawW += g.PowerDrawW
		}
		n := len(samples)
		s.TempC /= n
		s.VramTempC /= n
		s.MemUsedMB /= n
		s.MemTotalMB /= n
		s.GpuUtilPct /= float64(n)
		s.MemUtilPct /= float64(n)
		s.FanSpeedPct /= float64(n)
		s.PowerDrawW /= float64(n)
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].ID != out[j].ID {
			return out[i].ID < out[j].ID
		}
		if out[i].UUID != out[j].UUID {
			return out[i].UUID < out[j].UUID
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// History returns a bounded snapshot of the requested rolling window. The
// unfinished bucket is included and may change on the next call. History is
// in memory, starting at process startup or the last performance config change.
func (m *Monitor) History(window time.Duration, now time.Time) ([]SysStat, []GpuStat) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()
	var sys []SysStat
	var snapshots [][]GpuStat
	switch {
	case window > 24*time.Hour:
		sys, snapshots = m.sysWeek.current(), m.gpuWeek.current()
	case window > time.Hour:
		sys, snapshots = m.sysDay.current(), m.gpuDay.current()
	default:
		sys, snapshots = m.sysRing.Slice(), m.gpuRing.Slice()
	}
	cutoff := now.Add(-window)
	filteredSys := make([]SysStat, 0, len(sys))
	for _, s := range sys {
		if !s.Timestamp.Before(cutoff) && !s.Timestamp.After(now) {
			filteredSys = append(filteredSys, s)
		}
	}
	gpu := make([]GpuStat, 0)
	for _, snapshot := range snapshots {
		for _, s := range snapshot {
			if !s.Timestamp.Before(cutoff) && !s.Timestamp.After(now) {
				gpu = append(gpu, s)
			}
		}
	}
	return filteredSys, gpu
}

func (m *Monitor) resetHistory() {
	m.sysDay = newRollup(24*time.Hour, 5*time.Minute, averageSys)
	m.sysWeek = newRollup(7*24*time.Hour, 30*time.Minute, averageSys)
	m.gpuDay = newRollup(24*time.Hour, 5*time.Minute, averageGPU)
	m.gpuWeek = newRollup(7*24*time.Hour, 30*time.Minute, averageGPU)
}

func (m *Monitor) recordSys(s SysStat) {
	m.sysRing.Push(s)
	m.sysDay.push(s.Timestamp, s)
	m.sysWeek.push(s.Timestamp, s)
}

func (m *Monitor) recordGPU(snapshot []GpuStat) {
	m.gpuRing.Push(snapshot)
	if len(snapshot) > 0 {
		m.gpuDay.push(snapshot[0].Timestamp, snapshot)
		m.gpuWeek.push(snapshot[0].Timestamp, snapshot)
	}
}
