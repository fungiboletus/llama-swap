package router

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/router/scheduler"
	"github.com/mostlygeek/llama-swap/internal/store"
)

// Auto uses one configured memory pool to keep models resident until the next
// start requires evictions. A configured model memory value is authoritative.
// On Apple Silicon, an omitted value is learned from the model's OS process
// tree after its first request and persisted for subsequent launches.
type Auto struct {
	*baseRouter
}

const memoryObservationVersion = "process-tree-rss-v1"

func NewAuto(conf config.Config, proxylog, upstreamlog *logmon.Monitor, st store.Store) (*Auto, error) {
	if st == nil {
		return nil, fmt.Errorf("auto router requires a store")
	}
	if runtime.GOOS != "darwin" {
		for id, model := range conf.Models {
			if model.Memory == 0 {
				return nil, fmt.Errorf("auto router on %s requires models.%s.memory; automatic calibration is currently Apple-only", runtime.GOOS, id)
			}
		}
	}

	swapper, err := newAutoSwapper(conf, proxylog, st)
	if err != nil {
		return nil, err
	}
	processes := make(map[string]process.Process, len(conf.Models))
	base, err := newBaseRouter("auto", conf, processes, proxylog, swapper)
	if err != nil {
		return nil, fmt.Errorf("creating base router: %w", err)
	}
	for id, model := range conf.Models {
		procLog := logmon.NewWriter(upstreamlog)
		p, err := process.New(base.procCtx, id, model, procLog, proxylog)
		if err != nil {
			base.shutdownFn()
			base.procCancel()
			return nil, fmt.Errorf("creating process for %q: %w", id, err)
		}
		processes[id] = p
	}

	r := &Auto{baseRouter: base}
	go base.run()
	if runtime.GOOS == "darwin" {
		go r.observeProcessMemoryHighWater()
	}
	return r, nil
}

// observeProcessMemoryHighWater periodically refreshes each ready model's own
// process-tree measurement. Because observations are attributable by process,
// every resident model can be sampled even while other engines are running.
func (r *Auto) observeProcessMemoryHighWater() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-r.shutdownCtx.Done():
			return
		case <-ticker.C:
			running := r.RunningModels()
			for modelID, state := range running {
				if state == process.StateReady {
					r.observeProcessMemory(modelID, r.processes[modelID])
				}
			}
		}
	}
}

// autoSwapper is intentionally a small online admission policy, not a general
// placement solver. Its mutable measurements are protected because process
// observations run in serving and periodic-observer goroutines while scheduler
// lifecycle notifications run in the base router's event loop.
type autoSwapper struct {
	conf     config.Config
	logger   *logmon.Monitor
	store    store.Store
	capacity uint64

	mu           sync.Mutex
	profileMu    sync.Mutex
	profiles     map[string]uint64
	loadDuration map[string]time.Duration
	lastUse      map[string]time.Time
}

func newAutoSwapper(conf config.Config, logger *logmon.Monitor, st store.Store) (*autoSwapper, error) {
	settings := conf.Routing.Router.Settings.Auto
	if settings == nil || settings.Capacity <= settings.Reserve {
		return nil, fmt.Errorf("auto router requires a positive usable capacity")
	}
	p := &autoSwapper{
		conf:         conf,
		logger:       logger,
		store:        st,
		capacity:     uint64(settings.Capacity - settings.Reserve),
		profiles:     make(map[string]uint64),
		loadDuration: make(map[string]time.Duration),
		lastUse:      make(map[string]time.Time),
	}
	for id, model := range conf.Models {
		if model.Memory != 0 {
			continue
		}
		fingerprint := memoryProfileFingerprint(id, model)
		profile, found, err := st.MemoryProfiles().Load(context.Background(), id, fingerprint)
		if err != nil {
			return nil, err
		}
		if found && profile.ObservedBytes > 0 {
			p.profiles[id] = profile.ObservedBytes
		}
	}
	return p, nil
}

// EvictionFor calculates the smallest prefix of deterministically ranked idle
// candidates that allows target to fit. An unknown target is calibrated alone:
// every running model is selected for eviction before its first start.
func (p *autoSwapper) EvictionFor(target string, running []string) []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	if contains(running, target) {
		return nil
	}
	targetBytes, known := p.reservationLocked(target)
	if !known {
		return sortedCopy(running)
	}

	used := uint64(0)
	allKnown := true
	for _, id := range running {
		bytes, known := p.reservationLocked(id)
		if !known {
			// An in-flight first calibration is conservatively treated as using
			// the whole pool until it reports a profile.
			used = p.capacity
			allKnown = false
			break
		}
		used += bytes
	}
	if used <= p.capacity && targetBytes <= p.capacity-used {
		return nil
	}

	candidates := sortedCopy(running)
	originalUsed := used
	now := time.Now()
	sort.SliceStable(candidates, func(i, j int) bool {
		return p.moreEvictableLocked(candidates[i], candidates[j], now)
	})
	result := make([]string, 0, len(candidates))
	for _, id := range candidates {
		result = append(result, id)
		bytes, known := p.reservationLocked(id)
		if !known || bytes >= used {
			used = 0
		} else {
			used -= bytes
		}
		if targetBytes <= p.capacity-used {
			if allKnown {
				result = p.pruneRedundantEvictionsLocked(result, originalUsed, targetBytes)
			}
			return result
		}
	}
	// Configuration validation guarantees an individually known target fits.
	return result
}

// pruneRedundantEvictionsLocked removes models that a ranked-prefix decision
// selected before a later, larger eviction made them unnecessary. Iterating in
// reverse preserves the planner's preference order: less-evictable models are
// kept in the eviction set only when capacity actually requires them.
func (p *autoSwapper) pruneRedundantEvictionsLocked(selected []string, used, targetBytes uint64) []string {
	remainingUsed := used
	for _, id := range selected {
		bytes, _ := p.reservationLocked(id)
		if bytes >= remainingUsed {
			remainingUsed = 0
		} else {
			remainingUsed -= bytes
		}
	}

	keep := make([]bool, len(selected))
	for i := len(selected) - 1; i >= 0; i-- {
		bytes, _ := p.reservationLocked(selected[i])
		if bytes <= p.capacity-remainingUsed && targetBytes <= p.capacity-(remainingUsed+bytes) {
			remainingUsed += bytes
			continue
		}
		keep[i] = true
	}

	result := make([]string, 0, len(selected))
	for i, id := range selected {
		if keep[i] {
			result = append(result, id)
		}
	}
	return result
}

func (p *autoSwapper) OnSwapStart(target string, running []string) {
	evict := p.EvictionFor(target, running)
	preferredEvictions := make([]string, 0, len(evict))
	for _, modelID := range evict {
		if model, found := p.conf.Models[modelID]; found && model.KeepResident {
			preferredEvictions = append(preferredEvictions, modelID)
		}
	}
	p.mu.Lock()
	targetBytes, known := p.reservationLocked(target)
	used := uint64(0)
	for _, id := range running {
		if bytes, ok := p.reservationLocked(id); ok {
			used += bytes
		}
	}
	p.mu.Unlock()
	if p.logger != nil {
		if len(preferredEvictions) > 0 {
			p.logger.Warnf("auto: evicting keep-resident model(s)=%v for target=%s reason=capacity required eviction", preferredEvictions, target)
		}
		reason := "fits configured or calibrated reservation"
		if !known {
			reason = "unprofiled model is calibrated in isolation"
		}
		p.logger.Infof("auto: model=%s reservation_bytes=%d used_bytes=%d capacity_bytes=%d evict=%v reason=%s", target, targetBytes, used, p.capacity, evict, reason)
	}
}

func (p *autoSwapper) OnSwapDone(ev scheduler.SwapDone) {
	p.mu.Lock()
	p.loadDuration[ev.ModelID] = ev.LoadDuration
	p.mu.Unlock()
}

func (p *autoSwapper) MemoryObservationMode(modelID string) (observe, synchronous bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	model, found := p.conf.Models[modelID]
	if !found || model.Memory != 0 {
		return false, false
	}
	_, profiled := p.profiles[modelID]
	return true, !profiled
}

// OnMemoryObserved records an OS-attributed process-tree RSS high-water mark.
// It may run concurrently from serving and periodic-observer goroutines, so
// profileMu serializes persistence and prevents an older, smaller observation
// from winning an out-of-order database write.
func (p *autoSwapper) OnMemoryObserved(modelID string, observed uint64, sampleErr error) {
	model, found := p.conf.Models[modelID]
	if !found || model.Memory != 0 {
		return
	}
	if sampleErr != nil {
		if p.logger != nil {
			p.logger.Debugf("auto: sampling process-tree memory for %s: %v", modelID, sampleErr)
		}
		return
	}
	if observed == 0 {
		return
	}

	p.profileMu.Lock()
	defer p.profileMu.Unlock()
	p.mu.Lock()
	previous := p.profiles[modelID]
	p.mu.Unlock()
	if observed <= previous {
		return
	}

	profile := store.MemoryProfile{
		ModelID:       modelID,
		Fingerprint:   memoryProfileFingerprint(modelID, model),
		ObservedBytes: observed,
		ObservedAt:    time.Now(),
	}
	if err := p.store.MemoryProfiles().Save(context.Background(), profile); err != nil {
		if p.logger != nil {
			p.logger.Warnf("auto: saving process-tree memory profile for %s: %v", modelID, err)
		}
		return
	}
	p.mu.Lock()
	p.profiles[modelID] = observed
	p.mu.Unlock()
	if p.logger != nil {
		p.logger.Infof("auto: calibrated model=%s observed_bytes=%d", modelID, observed)
	}
}

func (p *autoSwapper) OnServeDone(modelID string) {
	p.mu.Lock()
	p.lastUse[modelID] = time.Now()
	p.mu.Unlock()
}

func (p *autoSwapper) reservationLocked(modelID string) (uint64, bool) {
	if model, ok := p.conf.Models[modelID]; ok && model.Memory != 0 {
		return uint64(model.Memory), true
	}
	bytes, ok := p.profiles[modelID]
	return bytes, ok
}

// ResidencyInfo reports the effective model reservations used by auto routing.
func (r *Auto) ResidencyInfo() map[string]ResidencyInfo {
	swapper, ok := r.planner.(*autoSwapper)
	if !ok {
		return nil
	}
	swapper.mu.Lock()
	defer swapper.mu.Unlock()
	result := make(map[string]ResidencyInfo, len(swapper.conf.Models))
	for id, model := range swapper.conf.Models {
		if model.Memory != 0 {
			result[id] = ResidencyInfo{Bytes: uint64(model.Memory), Source: "configured"}
			continue
		}
		if bytes, ok := swapper.profiles[id]; ok {
			result[id] = ResidencyInfo{Bytes: bytes, Source: "calibrated"}
			continue
		}
		result[id] = ResidencyInfo{Source: "unprofiled"}
	}
	return result
}

func (p *autoSwapper) moreEvictableLocked(a, b string, now time.Time) bool {
	aPreferred := p.conf.Models[a].KeepResident
	bPreferred := p.conf.Models[b].KeepResident
	if aPreferred != bPreferred {
		return !aPreferred
	}
	score := func(id string) int64 {
		idle := time.Duration(0)
		if last, ok := p.lastUse[id]; ok {
			idle = now.Sub(last)
		} else {
			idle = time.Duration(1<<63 - 1)
		}
		load := p.loadDuration[id]
		if load <= 0 {
			load = time.Second
		}
		return int64(idle/time.Second) / max(1, int64(load/time.Second))
	}
	aScore, bScore := score(a), score(b)
	if aScore != bScore {
		return aScore > bScore
	}
	return a < b
}

func memoryProfileFingerprint(modelID string, model config.ModelConfig) string {
	encoded, _ := json.Marshal(struct {
		ObservationVersion string
		ID                 string
		Cmd                string
		CmdStop            string
		Env                []string
		Proxy              string
		ConcurrencyLimit   int
	}{memoryObservationVersion, modelID, model.Cmd, model.CmdStop, model.Env, model.Proxy, model.ConcurrencyLimit})
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:])
}

func sortedCopy(ids []string) []string {
	result := append([]string(nil), ids...)
	sort.Strings(result)
	return result
}

func contains(ids []string, target string) bool {
	for _, id := range ids {
		if id == target {
			return true
		}
	}
	return false
}
