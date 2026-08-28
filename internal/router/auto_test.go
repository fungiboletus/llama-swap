package router

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/router/scheduler"
	"github.com/mostlygeek/llama-swap/internal/store"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func autoConfig(models map[string]config.ModelConfig, capacity uint64) config.Config {
	return config.Config{
		HealthCheckTimeout: 5,
		Models:             models,
		Routing: config.RoutingConfig{Router: config.RouterConfig{
			Use: "auto",
			Settings: config.RouterSettings{Auto: &config.AutoRouterConfig{
				Capacity: config.ByteSize(capacity),
			}},
		}},
	}
}

func newTestAuto(t *testing.T, conf config.Config, processes map[string]process.Process) (*Auto, *autoSwapper) {
	t.Helper()
	st, err := store.New("")
	require.NoError(t, err)
	t.Cleanup(func() { _ = st.Close() })
	swapper, err := newAutoSwapper(conf, logmon.NewWriter(io.Discard), st, func() (uint64, error) { return 0, nil })
	require.NoError(t, err)
	base, err := newBaseRouter("auto", conf, processes, logmon.NewWriter(io.Discard), swapper)
	require.NoError(t, err)
	base.testProcessed = make(chan struct{}, 64)
	r := &Auto{baseRouter: base}
	go base.run()
	t.Cleanup(func() {
		if !r.shuttingDown.Load() {
			_ = r.Shutdown(time.Second)
		}
	})
	return r, swapper
}

func TestAutoSwapper_FitAndDeterministicEviction(t *testing.T) {
	conf := autoConfig(map[string]config.ModelConfig{
		"a": {Memory: 4},
		"b": {Memory: 4},
		"c": {Memory: 5},
	}, 11)
	st, err := store.New("")
	require.NoError(t, err)
	defer st.Close()
	p, err := newAutoSwapper(conf, logmon.NewWriter(io.Discard), st, nil)
	require.NoError(t, err)

	assert.Empty(t, p.EvictionFor("b", []string{"a"}), "4 + 4 fits")
	assert.Equal(t, []string{"a"}, p.EvictionFor("c", []string{"b", "a"}), "ties break by model ID")
	assert.Empty(t, p.EvictionFor("a", []string{"a", "b"}), "already-running model is never evicted")
}

func TestAutoSwapper_PrefersLongerIdleAndCheaperLoad(t *testing.T) {
	conf := autoConfig(map[string]config.ModelConfig{
		"old":  {Memory: 5},
		"busy": {Memory: 5},
		"next": {Memory: 6},
	}, 11)
	st, err := store.New("")
	require.NoError(t, err)
	defer st.Close()
	p, err := newAutoSwapper(conf, logmon.NewWriter(io.Discard), st, nil)
	require.NoError(t, err)
	p.lastUse["old"] = time.Now().Add(-10 * time.Minute)
	p.lastUse["busy"] = time.Now().Add(-time.Minute)
	p.loadDuration["old"] = time.Second
	p.loadDuration["busy"] = time.Minute

	assert.Equal(t, []string{"old"}, p.EvictionFor("next", []string{"busy", "old"}))
}

func TestAutoSwapper_PrefersOrdinaryModelsOverKeepResidentModels(t *testing.T) {
	conf := autoConfig(map[string]config.ModelConfig{
		"preferred": {Memory: 5, KeepResident: true},
		"ordinary":  {Memory: 5},
		"next":      {Memory: 6},
	}, 11)
	st, err := store.New("")
	require.NoError(t, err)
	defer st.Close()
	p, err := newAutoSwapper(conf, logmon.NewWriter(io.Discard), st, nil)
	require.NoError(t, err)

	// The ordinary model is more recently used, but it remains the first
	// eviction candidate because keepResident is a stronger soft preference.
	p.lastUse["preferred"] = time.Now().Add(-time.Hour)
	p.lastUse["ordinary"] = time.Now()
	assert.Equal(t, []string{"ordinary"}, p.EvictionFor("next", []string{"preferred", "ordinary"}))
}

func TestAutoSwapper_EvictsKeepResidentModelWhenCapacityRequiresIt(t *testing.T) {
	conf := autoConfig(map[string]config.ModelConfig{
		"preferred": {Memory: 5, KeepResident: true},
		"next":      {Memory: 6},
	}, 10)
	st, err := store.New("")
	require.NoError(t, err)
	defer st.Close()
	p, err := newAutoSwapper(conf, logmon.NewWriter(io.Discard), st, nil)
	require.NoError(t, err)

	assert.Equal(t, []string{"preferred"}, p.EvictionFor("next", []string{"preferred"}))
}

func TestAutoSwapper_LogsPreferredEviction(t *testing.T) {
	conf := autoConfig(map[string]config.ModelConfig{
		"preferred": {Memory: 5, KeepResident: true},
		"next":      {Memory: 6},
	}, 10)
	st, err := store.New("")
	require.NoError(t, err)
	defer st.Close()
	var logs bytes.Buffer
	p, err := newAutoSwapper(conf, logmon.NewWriter(&logs), st, nil)
	require.NoError(t, err)

	p.OnSwapStart("next", []string{"preferred"})
	assert.Contains(t, logs.String(), "evicting keep-resident model(s)=[preferred] for target=next reason=capacity required eviction")
}

func TestAutoSwapper_UnprofiledModelCalibratesAloneAndPersists(t *testing.T) {
	conf := autoConfig(map[string]config.ModelConfig{"model": {}}, 100)
	st, err := store.New("")
	require.NoError(t, err)
	defer st.Close()
	samples := []uint64{100, 140}
	p, err := newAutoSwapper(conf, logmon.NewWriter(io.Discard), st, func() (uint64, error) {
		value := samples[0]
		samples = samples[1:]
		return value, nil
	})
	require.NoError(t, err)

	assert.Equal(t, []string{"a", "b"}, p.EvictionFor("model", []string{"b", "a"}))
	p.BeforeStart("model")
	p.OnSwapDone(scheduler.SwapDone{ModelID: "model", LoadDuration: time.Second})
	assert.Empty(t, p.EvictionFor("model", nil), "calibrated model fits with its learned reservation")

	profile, found, err := st.LoadMemoryProfile(context.Background(), "model", memoryProfileFingerprint("model", conf.Models["model"]))
	require.NoError(t, err)
	require.True(t, found)
	assert.Equal(t, uint64(40), profile.ObservedBytes)
}

func TestAuto_ServesTwoResidentModelsThenEvictsOne(t *testing.T) {
	a := newFakeProcess("a")
	a.markReady()
	go a.Run(0)
	b := newFakeProcess("b")
	b.autoReady = true
	c := newFakeProcess("c")
	c.autoReady = true
	conf := autoConfig(map[string]config.ModelConfig{
		"a": {Memory: 4},
		"b": {Memory: 4},
		"c": {Memory: 5},
	}, 10)
	r, _ := newTestAuto(t, conf, map[string]process.Process{"a": a, "b": b, "c": c})

	for _, model := range []string{"b", "c"} {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, newRequest(model))
		if w.Code != http.StatusOK {
			t.Fatalf("%s status = %d body=%q", model, w.Code, w.Body.String())
		}
	}
	if got := a.stopCalls.Load(); got != 1 {
		t.Errorf("a stop calls = %d, want 1", got)
	}
	if got := b.stopCalls.Load(); got != 0 {
		t.Errorf("b stop calls = %d, want 0", got)
	}
	if got := c.runCalls.Load(); got != 1 {
		t.Errorf("c start calls = %d, want 1", got)
	}
}

func TestAuto_SerialPreloadRespectsCapacity(t *testing.T) {
	first := newFakeProcess("first")
	first.autoReady = true
	second := newFakeProcess("second")
	second.autoReady = true
	conf := autoConfig(map[string]config.ModelConfig{
		"first":  {Memory: 6, KeepResident: true},
		"second": {Memory: 6},
	}, 10)
	r, _ := newTestAuto(t, conf, map[string]process.Process{"first": first, "second": second})

	// startPreload invokes the local router serially; these requests exercise
	// the same deterministic order and prove that preload cannot overcommit
	// the auto-router budget.
	for _, modelID := range []string{"first", "second"} {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, newRequest(modelID))
		if w.Code != http.StatusOK {
			t.Fatalf("preload %s status = %d body=%q", modelID, w.Code, w.Body.String())
		}
	}

	running := r.RunningModels()
	if _, found := running["first"]; found {
		t.Errorf("first remains running after preload overflow: %v", running)
	}
	if running["second"] != process.StateReady {
		t.Errorf("second state = %q, want ready; running=%v", running["second"], running)
	}
	if got := first.stopCalls.Load(); got != 1 {
		t.Errorf("first stop calls = %d, want 1", got)
	}
}

var _ http.Handler = (*Auto)(nil)

func BenchmarkAutoSwapper_EvictionFor500Models(b *testing.B) {
	models := make(map[string]config.ModelConfig, 501)
	running := make([]string, 0, 500)
	for i := range 500 {
		id := fmt.Sprintf("model-%03d", i)
		models[id] = config.ModelConfig{Memory: 1}
		running = append(running, id)
	}
	models["target"] = config.ModelConfig{Memory: 2}
	conf := autoConfig(models, 500)
	st, err := store.New("")
	if err != nil {
		b.Fatal(err)
	}
	defer st.Close()
	p, err := newAutoSwapper(conf, logmon.NewWriter(io.Discard), st, nil)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		_ = p.EvictionFor("target", running)
	}
}

func TestAutoSwapper_500ModelsRegressionGuard(t *testing.T) {
	models := make(map[string]config.ModelConfig, 501)
	running := make([]string, 0, 500)
	for i := range 500 {
		id := fmt.Sprintf("model-%03d", i)
		models[id] = config.ModelConfig{Memory: 1}
		running = append(running, id)
	}
	models["target"] = config.ModelConfig{Memory: 2}
	st, err := store.New("")
	require.NoError(t, err)
	defer st.Close()
	p, err := newAutoSwapper(autoConfig(models, 500), logmon.NewWriter(io.Discard), st, nil)
	require.NoError(t, err)

	started := time.Now()
	for range 1_000 {
		_ = p.EvictionFor("target", running)
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("1,000 decisions across 500 models took %v, want under 1s", elapsed)
	}
}
