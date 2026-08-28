package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mostlygeek/llama-swap/internal/chain"
	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/event"
	"github.com/mostlygeek/llama-swap/internal/hw"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/perf"
	"github.com/mostlygeek/llama-swap/internal/router"
	"github.com/mostlygeek/llama-swap/internal/store"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// Server owns the HTTP mux, cross-cutting middleware, and the local/peer model
// dispatch. It supersedes router.Server: it builds the local and peer routers
// directly and dispatches between them itself.
type Server struct {
	cfg config.Config

	muxlog      *logmon.Monitor
	proxylog    *logmon.Monitor
	upstreamlog *logmon.Monitor

	perf     *perf.Monitor
	inflight *inflightTracker
	metrics  *metricsMonitor
	store    *store.Store
	build    BuildInfo
	hardware *hw.HardwareSnapshot

	profileMu     sync.RWMutex
	activeProfile string

	local router.LocalRouter
	peer  router.Router

	mux     *http.ServeMux
	handler http.Handler

	shutdownCtx  context.Context
	shutdownFn   context.CancelFunc
	shuttingDown atomic.Bool
}

// ActiveProfile returns the active runtime profile, or an empty string when no
// profile is active.
func (s *Server) ActiveProfile() string {
	s.profileMu.RLock()
	defer s.profileMu.RUnlock()
	return s.activeProfile
}

// setActiveProfile updates the runtime selection. An empty name deactivates
// profiles. It returns whether the selection changed.
func (s *Server) setActiveProfile(name string) (bool, error) {
	if name != "" {
		if _, ok := s.cfg.Profiles[name]; !ok {
			return false, fmt.Errorf("profile %q not found", name)
		}
	}
	s.profileMu.Lock()
	if s.activeProfile == name {
		s.profileMu.Unlock()
		return false, nil
	}
	s.activeProfile = name
	s.profileMu.Unlock()

	s.proxylog.Infof("active profile changed to %q", name)
	event.Emit(swaputil.ProfileChangedEvent{Active: name})
	return true, nil
}

// modelPostJSONRoutes are endpoints with a model id in the JSON request body.
var modelPostJSONRoutes = []string{
	"/v1/chat/completions",
	"/v1/responses",
	"/v1/completions",
	"/v1/messages",
	"/v1/messages/count_tokens",
	"/v1/embeddings",
	"/reranking",
	"/rerank",
	"/v1/rerank",
	"/v1/reranking",
	"/infill",
	"/completion",
	"/v1/audio/speech",
	"/v1/audio/voices",
	"/v1/images/generations",
	"/sdapi/v1/txt2img",
	"/sdapi/v1/img2img",

	// audio.cpp generic task API
	"/audioapi/v1/tasks/run",

	// versionless routes, the /v/ is stripped before the request is forwarded upstream
	// see issue #728
	"/v/chat/completions",
	"/v/responses",
	"/v/completions",
	"/v/messages",
	"/v/messages/count_tokens",
	"/v/embeddings",
	"/v/rerank",
	"/v/reranking",
}

// modelPostFormRoutes are multipart/form-data endpoints with a model id in the form data
var modelPostFormRoutes = []string{
	"/v1/audio/transcriptions",
	"/v1/images/edits",
}

// modelGetRoutes are model-dispatched GET endpoints (the model arrives as a
// query parameter).
var modelGetRoutes = []string{
	"/v1/audio/voices",
	"/sdapi/v1/loras",
	"/props",
}

// routeAccess determines which key set, if any, a route requires. Keeping this
// classification next to route registration makes a newly added endpoint fail
// fast during startup until its access policy is explicitly chosen.
type routeAccess uint8

const (
	routePublic routeAccess = iota
	routeInference
	routeAdmin
)

func routeAccessForPattern(pattern string) (routeAccess, bool) {
	for _, path := range modelPostJSONRoutes {
		if pattern == "POST "+path {
			return routeInference, true
		}
	}
	for _, path := range modelPostFormRoutes {
		if pattern == "POST "+path {
			return routeInference, true
		}
	}
	for _, path := range modelGetRoutes {
		if pattern == "GET "+path {
			return routeInference, true
		}
	}

	switch pattern {
	case "GET /v1/models", "GET /models", "GET /health", "GET /wol-health", "GET /{$}", "HEAD /{$}",
		"GET /api/version", "GET /api/tags", "GET /api/ps", "POST /api/show", "POST /api/chat", "POST /api/generate":
		return routeInference, true
	case "GET /favicon.ico":
		return routePublic, true
	case "GET /logs", "GET /logs/stream", "GET /logs/stream/{logMonitorID...}",
		"GET /ui/", "GET /metrics", "GET /unload", "GET /running", "GET /upstream",
		"/upstream/{upstreamPath...}", "/comfyui", "/comfyui/{comfyPath...}",
		"POST /api/models/unload", "POST /api/models/unload/{model...}", "GET /api/profiles",
		"PUT /api/profiles/active", "POST /api/inflight/{id}/cancel", "GET /api/events",
		"GET /api/metrics/activity", "GET /api/metrics/stats", "GET /api/performance",
		"GET /api/hardware", "GET /api/residency", "GET /api/captures/{id}":
		return routeAdmin, true
	default:
		return routePublic, false
	}
}

// isMetricsRecordPath reports whether path is one of the model-dispatched
// endpoints that the metrics middleware records in the activity log.
func isMetricsRecordPath(path string) bool {
	for _, p := range modelPostJSONRoutes {
		if p == path {
			return true
		}
	}
	for _, p := range modelPostFormRoutes {
		if p == path {
			return true
		}
	}
	for _, p := range modelGetRoutes {
		if p == path {
			return true
		}
	}
	return false
}

// BuildInfo carries version metadata surfaced by GET /api/version.
type BuildInfo struct {
	Version string
	Commit  string
	Date    string
}

func New(cfg config.Config, muxlog *logmon.Monitor, proxylog *logmon.Monitor, upstreamlog *logmon.Monitor, perfMon *perf.Monitor, st *store.Store, build BuildInfo, hardware *hw.HardwareSnapshot) (*Server, error) {
	if err := prepareOllamaModels(context.Background(), &cfg); err != nil {
		return nil, fmt.Errorf("preparing Ollama models: %w", err)
	}

	var local router.LocalRouter
	var err error

	switch cfg.Routing.Router.Use {
	case "matrix":
		local, err = router.NewMatrix(cfg, proxylog, upstreamlog)
		if err != nil {
			return nil, fmt.Errorf("creating matrix router: %w", err)
		}
	case "auto":
		local, err = router.NewAuto(cfg, proxylog, upstreamlog, st)
		if err != nil {
			return nil, fmt.Errorf("creating auto router: %w", err)
		}
	default: // "group"
		local, err = router.NewGroup(cfg, proxylog, upstreamlog)
		if err != nil {
			return nil, fmt.Errorf("creating group router: %w", err)
		}
	}

	peer, err := router.NewPeer(cfg, proxylog)
	if err != nil {
		return nil, fmt.Errorf("creating peer router: %w", err)
	}

	if st == nil {
		return nil, fmt.Errorf("store is required")
	}
	shutdownCtx, shutdownFn := context.WithCancel(context.Background())
	s := &Server{
		cfg:         cfg,
		muxlog:      muxlog,
		proxylog:    proxylog,
		upstreamlog: upstreamlog,
		perf:        perfMon,
		inflight:    newInflightTracker(),
		metrics:     newMetricsMonitor(proxylog, cfg.MetricsMaxInMemory, cfg.CaptureBuffer, st),
		store:       st,
		build:       build,
		hardware:    hardware,
		local:       local,
		peer:        peer,
		shutdownCtx: shutdownCtx,
		shutdownFn:  shutdownFn,
	}
	s.routes()
	if len(cfg.RequiredAPIKeys) == 0 && len(cfg.AdminAPIKeys) == 0 && proxylog != nil {
		proxylog.Warnf("no apiKeys or adminApiKeys configured; inference and administration are open")
	}
	s.startPreload()
	return s, nil
}

// localPeerHandler dispatches a model-routed request to the local or peer
// router. The model is resolved once via swaputil.FetchContext.
func (s *Server) localPeerHandler(w http.ResponseWriter, r *http.Request) {
	stripVersionPrefix(r)
	stripAudioAPIPrefix(r)

	data, err := swaputil.FetchContext(r, s.cfg)
	if err != nil {
		swaputil.SendError(w, r, swaputil.ErrNoModelInContext)
		return
	}

	switch {
	case s.local.Handles(data.ModelID):
		s.proxylog.Debugf("dispatch: using local process for model: %s", data.ModelID)
		s.local.ServeHTTP(w, r)
	case s.peer.Handles(data.ModelID):
		s.proxylog.Debugf("dispatch: using peer for model: %s", data.ModelID)
		s.peer.ServeHTTP(w, r)
	default:
		swaputil.SendError(w, r, router.ErrNoRouterFound)
	}
}

// stripVersionPrefix rewrites versionless /v/... requests to their /... form
// before forwarding upstream (issue #728).
func stripVersionPrefix(r *http.Request) {
	if strings.HasPrefix(r.URL.Path, "/v/") {
		r.URL.Path = strings.TrimPrefix(r.URL.Path, "/v")
	}
}

// stripAudioAPIPrefix rewrites /audioapi/... requests to their /... form
// before forwarding upstream, so /audioapi/v1/tasks/run reaches the upstream
// as /v1/tasks/run.
func stripAudioAPIPrefix(r *http.Request) {
	if strings.HasPrefix(r.URL.Path, "/audioapi") {
		r.URL.Path = strings.TrimPrefix(r.URL.Path, "/audioapi")
	}
}

// routes builds the mux, registers every route, and wraps the mux with the
// global CORS middleware.
func (s *Server) routes() {
	inferenceAuthMW := CreateInferenceAuthMiddleware(s.cfg)
	adminAuthMW := CreateAdminAuthMiddleware(s.cfg)
	modelChain := chain.New(
		CreateProfileMiddleware(s),
		CreateSelectorMiddleware(s),
		CreateRequestContextMiddleware(s.cfg),
		CreateInflightMiddleware(s.inflight, s.cfg),
		CreateFilterMiddleware(s.cfg),
		CreateFormFilterMiddleware(s.cfg),
		CreateMetricsMiddleware(s.metrics, s.cfg),
	)

	mux := http.NewServeMux()
	register := func(pattern string, handler http.Handler) {
		access, found := routeAccessForPattern(pattern)
		if !found {
			panic("server route has no access classification: " + pattern)
		}
		switch access {
		case routeInference:
			handler = inferenceAuthMW(handler)
		case routeAdmin:
			handler = adminAuthMW(handler)
		}
		mux.Handle(pattern, handler)
	}
	dispatch := http.HandlerFunc(s.localPeerHandler)

	for _, path := range modelPostJSONRoutes {
		register("POST "+path, modelChain.Then(dispatch))
	}
	for _, path := range modelPostFormRoutes {
		register("POST "+path, modelChain.Then(dispatch))
	}
	for _, path := range modelGetRoutes {
		register("GET "+path, modelChain.Then(dispatch))
	}

	// llama-swap API + custom endpoints.
	register("GET /v1/models", http.HandlerFunc(s.handleListModels))
	register("GET /models", http.HandlerFunc(s.handleListModels))
	register("GET /logs", http.HandlerFunc(s.handleLogs))
	register("GET /logs/stream", http.HandlerFunc(s.handleLogStream))
	register("GET /logs/stream/{logMonitorID...}", http.HandlerFunc(s.handleLogStream))

	register("GET /health", http.HandlerFunc(handleHealth))
	register("GET /wol-health", http.HandlerFunc(handleHealth))
	register("GET /{$}", http.HandlerFunc(handleRootRedirect))
	register("HEAD /{$}", http.HandlerFunc(s.handleOllamaRoot))

	// Minimal Ollama compatibility API. These endpoints translate requests before
	// entering the same local router and memory scheduler.
	register("GET /api/version", http.HandlerFunc(s.handleAPIVersion))
	register("GET /api/tags", http.HandlerFunc(s.handleOllamaTags))
	register("GET /api/ps", http.HandlerFunc(s.handleOllamaPS))
	register("POST /api/show", http.HandlerFunc(s.handleOllamaShow))
	register("POST /api/chat", http.HandlerFunc(s.handleOllamaChat))
	register("POST /api/generate", http.HandlerFunc(s.handleOllamaGenerate))

	// Embedded UI.
	register("GET /ui/", http.HandlerFunc(s.handleUI))
	register("GET /favicon.ico", http.HandlerFunc(s.handleFavicon))

	// Prometheus metrics are administrative because they expose host activity.
	register("GET /metrics", http.HandlerFunc(s.handleMetrics))

	// Operations endpoints.
	register("GET /unload", http.HandlerFunc(s.handleUnload))
	register("GET /running", http.HandlerFunc(s.handleRunning))

	// Upstream passthrough. Meter only the model-dispatched endpoints that can
	// produce token usage/timings.
	upstreamChain := chain.New(
		CreateProfileMiddleware(s),
		CreateUpstreamInflightMiddleware(s.inflight, s.cfg),
		CreateMetricsMiddleware(s.metrics, s.cfg),
	)
	register("GET /upstream", http.HandlerFunc(handleUpstreamRedirect))
	register("/upstream/{upstreamPath...}", upstreamChain.ThenFunc(s.handleUpstream))

	// ComfyUI compatibility passthrough. This uses the fixed comfyui_auto model,
	// whose compatibility settings are applied while loading config. Only the
	// root path may start an unloaded model.
	register("/comfyui", http.HandlerFunc(handleComfyUIRedirect))
	register("/comfyui/{comfyPath...}", http.HandlerFunc(s.handleComfyUI))

	// API group consumed by the administrative UI.
	register("POST /api/models/unload", http.HandlerFunc(s.handleAPIUnloadAll))
	register("POST /api/models/unload/{model...}", http.HandlerFunc(s.handleAPIUnloadModel))
	register("GET /api/profiles", http.HandlerFunc(s.handleAPIProfiles))
	register("PUT /api/profiles/active", http.HandlerFunc(s.handleAPIActiveProfile))
	register("POST /api/inflight/{id}/cancel", http.HandlerFunc(s.handleAPICancelInflight))
	register("GET /api/events", http.HandlerFunc(s.handleAPIEvents))
	register("GET /api/metrics/activity", http.HandlerFunc(s.handleAPIActivity))
	register("GET /api/metrics/stats", http.HandlerFunc(s.handleAPIActivityStats))
	register("GET /api/performance", http.HandlerFunc(s.handleAPIPerformance))
	register("GET /api/hardware", http.HandlerFunc(s.handleAPIHardware))
	register("GET /api/residency", http.HandlerFunc(s.handleAPIResidency))
	register("GET /api/captures/{id}", http.HandlerFunc(s.handleAPICapture))

	s.mux = mux
	s.handler = chain.New(CreateRequestLogMiddleware(s.proxylog), CreateCORSMiddleware()).Then(mux)
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.handler.ServeHTTP(w, r)
}

// CloseStreams cancels long-lived response streams (Server-Sent Events) so a
// graceful httpServer.Shutdown can drain without blocking on them. It does not
// tear down routers; call Shutdown for that. Safe to call repeatedly.
func (s *Server) CloseStreams() {
	s.shutdownFn()
}

// Shutdown stops the local and peer routers in parallel. It is idempotent;
// repeated calls return nil without re-running shutdown.
//
// Callers must drain inflight HTTP requests (httpServer.Shutdown) before
// calling this, otherwise inflight requests 502 when their processes are torn
// down. Call CloseStreams before httpServer.Shutdown so SSE streams do not
// block the drain.
func (s *Server) Shutdown(timeout time.Duration) error {
	if !s.shuttingDown.CompareAndSwap(false, true) {
		return nil
	}
	s.shutdownFn()

	var wg sync.WaitGroup
	var mu sync.Mutex
	var errs []error

	for _, rt := range []router.Router{s.local, s.peer} {
		if rt == nil {
			continue
		}
		wg.Add(1)
		go func(rt router.Router) {
			defer wg.Done()
			if err := rt.Shutdown(timeout); err != nil {
				mu.Lock()
				errs = append(errs, err)
				mu.Unlock()
			}
		}(rt)
	}

	wg.Wait()
	return errors.Join(errs...)
}
