package server

import (
	"bufio"
	"bytes"
	"compress/flate"
	"compress/gzip"
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/mostlygeek/llama-swap/internal/event"
	"github.com/mostlygeek/llama-swap/internal/hint"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/server/usage"
	"github.com/mostlygeek/llama-swap/internal/store"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
	"github.com/mostlygeek/llama-swap/internal/tailcat"
	"github.com/tidwall/gjson"
)

type TokenMetrics = store.TokenMetrics
type ActivityLogEntry = store.ActivityLogEntry

// ActivityLogEvent carries a single activity log entry to event subscribers.
type ActivityLogEvent struct {
	Metrics ActivityLogEntry
}

func (e ActivityLogEvent) Type() uint32 {
	return swaputil.ActivityLogEventID
}

// metricsMonitor parses upstream responses for token statistics, stores
// activity in a store, and (when captures are enabled) stores
// zstd+CBOR-compressed request/response captures in the store's capture
// repository, bounded to maxCaptureBytes in total.
//
// Captures are written on the request path, but pruning them back under
// maxCaptureBytes runs on a background goroutine so a request never waits for
// old captures to be deleted. The total can briefly exceed maxCaptureBytes
// while a prune is pending.
type metricsMonitor struct {
	store           store.Store
	maxMetrics      int
	logger          *logmon.Monitor
	enableCaptures  bool
	maxCaptureBytes int64

	// pruneCh wakes the prune goroutine. It holds one pending request, so
	// many captures written while a prune runs cause one more prune, not
	// one each. Nil when captures are disabled.
	pruneCh   chan struct{}
	pruneStop context.CancelFunc
	pruneDone chan struct{}
	closeOnce sync.Once
}

// capturePruneTimeout bounds one background prune. A prune after
// captureBuffer is lowered can delete a lot; one that times out is resumed
// by the next capture.
const capturePruneTimeout = 5 * time.Minute

func newMetricsMonitor(logger *logmon.Monitor, maxMetrics int, captureBufferMB int, st store.Store) *metricsMonitor {
	if maxMetrics <= 0 {
		maxMetrics = 1000
	}
	mm := &metricsMonitor{
		logger:         logger,
		store:          st,
		maxMetrics:     maxMetrics,
		enableCaptures: captureBufferMB > 0,
	}
	if captureBufferMB > 0 {
		mm.maxCaptureBytes = int64(captureBufferMB) * 1024 * 1024
		mm.startCapturePruner()
		// The store outlives a config reload, so apply a lowered
		// captureBuffer now rather than on the next capture.
		mm.requestCapturePrune()
	}
	return mm
}

// startCapturePruner starts the goroutine that prunes captures. Close stops
// it.
func (mp *metricsMonitor) startCapturePruner() {
	ctx, cancel := context.WithCancel(context.Background())
	mp.pruneCh = make(chan struct{}, 1)
	mp.pruneStop = cancel
	mp.pruneDone = make(chan struct{})
	go func() {
		defer close(mp.pruneDone)
		for {
			select {
			case <-ctx.Done():
				return
			case <-mp.pruneCh:
				mp.pruneCaptures(ctx)
			}
		}
	}()
}

// requestCapturePrune asks the prune goroutine to bring captures back under
// maxCaptureBytes. It never blocks: if a request is already pending, this one
// is folded into it.
func (mp *metricsMonitor) requestCapturePrune() {
	select {
	case mp.pruneCh <- struct{}{}:
	default:
	}
}

func (mp *metricsMonitor) pruneCaptures(ctx context.Context) {
	ctx, cancel := context.WithTimeout(ctx, capturePruneTimeout)
	defer cancel()
	if err := mp.store.Captures().Prune(ctx, mp.maxCaptureBytes); err != nil && !errors.Is(err, context.Canceled) {
		mp.warnf("failed to prune captures: %v", err)
	}
}

// queueMetrics persists a metric and returns the store-assigned row. It
// deliberately does not take the request context: record runs after the
// handler returns, and an aborted request (canceled context) must still be
// recorded — that is exactly when the error entry matters.
func (mp *metricsMonitor) queueMetrics(metric ActivityLogEntry) (ActivityLogEntry, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	stored, err := mp.store.Activity().Insert(ctx, metric)
	if err != nil {
		mp.warnf("failed to persist activity metric: %v", err)
		return ActivityLogEntry{}, false
	}
	if mp.store.IsInMemory() {
		if err := mp.store.Activity().Prune(ctx, mp.maxMetrics); err != nil {
			mp.warnf("failed to prune activity metrics: %v", err)
		}
	}
	return stored, true
}

// emitMetric publishes an ActivityLogEvent for the given metric.
func (mp *metricsMonitor) emitMetric(metric ActivityLogEntry) {
	event.Emit(ActivityLogEvent{Metrics: metric})
}

// overlayCaptureState sets HasCapture on entries whose capture is still in
// the capture repository. Stored captures are hidden while captures are
// disabled.
func (mp *metricsMonitor) overlayCaptureState(entries []ActivityLogEntry) {
	var found map[string]bool
	if mp.enableCaptures {
		ids := make([]string, 0, len(entries))
		for i := range entries {
			if entries[i].CaptureID != "" {
				ids = append(ids, entries[i].CaptureID)
			}
		}
		if len(ids) > 0 {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			var err error
			if found, err = mp.store.Captures().Has(ctx, ids); err != nil {
				mp.warnf("failed to look up captures: %v", err)
			}
		}
	}
	for i := range entries {
		entries[i].HasCapture = entries[i].CaptureID != "" && found[entries[i].CaptureID]
	}
}

// newCaptureID returns a capture ID for an activity row recorded at t that
// will get a capture, or "" when captures are disabled. The ID is recorded on
// the row before the capture is written; if writing fails or the capture is
// pruned later, the row simply has no capture.
func (mp *metricsMonitor) newCaptureID(t time.Time) string {
	if !mp.enableCaptures {
		return ""
	}
	return store.NewCaptureID(t)
}

func (mp *metricsMonitor) warnf(format string, args ...any) {
	if mp.logger != nil {
		mp.logger.Warnf(format, args...)
	}
}

func (mp *metricsMonitor) debugf(format string, args ...any) {
	if mp.logger != nil {
		mp.logger.Debugf(format, args...)
	}
}

// Close stops the capture prune goroutine, cancelling a prune in progress,
// and waits for it to exit so the store can be closed after it. It is safe to
// call more than once.
func (mp *metricsMonitor) Close() error {
	mp.closeOnce.Do(func() {
		if mp.pruneStop != nil {
			mp.pruneStop()
			<-mp.pruneDone
		}
	})
	return nil
}

// activitySource returns connection metadata for activity records. It
// prefers Tailcat's authenticated source, then falls back to the
// X-Forwarded-For/X-Real-IP headers (prefixed with "xff:" since a proxy
// header is client-supplied and can be spoofed), and finally the raw
// connection address.
func activitySource(r *http.Request) string {
	if source, ok := tailcat.SourceFromContext(r.Context()); ok {
		return source
	}
	if ip, ok := forwardedIP(r); ok {
		return "xff:" + ip
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	return "ip:" + host
}

// responseFormat names the wire format of a buffered upstream response.
type responseFormat uint8

const (
	// formatJSON is a single JSON document, usually OpenAI-compatible.
	formatJSON responseFormat = iota
	// formatSSE is an upstream server-sent-events stream.
	formatSSE
	// formatOllama is a native Ollama response: one JSON object, or NDJSON
	// when streamed, whose final object carries the eval counts and durations.
	formatOllama
)

func (mp *metricsMonitor) record(modelID string, task hint.Task, r *http.Request, recorder *responseBodyCopier, cf captureFields, reqBody []byte, reqHeaders map[string]string) {
	mp.recordResponse(modelID, task, r, recorder, cf, reqBody, reqHeaders, formatJSON)
}

// recordStreaming records an upstream SSE response whose client-facing
// representation was converted to a non-SSE format before reaching the caller.
func (mp *metricsMonitor) recordStreaming(modelID string, task hint.Task, r *http.Request, recorder *responseBodyCopier, cf captureFields, reqBody []byte, reqHeaders map[string]string) {
	mp.recordResponse(modelID, task, r, recorder, cf, reqBody, reqHeaders, formatSSE)
}

// recordOllama records a response that was proxied unmodified from a managed
// Ollama lane, so it is in Ollama's native format rather than OpenAI's.
func (mp *metricsMonitor) recordOllama(modelID string, task hint.Task, r *http.Request, recorder *responseBodyCopier, cf captureFields, reqBody []byte, reqHeaders map[string]string) {
	mp.recordResponse(modelID, task, r, recorder, cf, reqBody, reqHeaders, formatOllama)
}

// newEntry starts the activity row for a finished request: what was asked,
// who asked, where it was routed, and the request's metadata. It also returns
// the upstream the router detected.
func (mp *metricsMonitor) newEntry(modelID string, task hint.Task, r *http.Request, recorder *responseBodyCopier) (ActivityLogEntry, hint.Upstream) {
	tm := ActivityLogEntry{
		Timestamp:         time.Now(),
		Model:             modelID,
		Task:              string(task),
		ReqPath:           r.URL.Path,
		RespContentType:   recorder.Header().Get("Content-Type"),
		RespStatusCode:    recorder.Status(),
		DurationMs:        int(time.Since(recorder.StartTime()).Milliseconds()),
		AuthenticatedUser: strings.TrimSpace(r.Header.Get("X-Authenticated-User")),
		Tokens:            TokenMetrics{ReasoningTokens: -1},
	}
	tm.Src = activitySource(r)

	// The router or peer proxy fills the hint while dispatching; it says
	// where the request went and which extractor understands the response.
	upstream := hint.Unknown
	if h := hint.From(r.Context()); h != nil {
		tm.Route = string(h.Route)
		tm.PeerID = h.PeerID
		upstream = h.Upstream
	}

	if ctxData, ok := swaputil.ReadContext(r.Context()); ok && len(ctxData.Metadata) > 0 {
		tm.Metadata = make(map[string]string, len(ctxData.Metadata))
		for k, v := range ctxData.Metadata {
			tm.Metadata[k] = v
		}
	}
	if selectorID := selectorFromContext(r.Context()); selectorID != "" {
		if tm.Metadata == nil {
			tm.Metadata = make(map[string]string)
		}
		tm.Metadata["selector"] = selectorID
	}
	return tm, upstream
}

// recordRealtime files one row for a finished /v1/realtime websocket. Its
// duration is the whole session, and its tokens and metadata are the
// session's tallies. No capture is kept: the traffic is mostly audio.
func (mp *metricsMonitor) recordRealtime(modelID string, task hint.Task, r *http.Request, recorder *responseBodyCopier) {
	tm, _ := mp.newEntry(modelID, task, r, recorder)
	recorder.realtime.fill(&tm)
	stored, ok := mp.queueMetrics(tm)
	if !ok {
		return
	}
	mp.emitMetric(stored)
}

// recordResponse parses a completed response body and stores/emits an
// activity entry. Successful requests store a zstd+CBOR capture (when
// enabled) with cf controlling which parts are retained. Failed (non-200)
// requests capture the request only and set ErrorMsg to a description of the
// failure, so the error can be inspected without storing unreadable raw
// response bytes. reqBody and reqHeaders are the request data buffered before
// dispatch. task is the kind of inference. format names the wire format of
// the buffered body.
func (mp *metricsMonitor) recordResponse(modelID string, task hint.Task, r *http.Request, recorder *responseBodyCopier, cf captureFields, reqBody []byte, reqHeaders map[string]string, format responseFormat) {
	tm, upstream := mp.newEntry(modelID, task, r, recorder)

	queueAndEmit := func() {
		stored, ok := mp.queueMetrics(tm)
		if !ok {
			return
		}
		tm = stored
		mp.emitMetric(tm)
	}

	// A client that hangs up is normal traffic, not a server fault: record the
	// entry so the abort is visible, but at debug level and without the
	// synthesized upstream error message the failure path below would attach.
	// No capture is stored — there is no response to inspect, and a client
	// retrying in a loop would otherwise flood the capture store. See #1029.
	if recorder.Status() == swaputil.StatusClientClosedRequest {
		mp.debugf("metrics: client disconnected before response, path=%s", r.URL.Path)
		tm.ErrorMsg = "client disconnected before response"
		queueAndEmit()
		return
	}

	if recorder.Status() != http.StatusOK {
		mp.logger.Warnf("non-200 response, recording partial metrics: status=%d, path=%s", recorder.Status(), r.URL.Path)
		decoded, decErr := mp.decodeResponseBody(recorder, r.URL.Path)
		tm.ErrorMsg = failedErrorMessage(recorder.Status(), decoded, decErr)
		tm.CaptureID = mp.newCaptureID(tm.Timestamp)
		stored, ok := mp.queueMetrics(tm)
		if !ok {
			return
		}
		tm = stored
		// Capture the request only; the failure is surfaced via ErrorMsg
		// rather than storing the (possibly undisplayable) response body.
		tm.HasCapture = mp.storeCapture(tm, r, recorder, cf&^captureRespBody, reqBody, reqHeaders, nil)
		mp.emitMetric(tm)
		return
	}

	body := recorder.body.Bytes()
	if len(body) == 0 {
		mp.logger.Warn("metrics: empty body, recording minimal metrics")
		queueAndEmit()
		return
	}

	if encoding := recorder.Header().Get("Content-Encoding"); encoding != "" {
		decoded, err := decompressBody(body, encoding)
		if err != nil {
			mp.logger.Warnf("metrics: decompression failed: %v, path=%s, recording minimal metrics", err, r.URL.Path)
			tm.ErrorMsg = fmt.Sprintf("response decompression failed: %v", err)
			queueAndEmit()
			return
		}
		body = decoded
	}

	if format == formatOllama {
		parsed, errMsg, err := processOllamaResponse(modelID, recorder.StartTime(), body)
		switch {
		case err != nil:
			mp.logger.Warnf("error processing Ollama response: %v, path=%s, recording minimal metrics", err, r.URL.Path)
		case errMsg != "":
			tm.ErrorMsg = errMsg
		default:
			tm.Tokens = parsed.Tokens
			tm.DurationMs = parsed.DurationMs
		}
	} else if u, err := usage.Extract(upstream, r.URL.Path, usageContentType(format, recorder), body, recorder.StartTime()); err != nil {
		mp.logger.Warnf("metrics: %v, path=%s, recording minimal metrics", err, r.URL.Path)
	} else if u.Found {
		tm.Tokens = u.Tokens
		applyObservedStreamingRates(&tm.Tokens, recorder)
		tm.DurationMs = u.DurationMs
	}

	tm.CaptureID = mp.newCaptureID(tm.Timestamp)
	stored, ok := mp.queueMetrics(tm)
	if !ok {
		return
	}
	tm = stored
	tm.HasCapture = mp.storeCapture(tm, r, recorder, cf, reqBody, reqHeaders, body)
	mp.emitMetric(tm)
}

// applyObservedStreamingRates fills rates that the upstream did not report.
// The estimates use only response arrival timestamps: prompt throughput covers
// request start through the first content delta, while generation throughput
// covers the intervals between the first and last content deltas. Upstream
// timings and metrics always take precedence.
func applyObservedStreamingRates(tokens *TokenMetrics, recorder *responseBodyCopier) {
	if tokens.PromptPerSecond < 0 && tokens.InputTokens > 0 && !recorder.firstTokenAt.IsZero() {
		promptTokens := tokens.InputTokens
		if tokens.CachedTokens > 0 {
			promptTokens -= tokens.CachedTokens
		}
		if elapsed := recorder.firstTokenAt.Sub(recorder.start); promptTokens > 0 && elapsed > 0 {
			tokens.PromptPerSecond = float64(promptTokens) / elapsed.Seconds()
		}
	}

	if tokens.TokensPerSecond < 0 && tokens.OutputTokens > 1 &&
		!recorder.firstTokenAt.IsZero() && !recorder.lastTokenAt.IsZero() {
		if elapsed := recorder.lastTokenAt.Sub(recorder.firstTokenAt); elapsed > 0 {
			// N tokens contain N-1 observable inter-token intervals.
			tokens.TokensPerSecond = float64(tokens.OutputTokens-1) / elapsed.Seconds()
		}
	}
}

// storeCapture assembles a ReqRespCapture for the stored activity row tm,
// honoring the captureFields mask, and stores it under tm.CaptureID when one
// was assigned. body is the response body to
// capture (already decompressed by the caller); pass nil to omit it. Returns
// true if a capture was stored.
func (mp *metricsMonitor) storeCapture(tm ActivityLogEntry, r *http.Request, recorder *responseBodyCopier, cf captureFields, reqBody []byte, reqHeaders map[string]string, body []byte) bool {
	if !mp.enableCaptures || tm.CaptureID == "" {
		return false
	}
	capture := ReqRespCapture{
		ID:         tm.ID,
		ReqPath:    r.URL.Path,
		ReqHeaders: reqHeaders,
	}
	if cf&captureReqBody != 0 {
		capture.ReqBody = reqBody
	}
	if cf&captureRespHeaders != 0 {
		capture.RespHeaders = headerMap(recorder.Header())
		redactHeaders(capture.RespHeaders)
		delete(capture.RespHeaders, "Content-Encoding")
	}
	if cf&captureRespBody != 0 {
		capture.RespBody = body
	}
	return mp.addCapture(tm.CaptureID, capture)
}

// decodeResponseBody returns the buffered response body, decompressing it when
// the upstream set a Content-Encoding we recognize. On decompression failure it
// logs a warning and returns an error so the caller can record a description
// (via ErrorMsg) instead of storing unreadable raw bytes.
func (mp *metricsMonitor) decodeResponseBody(recorder *responseBodyCopier, path string) ([]byte, error) {
	body := recorder.body.Bytes()
	if len(body) == 0 {
		return nil, nil
	}
	encoding := recorder.Header().Get("Content-Encoding")
	if encoding == "" {
		return body, nil
	}
	decoded, err := decompressBody(body, encoding)
	if err != nil {
		mp.logger.Warnf("metrics: response decompression failed: %v, path=%s", err, path)
		return nil, err
	}
	return decoded, nil
}

// errorMessagePaths lists JSON paths where a human-readable error message can
// live across OpenAI- and llama.cpp-style error responses.
var errorMessagePaths = []string{"error.message", "error", "message", "detail"}

// extractErrorMessage pulls a human-readable error string from a JSON error
// response. Returns "" if no message is found or the body is not valid JSON.
func extractErrorMessage(body []byte) string {
	if !gjson.ValidBytes(body) {
		return ""
	}
	parsed := gjson.ParseBytes(body)
	for _, path := range errorMessagePaths {
		v := parsed.Get(path)
		if v.Exists() && v.Type == gjson.String {
			if s := strings.TrimSpace(v.String()); s != "" {
				return s
			}
		}
	}
	return ""
}

// failedErrorMessage builds a human-readable description for a non-200 response.
// It prefers an error message parsed from the (decompressed) body and falls back
// to the HTTP status text. A non-nil decErr indicates the body could not be
// decoded, in which case the decode error is described instead.
func failedErrorMessage(status int, body []byte, decErr error) string {
	const maxLen = 500
	if decErr != nil {
		return fmt.Sprintf("response decode failed: %v", decErr)
	}
	if msg := extractErrorMessage(body); msg != "" {
		if len(msg) > maxLen {
			msg = msg[:maxLen] + "..."
		}
		return msg
	}
	if text := http.StatusText(status); text != "" {
		return fmt.Sprintf("%d %s", status, text)
	}
	return fmt.Sprintf("HTTP %d", status)
}

// processOllamaResponse reads the counts and timings from a native Ollama
// /api/generate response: either one JSON object or NDJSON, where the object
// with "done": true closes the stream. Ollama reports durations in
// nanoseconds, and omits prompt_eval_count when it served the prompt from its
// cache, in which case no prompt rate can be derived. An error object that
// arrives instead of a final one is returned as errMsg.
func processOllamaResponse(modelID string, start time.Time, body []byte) (entry ActivityLogEntry, errMsg string, err error) {
	var final, failure gjson.Result
	scan := func(obj gjson.Result) {
		if !obj.IsObject() {
			return
		}
		if obj.Get("done").Bool() {
			final = obj
		}
		if e := obj.Get("error"); e.Exists() && e.Type == gjson.String {
			failure = obj
		}
	}
	if gjson.ValidBytes(body) {
		scan(gjson.ParseBytes(body))
	} else {
		for _, line := range bytes.Split(body, []byte("\n")) {
			line = bytes.TrimSpace(line)
			if len(line) > 0 && gjson.ValidBytes(line) {
				scan(gjson.ParseBytes(line))
			}
		}
	}

	if !final.Exists() {
		if failure.Exists() {
			return ActivityLogEntry{}, strings.TrimSpace(failure.Get("error").String()), nil
		}
		return ActivityLogEntry{}, "", fmt.Errorf("no final Ollama object found")
	}

	input := final.Get("prompt_eval_count").Int()
	output := final.Get("eval_count").Int()
	promptPerSecond, tokensPerSecond := -1.0, -1.0
	if ns := final.Get("prompt_eval_duration").Float(); input > 0 && ns > 0 {
		promptPerSecond = float64(input) / (ns / 1e9)
	}
	// A single token has no interval to measure: Ollama reports a near-zero
	// eval_duration for it, which would yield an absurd rate.
	if ns := final.Get("eval_duration").Float(); output > 1 && ns > 0 {
		tokensPerSecond = float64(output) / (ns / 1e9)
	}
	durationMs := int(time.Since(start).Milliseconds())
	if total := int(final.Get("total_duration").Float() / 1e6); total > durationMs {
		durationMs = total
	}

	return ActivityLogEntry{
		Timestamp: time.Now(),
		Model:     modelID,
		Tokens: TokenMetrics{
			CachedTokens:    -1,
			DraftTokens:     -1,
			DraftAccTokens:  -1,
			InputTokens:     int(input),
			OutputTokens:    int(output),
			PromptPerSecond: promptPerSecond,
			TokensPerSecond: tokensPerSecond,
		},
		DurationMs: durationMs,
	}, "", nil
}

// decompressBody decompresses the body based on the Content-Encoding header.
func decompressBody(body []byte, encoding string) ([]byte, error) {
	switch strings.ToLower(strings.TrimSpace(encoding)) {
	case "gzip":
		reader, err := gzip.NewReader(bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		defer reader.Close()
		return io.ReadAll(reader)
	case "deflate":
		reader := flate.NewReader(bytes.NewReader(body))
		defer reader.Close()
		return io.ReadAll(reader)
	default:
		return body, nil
	}
}

// filterAcceptEncoding filters Accept-Encoding to only gzip/deflate so response
// bodies remain decompressible for metrics parsing.
func filterAcceptEncoding(acceptEncoding string) string {
	if acceptEncoding == "" {
		return ""
	}

	supported := map[string]bool{"gzip": true, "deflate": true}
	var filtered []string
	for part := range strings.SplitSeq(acceptEncoding, ",") {
		encoding, _, _ := strings.Cut(strings.TrimSpace(part), ";")
		if supported[strings.ToLower(encoding)] {
			filtered = append(filtered, strings.TrimSpace(part))
		}
	}
	return strings.Join(filtered, ", ")
}

// responseBodyCopier tees the upstream response to the client while buffering
// it for metrics parsing. Status defaults to 200 until WriteHeader is called.
type responseBodyCopier struct {
	http.ResponseWriter
	body         *bytes.Buffer
	tee          io.Writer
	status       int
	wroteHeader  bool
	start        time.Time
	observeSSE   bool
	ssePending   []byte
	firstTokenAt time.Time
	lastTokenAt  time.Time

	// realtime, when set, taps the connection Hijack hands out so a
	// /v1/realtime websocket can be tallied. See realtime.go.
	realtime *realtimeSession
}

func newBodyCopier(w http.ResponseWriter) *responseBodyCopier {
	buf := &bytes.Buffer{}
	return &responseBodyCopier{
		ResponseWriter: w,
		body:           buf,
		tee:            io.MultiWriter(w, buf),
		status:         http.StatusOK,
		start:          time.Now(),
	}
}

func (w *responseBodyCopier) Write(b []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	// On a protocol upgrade (e.g. websocket) the body is raw framed data, not a
	// metrics-parseable response, so write straight to the client without
	// buffering a copy we can't use.
	if w.status == http.StatusSwitchingProtocols {
		return w.ResponseWriter.Write(b)
	}
	n, err := w.tee.Write(b)
	if n > 0 && (w.observeSSE || strings.Contains(w.Header().Get("Content-Type"), "text/event-stream")) {
		w.observeStreamingChunk(b[:n], time.Now())
	}
	return n, err
}

// observeStreamingChunk records when content-bearing SSE deltas arrive. It
// retains only an unfinished SSE line between writes; the response body itself
// is already buffered transiently for usage parsing and capture policy decides
// whether any content is retained after the request.
func (w *responseBodyCopier) observeStreamingChunk(chunk []byte, observedAt time.Time) {
	w.ssePending = append(w.ssePending, chunk...)
	for {
		nl := bytes.IndexByte(w.ssePending, '\n')
		if nl < 0 {
			return
		}
		line := bytes.TrimSpace(w.ssePending[:nl])
		w.ssePending = w.ssePending[nl+1:]
		if !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}
		data := bytes.TrimSpace(line[len("data:"):])
		if !gjson.ValidBytes(data) || !streamingDeltaHasContent(gjson.ParseBytes(data)) {
			continue
		}
		if w.firstTokenAt.IsZero() {
			w.firstTokenAt = observedAt
		}
		w.lastTokenAt = observedAt
	}
}

func streamingDeltaHasContent(delta gjson.Result) bool {
	for _, path := range []string{
		"choices.0.delta.content",
		"choices.0.delta.reasoning",
		"choices.0.delta.reasoning_content",
		"choices.0.text",
		"delta.text",
		"delta.thinking",
	} {
		if value := delta.Get(path); value.Exists() && value.String() != "" {
			return true
		}
	}
	return delta.Get("type").String() == "response.output_text.delta" && delta.Get("delta").String() != ""
}

func (w *responseBodyCopier) WriteHeader(statusCode int) {
	// Forward 1xx interim responses (e.g. 100 Continue) without treating them
	// as the final status.
	if swaputil.IsInformationalStatus(statusCode) {
		w.ResponseWriter.WriteHeader(statusCode)
		return
	}
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	w.status = statusCode
	w.ResponseWriter.WriteHeader(statusCode)
}

// MarkStatus records code for metrics without writing to the client, and
// forwards to the wrapped writer so the access-log recorder agrees.
func (w *responseBodyCopier) MarkStatus(code int) {
	w.status = code
	if marker, ok := w.ResponseWriter.(swaputil.StatusMarker); ok {
		marker.MarkStatus(code)
	}
}

// WroteHeader reports whether a response status reached the client.
func (w *responseBodyCopier) WroteHeader() bool { return w.wroteHeader }

// Flush forwards to the underlying writer so streaming responses still flush.
func (w *responseBodyCopier) Flush() {
	f, ok := w.ResponseWriter.(http.Flusher)
	if !ok {
		return
	}
	// Flushing commits the implicit 200, so the response has started even if
	// nothing wrote a header. Recorded here for the same reason as in
	// statusRecorder: the client-closed sentinel must not overwrite it.
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	f.Flush()
}

// Hijack forwards to the underlying writer so httputil.ReverseProxy can take
// over the connection for websocket upgrades.
func (w *responseBodyCopier) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if hj, ok := w.ResponseWriter.(http.Hijacker); ok {
		conn, brw, err := hj.Hijack()
		if err != nil {
			return conn, brw, err
		}
		// ReverseProxy hijacks only once the upstream has agreed to switch
		// protocols, and writes that 101 straight to the connection rather
		// than through WriteHeader, so note it here.
		w.status = http.StatusSwitchingProtocols
		w.wroteHeader = true
		if w.realtime != nil {
			conn = &realtimeConn{Conn: conn, session: w.realtime}
		}
		return conn, brw, nil
	}
	return nil, nil, fmt.Errorf("underlying ResponseWriter does not support hijacking")
}

func (w *responseBodyCopier) Status() int          { return w.status }
func (w *responseBodyCopier) StartTime() time.Time { return w.start }

// usageContentType is the content type to hand the usage extractors. When the
// client-facing response was converted away from SSE the recorder's header no
// longer says the buffered body is an event stream, so say it explicitly.
func usageContentType(format responseFormat, recorder *responseBodyCopier) string {
	if format == formatSSE {
		return "text/event-stream"
	}
	return recorder.Header().Get("Content-Type")
}
