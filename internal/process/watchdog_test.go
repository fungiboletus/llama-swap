package process

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
)

// watchdogModelConfig builds a model whose upstream is a mock server, so the test
// controls exactly how long the "inference server" takes to answer.
func watchdogModelConfig(t *testing.T, proxyURL string, stallTimeout, requestTimeout int) config.ModelConfig {
	t.Helper()
	cmd, _ := simpleResponderCmd(t, "-silent")
	cfg := config.ModelConfig{
		Cmd:                cmd,
		Proxy:              proxyURL,
		CheckEndpoint:      "/health",
		HealthCheckTimeout: 10,
		UnloadTimeout:      1,
		StallTimeout:       stallTimeout,
		RequestTimeout:     requestTimeout,
	}
	if runtime.GOOS == "windows" {
		cfg.CmdStop = "taskkill /f /t /pid ${PID}"
	}
	return cfg
}

// TestProcessCommand_StallTimeout_FailsRequestAndStopsProcess covers the whole
// point of the watchdog: an upstream that accepts a request and then goes
// silent forever fails that one request with a 504 and is stopped, rather than
// holding the caller open indefinitely.
func TestProcessCommand_StallTimeout_FailsRequestAndStopsProcess(t *testing.T) {
	skipIfNoSimpleResponder(t)

	release := make(chan struct{})
	t.Cleanup(func() { close(release) })

	mock := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(mock.Close)

	p := newProcessCommand(t, watchdogModelConfig(t, mock.URL, 1, 0))
	runErr := runAsync(t, p)

	rr := httptest.NewRecorder()
	p.ServeHTTP(rr, httptest.NewRequest("POST", "/v1/chat/completions", nil))

	if rr.Code != http.StatusGatewayTimeout {
		t.Errorf("stalled request: expected %d, got %d", http.StatusGatewayTimeout, rr.Code)
	}

	waitForState(t, p, StateStopped)

	select {
	case err := <-runErr:
		if err != nil {
			t.Errorf("Run() after stall stop: expected nil, got %v", err)
		}
	case <-time.After(testReturnTimeout):
		t.Fatal("Run() did not return after the stalled process was stopped")
	}
}

// TestProcessCommand_StallTimeout_ResetsOnProgress verifies the watchdog
// measures progress rather than duration: a stream that keeps producing chunks
// runs well past the stall budget without tripping it.
func TestProcessCommand_StallTimeout_ResetsOnProgress(t *testing.T) {
	skipIfNoSimpleResponder(t)

	mock := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		for i := 0; i < 8; i++ {
			fmt.Fprintf(w, "data: chunk %d\n\n", i)
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			select {
			case <-time.After(250 * time.Millisecond):
			case <-r.Context().Done():
				return
			}
		}
	}))
	t.Cleanup(mock.Close)

	p := newProcessCommand(t, watchdogModelConfig(t, mock.URL, 1, 0))
	runAsync(t, p)
	t.Cleanup(func() { p.Stop(testStopTimeout) })

	rr := httptest.NewRecorder()
	// Total response time is ~2s, well over the 1s stall budget, but no
	// individual gap between chunks is.
	p.ServeHTTP(rr, httptest.NewRequest("POST", "/v1/chat/completions", nil))

	if rr.Code != http.StatusOK {
		t.Errorf("streaming request: expected 200, got %d", rr.Code)
	}
	if got := rr.Body.String(); !strings.Contains(got, "chunk 7") {
		t.Errorf("streaming request was cut short, body: %q", got)
	}
	if got := p.State(); got != StateReady {
		t.Errorf("a progressing stream must not stop the process; state is %s", got)
	}
}

// TestProcessCommand_StallTimeout_ZeroDisables confirms a model with no stall
// budget is left alone, however long the upstream takes.
func TestProcessCommand_StallTimeout_ZeroDisables(t *testing.T) {
	skipIfNoSimpleResponder(t)

	mock := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			time.Sleep(1500 * time.Millisecond)
		}
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(mock.Close)

	p := newProcessCommand(t, watchdogModelConfig(t, mock.URL, 0, 0))
	runAsync(t, p)
	t.Cleanup(func() { p.Stop(testStopTimeout) })

	rr := httptest.NewRecorder()
	p.ServeHTTP(rr, httptest.NewRequest("POST", "/v1/chat/completions", nil))

	if rr.Code != http.StatusOK {
		t.Errorf("slow request with stall detection off: expected 200, got %d", rr.Code)
	}
	if got := p.State(); got != StateReady {
		t.Errorf("expected process to stay ready, state is %s", got)
	}
}

// TestProcessCommand_RequestTimeout_EndsRunawayGeneration covers the failure
// the stall budget structurally cannot see: an upstream that keeps producing
// tokens, and would reset the stall clock forever, is still ended once the
// whole request outlives its budget.
func TestProcessCommand_RequestTimeout_EndsRunawayGeneration(t *testing.T) {
	skipIfNoSimpleResponder(t)

	mock := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// The repetition loop: never stops, never goes quiet.
		for {
			fmt.Fprint(w, "data: the\n\n")
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			select {
			case <-time.After(50 * time.Millisecond):
			case <-r.Context().Done():
				return
			}
		}
	}))
	t.Cleanup(mock.Close)

	// No stall budget at all, so only the total-time ceiling can end this.
	p := newProcessCommand(t, watchdogModelConfig(t, mock.URL, 0, 1))
	runAsync(t, p)
	t.Cleanup(func() { p.Stop(testStopTimeout) })

	started := time.Now()
	rr := httptest.NewRecorder()
	p.ServeHTTP(rr, httptest.NewRequest("POST", "/v1/chat/completions", nil))

	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("runaway request was not ended: ran for %v", elapsed)
	}
	// A model that is still producing tokens is healthy; only the request was
	// at fault, so the process must survive it.
	if got := p.State(); got != StateReady {
		t.Errorf("a runaway request must not stop the process; state is %s", got)
	}
}

// TestProcessCommand_RequestTimeout_LeavesNormalRequestsAlone guards against
// the ceiling firing on requests that finish well inside it.
func TestProcessCommand_RequestTimeout_LeavesNormalRequestsAlone(t *testing.T) {
	skipIfNoSimpleResponder(t)

	mock := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "done")
	}))
	t.Cleanup(mock.Close)

	p := newProcessCommand(t, watchdogModelConfig(t, mock.URL, 600, 1800))
	runAsync(t, p)
	t.Cleanup(func() { p.Stop(testStopTimeout) })

	rr := httptest.NewRecorder()
	p.ServeHTTP(rr, httptest.NewRequest("POST", "/v1/chat/completions", nil))

	if rr.Code != http.StatusOK {
		t.Errorf("ordinary request: expected 200, got %d", rr.Code)
	}
	if got := rr.Body.String(); got != "done" {
		t.Errorf("ordinary request body = %q, want %q", got, "done")
	}
}
