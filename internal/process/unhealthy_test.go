package process

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
)

// startWithUpstreamError runs a process whose upstream answers every chat
// request with status and body, and returns the process and that response.
func startWithUpstreamError(t *testing.T, rules []config.UnhealthyResponse, status int, body string) (*ProcessCommand, *httptest.ResponseRecorder) {
	t.Helper()
	skipIfNoSimpleResponder(t)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/chat/completions" {
			w.WriteHeader(status)
			io.WriteString(w, body) //nolint: errcheck
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(upstream.Close)

	cmd, _ := simpleResponderCmd(t, "-silent")
	p := newProcessCommand(t, config.ModelConfig{
		Cmd:                cmd,
		Proxy:              upstream.URL,
		CheckEndpoint:      "/health",
		HealthCheckTimeout: 10,
		UnhealthyResponses: rules,
	})
	t.Cleanup(func() { p.Stop(testStopTimeout) }) //nolint: errcheck
	_ = runAsync(t, p)

	rr := httptest.NewRecorder()
	p.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil))
	return p, rr
}

func TestProcessCommand_UnhealthyResponseStopsProcess(t *testing.T) {
	const body = `{"error": "generation thread died"}`
	p, rr := startWithUpstreamError(t,
		[]config.UnhealthyResponse{{Status: 404, BodyContains: "generation thread died"}},
		http.StatusNotFound, body)

	// The caller still gets the upstream's answer, unchanged.
	if rr.Code != http.StatusNotFound || rr.Body.String() != body {
		t.Fatalf("response = %d %q, want the upstream's 404 and body", rr.Code, rr.Body.String())
	}
	waitForState(t, p, StateStopped)
}

func TestProcessCommand_UnhealthyResponseIgnoresOtherErrors(t *testing.T) {
	tests := []struct {
		name   string
		rules  []config.UnhealthyResponse
		status int
		body   string
	}{
		{"other body", []config.UnhealthyResponse{{Status: 404, BodyContains: "generation thread died"}}, 404, `{"error":"model not found"}`},
		{"other status", []config.UnhealthyResponse{{Status: 404}}, 500, `{"error":"boom"}`},
		{"success", []config.UnhealthyResponse{{Status: 404}}, 200, `ok`},
		{"no rules", nil, 404, `{"error":"generation thread died"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, rr := startWithUpstreamError(t, tt.rules, tt.status, tt.body)
			if rr.Code != tt.status || !strings.Contains(rr.Body.String(), tt.body) {
				t.Fatalf("response = %d %q", rr.Code, rr.Body.String())
			}
			if got := p.State(); got != StateReady {
				t.Errorf("state = %s, want %s", got, StateReady)
			}
		})
	}
}
