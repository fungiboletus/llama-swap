package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
)

func newOllamaAPITestServer(t *testing.T, cfg config.Config, local *stubRouter) *Server {
	t.Helper()
	s := newTestServer(local, newStubRouter(nil, ""))
	s.cfg = cfg
	s.routes()
	return s
}

func TestOllamaAPI_DiscoveryAndShow(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  local-text:
    cmd: server --port ${PORT}
    aliases: [text-alias]
    name: Local text
    capabilities:
      in: [text]
      out: [text]
      context: 8192
  hidden:
    cmd: server --port ${PORT}
    unlisted: true
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	s := newOllamaAPITestServer(t, cfg, newStubRouter([]string{"local-text", "hidden"}, ""))

	t.Run("tags list canonical local names", func(t *testing.T) {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/tags", nil))
		if w.Code != http.StatusOK {
			t.Fatalf("status = %d", w.Code)
		}
		var response struct {
			Models []struct {
				Name  string `json:"name"`
				Model string `json:"model"`
			} `json:"models"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode tags: %v", err)
		}
		if got := len(response.Models); got != 1 {
			t.Fatalf("models = %#v", response.Models)
		}
		if response.Models[0].Name != "local-text" {
			t.Errorf("names = %#v", response.Models)
		}
	})

	t.Run("show resolves alias and latest spelling", func(t *testing.T) {
		for _, model := range []string{"text-alias", "local-text:latest"} {
			w := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"`+model+`"}`))
			s.ServeHTTP(w, req)
			if w.Code != http.StatusOK {
				t.Fatalf("%s: status = %d, body=%s", model, w.Code, w.Body.String())
			}
			var response struct {
				Capabilities []string       `json:"capabilities"`
				ModelInfo    map[string]any `json:"model_info"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
				t.Fatalf("%s: decode show: %v", model, err)
			}
			if !contains(response.Capabilities, "completion") || response.ModelInfo["llama.context_length"] != float64(8192) {
				t.Errorf("%s: response = %#v", model, response)
			}
		}
	})

	t.Run("unknown model has Ollama error shape", func(t *testing.T) {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/show", strings.NewReader(`{"model":"missing"}`)))
		if w.Code != http.StatusNotFound || w.Body.String() != "{\"error\":\"model not found\"}\n" {
			t.Errorf("status=%d body=%q", w.Code, w.Body.String())
		}
	})
}

func TestOllamaAPI_LocalChatAndGenerateTranslation(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  local:
    cmd: server --port ${PORT}
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	local := newStubRouter([]string{"local"}, "")
	local.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode translated request: %v", err)
		}
		if request["model"] != "local" || request["temperature"] != float64(0.2) {
			t.Errorf("translated request = %#v", request)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message":       map[string]any{"content": "hello back"},
				"finish_reason": "stop",
			}},
		})
	}
	s := newOllamaAPITestServer(t, cfg, local)

	for _, tt := range []struct {
		path string
		body string
		want string
	}{
		{"/api/chat", `{"model":"local","messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"temperature":0.2}}`, `{"model":"local","message":{"role":"assistant","content":"hello back"},"done":true,"done_reason":"stop"}` + "\n"},
		{"/api/generate", `{"model":"local","prompt":"hi","stream":false,"options":{"temperature":0.2}}`, `{"model":"local","response":"hello back","done":true,"done_reason":"stop"}` + "\n"},
	} {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body)))
		if w.Code != http.StatusOK || w.Body.String() != tt.want {
			t.Errorf("%s: status=%d body=%q", tt.path, w.Code, w.Body.String())
		}
	}
}

func TestOllamaAPI_LocalChatStreamsNDJSON(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  local:
    cmd: server --port ${PORT}
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	local := newStubRouter([]string{"local"}, "")
	local.serveHTTP = func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":null}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}
	s := newOllamaAPITestServer(t, cfg, local)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"local","messages":[{"role":"user","content":"hi"}]}`)))
	if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "application/x-ndjson" {
		t.Fatalf("status=%d content-type=%q", w.Code, w.Header().Get("Content-Type"))
	}
	if got, want := w.Body.String(), "{\"model\":\"local\",\"message\":{\"role\":\"assistant\",\"content\":\"hello\"},\"done\":false}\n{\"model\":\"local\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}\n"; got != want {
		t.Errorf("stream body = %q, want %q", got, want)
	}
}

func TestOllamaAPI_LocalUnsupportedInputDoesNotReachModel(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  local:
    cmd: server --port ${PORT}
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	local := newStubRouter([]string{"local"}, "")
	called := false
	local.serveHTTP = func(http.ResponseWriter, *http.Request) { called = true }
	s := newOllamaAPITestServer(t, cfg, local)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"local","messages":[{"role":"user","content":"what is this?","images":["not-an-image"]}]}`)))
	if w.Code != http.StatusNotImplemented || !strings.Contains(w.Body.String(), "images are not supported") {
		t.Errorf("status=%d body=%q", w.Code, w.Body.String())
	}
	if called {
		t.Error("unsupported request reached local model")
	}
}

func TestOllamaAPI_LocalStreamingCancellationReachesModel(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  local:
    cmd: server --port ${PORT}
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	local := newStubRouter([]string{"local"}, "")
	cancelled := make(chan struct{})
	local.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		<-r.Context().Done()
		close(cancelled)
	}
	s := newOllamaAPITestServer(t, cfg, local)
	ctx, cancel := context.WithCancel(context.Background())
	request := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"local","messages":[{"role":"user","content":"hi"}]}`)).WithContext(ctx)
	done := make(chan struct{})
	go func() {
		s.ServeHTTP(httptest.NewRecorder(), request)
		close(done)
	}()
	cancel()
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("cancellation did not reach local model")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("streaming handler did not return after cancellation")
	}
}
