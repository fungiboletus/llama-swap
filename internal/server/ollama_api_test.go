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
	"github.com/mostlygeek/llama-swap/internal/ollamaapi"
	"github.com/mostlygeek/llama-swap/internal/process"
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
			Models []ollamaapi.Tag `json:"models"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode tags: %v", err)
		}
		if got := len(response.Models); got != 1 {
			t.Fatalf("models = %#v", response.Models)
		}
		model := response.Models[0]
		if model.Name != "local-text" || model.Model != "local-text" || model.Digest != unknownOllamaDigest || model.Size != 0 || !model.ModifiedAt.IsZero() {
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

func TestOllamaAPI_PSReflectsReadyLocalModels(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  alpha:
    cmd: server --port ${PORT}
    memory: 2GiB
    capabilities:
      context: 4096
  zeta:
    cmd: server --port ${PORT}
  not-ready:
    cmd: server --port ${PORT}
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	local := newStubRouter([]string{"alpha", "zeta", "not-ready"}, "")
	s := newOllamaAPITestServer(t, cfg, local)

	assertPS := func(want string) {
		t.Helper()
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/ps", nil))
		if w.Code != http.StatusOK || w.Header().Get("Content-Type") != "application/json" || w.Body.String() != want {
			t.Errorf("status=%d content-type=%q body=%q, want %q", w.Code, w.Header().Get("Content-Type"), w.Body.String(), want)
		}
	}

	t.Run("empty until a model is ready", func(t *testing.T) {
		assertPS("{\"models\":[]}\n")
	})

	t.Run("ready entries are sorted and use only truthful metadata", func(t *testing.T) {
		local.running = map[string]process.ProcessState{
			"zeta":         process.StateReady,
			"alpha":        process.StateReady,
			"not-ready":    process.StateStarting,
			"unconfigured": process.StateReady,
		}
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/api/ps", nil))
		var response ollamaapi.ProcessResponse
		if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode ps response: %v", err)
		}
		if got, want := response.Models, 2; len(got) != want {
			t.Fatalf("models = %#v", got)
		}
		alpha, zeta := response.Models[0], response.Models[1]
		if alpha.Name != "alpha" || alpha.Model != "alpha" || alpha.Size != 2<<30 || alpha.SizeVRAM != 0 || alpha.ContextLength != 4096 || alpha.Digest != unknownOllamaDigest {
			t.Errorf("alpha = %#v", alpha)
		}
		if zeta.Name != "zeta" || zeta.Model != "zeta" || zeta.Size != 0 || zeta.SizeVRAM != 0 || zeta.ContextLength != 0 || zeta.Digest != unknownOllamaDigest {
			t.Errorf("zeta = %#v", zeta)
		}
		if alpha.ExpiresAt.Before(time.Now().Add(20*365*24*time.Hour)) || zeta.ExpiresAt.Before(time.Now().Add(20*365*24*time.Hour)) {
			t.Errorf("no-expiry sentinel = %s, %s", alpha.ExpiresAt, zeta.ExpiresAt)
		}
	})

	t.Run("unloaded transition removes the model", func(t *testing.T) {
		local.running = map[string]process.ProcessState{"zeta": process.StateStopping}
		assertPS("{\"models\":[]}\n")
	})
}

func TestOllamaAPI_LocalChatAndGenerateTranslation(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  local:
    cmd: server --port ${PORT}
    useModelName: upstream-local
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
		if request["model"] != "upstream-local" || request["temperature"] != float64(0.2) {
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
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"local","messages":[{"role":"user","content":"what is this?","images":["iVBORw0KGgo="]}]}`)))
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "does not declare image input capability") {
		t.Errorf("status=%d body=%q", w.Code, w.Body.String())
	}
	if called {
		t.Error("unsupported request reached local model")
	}
}

func TestOllamaAPI_LocalVisionTranslation(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  vision:
    cmd: server --port ${PORT}
    capabilities:
      in: [text, image]
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	const image = "iVBORw0KGgo="
	local := newStubRouter([]string{"vision"}, "")
	local.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode translated request: %v", err)
		}
		if request.Model != "vision" || len(request.Messages) != 1 || request.Messages[0].Role != "user" {
			t.Errorf("translated request = %#v", request)
		}
		var content []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			ImageURL struct {
				URL string `json:"url"`
			} `json:"image_url"`
		}
		if err := json.Unmarshal(request.Messages[0].Content, &content); err != nil {
			t.Fatalf("decode vision content: %v", err)
		}
		if len(content) != 2 || content[0].Type != "text" || content[0].Text != "describe this" || content[1].Type != "image_url" || content[1].ImageURL.URL != "data:image/png;base64,"+image {
			t.Errorf("vision content = %#v", content)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message":       map[string]any{"content": "an image"},
				"finish_reason": "stop",
			}},
		})
	}
	s := newOllamaAPITestServer(t, cfg, local)

	for _, tt := range []struct {
		name string
		path string
		body string
	}{
		{"chat", "/api/chat", `{"model":"vision","messages":[{"role":"user","content":"describe this","images":["` + image + `"]}],"stream":false}`},
		{"generate", "/api/generate", `{"model":"vision","prompt":"describe this","images":["` + image + `"],"stream":false}`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body)))
			if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "an image") {
				t.Errorf("status=%d body=%q", w.Code, w.Body.String())
			}
		})
	}
}

func TestOllamaAPI_RejectsInvalidImagesBeforeModelStart(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  vision:
    cmd: server --port ${PORT}
    capabilities:
      in: [image]
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	local := newStubRouter([]string{"vision"}, "")
	called := false
	local.serveHTTP = func(http.ResponseWriter, *http.Request) { called = true }
	s := newOllamaAPITestServer(t, cfg, local)

	for _, body := range []string{
		`{"model":"vision","messages":[{"role":"user","content":"x","images":"not-an-array"}]}`,
		`{"model":"vision","messages":[{"role":"user","content":"x","images":["not base64"]}]}`,
		`{"model":"vision","messages":[{"role":"user","content":"x","images":["dGV4dA=="]}]}`,
	} {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))
		if w.Code != http.StatusBadRequest {
			t.Errorf("status=%d body=%q", w.Code, w.Body.String())
		}
	}
	if called {
		t.Error("invalid image reached local model")
	}
}

func TestOllamaImageURLs_RejectsOversizedImage(t *testing.T) {
	_, err := ollamaImageURLs(json.RawMessage(`["iVBORw0KGgo="]`), 7)
	if err == nil || !strings.Contains(err.Error(), "maximum size") {
		t.Errorf("error = %v", err)
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
