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
	"github.com/mostlygeek/llama-swap/internal/router"
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
	local.residency = map[string]router.ResidencyInfo{
		"zeta": {Bytes: 9_426_059_264, Source: "calibrated"},
	}
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
		if zeta.Name != "zeta" || zeta.Model != "zeta" || zeta.Size != 9_426_059_264 || zeta.SizeVRAM != 0 || zeta.ContextLength != 0 || zeta.Digest != unknownOllamaDigest {
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

func TestOllamaAPI_GenerateSystemPromptForCommandBackedModel(t *testing.T) {
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
		var request struct {
			Messages []openAIChatMessage `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode translated request: %v", err)
		}
		if len(request.Messages) != 2 || request.Messages[0].Role != "system" || request.Messages[0].Content != "be brief" || request.Messages[1].Role != "user" || request.Messages[1].Content != "hello" {
			t.Errorf("messages = %#v", request.Messages)
		}
		json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{
			"message": map[string]any{"content": "hi"}, "finish_reason": "stop",
		}}})
	}
	s := newOllamaAPITestServer(t, cfg, local)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"local","prompt":"hello","system":"be brief","stream":false}`)))
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
}

func TestOllamaAPI_ManagedGenerateUsesNativeOllamaRequest(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  native:
    cmd: server --port ${PORT}
    aliases: [alias]
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	model := cfg.Models["native"]
	model.OllamaManaged = true
	cfg.Models["native"] = model

	local := newStubRouter([]string{"native"}, "")
	local.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Errorf("path = %q, want native Ollama endpoint", r.URL.Path)
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode native request: %v", err)
		}
		if request["model"] != "native" || request["raw"] != true || request["system"] != "be precise" || request["suffix"] != "END" || request["template"] != "{{ .Prompt }}" {
			t.Errorf("native request = %#v", request)
		}
		context, ok := request["context"].([]any)
		if !ok || len(context) != 3 || context[0] != float64(1) || context[2] != float64(3) {
			t.Errorf("context = %#v", request["context"])
		}
		if _, found := request["keep_alive"]; found {
			t.Errorf("keep_alive must remain owned by llama-swap: %#v", request)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"native","response":"continued","context":[4,5],"done":true}` + "\n"))
	}
	s := newOllamaAPITestServer(t, cfg, local)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"alias","prompt":"hello","context":[1,2,3],"suffix":"END","system":"be precise","template":"{{ .Prompt }}","raw":true,"keep_alive":"10m","stream":false}`)))
	if w.Code != http.StatusOK || w.Body.String() != `{"model":"native","response":"continued","context":[4,5],"done":true}`+"\n" {
		t.Errorf("status=%d body=%q", w.Code, w.Body.String())
	}
}

func TestOllamaAPI_RecordsActivity(t *testing.T) {
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
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message":       map[string]any{"content": "hello back"},
				"finish_reason": "stop",
			}},
			"usage": map[string]int{"prompt_tokens": 12, "completion_tokens": 7},
		})
	}
	s := newOllamaAPITestServer(t, cfg, local)

	for _, request := range []struct {
		path string
		body string
	}{
		{"/api/chat", `{"model":"local","messages":[{"role":"user","content":"hi"}],"stream":false}`},
		{"/api/generate", `{"model":"local","prompt":"hi","stream":false}`},
	} {
		w := httptest.NewRecorder()
		r := httptest.NewRequest(http.MethodPost, request.path, strings.NewReader(request.body))
		r.Header.Set("X-Authenticated-User", "  alice@example.com  ")
		s.ServeHTTP(w, r)
		if w.Code != http.StatusOK {
			t.Errorf("%s: status=%d body=%q", request.path, w.Code, w.Body.String())
		}
	}

	entries := metricsEntries(t, s.metrics)
	if len(entries) != 2 {
		t.Fatalf("activity entries = %d, want 2", len(entries))
	}
	for _, entry := range entries {
		if entry.Model != "local" {
			t.Errorf("model = %q, want local", entry.Model)
		}
		if entry.ReqPath != "/api/chat" && entry.ReqPath != "/api/generate" {
			t.Errorf("request path = %q", entry.ReqPath)
		}
		if entry.RespStatusCode != http.StatusOK {
			t.Errorf("response status = %d, want 200", entry.RespStatusCode)
		}
		if entry.AuthenticatedUser != "alice@example.com" {
			t.Errorf("authenticated user = %q, want alice@example.com", entry.AuthenticatedUser)
		}
		if entry.Tokens.InputTokens != 12 || entry.Tokens.OutputTokens != 7 {
			t.Errorf("tokens = %+v, want input=12 output=7", entry.Tokens)
		}
	}
}

func TestOllamaAPI_RecordsStreamingActivity(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  local:
    cmd: server --port ${PORT}
    filters:
      setParams:
        stream_options.include_usage: true
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	local := newStubRouter([]string{"local"}, "")
	local.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode translated request: %v", err)
		}
		streamOptions, ok := request["stream_options"].(map[string]any)
		if !ok || streamOptions["include_usage"] != true {
			t.Errorf("stream_options = %#v, want include_usage=true", request["stream_options"])
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n" +
			"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7}}\n\n" +
			"data: [DONE]\n\n"))
	}
	s := newOllamaAPITestServer(t, cfg, local)

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"local","messages":[{"role":"user","content":"hi"}]}`))
	r.Header.Set("X-Authenticated-User", "alice@example.com")
	s.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}

	entries := metricsEntries(t, s.metrics)
	if len(entries) != 1 {
		t.Fatalf("activity entries = %d, want 1", len(entries))
	}
	entry := entries[0]
	if entry.ReqPath != "/api/chat" || entry.AuthenticatedUser != "alice@example.com" {
		t.Errorf("activity entry = %#v", entry)
	}
	if entry.Tokens.InputTokens != 12 || entry.Tokens.OutputTokens != 7 {
		t.Errorf("tokens = %+v, want input=12 output=7", entry.Tokens)
	}
}

// managedGenerateTestServer wires a Server whose only model is an
// Ollama-managed lane answering /api/generate with handler.
func managedGenerateTestServer(t *testing.T, handler http.HandlerFunc) *Server {
	t.Helper()
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  native:
    cmd: server --port ${PORT}
    aliases: [alias]
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	model := cfg.Models["native"]
	model.OllamaManaged = true
	cfg.Models["native"] = model

	local := newStubRouter([]string{"native"}, "")
	local.serveHTTP = handler
	return newOllamaAPITestServer(t, cfg, local)
}

func TestOllamaAPI_ManagedGenerateRecordsActivity(t *testing.T) {
	s := managedGenerateTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"native","response":"e4","done":true,` +
			`"total_duration":1500000000,"prompt_eval_count":40,"prompt_eval_duration":400000000,` +
			`"eval_count":10,"eval_duration":500000000}` + "\n"))
	})

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"alias","prompt":"1.","stream":false}`))
	r.Header.Set("X-Authenticated-User", "alice@example.com")
	s.ServeHTTP(w, r)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"response":"e4"`) {
		t.Fatalf("client response changed: status=%d body=%q", w.Code, w.Body.String())
	}

	entries := metricsEntries(t, s.metrics)
	if len(entries) != 1 {
		t.Fatalf("activity entries = %d, want 1", len(entries))
	}
	entry := entries[0]
	if entry.Model != "native" || entry.ReqPath != "/api/generate" || entry.AuthenticatedUser != "alice@example.com" {
		t.Errorf("activity entry = %#v", entry)
	}
	if entry.Tokens.InputTokens != 40 || entry.Tokens.OutputTokens != 10 {
		t.Errorf("tokens = %+v, want input=40 output=10", entry.Tokens)
	}
	if entry.Tokens.PromptPerSecond != 100 || entry.Tokens.TokensPerSecond != 20 {
		t.Errorf("rates = prompt %v, generation %v; want 100 and 20", entry.Tokens.PromptPerSecond, entry.Tokens.TokensPerSecond)
	}
	if entry.DurationMs < 1500 {
		t.Errorf("duration = %dms, want at least the reported 1500ms", entry.DurationMs)
	}
}

func TestOllamaAPI_ManagedGenerateRecordsStreamingActivity(t *testing.T) {
	s := managedGenerateTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"model":"native","response":"e","done":false}` + "\n" +
			`{"model":"native","response":"4","done":false}` + "\n" +
			`{"model":"native","response":"","done":true,"total_duration":900000000,` +
			`"prompt_eval_count":8,"prompt_eval_duration":100000000,"eval_count":2,"eval_duration":200000000}` + "\n"))
	})

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"native","prompt":"1."}`)))
	if w.Code != http.StatusOK || strings.Count(w.Body.String(), "\n") != 3 {
		t.Fatalf("client stream changed: status=%d body=%q", w.Code, w.Body.String())
	}

	entries := metricsEntries(t, s.metrics)
	if len(entries) != 1 {
		t.Fatalf("activity entries = %d, want 1", len(entries))
	}
	if tokens := entries[0].Tokens; tokens.InputTokens != 8 || tokens.OutputTokens != 2 || tokens.TokensPerSecond != 10 {
		t.Errorf("tokens = %+v, want input=8 output=2 at 10 tokens/s", tokens)
	}
}

func TestOllamaAPI_ManagedGenerateRecordsOllamaError(t *testing.T) {
	s := managedGenerateTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte(`{"error":"model runner has unexpectedly stopped"}` + "\n"))
	})

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/generate", strings.NewReader(`{"model":"native","prompt":"1."}`)))

	entries := metricsEntries(t, s.metrics)
	if len(entries) != 1 {
		t.Fatalf("activity entries = %d, want 1", len(entries))
	}
	if entries[0].ErrorMsg != "model runner has unexpectedly stopped" {
		t.Errorf("error message = %q", entries[0].ErrorMsg)
	}
}

func TestOllamaAPI_ProcessOllamaResponse(t *testing.T) {
	start := time.Now()

	t.Run("prompt served from cache has no prompt rate", func(t *testing.T) {
		entry, errMsg, err := processOllamaResponse("m", start,
			[]byte(`{"done":true,"eval_count":5,"eval_duration":1000000000}`))
		if err != nil || errMsg != "" {
			t.Fatalf("err=%v errMsg=%q", err, errMsg)
		}
		if entry.Tokens.InputTokens != 0 || entry.Tokens.PromptPerSecond != -1 {
			t.Errorf("tokens = %+v, want no input and unknown prompt rate", entry.Tokens)
		}
		if entry.Tokens.OutputTokens != 5 || entry.Tokens.TokensPerSecond != 5 {
			t.Errorf("tokens = %+v, want 5 tokens at 5/s", entry.Tokens)
		}
	})

	t.Run("single token has no generation rate", func(t *testing.T) {
		entry, _, err := processOllamaResponse("m", start,
			[]byte(`{"done":true,"eval_count":1,"eval_duration":1000,"prompt_eval_count":10,"prompt_eval_duration":20000000}`))
		if err != nil {
			t.Fatal(err)
		}
		if entry.Tokens.OutputTokens != 1 || entry.Tokens.TokensPerSecond != -1 {
			t.Errorf("tokens = %+v, want 1 output token and unknown generation rate", entry.Tokens)
		}
		if entry.Tokens.PromptPerSecond != 500 {
			t.Errorf("prompt rate = %v, want 500", entry.Tokens.PromptPerSecond)
		}
	})

	t.Run("pretty printed single object", func(t *testing.T) {
		entry, _, err := processOllamaResponse("m", start,
			[]byte("{\n  \"done\": true,\n  \"eval_count\": 3,\n  \"prompt_eval_count\": 4\n}\n"))
		if err != nil || entry.Tokens.InputTokens != 4 || entry.Tokens.OutputTokens != 3 {
			t.Errorf("entry=%+v err=%v", entry.Tokens, err)
		}
	})

	t.Run("no final object", func(t *testing.T) {
		if _, _, err := processOllamaResponse("m", start, []byte(`{"response":"x","done":false}`+"\n")); err == nil {
			t.Error("expected an error when the stream never finished")
		}
	})
}

func TestOllamaAPI_EmptyToolsAreNotToolCalls(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  local:
    cmd: server --port ${PORT}
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	local := newStubRouter([]string{"local"}, "")
	called := 0
	local.serveHTTP = func(w http.ResponseWriter, _ *http.Request) {
		called++
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message":       map[string]any{"content": "hello back"},
				"finish_reason": "stop",
			}},
		})
	}
	s := newOllamaAPITestServer(t, cfg, local)

	for _, tools := range []string{"", `,"tools":null`, `,"tools":[]`} {
		called = 0
		w := httptest.NewRecorder()
		body := `{"model":"local","messages":[{"role":"user","content":"hi"}],"stream":false` + tools + `}`
		s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(body)))
		if w.Code != http.StatusOK || called != 1 {
			t.Errorf("tools %q: status=%d calls=%d body=%q", tools, w.Code, called, w.Body.String())
		}
	}

	called = 0
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"local","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"weather"}}]}`)))
	if w.Code != http.StatusNotImplemented || called != 0 {
		t.Errorf("tool definition: status=%d calls=%d body=%q", w.Code, called, w.Body.String())
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

// TestOllamaAPI_LatestTagFallsBackBothWays covers the two ways an Ollama client
// can spell the same model. Which one is configured depends on where the model
// came from — the discovered catalogue reports `name:latest`, while a
// command-backed model usually has a bare ID — and clients type either.
func TestOllamaAPI_LatestTagFallsBackBothWays(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  bare-id:
    cmd: server --port ${PORT}
  tagged-id:latest:
    cmd: server --port ${PORT}
  other:q4:
    cmd: server --port ${PORT}
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	s := newOllamaAPITestServer(t, cfg, newStubRouter([]string{"bare-id", "tagged-id:latest", "other:q4"}, ""))

	resolves := []struct{ requested, wantLocalID string }{
		{"bare-id", "bare-id"},
		// `ollama run bare-id:latest` against a bare configured ID.
		{"bare-id:latest", "bare-id"},
		{"tagged-id:latest", "tagged-id:latest"},
		// `ollama run tagged-id` against a catalogue name that carries the tag.
		{"tagged-id", "tagged-id:latest"},
	}
	for _, tc := range resolves {
		target, found := s.resolveOllamaModel(tc.requested)
		if !found {
			t.Errorf("%s: expected to resolve", tc.requested)
			continue
		}
		if target.localID != tc.wantLocalID {
			t.Errorf("%s resolved to %s, want %s", tc.requested, target.localID, tc.wantLocalID)
		}
	}

	// An explicit non-latest tag names one specific model. Missing it is a real
	// 404, not an invitation to serve a different quantization.
	for _, requested := range []string{"other", "other:q8", "bare-id:q4", "missing"} {
		if _, found := s.resolveOllamaModel(requested); found {
			t.Errorf("%s: expected no match", requested)
		}
	}
}

// TestOllamaAPI_ThinkTranslation covers both directions of thinking support:
// the `think` request field reaching the upstream in the spellings the
// supported backends read, and a reasoning trace coming back as Ollama's
// separate `thinking` field rather than being dropped.
func TestOllamaAPI_ThinkTranslation(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  local:
    cmd: server --port ${PORT}
    capabilities:
      thinking: true
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}

	var seen map[string]any
	local := newStubRouter([]string{"local"}, "")
	local.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&seen); err != nil {
			t.Fatalf("decode translated request: %v", err)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{
				"message": map[string]any{
					"content":           "4",
					"reasoning_content": "2 plus 2 is 4",
				},
				"finish_reason": "stop",
			}},
		})
	}
	s := newOllamaAPITestServer(t, cfg, local)

	post := func(t *testing.T, path, body string) *httptest.ResponseRecorder {
		t.Helper()
		seen = nil
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
		return w
	}

	t.Run("show advertises the thinking capability", func(t *testing.T) {
		w := post(t, "/api/show", `{"model":"local"}`)
		var response struct {
			Capabilities []string `json:"capabilities"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode show: %v", err)
		}
		if !contains(response.Capabilities, "thinking") {
			t.Errorf("capabilities = %v", response.Capabilities)
		}
	})

	t.Run("think true enables thinking upstream", func(t *testing.T) {
		w := post(t, "/api/chat", `{"model":"local","messages":[{"role":"user","content":"2+2"}],"stream":false,"think":true}`)
		if w.Code != http.StatusOK {
			t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
		}
		kwargs, ok := seen["chat_template_kwargs"].(map[string]any)
		if !ok || kwargs["enable_thinking"] != true {
			t.Errorf("chat_template_kwargs = %#v", seen["chat_template_kwargs"])
		}
		if seen["reasoning_effort"] != "medium" {
			t.Errorf("reasoning_effort = %#v", seen["reasoning_effort"])
		}
		want := `{"model":"local","message":{"role":"assistant","content":"4","thinking":"2 plus 2 is 4"},"done":true,"done_reason":"stop"}` + "\n"
		if got := w.Body.String(); got != want {
			t.Errorf("body = %q, want %q", got, want)
		}
	})

	t.Run("think false asks the model to stop thinking", func(t *testing.T) {
		post(t, "/api/chat", `{"model":"local","messages":[{"role":"user","content":"2+2"}],"stream":false,"think":false}`)
		kwargs, ok := seen["chat_template_kwargs"].(map[string]any)
		if !ok || kwargs["enable_thinking"] != false {
			t.Errorf("chat_template_kwargs = %#v", seen["chat_template_kwargs"])
		}
		// An Ollama upstream ignores chat_template_kwargs entirely, so the
		// same "off" also goes out as the reasoning_effort it does read.
		if seen["reasoning_effort"] != "none" {
			t.Errorf("reasoning_effort = %#v", seen["reasoning_effort"])
		}
	})

	t.Run("an effort level carries through", func(t *testing.T) {
		post(t, "/api/chat", `{"model":"local","messages":[{"role":"user","content":"2+2"}],"stream":false,"think":"high"}`)
		if seen["reasoning_effort"] != "high" {
			t.Errorf("reasoning_effort = %#v", seen["reasoning_effort"])
		}
	})

	t.Run("an unset think leaves the model's default alone", func(t *testing.T) {
		w := post(t, "/api/chat", `{"model":"local","messages":[{"role":"user","content":"2+2"}],"stream":false,"think":null}`)
		if _, present := seen["chat_template_kwargs"]; present {
			t.Errorf("chat_template_kwargs should be absent, got %#v", seen["chat_template_kwargs"])
		}
		// A model that thinks by default still gets its reasoning surfaced.
		if !strings.Contains(w.Body.String(), `"thinking":"2 plus 2 is 4"`) {
			t.Errorf("body = %q", w.Body.String())
		}
	})

	t.Run("an invalid think is rejected before reaching the model", func(t *testing.T) {
		w := post(t, "/api/chat", `{"model":"local","messages":[{"role":"user","content":"hi"}],"think":"sometimes"}`)
		if w.Code != http.StatusBadRequest {
			t.Errorf("status = %d, want %d", w.Code, http.StatusBadRequest)
		}
		if seen != nil {
			t.Errorf("request reached the model: %#v", seen)
		}
	})

	t.Run("generate surfaces thinking too", func(t *testing.T) {
		w := post(t, "/api/generate", `{"model":"local","prompt":"2+2","stream":false,"think":true}`)
		want := `{"model":"local","response":"4","thinking":"2 plus 2 is 4","done":true,"done_reason":"stop"}` + "\n"
		if got := w.Body.String(); got != want {
			t.Errorf("body = %q, want %q", got, want)
		}
	})
}

// TestOllamaAPI_ThinkStreamsReasoningChunks verifies reasoning is streamed as
// its own NDJSON chunks, the way Ollama does it, so a client can render the
// chain of thought separately from the answer.
func TestOllamaAPI_ThinkStreamsReasoningChunks(t *testing.T) {
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
		// llama.cpp's spelling, then Ollama's, to prove both are read.
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"hmm\"},\"finish_reason\":null}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"reasoning\":\" ok\"},\"finish_reason\":null}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"4\"},\"finish_reason\":null}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}
	s := newOllamaAPITestServer(t, cfg, local)

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/api/chat", strings.NewReader(`{"model":"local","messages":[{"role":"user","content":"2+2"}],"think":true}`)))

	want := `{"model":"local","message":{"role":"assistant","content":"","thinking":"hmm"},"done":false}` + "\n" +
		`{"model":"local","message":{"role":"assistant","content":"","thinking":" ok"},"done":false}` + "\n" +
		`{"model":"local","message":{"role":"assistant","content":"4"},"done":false}` + "\n" +
		`{"model":"local","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}` + "\n"
	if got := w.Body.String(); got != want {
		t.Errorf("stream body =\n%q\nwant\n%q", got, want)
	}
}

// TestOllamaAPI_LoadIdiom covers Ollama's "load a model" request: /api/chat
// with no messages and /api/generate with an empty prompt ask for residency
// and nothing else. `ollama run` sends one before the first prompt, so getting
// this wrong breaks the CLI on the step before anything is typed.
func TestOllamaAPI_LoadIdiom(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  local:
    cmd: server --port ${PORT}
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}

	var reached []string
	okServe := func(w http.ResponseWriter, r *http.Request) {
		reached = append(reached, r.Method+" "+r.URL.Path)
		w.WriteHeader(http.StatusOK)
	}
	local := newStubRouter([]string{"local"}, "")
	local.serveHTTP = okServe
	s := newOllamaAPITestServer(t, cfg, local)

	post := func(t *testing.T, path, body string) *httptest.ResponseRecorder {
		t.Helper()
		reached = nil
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodPost, path, strings.NewReader(body)))
		return w
	}

	t.Run("chat with no messages loads the model", func(t *testing.T) {
		w := post(t, "/api/chat", `{"model":"local","messages":[]}`)
		if w.Code != http.StatusOK {
			t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
		}
		want := `{"model":"local","message":{"role":"assistant","content":""},"done":true,"done_reason":"load"}` + "\n"
		if got := w.Body.String(); got != want {
			t.Errorf("body = %q, want %q", got, want)
		}
		if len(reached) != 1 {
			t.Errorf("expected the model to be loaded once, got %v", reached)
		}
	})

	t.Run("generate with an empty prompt loads the model", func(t *testing.T) {
		w := post(t, "/api/generate", `{"model":"local","prompt":""}`)
		want := `{"model":"local","response":"","done":true,"done_reason":"load"}` + "\n"
		if got := w.Body.String(); got != want {
			t.Errorf("body = %q, want %q", got, want)
		}
		if len(reached) != 1 {
			t.Errorf("expected the model to be loaded once, got %v", reached)
		}
	})

	// mlx_lm serves nothing at the root, so the load probe comes back 404 from
	// a server that is running perfectly well. Judging the load by that status
	// reported "model could not be loaded" for a model that was already up.
	t.Run("an upstream that has no root route still counts as loaded", func(t *testing.T) {
		local.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
			reached = append(reached, r.Method+" "+r.URL.Path)
			http.NotFound(w, r)
		}
		local.running = map[string]process.ProcessState{"local": process.StateReady}
		t.Cleanup(func() {
			local.serveHTTP = okServe
			local.running = nil
		})

		w := post(t, "/api/chat", `{"model":"local","messages":[]}`)
		if w.Code != http.StatusOK {
			t.Fatalf("status=%d body=%s", w.Code, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), `"done_reason":"load"`) {
			t.Errorf("body = %q", w.Body.String())
		}
	})

	// The failure this must still catch: nothing came up at all.
	t.Run("a model that does not come up is reported", func(t *testing.T) {
		local.serveHTTP = func(w http.ResponseWriter, r *http.Request) {
			reached = append(reached, r.Method+" "+r.URL.Path)
			w.WriteHeader(http.StatusServiceUnavailable)
		}
		t.Cleanup(func() { local.serveHTTP = okServe })

		w := post(t, "/api/chat", `{"model":"local","messages":[]}`)
		if w.Code != http.StatusBadGateway {
			t.Errorf("status = %d, want %d", w.Code, http.StatusBadGateway)
		}
	})

	t.Run("a residency hint is still a load", func(t *testing.T) {
		w := post(t, "/api/chat", `{"model":"local","messages":[],"keep_alive":"5m"}`)
		if w.Code != http.StatusOK || len(reached) != 1 {
			t.Errorf("status=%d reached=%v", w.Code, reached)
		}
	})

	// `ollama stop` sends the load shape with keep_alive: 0. Loading the model
	// there would turn "stop" into "start".
	for _, keepAlive := range []string{`0`, `"0"`, `"0s"`} {
		t.Run("ollama stop with keep_alive "+keepAlive+" is refused, not loaded", func(t *testing.T) {
			w := post(t, "/api/chat", `{"model":"local","messages":[],"keep_alive":`+keepAlive+`}`)
			if w.Code != http.StatusNotImplemented {
				t.Errorf("status = %d, want %d", w.Code, http.StatusNotImplemented)
			}
			if len(reached) != 0 {
				t.Errorf("model must not be loaded by a stop request, got %v", reached)
			}
		})
	}
}
