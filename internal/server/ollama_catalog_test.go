package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/config"
)

func TestOllamaCatalog_ParseTags(t *testing.T) {
	names, err := parseOllamaTags(strings.NewReader(`{
  "models": [
    {"name":" qwen3:8b "},
    {"name":"llama3.2:3b"}
  ]
}`))
	if err != nil {
		t.Fatalf("parse tags: %v", err)
	}
	if got, want := strings.Join(names, ","), "qwen3:8b,llama3.2:3b"; got != want {
		t.Errorf("names = %q, want %q", got, want)
	}
}

func TestOllamaCatalog_ParseTagsRejectsMalformedJSON(t *testing.T) {
	if _, err := parseOllamaTags(strings.NewReader(`{"models":`)); err == nil {
		t.Fatal("parse malformed tags succeeded")
	}
}

func TestOllamaCatalog_UsesShowCapabilities(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/api/show" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var request struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		capabilities := []string{"completion"}
		if request.Model == "vision" {
			capabilities = append(capabilities, "vision")
		}
		if request.Model == "thinking" {
			capabilities = append(capabilities, "thinking")
		}
		json.NewEncoder(w).Encode(ollamaShowResponse{Capabilities: capabilities})
	}))
	defer server.Close()

	models := discoverOllamaCatalogues(context.Background(), server.Client(), strings.TrimPrefix(server.URL, "http://"), []string{"text", "vision", "thinking"})
	want := []ollamaCatalogModel{
		{Name: "text"},
		{Name: "vision", Vision: true},
		{Name: "thinking", Thinking: true},
	}
	if !reflect.DeepEqual(models, want) {
		t.Errorf("models = %#v, want %#v", models, want)
	}
}

// TestOllamaCatalog_AppliesShowCapabilitiesToLanes checks the discovered
// capabilities land on the generated lane, which is what /api/show reports and
// what the Ollama CLI reads before offering `/set think`.
func TestOllamaCatalog_AppliesShowCapabilitiesToLanes(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
routing:
  router:
    use: auto
    settings:
      auto:
        capacity: 8GiB
        reserve: 1GiB
ollama:
  memory: 1GiB
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	catalogue := []ollamaCatalogModel{
		{Name: "text:latest"},
		{Name: "qwen3:4b", Thinking: true},
		{Name: "gemma3:4b", Vision: true},
	}
	names := make([]string, 0, len(catalogue))
	for _, model := range catalogue {
		names = append(names, model.Name)
	}
	if err := cfg.AddOllamaModels(names); err != nil {
		t.Fatalf("add ollama models: %v", err)
	}
	applyOllamaCatalogCapabilities(&cfg, catalogue)

	if capabilities := cfg.Models["text:latest"].Capabilities; capabilities.Thinking || contains(capabilities.In, "image") {
		t.Errorf("text capabilities = %#v", capabilities)
	}
	if capabilities := cfg.Models["qwen3:4b"].Capabilities; !capabilities.Thinking || contains(capabilities.In, "image") {
		t.Errorf("thinking capabilities = %#v", capabilities)
	}
	if capabilities := cfg.Models["gemma3:4b"].Capabilities; capabilities.Thinking || !contains(capabilities.In, "image") {
		t.Errorf("vision capabilities = %#v", capabilities)
	}
}
