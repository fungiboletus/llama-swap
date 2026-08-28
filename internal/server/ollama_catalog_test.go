package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
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

func TestOllamaCatalog_UsesShowVisionCapability(t *testing.T) {
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
		json.NewEncoder(w).Encode(ollamaShowResponse{Capabilities: capabilities})
	}))
	defer server.Close()

	models := discoverOllamaCatalogues(context.Background(), server.Client(), strings.TrimPrefix(server.URL, "http://"), []string{"text", "vision"})
	if len(models) != 2 || models[0].Name != "text" || models[0].Vision || models[1].Name != "vision" || !models[1].Vision {
		t.Errorf("models = %#v", models)
	}
}
