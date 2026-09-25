package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mostlygeek/llama-swap/internal/chain"
	"github.com/mostlygeek/llama-swap/internal/config"
)

func TestServer_RequestContextMiddleware(t *testing.T) {
	cfg := config.Config{
		Models: map[string]config.ModelConfig{
			"llama3": {},
		},
	}

	final := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	mw := CreateRequestContextMiddleware(cfg)

	t.Run("known model passes through", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"llama3"}`))
		r.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		mw(final).ServeHTTP(w, r)
		if w.Code != http.StatusOK {
			t.Errorf("status = %d, want 200", w.Code)
		}
	})

	t.Run("missing model returns 404", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{}`))
		r.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		mw(final).ServeHTTP(w, r)
		if w.Code != http.StatusNotFound {
			t.Errorf("status = %d, want 404", w.Code)
		}
	})
}

func TestServer_AuthMiddleware(t *testing.T) {
	final := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	t.Run("no keys configured passes through", func(t *testing.T) {
		mw := CreateAuthMiddleware(config.Config{})
		w := httptest.NewRecorder()
		mw(final).ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/", nil))
		if w.Code != http.StatusOK {
			t.Errorf("status = %d, want 200", w.Code)
		}
	})

	cfg := config.Config{RequiredAPIKeys: []string{"secret"}}

	t.Run("valid key", func(t *testing.T) {
		mw := CreateAuthMiddleware(cfg)
		r := httptest.NewRequest(http.MethodGet, "/", nil)
		r.Header.Set("Authorization", "Bearer secret")
		w := httptest.NewRecorder()
		mw(final).ServeHTTP(w, r)
		if w.Code != http.StatusOK {
			t.Errorf("status = %d, want 200", w.Code)
		}
	})

	t.Run("invalid key", func(t *testing.T) {
		mw := CreateAuthMiddleware(cfg)
		r := httptest.NewRequest(http.MethodGet, "/", nil)
		r.Header.Set("Authorization", "Bearer wrong")
		w := httptest.NewRecorder()
		mw(final).ServeHTTP(w, r)
		if w.Code != http.StatusUnauthorized {
			t.Errorf("status = %d, want 401", w.Code)
		}
		if w.Header().Get("WWW-Authenticate") == "" {
			t.Error("missing WWW-Authenticate header")
		}
	})
}

func TestServer_AuthMiddleware_AdminAndInferencePolicy(t *testing.T) {
	final := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	requestWithKey := func(key string) *http.Request {
		r := httptest.NewRequest(http.MethodGet, "/", nil)
		r.Header.Set("Authorization", "Bearer "+key)
		return r
	}

	cfg := config.Config{
		RequiredAPIKeys: []string{"inference-key"},
		AdminAPIKeys:    []string{"admin-key"},
	}
	for _, tt := range []struct {
		name string
		mw   chain.Middleware
		key  string
		want int
	}{
		{"inference key reaches inference", CreateInferenceAuthMiddleware(cfg), "inference-key", http.StatusOK},
		{"admin key reaches inference", CreateInferenceAuthMiddleware(cfg), "admin-key", http.StatusOK},
		{"inference key cannot administer", CreateAdminAuthMiddleware(cfg), "inference-key", http.StatusUnauthorized},
		{"admin key can administer", CreateAdminAuthMiddleware(cfg), "admin-key", http.StatusOK},
		{"legacy api key administers", CreateAdminAuthMiddleware(config.Config{RequiredAPIKeys: []string{"legacy"}}), "legacy", http.StatusOK},
	} {
		t.Run(tt.name, func(t *testing.T) {
			w := httptest.NewRecorder()
			tt.mw(final).ServeHTTP(w, requestWithKey(tt.key))
			if w.Code != tt.want {
				t.Errorf("status = %d, want %d", w.Code, tt.want)
			}
		})
	}
}

func TestServer_MatchesAPIKey(t *testing.T) {
	if !matchesAPIKey("expected", []string{"wrong", "expected"}) {
		t.Fatal("expected matching key to be accepted")
	}
	if matchesAPIKey("expected", []string{"unexpected"}) {
		t.Fatal("unexpected key was accepted")
	}
}
