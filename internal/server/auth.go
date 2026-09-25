package server

import (
	"crypto/subtle"
	"net/http"

	"github.com/mostlygeek/llama-swap/internal/chain"
	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// CreateAuthMiddleware returns the inference authorization middleware. It is
// kept as the public compatibility name for callers that previously used the
// single API-key policy.
func CreateAuthMiddleware(cfg config.Config) chain.Middleware {
	return CreateInferenceAuthMiddleware(cfg)
}

// CreateInferenceAuthMiddleware protects inference only when apiKeys are
// configured. In that case it also accepts admin keys, so administrators can
// use one key for both surfaces. adminApiKeys alone intentionally leaves
// inference open.
func CreateInferenceAuthMiddleware(cfg config.Config) chain.Middleware {
	keys := append([]string{}, cfg.RequiredAPIKeys...)
	if len(keys) > 0 {
		keys = append(keys, cfg.AdminAPIKeys...)
	}
	return createAuthMiddleware(keys)
}

// CreateAdminAuthMiddleware uses adminApiKeys when configured. If no admin
// keys are configured, apiKeys retains its historical behavior of protecting
// every protected route.
func CreateAdminAuthMiddleware(cfg config.Config) chain.Middleware {
	keys := cfg.AdminAPIKeys
	if len(keys) == 0 {
		keys = cfg.RequiredAPIKeys
	}
	return createAuthMiddleware(keys)
}

// createAuthMiddleware validates keys supplied via Authorization: Bearer,
// Authorization: Basic (password field), or x-api-key. When no keys are
// configured the middleware is a pass-through.
func createAuthMiddleware(keys []string) chain.Middleware {
	return func(next http.Handler) http.Handler {
		if len(keys) == 0 {
			return next
		}
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			provided := swaputil.ExtractAPIKey(r)

			if !matchesAPIKey(provided, keys) {
				w.Header().Set("WWW-Authenticate", `Basic realm="llama-swap"`)
				swaputil.SendResponse(w, r, http.StatusUnauthorized, "unauthorized: invalid or missing API key")
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

// matchesAPIKey performs a constant-time comparison for every candidate that
// has the same length as provided. Different lengths are rejected before the
// comparison because ConstantTimeCompare otherwise returns immediately.
func matchesAPIKey(provided string, keys []string) bool {
	valid := 0
	for _, key := range keys {
		if len(provided) != len(key) {
			continue
		}
		valid |= subtle.ConstantTimeCompare([]byte(provided), []byte(key))
	}
	return valid == 1
}

// CreateRequestContextMiddleware returns middleware that extracts model and
// auth info from the request into the context. Requests where no model can be
// identified are rejected with a 404.
func CreateRequestContextMiddleware(cfg config.Config) chain.Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			r = markInflightStart(r)
			data, err := swaputil.FetchContext(r, cfg)
			if err != nil {
				swaputil.SendError(w, r, swaputil.ErrNoModelInContext)
				return
			}
			_ = data
			next.ServeHTTP(w, r)
		})
	}
}
