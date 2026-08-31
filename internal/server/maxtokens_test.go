package server

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestServer_ApplyMaxTokens(t *testing.T) {
	tests := []struct {
		name  string
		body  string
		path  string
		limit int
		want  map[string]int64
		gone  []string
	}{
		{
			name:  "injects the ceiling when the client asked for none",
			body:  `{"model":"m","messages":[]}`,
			path:  "/v1/chat/completions",
			limit: 128,
			want:  map[string]int64{"max_tokens": 128},
		},
		{
			name:  "clamps a request over the ceiling",
			body:  `{"model":"m","max_tokens":100000}`,
			path:  "/v1/chat/completions",
			limit: 128,
			want:  map[string]int64{"max_tokens": 128},
		},
		{
			name:  "leaves a request under the ceiling alone",
			body:  `{"model":"m","max_tokens":16}`,
			path:  "/v1/chat/completions",
			limit: 128,
			want:  map[string]int64{"max_tokens": 16},
		},
		{
			name:  "treats a non-positive value as unlimited",
			body:  `{"model":"m","max_tokens":-1}`,
			path:  "/v1/chat/completions",
			limit: 128,
			want:  map[string]int64{"max_tokens": 128},
		},
		{
			name:  "caps the field the client used without adding a second one",
			body:  `{"model":"m","max_completion_tokens":9000}`,
			path:  "/v1/chat/completions",
			limit: 128,
			want:  map[string]int64{"max_completion_tokens": 128},
			gone:  []string{"max_tokens"},
		},
		{
			name:  "uses the llama.cpp field on native completion routes",
			body:  `{"model":"m","prompt":"hi"}`,
			path:  "/completion",
			limit: 64,
			want:  map[string]int64{"n_predict": 64},
		},
		{
			name:  "uses the responses field on /v1/responses",
			body:  `{"model":"m"}`,
			path:  "/v1/responses",
			limit: 64,
			want:  map[string]int64{"max_output_tokens": 64},
		},
		{
			name:  "applies to versionless aliases",
			body:  `{"model":"m"}`,
			path:  "/v/chat/completions",
			limit: 64,
			want:  map[string]int64{"max_tokens": 64},
		},
		{
			name:  "ignores endpoints that generate no tokens",
			body:  `{"model":"m","input":"hi"}`,
			path:  "/v1/embeddings",
			limit: 64,
			gone:  []string{"max_tokens", "n_predict"},
		},
		{
			name:  "does nothing when no ceiling is configured",
			body:  `{"model":"m"}`,
			path:  "/v1/chat/completions",
			limit: 0,
			gone:  []string{"max_tokens"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, err := applyMaxTokens([]byte(tc.body), tc.path, tc.limit)
			if err != nil {
				t.Fatalf("applyMaxTokens: %v", err)
			}
			for field, want := range tc.want {
				if got := gjson.GetBytes(out, field).Int(); got != want {
					t.Errorf("%s = %d, want %d (body: %s)", field, got, want, out)
				}
			}
			for _, field := range tc.gone {
				if gjson.GetBytes(out, field).Exists() {
					t.Errorf("%s should not have been added (body: %s)", field, out)
				}
			}
		})
	}
}
