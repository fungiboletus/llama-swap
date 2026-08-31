package server

import (
	"fmt"
	"strings"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// generationLimitField maps a generation endpoint onto the request field that
// carries its output-token ceiling. An endpoint that is not listed here either
// does not generate tokens (embeddings, reranking, audio, images) or has no
// field to cap, and is left alone.
var generationLimitField = map[string]string{
	"/v1/chat/completions": "max_tokens",
	"/v1/completions":      "max_tokens",
	"/v1/messages":         "max_tokens",
	"/v1/responses":        "max_output_tokens",
	"/completion":          "n_predict",
	"/infill":              "n_predict",
}

// clampableLimitFields are every spelling of "stop generating after N tokens"
// llama-swap may find in a request body. A client that sent one of these gets
// that field clamped rather than a second, conflicting one added: some servers
// reject a request carrying both max_tokens and max_completion_tokens.
var clampableLimitFields = []string{
	"max_tokens",
	"max_completion_tokens",
	"max_output_tokens",
	"n_predict",
	"num_predict",
	"options.num_predict",
}

// limitPath normalizes a request path to the form used by generationLimitField.
// The versionless /v/... aliases are rewritten to /v1/... on the way upstream
// (issue #728), so they carry the same ceiling as the routes they alias.
func limitPath(path string) string {
	if strings.HasPrefix(path, "/v/") {
		return "/v1/" + strings.TrimPrefix(path, "/v/")
	}
	return path
}

// applyMaxTokens caps how many tokens a request may ask the model to generate.
//
// A ceiling the client never asked for is still applied: an unbounded request
// is the one most likely to run for an hour, which is the case the cap exists
// for. A client that asked for fewer tokens than the cap keeps its own value —
// this is a ceiling, not an assignment.
//
// A value of 0 or less in the request means "no limit" to most inference
// servers, so it is treated as over the cap rather than under it.
func applyMaxTokens(body []byte, path string, limit int) ([]byte, error) {
	if limit <= 0 {
		return body, nil
	}
	field, ok := generationLimitField[limitPath(path)]
	if !ok {
		return body, nil
	}

	clamped := false
	for _, name := range clampableLimitFields {
		value := gjson.GetBytes(body, name)
		if !value.Exists() {
			continue
		}
		clamped = true
		// A non-numeric value (null, most commonly) means the client declined
		// to set a limit, which the cap now sets for it.
		if value.Type == gjson.Number && value.Int() > 0 && value.Int() <= int64(limit) {
			continue
		}
		updated, err := sjson.SetBytes(body, name, limit)
		if err != nil {
			return nil, fmt.Errorf("error capping %s at %d: %w", name, limit, err)
		}
		body = updated
	}
	if clamped {
		return body, nil
	}

	updated, err := sjson.SetBytes(body, field, limit)
	if err != nil {
		return nil, fmt.Errorf("error setting %s to %d: %w", field, limit, err)
	}
	return updated, nil
}

// resolveMaxTokens returns the configured generation cap for a requested model
// ID. Peers are excluded: their ceiling is the remote llama-swap's to enforce,
// and this instance does not manage the process that would need stopping.
func resolveMaxTokens(cfg config.Config, requested string) int {
	realName, found := cfg.RealModelName(requested)
	if !found {
		return 0
	}
	return cfg.Models[realName].MaxTokens
}
