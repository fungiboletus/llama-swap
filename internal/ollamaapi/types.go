// Package ollamaapi contains only the small request and response shapes used
// by llama-swap's deliberately limited Ollama compatibility surface.
package ollamaapi

import (
	"encoding/json"
	"time"
)

type Message struct {
	Role    string          `json:"role"`
	Content string          `json:"content"`
	Images  json.RawMessage `json:"images,omitempty"`

	// Thinking carries a reasoning model's chain of thought, which Ollama
	// keeps separate from Content so clients can render or hide it. Upstream
	// OpenAI-compatible servers spell the same thing "reasoning_content"
	// (llama.cpp, vLLM) or "reasoning" (Ollama's own OpenAI endpoint).
	Thinking string `json:"thinking,omitempty"`
}

type ChatRequest struct {
	Model    string                     `json:"model"`
	Messages []Message                  `json:"messages"`
	Stream   *bool                      `json:"stream,omitempty"`
	Options  map[string]json.RawMessage `json:"options,omitempty"`
	Tools    json.RawMessage            `json:"tools,omitempty"`
	Think    json.RawMessage            `json:"think,omitempty"`

	// KeepAlive is how long Ollama would hold the model resident. llama-swap
	// owns residency itself, so the value is ignored — except for the zero
	// that `ollama stop` sends, which has to be told apart from a load.
	KeepAlive json.RawMessage `json:"keep_alive,omitempty"`
}

type GenerateRequest struct {
	Model     string                     `json:"model"`
	Prompt    string                     `json:"prompt"`
	Stream    *bool                      `json:"stream,omitempty"`
	Options   map[string]json.RawMessage `json:"options,omitempty"`
	Context   json.RawMessage            `json:"context,omitempty"`
	Suffix    string                     `json:"suffix,omitempty"`
	System    string                     `json:"system,omitempty"`
	Template  string                     `json:"template,omitempty"`
	Raw       bool                       `json:"raw,omitempty"`
	Images    json.RawMessage            `json:"images,omitempty"`
	Think     json.RawMessage            `json:"think,omitempty"`
	KeepAlive json.RawMessage            `json:"keep_alive,omitempty"`
}

type TagsResponse struct {
	Models []Tag `json:"models"`
}

// ProcessResponse is the small /api/ps response shape used by the Ollama CLI.
// Command-backed models do not have Ollama manifests, so Details is deliberately
// an empty object rather than guessed metadata.
type ProcessResponse struct {
	Models []ProcessModel `json:"models"`
}

// ProcessModel describes one ready local model process. ExpiresAt uses Ollama's
// long-lived no-expiry sentinel when llama-swap does not track a deadline.
type ProcessModel struct {
	Name          string    `json:"name"`
	Model         string    `json:"model"`
	Size          int64     `json:"size"`
	Digest        string    `json:"digest"`
	Details       struct{}  `json:"details"`
	ExpiresAt     time.Time `json:"expires_at"`
	SizeVRAM      int64     `json:"size_vram"`
	ContextLength int       `json:"context_length"`
}

type Tag struct {
	Name       string    `json:"name"`
	Model      string    `json:"model"`
	ModifiedAt time.Time `json:"modified_at"`
	Size       int64     `json:"size"`
	Digest     string    `json:"digest"`
}

type ShowResponse struct {
	Capabilities []string       `json:"capabilities"`
	Details      map[string]any `json:"details,omitempty"`
	ModelInfo    map[string]any `json:"model_info,omitempty"`
}

type ChatResponse struct {
	Model   string  `json:"model"`
	Message Message `json:"message"`
	Done    bool    `json:"done"`
	Reason  string  `json:"done_reason,omitempty"`
}

type GenerateResponse struct {
	Model    string `json:"model"`
	Response string `json:"response"`
	Thinking string `json:"thinking,omitempty"`
	Done     bool   `json:"done"`
	Reason   string `json:"done_reason,omitempty"`
}
