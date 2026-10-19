package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"time"

	"github.com/mostlygeek/llama-swap/internal/ollamaapi"
)

// isUnloadKeepAlive reports whether keep_alive asks for the model to be
// released immediately, which is the only value llama-swap has to look at.
// Everything else is a residency hint, and residency here belongs to the
// memory scheduler rather than to whoever sent the request.
//
// Ollama accepts a duration either as a number of seconds or as a Go duration
// string, so both spellings of zero are recognized.
func isUnloadKeepAlive(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return false
	}

	var seconds float64
	if err := json.Unmarshal(trimmed, &seconds); err == nil {
		return seconds == 0
	}

	var text string
	if err := json.Unmarshal(trimmed, &text); err == nil {
		duration, err := time.ParseDuration(text)
		return err == nil && duration == 0
	}
	return false
}

// serveOllamaLoad answers Ollama's "load a model" idiom: /api/chat with no
// messages, or /api/generate with an empty prompt, asks for the model to be
// made resident and nothing else. This is the extra request that shows up
// before a prompt when switching models — `ollama run` sends one to warm the
// model before it sends anything you typed.
//
// It is answered here rather than translated upstream because it is purely an
// Ollama concept: an OpenAI chat request with no messages is an error, not a
// request to load anything.
//
// `ollama stop` sends this same shape with keep_alive: 0, which must not be
// mistaken for a load — doing so would turn "stop" into "start". It is
// refused rather than honoured: unloading is an administrative operation in
// this fork, guarded by adminApiKeys, and an inference client that could stop
// models on a shared host could disrupt everyone else on it.
func (s *Server) serveOllamaLoad(w http.ResponseWriter, r *http.Request, target ollamaModelTarget, keepAlive json.RawMessage, generate bool) {
	if isUnloadKeepAlive(keepAlive) {
		sendOllamaError(w, http.StatusNotImplemented,
			"unloading a model is an administrative operation in llama-swap; use the admin API or the web UI")
		return
	}

	if err := s.loadLocalModel(r.Context(), target.localID); err != nil {
		s.logs.ProxyLogs.Warnf("ollama load: %v", err)
		sendOllamaError(w, http.StatusBadGateway, "model could not be loaded")
		return
	}

	// Ollama answers a load with a single JSON object even when the request
	// asked to stream, because there is nothing to stream.
	w.Header().Set("Content-Type", "application/json")
	if generate {
		json.NewEncoder(w).Encode(ollamaapi.GenerateResponse{Model: target.publicName, Done: true, Reason: "load"})
		return
	}
	json.NewEncoder(w).Encode(ollamaapi.ChatResponse{
		Model:   target.publicName,
		Message: ollamaapi.Message{Role: "assistant"},
		Done:    true,
		Reason:  "load",
	})
}
