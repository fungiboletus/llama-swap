package server

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// thinkSetting is the parsed form of Ollama's `think` request field. Older
// clients send a boolean; newer ones send a reasoning-effort level. Both mean
// "turn thinking on", so they are parsed into one value and differ only in how
// much effort they ask for.
type thinkSetting struct {
	set    bool
	enable bool
	effort string // "low", "medium" or "high"; empty when think was a boolean
}

// defaultThinkEffort is what a bare `think: true` asks for. Ollama's own
// default for a model that supports levels is medium.
const defaultThinkEffort = "medium"

// noThinkEffort is how the OpenAI reasoning_effort field spells "off". Ollama
// maps it back to `think: false`; see applyTo.
const noThinkEffort = "none"

var validThinkEfforts = map[string]struct{}{
	"low":    {},
	"medium": {},
	"high":   {},
}

// parseOllamaThink reads the `think` field. An absent or null value leaves the
// setting unset, which is not the same as `think: false`: unset defers to
// whatever the model does by default, while false actively asks it to stop.
func parseOllamaThink(raw json.RawMessage) (thinkSetting, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return thinkSetting{}, nil
	}

	var enabled bool
	if err := json.Unmarshal(trimmed, &enabled); err == nil {
		return thinkSetting{set: true, enable: enabled}, nil
	}

	var level string
	if err := json.Unmarshal(trimmed, &level); err == nil {
		if _, ok := validThinkEfforts[level]; ok {
			return thinkSetting{set: true, enable: true, effort: level}, nil
		}
	}
	return thinkSetting{}, fmt.Errorf("think must be true, false, or one of: low, medium, high")
}

// applyTo writes the setting into an OpenAI-shaped request body.
//
// There is no single field every backend reads, so both spellings go out:
// llama.cpp, vLLM and SGLang toggle thinking through the chat template's
// enable_thinking argument, while Ollama ignores that argument entirely and
// reads reasoning_effort instead — including "none", which it translates back
// into the `think: false` this request started as. Each server ignores the key
// it does not know, and sending both is what makes one `think` field work
// across the backends this gateway fronts.
func (t thinkSetting) applyTo(body map[string]any) {
	if !t.set {
		return
	}
	body["chat_template_kwargs"] = map[string]any{"enable_thinking": t.enable}
	if !t.enable {
		body["reasoning_effort"] = noThinkEffort
		return
	}
	effort := t.effort
	if effort == "" {
		effort = defaultThinkEffort
	}
	body["reasoning_effort"] = effort
}

// reasoningText picks whichever spelling of the reasoning field the upstream
// used. Checked in llama.cpp's order first because that is the server most of
// these models run behind.
func reasoningText(reasoningContent, reasoning string) string {
	if reasoningContent != "" {
		return reasoningContent
	}
	return reasoning
}
