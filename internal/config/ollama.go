package config

import (
	"fmt"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

const defaultOllamaStartPort = 11450

// OllamaConfig enables locally managed Ollama model lanes. llama-swap
// discovers the installed Ollama catalogue at startup and owns one isolated
// Ollama server process per model while the auto router controls residency.
type OllamaConfig struct {
	Enabled       bool                `yaml:"-"`
	Command       string              `yaml:"command"`
	StartPort     int                 `yaml:"startPort"`
	ContextLength int                 `yaml:"contextLength"`
	NumParallel   int                 `yaml:"numParallel"`
	Memory        ByteSize            `yaml:"memory"`
	MemoryByModel map[string]ByteSize `yaml:"memoryByModel"`
}

// UnmarshalYAML applies managed-lane defaults and rejects unrecognized fields.
func (c *OllamaConfig) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind != yaml.MappingNode {
		return fmt.Errorf("ollama must be a mapping")
	}
	known := map[string]struct{}{
		"command":       {},
		"startPort":     {},
		"contextLength": {},
		"numParallel":   {},
		"memory":        {},
		"memoryByModel": {},
	}
	for i := 0; i < len(value.Content); i += 2 {
		var key string
		if err := value.Content[i].Decode(&key); err != nil {
			return fmt.Errorf("ollama field name: %w", err)
		}
		if _, found := known[key]; !found {
			return fmt.Errorf("ollama: unknown field %q", key)
		}
	}

	type raw OllamaConfig
	decoded := raw{
		Command:     "ollama",
		StartPort:   defaultOllamaStartPort,
		NumParallel: 1,
	}
	if err := value.Decode(&decoded); err != nil {
		return err
	}
	*c = OllamaConfig(decoded)
	c.Enabled = true
	return nil
}

// AddOllamaModels converts the startup catalogue into ordinary local model
// configurations. Each process is intentionally pinned to a single Ollama
// model so residency is owned by llama-swap rather than Ollama's global LRU.
func (c *Config) AddOllamaModels(names []string) error {
	if !c.Ollama.Enabled {
		return nil
	}
	if c.Routing.Router.Use != "auto" {
		return fmt.Errorf("ollama requires routing.router.use: auto")
	}
	if c.Models == nil {
		c.Models = make(map[string]ModelConfig)
	}

	names = append([]string(nil), names...)
	sort.Strings(names)
	seen := make(map[string]struct{}, len(names))
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		if _, duplicate := seen[name]; duplicate {
			continue
		}
		seen[name] = struct{}{}
		if _, exists := c.Models[name]; exists {
			return fmt.Errorf("ollama model %q conflicts with local model ID", name)
		}
		if local, exists := c.aliases[name]; exists {
			return fmt.Errorf("ollama model %q conflicts with alias for local model %s", name, local)
		}

		port := c.Ollama.StartPort + len(c.Models)
		env := []string{
			fmt.Sprintf("OLLAMA_HOST=127.0.0.1:%d", port),
			// Managed lanes are loopback-only, but browser requests proxied through
			// llama-swap retain their external Origin header. Allow those requests
			// instead of having Ollama reject the proxy hop with HTTP 403.
			"OLLAMA_ORIGINS=*",
			"OLLAMA_MAX_LOADED_MODELS=1",
			"OLLAMA_KEEP_ALIVE=-1",
			fmt.Sprintf("OLLAMA_NUM_PARALLEL=%d", c.Ollama.NumParallel),
		}
		if c.Ollama.ContextLength > 0 {
			env = append(env, fmt.Sprintf("OLLAMA_CONTEXT_LENGTH=%d", c.Ollama.ContextLength))
		}
		memory := c.Ollama.Memory
		if override, ok := c.Ollama.MemoryByModel[name]; ok {
			memory = override
		}
		c.Models[name] = ModelConfig{
			Cmd:                fmt.Sprintf("%q serve", c.Ollama.Command),
			Proxy:              fmt.Sprintf("http://127.0.0.1:%d", port),
			CheckEndpoint:      "/",
			UnloadAfter:        c.GlobalTTL,
			UnloadTimeout:      c.UnloadTimeout,
			Memory:             memory,
			ConcurrencyLimit:   c.Ollama.NumParallel,
			HealthCheckTimeout: c.HealthCheckTimeout,
			StallTimeout:       c.StallTimeout,
			RequestTimeout:     c.RequestTimeout,
			MaxTokens:          c.MaxTokens,
			OllamaManaged:      true,
			Capabilities:       ModelCapConfig{In: []string{"text"}, Context: c.Ollama.ContextLength},
			// Ollama's OpenAI-compatible streaming endpoint only includes the
			// final usage object when this option is requested. Inject it for
			// managed lanes so activity metrics include token counts regardless
			// of which client initiated the request.
			Filters: ModelFilters{Filters: Filters{SetParams: map[string]any{
				"stream_options.include_usage": true,
			}}},
			Env: env,
		}
	}
	return nil
}
