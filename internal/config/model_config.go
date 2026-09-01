package config

import (
	"errors"
	"fmt"
	"runtime"
)

const (
	MODEL_CONFIG_DEFAULT_TTL   = -1
	MODEL_CONFIG_DEFAULT_PROXY = "http://localhost:${PORT}"
	comfyUIConcurrencyLimit    = 50

	// ComfyUIModelID identifies the model used by the /comfyui endpoint.
	ComfyUIModelID = "comfyui_auto"
)

var validModalities = map[string]struct{}{
	"text":  {},
	"audio": {},
	"image": {},
	"video": {},
}

// ModelCapConfig defines what modalities and features a model supports.
// Used in /v1/models to inform clients. An empty block (all zero values) is
// treated as not configured.
type ModelCapConfig struct {
	In       []string `yaml:"in"`
	Out      []string `yaml:"out"`
	Tools    bool     `yaml:"tools"`
	Reranker bool     `yaml:"reranker"`
	Context  int      `yaml:"context"`

	// Thinking marks a reasoning model. It is advertised to clients (the
	// Ollama CLI uses it to decide whether `/set think` is offered) but never
	// gates a request: a `think` field is forwarded either way, because the
	// upstream is the only thing that really knows whether it can honour it.
	Thinking bool `yaml:"thinking"`
}

// Empty returns true when all fields are at their zero values.
func (c ModelCapConfig) Empty() bool {
	return len(c.In) == 0 && len(c.Out) == 0 && !c.Tools && !c.Reranker && !c.Thinking && c.Context == 0
}

// Validate checks that all modality values are recognized and context is
// non-negative. Returns an error if any value is invalid.
func (c ModelCapConfig) Validate() error {
	for _, m := range c.In {
		if _, ok := validModalities[m]; !ok {
			return fmt.Errorf("capabilities.in: invalid modality %q, must be one of: text, audio, image, video", m)
		}
	}
	for _, m := range c.Out {
		if _, ok := validModalities[m]; !ok {
			return fmt.Errorf("capabilities.out: invalid modality %q, must be one of: text, audio, image, video", m)
		}
	}
	if c.Context < 0 {
		return errors.New("capabilities.context: must be >= 0")
	}
	return nil
}

// TimeoutsConfig holds timeout settings for proxy connections
// 0 = no timeout
type TimeoutsConfig struct {
	Connect        int `yaml:"connect"`
	KeepAlive      int `yaml:"keepalive"`
	ResponseHeader int `yaml:"responseHeader"`
	TLSHandshake   int `yaml:"tlsHandshake"`
	ExpectContinue int `yaml:"expectContinue"`
	IdleConn       int `yaml:"idleConn"`
}

// CompatConfig holds compatibility settings for upstream applications.
type CompatConfig struct {
	// IgnoreWebsockets prevents websocket connections from participating in
	// model lifecycle activity such as swapping, concurrency, and TTL tracking.
	IgnoreWebsockets bool `yaml:"ignoreWebsockets"`
}

type ModelConfig struct {
	Cmd           string   `yaml:"cmd"`
	CmdStop       string   `yaml:"cmdStop"`
	Proxy         string   `yaml:"proxy"`
	Aliases       []string `yaml:"aliases"`
	Env           []string `yaml:"env"`
	CheckEndpoint string   `yaml:"checkEndpoint"`
	UnloadAfter   int      `yaml:"ttl"`
	UnloadTimeout int      `yaml:"unloadTimeout"`
	Unlisted      bool     `yaml:"unlisted"`
	UseModelName  string   `yaml:"useModelName"`

	// Memory is the conservative resident footprint used by the auto router.
	// It is an operator-provided reservation, not live process telemetry.
	Memory ByteSize `yaml:"memory"`

	// KeepResident is a soft auto-router eviction preference. It never prevents
	// eviction when capacity is needed for another admissible model.
	KeepResident bool `yaml:"keepResident"`

	// #179 for /v1/models
	Name        string `yaml:"name"`
	Description string `yaml:"description"`

	// Limit concurrency of HTTP requests to process
	ConcurrencyLimit int `yaml:"concurrencyLimit"`

	// Model filters see issue #174
	Filters ModelFilters `yaml:"filters"`

	// Macros: see #264
	// Model level macros take precedence over the global macros
	Macros MacroList `yaml:"macros"`

	// Metadata: see #264
	// Arbitrary metadata that can be exposed through the API
	Metadata map[string]any `yaml:"metadata"`

	// override global setting
	SendLoadingState *bool `yaml:"sendLoadingState"`

	// Timeout settings for proxy connections
	Timeouts TimeoutsConfig `yaml:"timeouts"`

	// StallTimeout is the number of seconds an in-flight request to this model
	// may go without progress from the upstream before llama-swap fails the
	// request and stops the process as presumed crashed. 0 inherits the global
	// stallTimeout; a negative value turns stall detection off for this model.
	// LoadConfigFromReader resolves both forms, so by the time a process reads
	// this field 0 simply means "disabled".
	StallTimeout int `yaml:"stallTimeout"`

	// RequestTimeout is the number of seconds one request to this model may
	// spend being answered before it is failed, however much progress it is
	// making. Follows the same inheritance rules as StallTimeout.
	RequestTimeout int `yaml:"requestTimeout"`

	// MaxTokens caps how many tokens a request may ask this model to generate.
	// 0 inherits the global maxTokens; a negative value removes the cap for
	// this model. Resolved by LoadConfigFromReader the same way as
	// StallTimeout, so 0 means "no cap" once loaded.
	MaxTokens int `yaml:"maxTokens"`

	// Compatibility settings for upstream applications.
	Compat CompatConfig `yaml:"compat"`

	// Capabilities defines what modalities and features the model supports.
	Capabilities ModelCapConfig `yaml:"capabilities"`

	// Copy of HealthCheckTimeout from global config
	HealthCheckTimeout int `yaml:"healthCheckTimeout"`

	// OllamaManaged identifies a lane generated from the local Ollama
	// catalogue. It is runtime-only metadata, not a user-configurable model
	// setting. Such a lane can safely receive native Ollama API requests.
	OllamaManaged bool `yaml:"-"`
}

func (m *ModelConfig) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type rawModelConfig ModelConfig
	defaults := rawModelConfig{
		Cmd:              "",
		CmdStop:          "",
		Proxy:            MODEL_CONFIG_DEFAULT_PROXY,
		Aliases:          []string{},
		Env:              []string{},
		CheckEndpoint:    "/health",
		UnloadAfter:      MODEL_CONFIG_DEFAULT_TTL, // use GlobalTTL
		UnloadTimeout:    0,                        // use global UnloadTimeout
		Unlisted:         false,
		UseModelName:     "",
		ConcurrencyLimit: 0,
		Name:             "",
		Description:      "",

		// matches http.DefaultTransport
		Timeouts: TimeoutsConfig{
			Connect:        30,
			KeepAlive:      30,
			ResponseHeader: 0,
			TLSHandshake:   10,
			ExpectContinue: 1,
			IdleConn:       90,
		},
	}

	// the default cmdStop to taskkill /f /t /pid ${PID}
	if runtime.GOOS == "windows" {
		defaults.CmdStop = "taskkill /f /t /pid ${PID}"
	}

	if err := unmarshal(&defaults); err != nil {
		return err
	}

	*m = ModelConfig(defaults)
	return nil
}

func (m *ModelConfig) SanitizedCommand() ([]string, error) {
	return SanitizeCommand(m.Cmd)
}

// ModelFilters embeds Filters and adds legacy support for strip_params field
// See issue #174
type ModelFilters struct {
	Filters `yaml:",inline"`
}

func (m *ModelFilters) UnmarshalYAML(unmarshal func(interface{}) error) error {
	type rawModelFilters ModelFilters
	defaults := rawModelFilters{}

	if err := unmarshal(&defaults); err != nil {
		return err
	}

	// Try to unmarshal with the old field name for backwards compatibility
	if defaults.StripParams == "" {
		var legacy struct {
			StripParams string `yaml:"strip_params"`
		}
		if legacyErr := unmarshal(&legacy); legacyErr != nil {
			return errors.New("failed to unmarshal legacy filters.strip_params: " + legacyErr.Error())
		}
		defaults.StripParams = legacy.StripParams
	}

	*m = ModelFilters(defaults)
	return nil
}

// SanitizedStripParams wraps Filters.SanitizedStripParams for backwards compatibility
// Returns ([]string, error) to match existing API
func (f ModelFilters) SanitizedStripParams() ([]string, error) {
	return f.Filters.SanitizedStripParams(), nil
}
