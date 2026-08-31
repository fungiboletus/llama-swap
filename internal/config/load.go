package config

import (
	"fmt"
	"io"
	"net/url"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

func LoadConfigFromReader(r io.Reader) (Config, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return Config{}, err
	}
	yamlStr := string(data)

	// Phase 1: Substitute all ${env.VAR} macros at string level
	// This is safe because env values are simple strings without YAML formatting
	yamlStr, err = substituteEnvMacros(yamlStr)
	if err != nil {
		return Config{}, err
	}

	raw, macroConfig, err := resolveConfigMacros(yamlStr)
	if err != nil {
		return Config{}, err
	}

	var node yaml.Node
	if err = node.Encode(raw); err != nil {
		return Config{}, err
	}

	// Decode the resolved values into the full Config with defaults.
	config := Config{
		HealthCheckTimeout: 120,
		StartPort:          5800,
		LogLevel:           "info",
		LogTimeFormat:      "",
		LogToStdout:        LogToStdoutProxy,
		MetricsMaxInMemory: 1000,
		CaptureBuffer:      5,
		GlobalTTL:          0,
		UnloadTimeout:      DEFAULT_UNLOAD_TIMEOUT,
		StallTimeout:       DEFAULT_STALL_TIMEOUT,
		RequestTimeout:     DEFAULT_REQUEST_TIMEOUT,
		MaxTokens:          DEFAULT_MAX_TOKENS,
		UI: UIConfig{Activity: UIActivityConfig{SessionID: []string{
			"X-Session-ID",
			"X-Litellm-Session-Id",
		}}},
	}
	if err = node.Decode(&config); err != nil {
		return Config{}, err
	}
	config.Macros = macroConfig.Macros
	for modelID, modelConfig := range config.Models {
		modelConfig.Macros = macroConfig.Models[modelID].Macros
		config.Models[modelID] = modelConfig
	}

	if config.HealthCheckTimeout < 15 {
		config.HealthCheckTimeout = 15
	}

	// Apply defaults for performance config when section is missing
	if config.Performance.Every == 0 {
		config.Performance.Every = 5 * time.Second
	}
	if err = config.Performance.Validate(); err != nil {
		return Config{}, fmt.Errorf("performance: %w", err)
	}

	if config.StartPort < 1 {
		return Config{}, fmt.Errorf("startPort must be greater than 1")
	}

	if config.GlobalTTL < 0 {
		return Config{}, fmt.Errorf("globalTTL must be >= 0")
	}

	if config.UnloadTimeout < 0 {
		return Config{}, fmt.Errorf("unloadTimeout must be >= 0")
	}
	if config.UnloadTimeout == 0 {
		config.UnloadTimeout = DEFAULT_UNLOAD_TIMEOUT
	}

	// Both of these are "off" at 0 rather than "use a default", so a negative
	// value has nothing to mean at the global level.
	if config.StallTimeout < 0 {
		return Config{}, fmt.Errorf("stallTimeout must be >= 0")
	}
	if config.RequestTimeout < 0 {
		return Config{}, fmt.Errorf("requestTimeout must be >= 0")
	}
	if config.MaxTokens < 0 {
		return Config{}, fmt.Errorf("maxTokens must be >= 0")
	}

	config.UI.Activity.SessionID = normalizeHeaderNames(config.UI.Activity.SessionID)

	if config.Store != nil {
		if err := validateStorePath(config.Store.Path); err != nil {
			return Config{}, err
		}
	}

	// Apply default for upstream.ignorePaths when not specified. The default
	// matches common static-asset suffixes so they do not trigger a swap.
	if len(config.Upstream.IgnorePaths) == 0 {
		config.Upstream.IgnorePaths = DefaultUpstreamIgnorePaths()
	}

	switch config.LogToStdout {
	case LogToStdoutProxy, LogToStdoutUpstream, LogToStdoutBoth, LogToStdoutNone:
	default:
		return Config{}, fmt.Errorf("logToStdout must be one of: proxy, upstream, both, none")
	}

	// Populate the aliases map
	config.aliases = make(map[string]string)
	for modelName, modelConfig := range config.Models {
		for _, alias := range modelConfig.Aliases {
			if _, found := config.aliases[alias]; found {
				return Config{}, fmt.Errorf("duplicate alias %s found in model: %s", alias, modelName)
			}
			config.aliases[alias] = modelName
		}
	}

	// Sort model IDs for deterministic validation and normalization.
	modelIds := make([]string, 0, len(config.Models))
	for modelId := range config.Models {
		modelIds = append(modelIds, modelId)
	}
	sort.Strings(modelIds)

	for _, modelId := range modelIds {
		modelConfig := config.Models[modelId]
		modelConfig.HealthCheckTimeout = config.HealthCheckTimeout
		if modelId == ComfyUIModelID {
			if modelConfig.ConcurrencyLimit < comfyUIConcurrencyLimit {
				modelConfig.ConcurrencyLimit = comfyUIConcurrencyLimit
			}
			modelConfig.Compat.IgnoreWebsockets = true
		}

		// set model TTL to globalTTL it is the default value
		if modelConfig.UnloadAfter == MODEL_CONFIG_DEFAULT_TTL {
			modelConfig.UnloadAfter = config.GlobalTTL
		}

		if modelConfig.UnloadAfter < 0 {
			return Config{}, fmt.Errorf("model %s: invalid TTL value %d", modelId, modelConfig.UnloadAfter)
		}

		// set model unloadTimeout to the global value when left at the default
		if modelConfig.UnloadTimeout < 0 {
			return Config{}, fmt.Errorf("model %s: invalid unloadTimeout value %d", modelId, modelConfig.UnloadTimeout)
		}
		if modelConfig.UnloadTimeout == 0 {
			modelConfig.UnloadTimeout = config.UnloadTimeout
		}

		// Resolve the inheritable limits down to their effective values so
		// nothing downstream has to know the global config exists.
		modelConfig.StallTimeout = resolveInherited(modelConfig.StallTimeout, config.StallTimeout)
		modelConfig.RequestTimeout = resolveInherited(modelConfig.RequestTimeout, config.RequestTimeout)
		modelConfig.MaxTokens = resolveInherited(modelConfig.MaxTokens, config.MaxTokens)

		if err := modelConfig.Capabilities.Validate(); err != nil {
			return Config{}, fmt.Errorf("model %s: %w", modelId, err)
		}

		// Auto-register setParamsByID keys as aliases (skip the model's own ID)
		for key := range modelConfig.Filters.SetParamsByID {
			if key == modelId {
				continue
			}
			if _, exists := config.Models[key]; exists {
				return Config{}, fmt.Errorf("model %s filters.setParamsByID: key '%s' conflicts with an existing model ID", modelId, key)
			}
			if existingModel, exists := config.aliases[key]; exists {
				if existingModel != modelId {
					return Config{}, fmt.Errorf("duplicate alias '%s' in model %s filters.setParamsByID, already used by model %s", key, modelId, existingModel)
				}
				continue // already registered as explicit alias for this model
			}
			config.aliases[key] = modelId
			modelConfig.Aliases = append(modelConfig.Aliases, key)
		}

		if _, err := url.Parse(modelConfig.Proxy); err != nil {
			return Config{}, fmt.Errorf("model %s: invalid proxy URL: %w", modelId, err)
		}

		if modelConfig.SendLoadingState == nil {
			v := config.SendLoadingState
			modelConfig.SendLoadingState = &v
		}

		config.Models[modelId] = modelConfig
	}

	if err := validateOllamaConfig(config); err != nil {
		return Config{}, err
	}

	// Normalize routing config. The legacy top-level `matrix`/`groups` keys and
	// the new `routing.router` block are mutually exclusive: a config may use
	// either style, never both.
	hasTopLevel := config.Matrix != nil || len(config.Groups) > 0
	rtr := config.Routing.Router
	hasRouting := rtr.Use != "" || rtr.Settings.Matrix != nil || len(rtr.Settings.Groups) > 0 || rtr.Settings.Auto != nil

	if hasTopLevel && hasRouting {
		return Config{}, fmt.Errorf("config uses both the legacy top-level 'matrix'/'groups' keys and the new 'routing.router' block; please migrate the top-level keys into 'routing.router' and remove them")
	}

	if !hasTopLevel {
		// Both groups and matrix may be defined under routing.router.settings;
		// routing.router.use selects which one is active, so there is no conflict.
		rs := config.Routing.Router.Settings
		switch config.Routing.Router.Use {
		case "matrix":
			if rs.Matrix == nil {
				return Config{}, fmt.Errorf("routing.router.use is 'matrix' but routing.router.settings.matrix is not set")
			}
			config.Matrix = rs.Matrix
		case "group", "":
			config.Groups = rs.Groups
		case "auto":
			if rs.Auto == nil {
				return Config{}, fmt.Errorf("routing.router.use is 'auto' but routing.router.settings.auto is not set")
			}
		default:
			return Config{}, fmt.Errorf("routing.router.use: unknown router %q (valid: group, matrix, auto)", config.Routing.Router.Use)
		}
	}

	// groups XOR matrix
	if config.Matrix != nil && len(config.Groups) > 0 {
		return Config{}, fmt.Errorf("config cannot use both 'groups' and 'matrix'")
	}

	if config.Matrix != nil {
		if err := ValidateMatrix(config.Matrix, config.Models); err != nil {
			return Config{}, fmt.Errorf("matrix: %w", err)
		}
	} else if config.Routing.Router.Use != "auto" {
		config = AddDefaultGroupToConfig(config)

		// Validate group members
		memberUsage := make(map[string]string)
		for groupID, groupConfig := range config.Groups {
			prevSet := make(map[string]bool)
			for _, member := range groupConfig.Members {
				if _, found := prevSet[member]; found {
					return Config{}, fmt.Errorf("duplicate model member %s found in group: %s", member, groupID)
				}
				prevSet[member] = true

				if existingGroup, exists := memberUsage[member]; exists {
					return Config{}, fmt.Errorf("model member %s is used in multiple groups: %s and %s", member, existingGroup, groupID)
				}
				memberUsage[member] = groupID
			}
		}
	}

	// Build the canonical Config.Routing from the effective result. Both legacy
	// and new-style configs converge here. The Matrix pointer is shared so the
	// compiled matrix program stays in one place.
	if config.Routing.Router.Use == "auto" {
		if err := validateAutoRouterConfig(config); err != nil {
			return Config{}, err
		}
	} else if config.Matrix != nil {
		config.Routing.Router.Use = "matrix"
	} else {
		config.Routing.Router.Use = "group"
	}
	config.Routing.Router.Settings.Matrix = config.Matrix
	config.Routing.Router.Settings.Groups = config.Groups
	if config.Ollama.Enabled && config.Routing.Router.Use != "auto" {
		return Config{}, fmt.Errorf("ollama requires routing.router.use: auto")
	}

	if config.Routing.Scheduler.Use == "" {
		config.Routing.Scheduler.Use = "fifo"
	}
	if config.Routing.Scheduler.Use != "fifo" {
		return Config{}, fmt.Errorf("routing.scheduler.use: unknown scheduler %q (valid: fifo)", config.Routing.Scheduler.Use)
	}
	for modelID := range config.Routing.Scheduler.Settings.Fifo.Priority {
		if _, found := config.RealModelName(modelID); !found {
			return Config{}, fmt.Errorf("routing.scheduler.settings.fifo.priority references unknown model %q", modelID)
		}
	}

	// Clean up hooks preload
	if len(config.Hooks.OnStartup.Preload) > 0 {
		var toPreload []string
		for _, modelID := range config.Hooks.OnStartup.Preload {
			modelID = strings.TrimSpace(modelID)
			if modelID == "" {
				continue
			}
			if real, found := config.RealModelName(modelID); found {
				toPreload = append(toPreload, real)
			}
		}
		config.Hooks.OnStartup.Preload = toPreload
	}

	// Validate API keys (env macros already substituted at string level).
	if err := validateAPIKeys("apiKeys", config.RequiredAPIKeys); err != nil {
		return Config{}, err
	}
	if err := validateAPIKeys("adminApiKeys", config.AdminAPIKeys); err != nil {
		return Config{}, err
	}

	if err := ValidatePeerNamespace(config); err != nil {
		return Config{}, err
	}

	if err := validateSelectors(config); err != nil {
		return Config{}, err
	}

	if err := validateProfiles(config); err != nil {
		return Config{}, err
	}

	return config, nil
}

func validateAutoRouterConfig(config Config) error {
	auto := config.Routing.Router.Settings.Auto
	if auto == nil {
		return fmt.Errorf("routing.router.use is 'auto' but routing.router.settings.auto is not set")
	}
	if auto.Capacity == 0 {
		return fmt.Errorf("routing.router.settings.auto.capacity must be greater than zero")
	}
	if auto.Reserve >= auto.Capacity {
		return fmt.Errorf("routing.router.settings.auto.reserve must be smaller than capacity")
	}

	usable := auto.Capacity - auto.Reserve
	modelIDs := make([]string, 0, len(config.Models))
	for modelID := range config.Models {
		modelIDs = append(modelIDs, modelID)
	}
	sort.Strings(modelIDs)
	for _, modelID := range modelIDs {
		memory := config.Models[modelID].Memory
		if memory > usable {
			return fmt.Errorf("model %s: memory reservation exceeds usable auto-router capacity", modelID)
		}
	}
	return nil
}

func validateOllamaConfig(config Config) error {
	ollama := config.Ollama
	if !ollama.Enabled {
		return nil
	}
	if strings.TrimSpace(ollama.Command) == "" {
		return fmt.Errorf("ollama.command must not be empty")
	}
	if ollama.StartPort < 1 || ollama.StartPort > 65535 {
		return fmt.Errorf("ollama.startPort must be between 1 and 65535")
	}
	if ollama.NumParallel < 1 {
		return fmt.Errorf("ollama.numParallel must be greater than zero")
	}
	if ollama.ContextLength < 0 {
		return fmt.Errorf("ollama.contextLength must be >= 0")
	}
	return nil
}

func validateAPIKeys(field string, keys []string) error {
	for i, apikey := range keys {
		if apikey == "" {
			return fmt.Errorf("empty api key found in %s", field)
		}
		if strings.Contains(apikey, " ") {
			return fmt.Errorf("%s[%d]: api key cannot contain spaces", field, i)
		}
	}
	return nil
}

func validateProfiles(config Config) error {
	for profileName, profile := range config.Profiles {
		if strings.TrimSpace(profileName) == "" {
			return fmt.Errorf("profiles: profile names cannot be empty")
		}
		if len(profile.Pins) == 0 {
			return fmt.Errorf("profiles.%s.pins must contain at least one entry", profileName)
		}
		for pin, target := range profile.Pins {
			if strings.TrimSpace(pin) == "" {
				return fmt.Errorf("profiles.%s.pins: pin names cannot be empty", profileName)
			}
			if target == "" {
				continue
			}
			if _, found := config.ResolveBaseModel(target); !found {
				if _, found := config.Selectors[target]; found {
					continue
				}
				return fmt.Errorf("profiles.%s.pins.%s references unknown model %q", profileName, pin, target)
			}
		}
	}
	if profile := config.Hooks.OnStartup.Profile; profile != "" {
		if _, found := config.Profiles[profile]; !found {
			return fmt.Errorf("hooks.on_startup.profile references unknown profile %q", profile)
		}
	}
	return nil
}

func normalizeHeaderNames(names []string) []string {
	normalized := make([]string, 0, len(names))
	seen := make(map[string]struct{}, len(names))
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		key := strings.ToLower(name)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		normalized = append(normalized, name)
	}
	return normalized
}

// resolveInherited normalizes a model-level limit against its global default.
// INHERIT_GLOBAL (0) takes the global value; a negative value is an explicit
// opt-out for that model and resolves to 0, which every consumer reads as
// "disabled". A positive value is the model's own override.
func resolveInherited(modelValue, globalValue int) int {
	switch {
	case modelValue == INHERIT_GLOBAL:
		return globalValue
	case modelValue < 0:
		return 0
	default:
		return modelValue
	}
}
