package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
)

const ollamaCatalogTimeout = 15 * time.Second

type ollamaTagsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

type ollamaShowResponse struct {
	Capabilities []string `json:"capabilities"`
}

type ollamaCatalogModel struct {
	Name     string
	Vision   bool
	Thinking bool
}

// prepareOllamaModels discovers the local catalogue through a
// short-lived, private Ollama server and adds one process configuration per
// discovered model. The temporary server does not receive inference traffic
// and is killed before any router is constructed, so the router remains the
// only owner of long-lived Ollama processes.
func prepareOllamaModels(ctx context.Context, cfg *config.Config) error {
	if !cfg.Ollama.Enabled {
		return nil
	}

	catalogue, err := discoverOllamaModels(ctx, cfg.Ollama.Command)
	if err != nil {
		return err
	}
	names := make([]string, 0, len(catalogue))
	for _, model := range catalogue {
		names = append(names, model.Name)
	}
	if err := cfg.AddOllamaModels(names); err != nil {
		return err
	}
	applyOllamaCatalogCapabilities(cfg, catalogue)
	return nil
}

// applyOllamaCatalogCapabilities carries what Ollama reported about each
// discovered model onto the lane configuration generated for it. The lanes are
// otherwise plain text lanes, and a client asking /api/show would be told the
// model can do less than it can.
func applyOllamaCatalogCapabilities(cfg *config.Config, catalogue []ollamaCatalogModel) {
	for _, catalogueModel := range catalogue {
		model, found := cfg.Models[catalogueModel.Name]
		if !found {
			continue
		}
		if catalogueModel.Vision {
			model.Capabilities.In = append(model.Capabilities.In, "image")
		}
		// Ollama clients read the thinking capability before deciding whether
		// `/set think` and `/set nothink` mean anything for this model, so a
		// lane that drops it warns that a plainly reasoning model "does not
		// support thinking output".
		model.Capabilities.Thinking = catalogueModel.Thinking
		cfg.Models[catalogueModel.Name] = model
	}
}

func discoverOllamaModels(parent context.Context, command string) ([]ollamaCatalogModel, error) {
	ctx, cancel := context.WithTimeout(parent, ollamaCatalogTimeout)
	defer cancel()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("reserve private Ollama catalogue port: %w", err)
	}
	address := listener.Addr().String()
	if err := listener.Close(); err != nil {
		return nil, fmt.Errorf("release private Ollama catalogue port: %w", err)
	}

	cmd := exec.CommandContext(ctx, command, "serve")
	cmd.Env = append(os.Environ(), "OLLAMA_HOST="+address)
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start temporary Ollama catalogue server: %w", err)
	}
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
	}()

	client := &http.Client{Timeout: time.Second}
	endpoint := "http://" + address + "/api/tags"
	for {
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return nil, fmt.Errorf("create Ollama catalogue request: %w", err)
		}
		response, err := client.Do(request)
		if err == nil {
			if response.StatusCode == http.StatusOK {
				names, parseErr := parseOllamaTags(response.Body)
				_ = response.Body.Close()
				if parseErr != nil {
					return nil, parseErr
				}
				return discoverOllamaCatalogues(ctx, client, address, names), nil
			}
			_ = response.Body.Close()
		}

		select {
		case <-ctx.Done():
			if err != nil {
				return nil, fmt.Errorf("wait for temporary Ollama catalogue server: %w", err)
			}
			return nil, fmt.Errorf("temporary Ollama catalogue server did not become ready")
		case <-time.After(100 * time.Millisecond):
		}
	}
}

// discoverOllamaCatalogues enriches each listed model with the capabilities
// Ollama already reports through /api/show. A failed show lookup remains a
// usable text lane; it merely does not advertise image input or thinking.
func discoverOllamaCatalogues(ctx context.Context, client *http.Client, address string, names []string) []ollamaCatalogModel {
	models := make([]ollamaCatalogModel, 0, len(names))
	for _, name := range names {
		model := ollamaCatalogModel{Name: name}
		body, _ := json.Marshal(map[string]string{"model": name})
		request, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://"+address+"/api/show", strings.NewReader(string(body)))
		if err == nil {
			request.Header.Set("Content-Type", "application/json")
			response, err := client.Do(request)
			if err == nil {
				var show ollamaShowResponse
				if response.StatusCode == http.StatusOK && json.NewDecoder(io.LimitReader(response.Body, 16<<20)).Decode(&show) == nil {
					model.Vision = containsString(show.Capabilities, "vision")
					model.Thinking = containsString(show.Capabilities, "thinking")
				}
				_ = response.Body.Close()
			}
		}
		models = append(models, model)
	}
	return models
}

func parseOllamaTags(body io.Reader) ([]string, error) {
	var tags ollamaTagsResponse
	decoder := json.NewDecoder(io.LimitReader(body, 16<<20))
	if err := decoder.Decode(&tags); err != nil {
		return nil, fmt.Errorf("decode Ollama catalogue: %w", err)
	}
	names := make([]string, 0, len(tags.Models))
	for _, model := range tags.Models {
		names = append(names, strings.TrimSpace(model.Name))
	}
	return names, nil
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
