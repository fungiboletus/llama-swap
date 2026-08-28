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

// prepareOllamaModels discovers the local catalogue through a
// short-lived, private Ollama server and adds one process configuration per
// discovered model. The temporary server does not receive inference traffic
// and is killed before any router is constructed, so the router remains the
// only owner of long-lived Ollama processes.
func prepareOllamaModels(ctx context.Context, cfg *config.Config) error {
	if !cfg.Ollama.Enabled {
		return nil
	}

	names, err := discoverOllamaModels(ctx, cfg.Ollama.Command)
	if err != nil {
		return err
	}
	if err := cfg.AddOllamaModels(names); err != nil {
		return err
	}
	return nil
}

func discoverOllamaModels(parent context.Context, command string) ([]string, error) {
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
			defer response.Body.Close()
			if response.StatusCode == http.StatusOK {
				return parseOllamaTags(response.Body)
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
