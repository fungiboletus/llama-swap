package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"

	"github.com/mostlygeek/llama-swap/internal/ollamaapi"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

type ollamaModelTarget struct {
	publicName string
	localID    string
}

func (s *Server) resolveOllamaModel(name string) (ollamaModelTarget, bool) {
	if localID, found := s.cfg.RealModelName(name); found {
		return ollamaModelTarget{publicName: name, localID: localID}, true
	}
	if strings.HasSuffix(name, ":latest") {
		return s.resolveOllamaExact(strings.TrimSuffix(name, ":latest"))
	}
	return ollamaModelTarget{}, false
}

func (s *Server) resolveOllamaExact(name string) (ollamaModelTarget, bool) {
	if localID, found := s.cfg.RealModelName(name); found {
		return ollamaModelTarget{publicName: name, localID: localID}, true
	}
	return ollamaModelTarget{}, false
}

func (s *Server) handleOllamaRoot(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
}

func (s *Server) handleOllamaTags(w http.ResponseWriter, _ *http.Request) {
	models := make([]ollamaapi.Tag, 0, len(s.cfg.Models))
	for name, model := range s.cfg.Models {
		if !model.Unlisted {
			models = append(models, ollamaapi.Tag{Name: name, Model: name})
		}
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ollamaapi.TagsResponse{Models: models})
}

func (s *Server) handleOllamaShow(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Model string `json:"model"`
	}
	if err := decodeOllamaRequest(r, &request); err != nil || strings.TrimSpace(request.Model) == "" {
		sendOllamaError(w, http.StatusBadRequest, "Ollama request requires model")
		return
	}
	target, found := s.resolveOllamaModel(request.Model)
	if !found {
		sendOllamaError(w, http.StatusNotFound, "model not found")
		return
	}
	model := s.cfg.Models[target.localID]
	capabilities := []string{"completion"}
	if contains(model.Capabilities.In, "image") {
		capabilities = append(capabilities, "vision")
	}
	if model.Capabilities.Tools {
		capabilities = append(capabilities, "tools")
	}
	response := ollamaapi.ShowResponse{Capabilities: capabilities}
	if model.Capabilities.Context > 0 {
		response.ModelInfo = map[string]any{"llama.context_length": model.Capabilities.Context}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func (s *Server) handleOllamaChat(w http.ResponseWriter, r *http.Request) {
	var request ollamaapi.ChatRequest
	if err := decodeOllamaRequest(r, &request); err != nil {
		sendOllamaError(w, http.StatusBadRequest, "invalid Ollama request")
		return
	}
	if strings.TrimSpace(request.Model) == "" {
		sendOllamaError(w, http.StatusBadRequest, "Ollama request requires model")
		return
	}
	target, found := s.resolveOllamaModel(request.Model)
	if !found {
		sendOllamaError(w, http.StatusNotFound, "model not found")
		return
	}
	if len(request.Tools) > 0 || len(request.Think) > 0 {
		sendOllamaError(w, http.StatusNotImplemented, "tools and think are not supported for local Ollama compatibility")
		return
	}
	for _, message := range request.Messages {
		if len(message.Images) > 0 {
			sendOllamaError(w, http.StatusNotImplemented, "images are not supported for local Ollama compatibility")
			return
		}
	}
	if len(request.Messages) == 0 {
		sendOllamaError(w, http.StatusBadRequest, "Ollama chat request requires messages")
		return
	}
	s.serveLocalOllamaChat(w, r, target, request.Messages, request.Options, request.Stream, false)
}

func (s *Server) handleOllamaGenerate(w http.ResponseWriter, r *http.Request) {
	var request ollamaapi.GenerateRequest
	if err := decodeOllamaRequest(r, &request); err != nil {
		sendOllamaError(w, http.StatusBadRequest, "invalid Ollama request")
		return
	}
	if strings.TrimSpace(request.Model) == "" {
		sendOllamaError(w, http.StatusBadRequest, "Ollama request requires model")
		return
	}
	target, found := s.resolveOllamaModel(request.Model)
	if !found {
		sendOllamaError(w, http.StatusNotFound, "model not found")
		return
	}
	if len(request.Context) > 0 || request.Suffix != "" || request.System != "" || request.Template != "" || request.Raw || len(request.Images) > 0 {
		sendOllamaError(w, http.StatusNotImplemented, "generate context, suffix, system, template, raw, and images are not supported for local Ollama compatibility")
		return
	}
	s.serveLocalOllamaChat(w, r, target, []ollamaapi.Message{{Role: "user", Content: request.Prompt}}, request.Options, request.Stream, true)
}

func decodeOllamaRequest(r *http.Request, destination any) error {
	body, err := io.ReadAll(io.LimitReader(r.Body, 128<<20))
	if closeErr := r.Body.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.Header.Set("Content-Length", fmt.Sprintf("%d", len(body)))
	return json.Unmarshal(body, destination)
}

func sendOllamaError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}

func (s *Server) serveLocalOllamaChat(w http.ResponseWriter, r *http.Request, target ollamaModelTarget, messages []ollamaapi.Message, options map[string]json.RawMessage, stream *bool, generate bool) {
	streaming := true
	if stream != nil {
		streaming = *stream
	}

	body := make(map[string]any, len(options)+3)
	body["model"] = target.localID
	body["messages"] = messages
	body["stream"] = streaming
	for key, value := range options {
		var option any
		if err := json.Unmarshal(value, &option); err != nil {
			sendOllamaError(w, http.StatusBadRequest, "invalid Ollama options")
			return
		}
		body[key] = option
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		sendOllamaError(w, http.StatusBadRequest, "invalid Ollama request")
		return
	}
	translated := r.Clone(r.Context())
	translated.URL.Path = "/v1/chat/completions"
	translated.URL.RawPath = ""
	translated.Body = io.NopCloser(bytes.NewReader(encoded))
	translated.ContentLength = int64(len(encoded))
	translated.Header = r.Header.Clone()
	translated.Header.Set("Content-Type", "application/json")
	translated.Header.Set("Content-Length", fmt.Sprintf("%d", len(encoded)))
	translated = translated.WithContext(swaputil.SetContext(translated.Context(), swaputil.ReqContextData{
		Model:     target.publicName,
		ModelID:   target.localID,
		Streaming: streaming,
		Metadata:  make(map[string]string),
	}))

	if streaming {
		converter := newOllamaStreamWriter(w, target.publicName, generate)
		s.local.ServeHTTP(converter, translated)
		converter.finish()
		return
	}

	recorder := httptest.NewRecorder()
	s.local.ServeHTTP(recorder, translated)
	copyOllamaLocalResponse(w, recorder.Result(), target.publicName, generate)
}

func copyOllamaLocalResponse(w http.ResponseWriter, response *http.Response, model string, generate bool) {
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		copyHeader(w.Header(), response.Header)
		w.WriteHeader(response.StatusCode)
		_, _ = io.Copy(w, response.Body)
		return
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, 128<<20))
	if err != nil {
		sendOllamaError(w, http.StatusBadGateway, "local model response could not be read")
		return
	}
	content, reason, err := openAIResponseContent(body)
	if err != nil {
		sendOllamaError(w, http.StatusBadGateway, "local model returned an unsupported response")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if generate {
		json.NewEncoder(w).Encode(ollamaapi.GenerateResponse{Model: model, Response: content, Done: true, Reason: reason})
		return
	}
	json.NewEncoder(w).Encode(ollamaapi.ChatResponse{Model: model, Message: ollamaapi.Message{Role: "assistant", Content: content}, Done: true, Reason: reason})
}

func openAIResponseContent(body []byte) (string, string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &response); err != nil || len(response.Choices) == 0 {
		return "", "", fmt.Errorf("invalid OpenAI response")
	}
	return response.Choices[0].Message.Content, response.Choices[0].FinishReason, nil
}

type ollamaStreamWriter struct {
	writer   http.ResponseWriter
	model    string
	generate bool
	buffer   bytes.Buffer
	started  bool
	done     bool
}

func newOllamaStreamWriter(writer http.ResponseWriter, model string, generate bool) *ollamaStreamWriter {
	return &ollamaStreamWriter{writer: writer, model: model, generate: generate}
}

func (w *ollamaStreamWriter) Header() http.Header { return w.writer.Header() }

func (w *ollamaStreamWriter) WriteHeader(status int) {
	if w.started {
		return
	}
	w.started = true
	if status < 200 || status >= 300 {
		w.writer.WriteHeader(status)
		return
	}
	w.writer.Header().Set("Content-Type", "application/x-ndjson")
	w.writer.Header().Set("X-Accel-Buffering", "no")
	w.writer.WriteHeader(http.StatusOK)
}

func (w *ollamaStreamWriter) Write(chunk []byte) (int, error) {
	if !w.started {
		w.WriteHeader(http.StatusOK)
	}
	if w.done {
		return len(chunk), nil
	}
	w.buffer.Write(chunk)
	for {
		pending := w.buffer.Bytes()
		lineEnd := bytes.IndexByte(pending, '\n')
		if lineEnd < 0 {
			break
		}
		line := strings.TrimSpace(string(pending[:lineEnd]))
		w.buffer.Next(lineEnd + 1)
		if err := w.processLine(line); err != nil {
			return len(chunk), err
		}
	}
	return len(chunk), nil
}

func (w *ollamaStreamWriter) Flush() {
	if flusher, ok := w.writer.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (w *ollamaStreamWriter) finish() {
	if !w.done && w.started {
		w.writeChunk("", "stop", true)
	}
}

func (w *ollamaStreamWriter) processLine(line string) error {
	if line == "" || !strings.HasPrefix(line, "data:") {
		return nil
	}
	payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if payload == "[DONE]" {
		w.writeChunk("", "stop", true)
		return nil
	}
	var chunk struct {
		Choices []struct {
			Delta struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil || len(chunk.Choices) == 0 {
		return nil
	}
	choice := chunk.Choices[0]
	if choice.Delta.Content != "" {
		w.writeChunk(choice.Delta.Content, "", false)
	}
	if choice.FinishReason != nil {
		w.writeChunk("", *choice.FinishReason, true)
	}
	return nil
}

func (w *ollamaStreamWriter) writeChunk(content, reason string, done bool) {
	if w.done {
		return
	}
	var payload any
	if w.generate {
		payload = ollamaapi.GenerateResponse{Model: w.model, Response: content, Done: done, Reason: reason}
	} else {
		payload = ollamaapi.ChatResponse{Model: w.model, Message: ollamaapi.Message{Role: "assistant", Content: content}, Done: done, Reason: reason}
	}
	if encoded, err := json.Marshal(payload); err == nil {
		_, _ = w.writer.Write(append(encoded, '\n'))
		w.Flush()
	}
	if done {
		w.done = true
	}
}

func copyHeader(destination, source http.Header) {
	for key, values := range source {
		destination[key] = append([]string(nil), values...)
	}
}

var _ http.ResponseWriter = (*ollamaStreamWriter)(nil)
var _ http.Flusher = (*ollamaStreamWriter)(nil)
