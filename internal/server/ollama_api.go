package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"time"

	"github.com/mostlygeek/llama-swap/internal/ollamaapi"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

type ollamaModelTarget struct {
	publicName string
	localID    string
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

// unknownOllamaDigest is long enough for the Ollama CLI's 12-character ID
// column while making it explicit that command-backed models have no Ollama
// manifest digest.
const unknownOllamaDigest = "unknown-----"

// ollamaNoExpiryDuration matches Ollama's representation of a negative
// OLLAMA_KEEP_ALIVE value. It is a protocol sentinel for no tracked deadline,
// not an actual promise that the process can never be evicted.
const ollamaNoExpiryDuration = time.Duration(math.MaxInt64)

const (
	maxOllamaRequestBytes = 128 << 20
	maxOllamaImageBytes   = 32 << 20
	maxOllamaImageCount   = 8
)

var errOllamaRequestTooLarge = errors.New("Ollama request exceeds the maximum size")

// resolveOllamaModel maps a requested Ollama name onto a configured model.
//
// ":latest" is Ollama's implicit tag, and the two spellings of a name are the
// same model to every Ollama client: `ollama run qwen3` and `ollama run
// qwen3:latest` are one command typed two ways. Which spelling llama-swap has
// configured depends on where the model came from — the discovered Ollama
// catalogue reports tagged names, while a command-backed model is usually
// given a bare ID — so the fallback runs in both directions rather than
// making operators guess which one their clients must type.
//
// Only the implicit tag is inferred. A name that already carries some other
// tag names a specific model, and a miss there is a genuine 404 rather than an
// invitation to serve a different quantization than the one asked for.
func (s *Server) resolveOllamaModel(name string) (ollamaModelTarget, bool) {
	if localID, found := s.cfg.RealModelName(name); found {
		return ollamaModelTarget{publicName: name, localID: localID}, true
	}
	if bare, tagged := strings.CutSuffix(name, ":latest"); tagged {
		return s.resolveOllamaExact(bare)
	}
	if !strings.Contains(name, ":") {
		return s.resolveOllamaExact(name + ":latest")
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
			// The Ollama CLI slices digest[:12] while rendering `ollama list`.
			// Command-backed and isolated managed lanes do not expose a stable
			// manifest digest to llama-swap, so use an explicit CLI-safe marker.
			// Size and modification time are intentionally zero rather than guessed.
			models = append(models, ollamaapi.Tag{
				Name:   name,
				Model:  name,
				Digest: unknownOllamaDigest,
			})
		}
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ollamaapi.TagsResponse{Models: models})
}

// handleOllamaPS reports ready local processes only. A configured reservation
// is a capacity-planning input, not measured resident memory; no per-model
// CPU/GPU split or expiry is available from the local router.
func (s *Server) handleOllamaPS(w http.ResponseWriter, _ *http.Request) {
	states := s.local.RunningModels()
	models := make([]ollamaapi.ProcessModel, 0, len(states))
	for modelID, state := range states {
		if state != process.StateReady {
			continue
		}
		model, found := s.cfg.Models[modelID]
		if !found {
			continue
		}
		models = append(models, ollamaapi.ProcessModel{
			Name:          modelID,
			Model:         modelID,
			Size:          int64(model.Memory),
			Digest:        unknownOllamaDigest,
			Details:       struct{}{},
			ExpiresAt:     time.Now().Add(ollamaNoExpiryDuration),
			SizeVRAM:      0,
			ContextLength: model.Capabilities.Context,
		})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ollamaapi.ProcessResponse{Models: models})
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
	if model.Capabilities.Thinking {
		capabilities = append(capabilities, "thinking")
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
		if errors.Is(err, errOllamaRequestTooLarge) {
			sendOllamaError(w, http.StatusRequestEntityTooLarge, err.Error())
		} else {
			sendOllamaError(w, http.StatusBadRequest, "invalid Ollama request")
		}
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
	if len(request.Tools) > 0 {
		sendOllamaError(w, http.StatusNotImplemented, "tools are not supported for local Ollama compatibility")
		return
	}
	think, err := parseOllamaThink(request.Think)
	if err != nil {
		sendOllamaError(w, http.StatusBadRequest, err.Error())
		return
	}
	if len(request.Messages) == 0 {
		s.serveOllamaLoad(w, r, target, request.KeepAlive, false)
		return
	}
	messages, err := translateOllamaMessages(request.Messages, contains(s.cfg.Models[target.localID].Capabilities.In, "image"))
	if err != nil {
		sendOllamaError(w, http.StatusBadRequest, err.Error())
		return
	}
	s.serveLocalOllamaChat(w, r, target, messages, request.Options, think, request.Stream, false)
}

func (s *Server) handleOllamaGenerate(w http.ResponseWriter, r *http.Request) {
	var request ollamaapi.GenerateRequest
	if err := decodeOllamaRequest(r, &request); err != nil {
		if errors.Is(err, errOllamaRequestTooLarge) {
			sendOllamaError(w, http.StatusRequestEntityTooLarge, err.Error())
		} else {
			sendOllamaError(w, http.StatusBadRequest, "invalid Ollama request")
		}
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
	if len(request.Context) > 0 || request.Suffix != "" || request.System != "" || request.Template != "" || request.Raw {
		sendOllamaError(w, http.StatusNotImplemented, "generate context, suffix, system, template, and raw are not supported for local Ollama compatibility")
		return
	}
	if request.Prompt == "" {
		s.serveOllamaLoad(w, r, target, request.KeepAlive, true)
		return
	}
	think, err := parseOllamaThink(request.Think)
	if err != nil {
		sendOllamaError(w, http.StatusBadRequest, err.Error())
		return
	}
	messages, err := translateOllamaMessages([]ollamaapi.Message{{Role: "user", Content: request.Prompt, Images: request.Images}}, contains(s.cfg.Models[target.localID].Capabilities.In, "image"))
	if err != nil {
		sendOllamaError(w, http.StatusBadRequest, err.Error())
		return
	}
	s.serveLocalOllamaChat(w, r, target, messages, request.Options, think, request.Stream, true)
}

func decodeOllamaRequest(r *http.Request, destination any) error {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxOllamaRequestBytes+1))
	if closeErr := r.Body.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if len(body) > maxOllamaRequestBytes {
		return errOllamaRequestTooLarge
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	r.ContentLength = int64(len(body))
	r.Header.Set("Content-Length", fmt.Sprintf("%d", len(body)))
	return json.Unmarshal(body, destination)
}

func translateOllamaMessages(messages []ollamaapi.Message, acceptsImages bool) ([]openAIChatMessage, error) {
	translated := make([]openAIChatMessage, 0, len(messages))
	for _, message := range messages {
		images, err := ollamaImageURLs(message.Images, maxOllamaImageBytes)
		if err != nil {
			return nil, err
		}
		if len(images) == 0 {
			translated = append(translated, openAIChatMessage{Role: message.Role, Content: message.Content})
			continue
		}
		if !acceptsImages {
			return nil, errors.New("model does not declare image input capability")
		}
		content := make([]any, 0, len(images)+1)
		if message.Content != "" {
			content = append(content, map[string]string{"type": "text", "text": message.Content})
		}
		for _, imageURL := range images {
			content = append(content, map[string]any{
				"type":      "image_url",
				"image_url": map[string]string{"url": imageURL},
			})
		}
		translated = append(translated, openAIChatMessage{Role: message.Role, Content: content})
	}
	return translated, nil
}

func ollamaImageURLs(raw json.RawMessage, maxBytes int) ([]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var encoded []string
	if err := json.Unmarshal(raw, &encoded); err != nil || encoded == nil {
		return nil, errors.New("images must be an array of base64 image strings")
	}
	if len(encoded) > maxOllamaImageCount {
		return nil, fmt.Errorf("images exceeds the maximum count of %d", maxOllamaImageCount)
	}
	urls := make([]string, 0, len(encoded))
	for _, value := range encoded {
		image, err := base64.StdEncoding.DecodeString(value)
		if err != nil || len(image) == 0 {
			return nil, errors.New("images contains invalid base64 image data")
		}
		if len(image) > maxBytes {
			return nil, fmt.Errorf("image exceeds the maximum size of %d bytes", maxBytes)
		}
		mime, ok := ollamaImageMIME(image)
		if !ok {
			return nil, errors.New("images contains an unsupported image format")
		}
		urls = append(urls, "data:"+mime+";base64,"+value)
	}
	return urls, nil
}

func ollamaImageMIME(image []byte) (string, bool) {
	switch {
	case len(image) >= 3 && bytes.Equal(image[:3], []byte{0xff, 0xd8, 0xff}):
		return "image/jpeg", true
	case len(image) >= 8 && bytes.Equal(image[:8], []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}):
		return "image/png", true
	case len(image) >= 12 && bytes.Equal(image[:4], []byte("RIFF")) && bytes.Equal(image[8:12], []byte("WEBP")):
		return "image/webp", true
	default:
		return "", false
	}
}

func sendOllamaError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}

func (s *Server) serveLocalOllamaChat(w http.ResponseWriter, r *http.Request, target ollamaModelTarget, messages []openAIChatMessage, options map[string]json.RawMessage, think thinkSetting, stream *bool, generate bool) {
	streaming := true
	if stream != nil {
		streaming = *stream
	}

	body := make(map[string]any, len(options)+3)
	upstreamModel := target.localID
	if configured := s.cfg.Models[target.localID].UseModelName; configured != "" {
		upstreamModel = configured
	}
	body["model"] = upstreamModel
	body["messages"] = messages
	body["stream"] = streaming
	for key, value := range options {
		var option any
		if err := json.Unmarshal(value, &option); err != nil {
			sendOllamaError(w, http.StatusBadRequest, "invalid Ollama options")
			return
		}
		// num_predict is Ollama's spelling of max_tokens. The body being built
		// here is an OpenAI chat request, where the original name means
		// nothing, so the option is carried across under the name the upstream
		// actually reads — and where the generation cap can see it. Ollama
		// spells "unlimited" as a non-positive num_predict; OpenAI spells it by
		// omitting the field, which also leaves the cap free to fill in a
		// ceiling of its own.
		if key == "num_predict" {
			count, numeric := option.(float64)
			if !numeric || count <= 0 {
				continue
			}
			key = "max_tokens"
		}
		body[key] = option
	}
	// After the options loop: `think` is an explicit request field, so it wins
	// over anything the same client smuggled through options.
	think.applyTo(body)
	encoded, err := json.Marshal(body)
	if err != nil {
		sendOllamaError(w, http.StatusBadRequest, "invalid Ollama request")
		return
	}
	encoded, err = applyMaxTokens(encoded, "/v1/chat/completions", s.cfg.Models[target.localID].MaxTokens)
	if err != nil {
		sendOllamaError(w, http.StatusInternalServerError, err.Error())
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
	content, thinking, reason, err := openAIResponseContent(body)
	if err != nil {
		sendOllamaError(w, http.StatusBadGateway, "local model returned an unsupported response")
		return
	}
	w.Header().Set("Content-Type", "application/json")
	if generate {
		json.NewEncoder(w).Encode(ollamaapi.GenerateResponse{Model: model, Response: content, Thinking: thinking, Done: true, Reason: reason})
		return
	}
	json.NewEncoder(w).Encode(ollamaapi.ChatResponse{Model: model, Message: ollamaapi.Message{Role: "assistant", Content: content, Thinking: thinking}, Done: true, Reason: reason})
}

// openAIResponseContent returns the answer, the reasoning trace, and the finish
// reason. Both spellings of the reasoning field are read; see reasoningText.
func openAIResponseContent(body []byte) (string, string, string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				Reasoning        string `json:"reasoning"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &response); err != nil || len(response.Choices) == 0 {
		return "", "", "", fmt.Errorf("invalid OpenAI response")
	}
	choice := response.Choices[0]
	thinking := reasoningText(choice.Message.ReasoningContent, choice.Message.Reasoning)
	return choice.Message.Content, thinking, choice.FinishReason, nil
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
		w.writeChunk("", "", "stop", true)
	}
}

func (w *ollamaStreamWriter) processLine(line string) error {
	if line == "" || !strings.HasPrefix(line, "data:") {
		return nil
	}
	payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if payload == "[DONE]" {
		w.writeChunk("", "", "stop", true)
		return nil
	}
	var chunk struct {
		Choices []struct {
			Delta struct {
				Role             string `json:"role"`
				Content          string `json:"content"`
				Reasoning        string `json:"reasoning"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil || len(chunk.Choices) == 0 {
		return nil
	}
	choice := chunk.Choices[0]
	// Ollama streams reasoning as its own chunks, with thinking set and
	// content empty, so a client can render the two differently as they
	// arrive. A delta carrying both is emitted as two chunks for the same
	// reason.
	if thinking := reasoningText(choice.Delta.ReasoningContent, choice.Delta.Reasoning); thinking != "" {
		w.writeChunk("", thinking, "", false)
	}
	if choice.Delta.Content != "" {
		w.writeChunk(choice.Delta.Content, "", "", false)
	}
	if choice.FinishReason != nil {
		w.writeChunk("", "", *choice.FinishReason, true)
	}
	return nil
}

func (w *ollamaStreamWriter) writeChunk(content, thinking, reason string, done bool) {
	if w.done {
		return
	}
	var payload any
	if w.generate {
		payload = ollamaapi.GenerateResponse{Model: w.model, Response: content, Thinking: thinking, Done: done, Reason: reason}
	} else {
		payload = ollamaapi.ChatResponse{Model: w.model, Message: ollamaapi.Message{Role: "assistant", Content: content, Thinking: thinking}, Done: done, Reason: reason}
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
