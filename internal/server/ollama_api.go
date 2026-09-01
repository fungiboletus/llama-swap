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
	if hasOllamaTools(request.Tools) {
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
	modelConfig := s.cfg.Models[target.localID]
	if modelConfig.OllamaManaged {
		// A managed lane is an isolated Ollama server. Let it interpret the
		// native request so prompt templates, token-ID context, raw prompts,
		// suffixes, and system prompts retain Ollama's exact semantics.
		if request.Prompt == "" && len(request.Context) == 0 {
			s.serveOllamaLoad(w, r, target, request.KeepAlive, true)
			return
		}
		s.serveManagedOllamaGenerate(w, r, target)
		return
	}
	if len(request.Context) > 0 || request.Suffix != "" || request.Template != "" || request.Raw {
		sendOllamaError(w, http.StatusNotImplemented,
			"generate context, suffix, template, and raw require a managed local Ollama model")
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
	generateMessages := make([]ollamaapi.Message, 0, 2)
	if request.System != "" {
		generateMessages = append(generateMessages, ollamaapi.Message{Role: "system", Content: request.System})
	}
	generateMessages = append(generateMessages, ollamaapi.Message{Role: "user", Content: request.Prompt, Images: request.Images})
	messages, err := translateOllamaMessages(generateMessages, contains(modelConfig.Capabilities.In, "image"))
	if err != nil {
		sendOllamaError(w, http.StatusBadRequest, err.Error())
		return
	}
	s.serveLocalOllamaChat(w, r, target, messages, request.Options, think, request.Stream, true)
}

// serveManagedOllamaGenerate forwards a native generate request to one of the
// isolated Ollama lanes. llama-swap remains responsible for choosing and
// keeping the process resident, so keep_alive is deliberately not forwarded.
func (s *Server) serveManagedOllamaGenerate(w http.ResponseWriter, r *http.Request, target ollamaModelTarget) {
	var body map[string]json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		sendOllamaError(w, http.StatusBadRequest, "invalid Ollama request")
		return
	}
	delete(body, "keep_alive")
	model, err := json.Marshal(target.localID)
	if err != nil {
		sendOllamaError(w, http.StatusBadRequest, "invalid Ollama request")
		return
	}
	body["model"] = model
	encoded, err := json.Marshal(body)
	if err != nil {
		sendOllamaError(w, http.StatusBadRequest, "invalid Ollama request")
		return
	}

	translated := r.Clone(r.Context())
	translated.URL.Path = "/api/generate"
	translated.URL.RawPath = ""
	translated.Body = io.NopCloser(bytes.NewReader(encoded))
	translated.ContentLength = int64(len(encoded))
	translated.Header = r.Header.Clone()
	translated.Header.Set("Content-Type", "application/json")
	translated.Header.Set("Content-Length", fmt.Sprintf("%d", len(encoded)))
	translated = translated.WithContext(swaputil.SetContext(translated.Context(), swaputil.ReqContextData{
		Model:     target.publicName,
		ModelID:   target.localID,
		Streaming: requestStreams(body),
		Metadata:  make(map[string]string),
	}))
	s.local.ServeHTTP(w, translated)
}

func requestStreams(body map[string]json.RawMessage) bool {
	stream, found := body["stream"]
	if !found {
		return true
	}
	var enabled bool
	return json.Unmarshal(stream, &enabled) == nil && enabled
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

// hasOllamaTools distinguishes a real tool definition from the empty value
// emitted by Ollama's Python SDK when callers omit its optional tools argument.
// An empty list cannot cause a model to call a tool, so it has the same meaning
// as an omitted or null field and is safe to translate as an ordinary chat.
func hasOllamaTools(raw json.RawMessage) bool {
	tools := bytes.TrimSpace(raw)
	return len(tools) > 0 && !bytes.Equal(tools, []byte("null")) && !bytes.Equal(tools, []byte("[]"))
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
	// Native Ollama requests bypass the normal JSON filter middleware because
	// their body is translated above. Apply the configured OpenAI-side filters
	// now so managed Ollama lanes still request the final streaming usage block
	// needed by the activity store.
	modelConfig := s.cfg.Models[target.localID]
	encoded, err = applyFilters(encoded, target.localID, upstreamModel, modelConfig.Filters.Filters)
	if err != nil {
		sendOllamaError(w, http.StatusInternalServerError, err.Error())
		return
	}
	encoded, err = applyMaxTokens(encoded, "/v1/chat/completions", modelConfig.MaxTokens)
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
	if acceptEncoding := translated.Header.Get("Accept-Encoding"); acceptEncoding != "" {
		translated.Header.Set("Accept-Encoding", filterAcceptEncoding(acceptEncoding))
	}
	translated = translated.WithContext(swaputil.SetContext(translated.Context(), swaputil.ReqContextData{
		Model:     target.publicName,
		ModelID:   target.localID,
		Streaming: streaming,
		Metadata:  make(map[string]string),
	}))
	captureFields, reqBody, reqHeaders := s.captureOllamaActivityRequest(r)

	if streaming {
		converter := newOllamaStreamWriter(w, target.publicName, generate)
		recorder := newBodyCopier(converter)
		// The converter changes the client response to NDJSON, but upstream is
		// still SSE. Keep observing its content deltas for rate estimates.
		recorder.observeSSE = true
		s.local.ServeHTTP(recorder, translated)
		converter.finish()
		s.recordOllamaActivity(target.localID, r, recorder, captureFields, reqBody, reqHeaders, true)
		return
	}

	response := httptest.NewRecorder()
	recorder := newBodyCopier(response)
	s.local.ServeHTTP(recorder, translated)
	copyOllamaLocalResponse(w, response.Result(), target.publicName, generate)
	s.recordOllamaActivity(target.localID, r, recorder, captureFields, reqBody, reqHeaders, false)
}

// captureOllamaActivityRequest preserves the original Ollama request for an
// optional activity capture. The translated OpenAI request is intentionally
// not captured: the Activity page should identify the API request the client
// actually sent.
func (s *Server) captureOllamaActivityRequest(r *http.Request) (captureFields, []byte, map[string]string) {
	fields := captureFieldsFor(r.URL.Path)
	if s.metrics == nil || !s.metrics.enableCaptures {
		return fields, nil, nil
	}

	var body []byte
	if fields&captureReqBody != 0 && r.Body != nil {
		if buffered, err := io.ReadAll(r.Body); err == nil {
			body = buffered
			r.Body.Close()
			r.Body = io.NopCloser(bytes.NewReader(body))
		}
	}
	var headers map[string]string
	if fields&captureReqHeaders != 0 {
		headers = headerMap(r.Header)
		redactHeaders(headers)
	}
	return fields, body, headers
}

// recordOllamaActivity records the unconverted OpenAI-compatible upstream
// response. That response is where usage and timings exist; the Ollama wire
// response intentionally does not expose them. The original request remains
// attached so the activity path and authenticated user reflect the caller.
func (s *Server) recordOllamaActivity(modelID string, r *http.Request, recorder *responseBodyCopier, fields captureFields, reqBody []byte, reqHeaders map[string]string, streaming bool) {
	if s.metrics == nil {
		return
	}
	swaputil.MarkClientClosed(recorder, r)
	if streaming {
		// ollamaStreamWriter replaces the client-facing SSE response with NDJSON,
		// while recorder retains the original SSE body that carries final usage.
		s.metrics.recordStreaming(modelID, r, recorder, fields, reqBody, reqHeaders)
		return
	}
	s.metrics.record(modelID, r, recorder, fields, reqBody, reqHeaders)
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
