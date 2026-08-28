# Manual Ollama text and ps smoke test

This opt-in smoke test uses the pinned Ollama CLI baseline `0.32.15`. It
requires a running llama-swap instance and one configured text-capable model.
It is not part of `make test-all`.

```bash
export OLLAMA_HOST=http://127.0.0.1:8080
ollama ps
ollama run configured-text-model 'Reply with exactly: ok'
ollama ps
```

The first `ps` may be empty. The final one should show the ready local model.
For a command-backed model, `ID` displays `unknown-----`, `PROCESSOR` is
`100% CPU` because per-model placement is unknown, and `UNTIL` may show
`Forever` even though automatic admission can evict it for capacity.

For a key-protected inference endpoint, use a direct API client that supplies
the configured bearer key or an authentication proxy that injects it; the
standard Ollama CLI does not send llama-swap bearer keys.
