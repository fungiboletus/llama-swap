# Migrating from Ollama

llama-swap can replace a fixed local Ollama server when you want a controlled
model catalogue, separate operational access, and memory-budgeted residency.
It is an Ollama-compatible gateway, not an Ollama model store.

Start with [the minimal local configuration](examples/ollama-local.yaml), or
use [the production example](examples/ollama-production.yaml) for split keys,
a command-backed model, and automatic residency.

## Supported workflow

Set the CLI target after llama-swap is running:

```bash
export OLLAMA_HOST=http://127.0.0.1:8080
ollama list
ollama ps
ollama run configured-model 'Reply with exactly: ok'
```

The supported API surface is intentionally small:

- `HEAD /`, `GET /api/version`, `GET /api/tags`, `POST /api/show`, and
  `GET /api/ps`;
- text `POST /api/chat` and `POST /api/generate`, including streaming;
- image input for models that declare `capabilities.in: [image]`; and
- configured local command-backed models plus managed, locally installed
  Ollama models.

`ollama ps` lists ready local processes, not every configured model. `size` is
llama-swap's configured reservation rather than measured resident memory,
`size_vram` is `0`, and its `ID` column is `unknown-----` for command-backed
models because no Ollama manifest digest exists. `Forever` means llama-swap
does not track an idle expiry; memory admission can still evict that model.

`ollama list` lists the configured catalogue. Its `ID` is likewise
`unknown-----` where llama-swap does not have a stable Ollama manifest digest;
its `SIZE` is `0 B` and `MODIFIED` is `Never` rather than guessed metadata.

## Catalogue and naming

Managed Ollama models are discovered once at startup from the local catalogue.
Restart llama-swap after `ollama pull` or removing a model. A local model's
configuration key is its canonical public name; aliases resolve for requests,
and `name:latest` falls back to `name` when no exact `:latest` model exists.

An undefined model is never pulled, created, or started. There is no remote
catalogue lookup or forwarding configuration in this fork. Command-backed
models are ordinary `models` entries and must be explicitly configured.

The following Ollama operations are not implemented: `pull`, `push`, `create`,
`copy`, `rm`/delete, blobs, remote catalogue discovery, and arbitrary model
execution. Do not point clients that require these model-management operations
at llama-swap.

## Authentication and the CLI

Set `adminApiKeys` to protect the UI, logs, metrics, unload, residency, and
other operational endpoints. Open `/ui/` in a browser and use any username
with the admin key as the HTTP Basic-auth password.

Use `apiKeys` when inference itself needs bearer-token protection:

```bash
curl http://127.0.0.1:8080/api/tags \
  -H 'Authorization: Bearer replace-with-inference-key'
```

The standard Ollama CLI does not add llama-swap bearer keys to requests. For
`ollama run`/`ps`, leave inference open on the intended trusted network, or
place an authentication proxy in front that injects the authorization header.
An admin key also grants inference access for direct API clients.

## Automatic residency

With `routing.router.use: auto`, declare a conservative total capacity and
reserve host headroom. Prefer an explicit `memory` value that includes model
weights, context/KV cache, backend allocations, and the configured request
parallelism. On Apple unified memory, an omitted value is conservatively
calibrated in isolation and persisted; NVIDIA and ROCm deployments require
explicit values.

The proxy log records each admission decision as `auto:` with target,
reservation, used capacity, evictions, and reason. `GET /api/residency` shows
the configured/calibrated reservation and current process state. A
`keepResident: true` model is preferred, not pinned: it is evicted with a
warning when that is required to admit another fitting request.

## Troubleshooting

For streaming, ensure every reverse proxy disables response buffering for
`/api/chat` and `/api/generate`; llama-swap emits newline-delimited JSON for
Ollama streaming. A client disconnect cancels the upstream request normally.

For upstream failures, first check the model's command, proxy, and health
endpoint in the configuration, then inspect the admin UI or proxy/process logs.
An invalid image, unsupported capability, or an oversized request fails before
the model starts; a later 5xx response comes from the selected local upstream.

## Smoke-test baseline and upgrades

The compatibility smoke baseline is Ollama CLI `0.32.15`. The documented
manual checks are [the text and ps smoke](ollama-smoke.md) and the optional
[vision smoke](ollama-vision-smoke.md). When upgrading the pinned CLI, capture
its requests for `ps`, `run`, and vision; update the compatibility research,
fixtures, and smoke results together before changing this baseline.
