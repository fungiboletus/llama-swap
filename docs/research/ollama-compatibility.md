# Ollama compatibility research

## Scope and reproducible baseline

This note defines the deliberately small Ollama surface for this fork. It is
an implementation input for Tasks 02, 07, 08, and 09; it is not a promise to
reimplement Ollama's model store or cloud features.

Research was performed on 2026-08-28 with these pinned inputs:

| Input | Version / revision |
| --- | --- |
| Ollama CLI and local server | `0.32.15` |
| Ollama source | `f96e7aa0513b9973a0ccc71be414c2ecb9d65b1a` (2026-08-27) |
| llama-swappo source | `c2f9b65f8bf91a6f1bc985aa47bfdd6ea121d621` (2026-05-26) |
| llama-swap base | `176206aca312e6673dcd016dc6050c36fd8cf3f1` (2026-08-27) |

The two source snapshots are shallow clones in the ignored `.research/`
directory. The captures below used a temporary local Ollama server on
`127.0.0.1:11435` and request logging, then unloaded the test models.

## What the current CLI actually uses

`ollama ps` first sends `HEAD /`, then `GET /api/ps`. The `HEAD` response must
be a successful plain-text heartbeat; its body is not used.

Non-interactive `ollama run <model> <prompt>` first sends `HEAD /`, then
`POST /api/show` with `{"model":"<model>"}`, and only then streams
`POST /api/generate`. It does **not** use `/api/chat` for that command. The
show response determines capabilities before generation, so returning a
successful generate response alone is not enough for CLI compatibility.

The text capture sent a generate request equivalent to:

```json
{
  "model": "configured-text-model",
  "prompt": "Respond with exactly: ok",
  "options": {}
}
```

`stream` is omitted by the CLI, which means streaming is the server default.
The response is newline-delimited JSON; intermediate records have
`done: false`, and the final record has `done: true`. Cancellation is the
normal HTTP request-context cancellation path, so translations must stop the
local upstream request when the client disconnects.

For a vision-capable model, the CLI recognizes an image path only when it is a
path such as `./image.jpg` or `/image.jpg`, not a bare filename. It removes the
path from `prompt` and sends the raw bytes as a JSON base64 `images` array:

```json
{
  "model": "configured-vision-model",
  "prompt": "Describe in three words.",
  "images": ["<base64 image bytes>"],
  "options": {}
}
```

The recorded image was a 267,532-byte JPEG and the resulting base64 string
contained 356,712 characters. This is the wire shape Task 09 must preserve
for forwarded models and translate for a local server that supports images.

## Minimal endpoint contract

Only the following fields are needed by the current CLI or are required for
the documented direct API subset. Unknown values must be omitted rather than
invented.

| Endpoint | Required request | Minimum response / behavior | Consumer |
| --- | --- | --- | --- |
| `HEAD /` | none | `200`; plain-text heartbeat | CLI connection check |
| `GET /api/version` | none | `{"version":"<llama-swap version>"}` | API clients |
| `GET /api/tags` | none | `{"models":[...]}`; each known model has `name`, `model`, a CLI-safe `digest`, `size`, and `modified_at` | discovery and CLI list |
| `POST /api/show` | `model` | `200` for an allowed model; include `capabilities`, with optional truthful `details`, `model_info`, and `messages` | `ollama run` preflight |
| `GET /api/ps` | none | `{"models":[...]}`; each loaded local model has `name`, `model`, `size`, `size_vram`, `details`, and `expires_at` | `ollama ps` |
| `POST /api/generate` | `model`, `prompt`; optional `images`, `options`, `stream`, `keep_alive` | NDJSON when streaming, one JSON object when `stream:false`; final response has `done:true` | `ollama run`, direct clients |
| `POST /api/chat` | `model`, `messages`; optional `images` per message, `options`, `stream`, `keep_alive` | Same streaming rule; chunks contain `message` and final `done:true` | direct chat clients |

`/api/tags` may expose configured aliases only if a later compatibility test
shows a client needs them. The canonical configured public name is otherwise
the stable listed name. Ollama CLI 0.32.15 unconditionally slices the digest
when rendering `ollama list`, so each entry must provide a string at least 12
characters long. llama-swap uses the explicit `unknown-----` marker for a
model without a stable Ollama manifest digest, plus zero size and modification
time rather than fabricating model-store metadata.

For `/api/show`, the current CLI inspects `capabilities` to decide whether a
model supports completion or vision. The first implementation should return
only capabilities declared by the configuration. In particular, do not infer
`vision` from a command line.

`/api/ps` is detailed further in Task 08. The current CLI renders the provided
fields but does not turn an unavailable CPU/GPU split into a hard error. A
configured total reservation is therefore valid for `size`; `size_vram` must
be `0` when the managed process has not reported that split.

For the `0.32.15` CLI, a zero `expires_at` is rendered as `Stopping...` because
the client compares it with the current time before checking whether it is
zero. Ollama represents an infinite `OLLAMA_KEEP_ALIVE` with its maximum
duration instead. llama-swap uses that same long-lived response sentinel for a
ready process with no tracked expiry, so the CLI renders `Forever`; it is not a
claim that automatic memory admission will never evict the process.

## Naming decision

Names are an operator-controlled catalogue, not Ollama registry references.

1. A local model ID is its canonical public name and is returned by discovery,
   status, and inference responses.
2. A configured alias resolves to that local model ID. Exact model IDs take
   precedence over aliases, matching existing llama-swap resolution.
3. Resolve an exact name first. If it does not resolve and it ends in
   `:latest`, retry the base name as an ID or alias. This permits the common
   Ollama spelling without silently changing any other tag.
4. Configuration must reject collisions across IDs, aliases, and the implicit
   `:latest` fallback. For example, `foo` and a distinct `foo:latest` cannot
   denote different configured models.
5. Do not synthesize, accept, or list digest names for local command-backed
   models. An unconfigured name is a `404` JSON error and must never trigger a
   pull or upstream discovery.

Task 07 owns the resolver and its exact fixtures. Task 02 applies the same
rules to allowlisted forwarded public names before it forwards the request.

## Support decision

| Endpoint family | Local configured model | Allowlisted forwarded model | Decision |
| --- | --- | --- | --- |
| heartbeat and version | local response | local response | local translation |
| tags | configured catalogue | configured catalogue entry only | local translation |
| show | synthesize declared capability/known metadata | proxy after Task 02, with public name restored | local translation / forward |
| ps | managed local process state | omit: it is not a llama-swap-managed process | local translation |
| chat and generate text | translate to the selected local upstream protocol | forward after Task 02 | local translation / forward |
| chat and generate images | Task 09; reject before model start until supported | preserve the Ollama payload in Task 09 | deferred |
| pull, push, create, copy, delete, blobs | none | none | reject; not planned |
| embed, embeddings, OpenAI `/v1/*`, cloud/status/experimental APIs | none in this fork-specific contract | none | not planned unless a consumer requires them |

Rejecting an unsupported or unconfigured operation uses the normal Ollama
error shape, `{"error":"..."}`, with a 404 for an absent endpoint/model or
a 501 for a recognized-but-not-implemented operation. The selection should be
consistent per endpoint and covered by fixtures; it must not accidentally
expose management behavior.

## llama-swappo review

The reviewed snapshot is a single squashed compatibility addition centred on
`proxy/ollama_api.go`. It is useful as a catalogue of endpoint shapes and for
the following implementation ideas:

- keep protocol-specific request/response types together;
- transform streaming OpenAI chunks into newline-delimited Ollama JSON and
  flush every chunk;
- use the standard `{"error":"..."}` response shape; and
- derive `ps` from managed-process readiness rather than from the model list.

It should not be copied directly. It is built around the older Gin proxy
architecture, reports a stale hard-coded Ollama version, guesses metadata from
model names, parses arbitrary command lines, and invents a digest from the
model ID. Those behaviours conflict with this fork's explicit-configuration
and truthful-metadata boundary. The new implementation belongs in a focused
compatibility package plus the existing `internal/server` routing layer.

## Fixture guidance for the next tasks

Use hermetic `httptest` local/upstream responders. Do not require a GPU, a
downloaded model, or a live Ollama instance. Cover at least:

- empty and one-model `tags`/`ps` responses;
- `show` for a canonical ID, alias, implicit `:latest`, and unknown name;
- streaming and non-streaming text generation, including cancellation;
- a forwarded allowed name whose upstream name differs from its public name;
- an unknown/disabled/management name that cannot cause a pull; and
- the base64 image array for a later vision test.

The smoke-test baseline for compatibility changes is `ollama 0.32.15`.
