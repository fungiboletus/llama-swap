# config.yaml

llama-swap is designed to be very simple: one binary, one configuration file.

## minimal viable config

```yaml
models:
  model1:
    cmd: llama-server --port ${PORT} --model /path/to/model.gguf
```

This is enough to launch `llama-server` to serve `model1`. Of course, llama-swap is about making it possible to serve many models:

```yaml
models:
  model1:
    cmd: llama-server --port ${PORT} -m /path/to/model.gguf
  model2:
    cmd: llama-server --port ${PORT} -m /path/to/another_model.gguf
  model3:
    cmd: llama-server --port ${PORT} -m /path/to/third_model.gguf
```

With this configuration models will be hot swapped and loaded on demand. The special `${PORT}` macro provides a unique port per model which is useful if you want to run multiple models at the same time with the `matrix` feature.

## Access control

Use `apiKeys` to protect inference and `adminApiKeys` to protect operational
endpoints such as the UI, logs, metrics, profiles, unload, and raw upstream
passthrough. An admin key can also call inference endpoints. For a shared LAN
where inference is open but administration is protected:

```yaml
adminApiKeys:
  - "${env.LLAMA_SWAP_ADMIN_KEY}"
```

For compatibility with existing deployments, when `adminApiKeys` is omitted,
`apiKeys` protects both inference and administration. When both lists are
empty, llama-swap is open; use that only behind suitable network controls.

When administrative keys are configured, open `/ui/` directly in a browser.
llama-swap responds with an HTTP Basic-auth challenge, so the browser prompts
for the key and reuses the resulting credentials for the UI API and event
stream requests. Enter any username and use the configured admin key as the
password.

## Ollama model sources

`ollama` enables managed local Ollama lanes. This is designed for Apple unified
memory: llama-swap, rather than Ollama's global scheduler, owns cross-model
admission and eviction. At startup llama-swap briefly starts a private Ollama
server to read the installed catalogue. It then makes each discovered model a
normal `auto`-router member and starts one `ollama serve` process for that
model only when it is admitted. Restart llama-swap after pulling or removing a
model so it can refresh that fixed runtime catalogue.

```yaml
ollama:
  command: ollama
  startPort: 11450
  contextLength: 32768 # optional; 0 keeps each model's Ollama default
  numParallel: 1
  # optional on Apple: first load is calibrated and persisted
  memory: 24GiB

routing:
  router:
    use: auto
    settings:
      auto:
        capacity: 96GiB
        reserve: 8GiB
```

Each lane gets a distinct localhost port, `OLLAMA_MAX_LOADED_MODELS=1`, and an
infinite Ollama keep-alive. `numParallel` becomes the per-model request-
concurrency limit and affects the model's actual memory use, so choose it
conservatively.
The isolated lane-port range must not overlap another local service.

## Vision through the Ollama API

Local models opt in to vision with their existing capability declaration:

```yaml
models:
  vision-model:
    cmd: vision-server --port ${PORT}
    capabilities:
      in: [text, image]
```

Managed Ollama lanes inherit this capability from Ollama's startup
`/api/show` response; a model whose capability cannot be confirmed remains
text-only. For `POST /api/chat` and `POST /api/generate`, llama-swap accepts
Ollama's raw base64 `images` array and passes it to the OpenAI-compatible
upstream as `image_url` data-URL content blocks. JPEG, PNG, and WebP are
supported. Each decoded image is limited to 32 MiB, with at most eight images
per message/request and the existing 128 MiB total request limit.

Image validation happens before router admission, so malformed, unsupported,
or oversized image data never starts or swaps a model. See
[the opt-in vision smoke guide](ollama-vision-smoke.md) for testing a real
vision model; it is not part of the automated test suite.

## Automatic residency memory budget

The `auto` router uses explicit, conservative reservations instead of trying
to infer process memory or GPU placement. Specify each local model's deployed
resident footprint and the one memory pool it may use:

```yaml
models:
  coder:
    cmd: llama-server --port ${PORT} --model /models/coder.gguf
    memory: 28GiB
  chat:
    cmd: llama-server --port ${PORT} --model /models/chat.gguf
    memory: 18GiB

routing:
  router:
    use: auto
    settings:
      auto:
        capacity: 96GiB
        reserve: 8GiB
```

Only exact integer values with `B`, `KiB`, `MiB`, `GiB`, or `TiB` are accepted.
The usable budget is `capacity - reserve`. A positive `memory` value is an
authoritative operator override. When it is omitted, the auto router initially
loads that model by itself and records a conservative observed footprint for
later starts. Group and matrix routers do not need memory fields.

Choose a reservation by running one model at a time with its production
command and context settings, then use a conservative peak from its startup
and runtime logs. Include model weights, KV cache, and backend allocations.
Leave host, driver, and unrelated-process headroom in `reserve`; do not add
separate CPU and shared/unified-GPU capacities together.

`GET /api/residency` exposes the configured or learned reservations and budget.
Use `/api/performance` separately for aggregate host telemetry.

On Apple Silicon, a model with no `memory` override starts by itself once. The
auto router records the increase in unified memory at readiness and keeps a
30-second high-water observation only while that model remains the sole
managed resident model. It never assigns aggregate growth while models share
the pool, and it never lowers a learned value. The resulting profile is keyed
to the deployed command/configuration fingerprint and is persisted in the
SQLite `store.path`; configure a file-backed store to retain it across restarts.
`GET /api/residency` labels each reservation as `configured`, `calibrated`, or
`unprofiled`.

On non-Apple hosts, automatic calibration is unavailable in this release, so
auto-routing configurations must provide `memory` for every model. This keeps
NVIDIA and ROCm deployments compatible without pretending an aggregate device
measurement identifies a particular process.

## Soft keep-resident preference and startup preload

For an important model that should survive ordinary churn, set
`keepResident: true`. This is a soft preference for the `auto` router: it
evicts ordinary resident models first, then evicts a preferred model only when
that is necessary to admit another model that fits by itself.

```yaml
models:
  embedding-small:
    cmd: embedding-server --port ${PORT}
    memory: 4GiB
    keepResident: true
    ttl: 0

hooks:
  on_startup:
    preload: [embedding-small]
```

`ttl: 0` remains the only way to disable idle-time unloading. `keepResident`
does not introduce another keep-alive timer and does not guarantee residency:
the auto router logs a warning when it must evict a preferred model for
capacity. Startup preloads run one at a time in the listed order through the
same router and budget as normal requests. If the list overflows the budget,
later preloads can evict earlier ones; failures are logged and reported through
the existing preload event.

`GET /api/residency` lists `keep_resident`, `loaded`, and process `state` next
to each reservation. Treat these as a snapshot, not an allocation guarantee.

## Advanced control with `cmd`

llama-swap is also about customizability. You can use any CLI flag available:

```yaml
models:
  model1:
    cmd: | # support for multi-line
      llama-server --PORT ${PORT} -m /path/to/model.gguf
      --ctx-size 8192
      --jinja
      --cache-type-k q8_0
      --cache-type-v q8_0
```

## Support for any OpenAI API compatible server

llama-swap supports any OpenAI API compatible server. If you can run it on the CLI llama-swap will be able to manage it. Even if it's run in Docker or Podman containers.

```yaml
models:
  "Q3-30B-CODER-VLLM":
    name: "Qwen3 30B Coder vllm AWQ (Q3-30B-CODER-VLLM)"
    # cmdStop provides a reliable way to stop containers
    cmdStop: docker stop vllm-coder
    cmd: |
      docker run --init --rm --name vllm-coder
        --runtime=nvidia --gpus '"device=2,3"'
        --shm-size=16g
        -v /mnt/nvme/vllm-cache:/root/.cache
        -v /mnt/ssd-extra/models:/models -p ${PORT}:8000
        vllm/vllm-openai:v0.10.0
        --model "/models/cpatonn/Qwen3-Coder-30B-A3B-Instruct-AWQ"
        --served-model-name "Q3-30B-CODER-VLLM"
        --enable-expert-parallel
        --swap-space 16
        --max-num-seqs 512
        --max-model-len 65536
        --max-seq-len-to-capture 65536
        --gpu-memory-utilization 0.9
        --tensor-parallel-size 2
        --trust-remote-code
```

## Many more features..

llama-swap supports many more features to customize how you want to manage your environment.

| Feature   | Description                                    |
| --------- | ---------------------------------------------- |
| `ttl`     | automatic unloading of models after a timeout  |
| `macros`  | reusable snippets to use in configurations     |
| `matrix`  | run multiple models at a time                  |
| `hooks`   | event driven functionality                     |
| `env`     | define environment variables per model         |
| `aliases` | serve a model with different names             |
| `filters` | modify requests before sending to the upstream |
| `profiles` | switch model ID replacements at runtime       |
| `...`     | And many more tweaks                           |

## Full Configuration Example

Check [config.example.yaml](https://github.com/mostlygeek/llama-swap/blob/main/config.example.yaml) for the most up to date reference for all example configurations. It has grown quite complex but your favorite local LLM can help with a local configuration.
