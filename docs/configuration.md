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
The usable budget is `capacity - reserve`; every configured model must have a
positive reservation that fits within it. Group and matrix routers do not need
memory fields.

Choose a reservation by running one model at a time with its production
command and context settings, then use a conservative peak from its startup
and runtime logs. Include model weights, KV cache, and backend allocations.
Leave host, driver, and unrelated-process headroom in `reserve`; do not add
separate CPU and shared/unified-GPU capacities together.

`GET /api/residency` exposes the configured reservations and budget. They are
planning inputs, not measured process usage; use `/api/performance` separately
for aggregate host telemetry.

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
