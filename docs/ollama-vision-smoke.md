# Manual Ollama vision smoke test

This opt-in check requires a configured model that advertises `vision` through
`POST /api/show`, a suitable local image, and hardware capable of serving the
model. It does not run in `make test-all`.

Start llama-swap, point the CLI to it, then pass an image path beginning with
`./` or `/`:

```bash
export OLLAMA_HOST=http://127.0.0.1:8080
ollama run configured-vision-model 'Describe this image in three words: ./image.jpg'
```

Before trying inference, confirm the model advertises vision:

```bash
curl -sS http://127.0.0.1:8080/api/show \
  -H 'Content-Type: application/json' \
  -d '{"model":"configured-vision-model"}'
```

The response must contain `"vision"` in `capabilities`. If it does not, add
`image` to `models.<id>.capabilities.in` for an ordinary local server, or check
the source model's own Ollama `/api/show` capability before restarting
llama-swap for a managed lane.

On failure, inspect llama-swap's proxy logs. Invalid base64, an unsupported
format, more than eight images, an image above 32 MiB, or a request above
128 MiB is rejected before the model starts. A valid request that fails later
is an upstream model/backend error rather than an image-validation failure.
