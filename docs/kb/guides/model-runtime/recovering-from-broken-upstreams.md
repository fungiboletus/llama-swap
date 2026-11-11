---
title: Recovering from hung or broken upstream servers
summary: How stallTimeout, requestTimeout and unhealthyResponses end a bad request and restart a model whose server is broken but still running.
category: guides
tags: [stall, timeout, crash, restart, mlx-lm, hung, unhealthy, watchdog]
config_keys: [stallTimeout, requestTimeout, models.*.stallTimeout, models.*.requestTimeout, models.*.unhealthyResponses]
updated: 2026-11-11
---

# Recovering from hung or broken upstream servers

Some inference servers fail without exiting. llama-swap never retries a failed
request; it ends the request and, where the server is presumed broken, stops
the process. The next request starts a fresh one.

| Failure | Setting | What happens |
| --- | --- | --- |
| The server goes silent | `stallTimeout` (seconds) | Request fails, process is stopped |
| The server generates forever | `requestTimeout` (seconds) | Request fails, process keeps running |
| The server answers with a known "I am broken" error | `models.*.unhealthyResponses` | Error is returned as is, process is stopped |

## Stopping a server that keeps answering with an error

mlx-lm can lose its generation thread and then answer every request with
`404 {"error": "generation thread died"}` until it is restarted. List that
response so llama-swap restarts it:

```yaml
models:
  qwen-mlx:
    cmd: mlx_lm.server --model ... --port ${PORT}
    unhealthyResponses:
      - status: 404
        bodyContains: generation thread died
```

`status` must be 400-599. `bodyContains` is optional and is matched against the
first 4 KB of the body; leave it out to match any body with that status.

The caller whose request hit the error still receives the upstream's response
unchanged, and that request is not retried. Only requests after it benefit from
the restart.

## What goes wrong

- A rule that is too broad (a bare `status: 404` on a server that legitimately
  returns 404 for unknown routes) restarts the model on harmless errors. Match
  on `bodyContains` where you can.
- Successful responses are never inspected, so a server that fails with `200`
  and an error inside a stream is not caught by this setting.
