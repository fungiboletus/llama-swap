# Performance history

The Performance page offers five rolling windows:

| Window | Resolution | Maximum points per system/device |
| --- | --- | --- |
| 5 minutes, 15 minutes, 1 hour | Configured `performance.every` sampling interval | One hour of raw samples |
| 1 day | 5-minute sample averages | 289 |
| 7 days | 30-minute sample averages | 337 |

CPU, memory, swap, load, and GPU gauges use arithmetic means of the samples
available in each bucket. Integer measurements are rounded down to their
existing integer units. Missing samples do not count as zero usage. The current
unfinished bucket is included and changes as new samples arrive.

Network counters retain the last reading and its timestamp in each bucket.
The charts calculate average bandwidth between those readings and suppress
negative deltas when an interface counter resets.

History is bounded in memory. It starts when llama-swap starts and resets when
performance configuration changes. It is not stored in the activity SQLite
database, and restarting does not recover earlier performance samples. A fresh
server therefore needs seven days to fill a week view. Prometheus scraping is
available through `/metrics` for external persistent telemetry storage.

## HTTP API

`GET /api/performance?window=5m|15m|1h|24h|7d` returns `enabled`, `sys_stats`,
`gpu_stats`, and `aggregation_seconds`. The last field is `0` for raw samples,
`300` for a day, and `1800` for a week. Windows are measured backwards from
server time. Unknown windows return HTTP 400; disabled monitoring returns 503.

Without `window`, the existing one-hour raw buffer response is preserved.
`after=<RFC3339 timestamp>` remains available for raw windows. Day and week
windows reject `after`: refresh their complete bounded snapshot, replacing the
previous response so unfinished buckets are updated without duplicates.
