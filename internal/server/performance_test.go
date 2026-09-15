package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/perf"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServer_PerformanceWindows(t *testing.T) {
	monitor, err := perf.New(config.PerformanceConfig{Every: 5 * time.Second}, logmon.NewWriter(io.Discard))
	require.NoError(t, err)
	s := &Server{perf: monitor}
	for _, tc := range []struct {
		query           string
		status, seconds int
	}{
		{"", 200, 0}, {"?window=5m", 200, 0}, {"?window=15m", 200, 0}, {"?window=1h", 200, 0},
		{"?window=24h", 200, 300}, {"?window=7d", 200, 1800},
		{"?window=8d", 400, 0}, {"?window=-1h", 400, 0},
		{"?after=invalid", 400, 0}, {"?window=7d&after=2026-09-15T00:00:00Z", 400, 0},
		{"?window=1h&after=2026-09-15T00:00:00Z", 200, 0},
	} {
		t.Run(tc.query, func(t *testing.T) {
			w := httptest.NewRecorder()
			s.handleAPIPerformance(w, httptest.NewRequest(http.MethodGet, "/api/performance"+tc.query, nil))
			require.Equal(t, tc.status, w.Code, w.Body.String())
			if tc.status != 200 {
				return
			}
			var body struct {
				Enabled bool           `json:"enabled"`
				Seconds int            `json:"aggregation_seconds"`
				Sys     []perf.SysStat `json:"sys_stats"`
				GPU     []perf.GpuStat `json:"gpu_stats"`
			}
			require.NoError(t, json.Unmarshal(w.Body.Bytes(), &body))
			assert.True(t, body.Enabled)
			assert.Equal(t, tc.seconds, body.Seconds)
			assert.Empty(t, body.Sys)
			assert.Empty(t, body.GPU)
		})
	}
}
