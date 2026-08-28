//go:build !windows

package process

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
)

// TestProcess_KillAllForceTerminates covers the second-Ctrl-C path: the
// upstream ignores SIGTERM, so the graceful stop would sit out its whole
// timeout, but KillAll SIGKILLs the process group and the process is reaped
// immediately.
func TestProcess_KillAllForceTerminates(t *testing.T) {
	skipIfNoSimpleResponder(t)

	cmd, port := simpleResponderCmd(t, "-silent", "-ignore-sig-term")
	p := newProcessCommand(t, config.ModelConfig{
		Cmd:                cmd,
		Proxy:              fmt.Sprintf("http://127.0.0.1:%d", port),
		CheckEndpoint:      "/health",
		HealthCheckTimeout: 10,
	})
	t.Cleanup(func() { p.Stop(testStopTimeout) }) //nolint: errcheck

	runErr := runAsync(t, p)

	killed := KillAll(2 * time.Second)
	if killed < 1 {
		t.Fatalf("KillAll: expected at least 1 process tree signalled, got %d", killed)
	}

	select {
	case <-runErr:
	case <-time.After(5 * time.Second):
		t.Fatal("process still running 5s after KillAll")
	}

	waitForState(t, p, StateStopped)
}

// TestProcess_KillAllIgnoresReapedProcesses asserts the registry does not leak
// entries: once a process has exited, KillAll has nothing left to signal.
func TestProcess_KillAllIgnoresReapedProcesses(t *testing.T) {
	skipIfNoSimpleResponder(t)

	cmd, port := simpleResponderCmd(t, "-silent")
	p := newProcessCommand(t, config.ModelConfig{
		Cmd:                cmd,
		Proxy:              fmt.Sprintf("http://127.0.0.1:%d", port),
		CheckEndpoint:      "/health",
		HealthCheckTimeout: 10,
	})

	ctx, cancel := context.WithTimeout(context.Background(), testStartTimeout)
	defer cancel()
	if err := p.EnsureReady(ctx, testStartTimeout); err != nil {
		t.Fatalf("EnsureReady: %v", err)
	}
	if err := p.Stop(testStopTimeout); err != nil {
		t.Fatalf("Stop: %v", err)
	}

	if killed := KillAll(2 * time.Second); killed != 0 {
		t.Errorf("KillAll after Stop: expected 0 live process trees, got %d", killed)
	}
}
