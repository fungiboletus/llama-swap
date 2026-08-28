package process

import (
	"os/exec"
	"sync"
	"time"
)

// liveCmds tracks every upstream *exec.Cmd that has been started and not yet
// reaped. The normal teardown path goes through the routers, which stop each
// process through its own state machine; this registry exists purely for the
// emergency path (KillAll), where going through the routers is not an option
// because they may be the thing that is stuck.
var (
	liveCmdsMu sync.Mutex
	liveCmds   = make(map[*exec.Cmd]struct{})
)

func registerLiveCmd(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	liveCmdsMu.Lock()
	liveCmds[cmd] = struct{}{}
	liveCmdsMu.Unlock()
}

func unregisterLiveCmd(cmd *exec.Cmd) {
	if cmd == nil {
		return
	}
	liveCmdsMu.Lock()
	delete(liveCmds, cmd)
	liveCmdsMu.Unlock()
}

// KillAll force-terminates every upstream process tree that is still running,
// bypassing the graceful shutdown sequence entirely. It is meant for the
// operator's "I said stop" path — a second Ctrl-C while a graceful shutdown is
// already under way — where waiting out the remaining timeouts is exactly what
// the operator is trying to avoid.
//
// Each tree is signalled concurrently (on Windows killProcessTree shells out to
// taskkill, which is slow enough to matter when several models are loaded) and
// the whole call is bounded by timeout, so the caller can exit promptly even if
// one kill hangs. Returns the number of process trees that were signalled.
func KillAll(timeout time.Duration) int {
	liveCmdsMu.Lock()
	cmds := make([]*exec.Cmd, 0, len(liveCmds))
	for cmd := range liveCmds {
		cmds = append(cmds, cmd)
	}
	liveCmdsMu.Unlock()

	var wg sync.WaitGroup
	for _, cmd := range cmds {
		wg.Add(1)
		go func(cmd *exec.Cmd) {
			defer wg.Done()
			_ = killProcessTree(cmd)
		}(cmd)
	}

	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(timeout):
	}

	return len(cmds)
}
