package process

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// watchdogTrip records why the watchdog gave up on a request. The proxy's
// ErrorHandler needs it because by the time the failure surfaces, both trips
// and an ordinary client disconnect look identical: a cancelled context.
type watchdogTrip int32

const (
	tripNone watchdogTrip = iota

	// tripStalled: the upstream produced nothing for stallTimeout. A local
	// inference server that has gone completely silent is not thinking, it is
	// wedged, so the process is stopped along with the request.
	tripStalled

	// tripTooLong: the upstream is still producing, but has been at it for
	// longer than requestTimeout. That is a runaway request, not a broken
	// server — most often a model that has fallen into a repetition loop — so
	// only the request is ended.
	tripTooLong
)

// watchdogState is shared between the watchdog goroutine and the ErrorHandler.
type watchdogState struct {
	trip     atomic.Int32
	stall    time.Duration
	lifetime time.Duration
}

func (s *watchdogState) tripped() watchdogTrip { return watchdogTrip(s.trip.Load()) }

// message explains the trip to the caller whose request it ended.
func (s *watchdogState) message() string {
	switch s.tripped() {
	case tripStalled:
		return fmt.Sprintf("upstream produced no response for %v and was stopped as unresponsive", s.stall)
	case tripTooLong:
		return fmt.Sprintf("request exceeded the %v generation limit and was ended", s.lifetime)
	default:
		return "request ended by llama-swap"
	}
}

// watchdogKey carries the watchdogState of the request being watched.
type watchdogKey struct{}

// watchdogStateFrom returns the watchdog state attached to ctx, if any.
func watchdogStateFrom(ctx context.Context) (*watchdogState, bool) {
	state, ok := ctx.Value(watchdogKey{}).(*watchdogState)
	return state, ok
}

// The watchdog polls rather than arming an exact timer, so the interval bounds
// how late a trip can be. It scales with the shorter budget so a ten minute
// stall allowance is not checked every second, while a short budget (tests,
// aggressive operators) is still enforced promptly.
const (
	minWatchdogPollInterval = 250 * time.Millisecond
	maxWatchdogPollInterval = 15 * time.Second
)

func watchdogPollInterval(budgets ...time.Duration) time.Duration {
	shortest := time.Duration(0)
	for _, budget := range budgets {
		if budget > 0 && (shortest == 0 || budget < shortest) {
			shortest = budget
		}
	}
	interval := shortest / 10
	if interval < minWatchdogPollInterval {
		interval = minWatchdogPollInterval
	}
	if interval > maxWatchdogPollInterval {
		interval = maxWatchdogPollInterval
	}
	return interval
}

// withWatchdog wraps the reverse-proxy handler with the two request budgets.
//
// They catch opposite failures, which is why one setting cannot do both work.
// stallTimeout measures progress: it is reset by every byte the upstream sends,
// so a slow-but-alive generation streams for as long as it likes while a server
// that has stopped producing anything trips it. requestTimeout measures total
// elapsed time, and is the only thing that catches the opposite failure — a
// model looping on "the the the" forever resets the stall clock with every
// token and would otherwise run until the client gave up.
//
// Neither trip is a retry. The request is failed and the caller decides what to
// do next. A stall additionally stops the process, on the assumption it has
// crashed; the next request routed there starts it again the same way the
// router starts any stopped model.
func (p *ProcessCommand) withWatchdog(next http.HandlerFunc) http.HandlerFunc {
	stall := time.Duration(p.config.StallTimeout) * time.Second
	lifetime := time.Duration(p.config.RequestTimeout) * time.Second
	if stall <= 0 && lifetime <= 0 {
		return next
	}

	return func(w http.ResponseWriter, r *http.Request) {
		// Websockets are long-lived by design and carry no request/response
		// rhythm to measure, so neither budget says anything useful about one.
		if swaputil.IsWebSocketUpgrade(r) {
			next(w, r)
			return
		}

		state := &watchdogState{stall: stall, lifetime: lifetime}
		ctx, cancel := context.WithCancel(r.Context())
		defer cancel()
		r = r.WithContext(context.WithValue(ctx, watchdogKey{}, state))

		watched := &watchdogWriter{ResponseWriter: w}
		watched.touch()

		done := make(chan struct{})
		defer close(done)
		go p.watchRequest(state, watched, done, cancel)

		next(watched, r)
	}
}

// watchRequest polls a request's progress until it finishes or outlives one of
// its budgets.
func (p *ProcessCommand) watchRequest(state *watchdogState, watched *watchdogWriter, done <-chan struct{}, cancel context.CancelFunc) {
	started := time.Now()
	ticker := time.NewTicker(watchdogPollInterval(state.stall, state.lifetime))
	defer ticker.Stop()

	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			switch {
			case state.stall > 0 && time.Since(watched.lastProgress()) >= state.stall:
				// Order matters: mark first, then cancel. The ErrorHandler
				// reads the trip as soon as cancellation reaches it, and an
				// unmarked cancellation is indistinguishable from a client
				// that hung up.
				state.trip.Store(int32(tripStalled))
				cancel()
				p.proxyLogger.Warnf("<%s> no progress from upstream for %v, assuming it has crashed and stopping it", p.id, state.stall)
				if err := p.Stop(time.Duration(p.config.UnloadTimeout) * time.Second); err != nil {
					p.proxyLogger.Warnf("<%s> stopping stalled process failed: %v", p.id, err)
				}
				return
			case state.lifetime > 0 && time.Since(started) >= state.lifetime:
				state.trip.Store(int32(tripTooLong))
				cancel()
				// Left running deliberately: an upstream that is still
				// producing tokens is healthy, and stopping it would punish
				// every other caller for one runaway request.
				p.proxyLogger.Warnf("<%s> request ran for %v without finishing, ending it; the model is still running", p.id, state.lifetime)
				return
			}
		}
	}
}

// watchdogWriter records when the upstream last made progress towards the
// client. Every write, header, and flush counts; nothing else does.
type watchdogWriter struct {
	http.ResponseWriter
	last        atomic.Int64 // unix nano
	wroteHeader atomic.Bool
}

func (w *watchdogWriter) touch() { w.last.Store(time.Now().UnixNano()) }

func (w *watchdogWriter) lastProgress() time.Time { return time.Unix(0, w.last.Load()) }

func (w *watchdogWriter) WriteHeader(statusCode int) {
	if !w.wroteHeader.CompareAndSwap(false, true) {
		return
	}
	w.touch()
	w.ResponseWriter.WriteHeader(statusCode)
}

func (w *watchdogWriter) Write(data []byte) (int, error) {
	if !w.wroteHeader.Load() {
		w.WriteHeader(http.StatusOK)
	}
	w.touch()
	return w.ResponseWriter.Write(data)
}

func (w *watchdogWriter) Flush() {
	if !w.wroteHeader.Load() {
		w.WriteHeader(http.StatusOK)
	}
	w.touch()
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// MarkStatus forwards a recorded-only status to the wrapped writer.
func (w *watchdogWriter) MarkStatus(code int) {
	if marker, ok := w.ResponseWriter.(swaputil.StatusMarker); ok {
		marker.MarkStatus(code)
	}
}

// WroteHeader reports whether a response status reached the client.
func (w *watchdogWriter) WroteHeader() bool { return w.wroteHeader.Load() }

// Unwrap exposes the wrapped writer to http.ResponseController, so capabilities
// this wrapper does not implement itself (deadline control, for one) still
// reach the real writer underneath.
func (w *watchdogWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

func (w *watchdogWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	if hijacker, ok := w.ResponseWriter.(http.Hijacker); ok {
		return hijacker.Hijack()
	}
	return nil, nil, fmt.Errorf("underlying ResponseWriter does not support hijacking")
}

var (
	_ http.Flusher          = (*watchdogWriter)(nil)
	_ http.Hijacker         = (*watchdogWriter)(nil)
	_ swaputil.StatusMarker = (*watchdogWriter)(nil)
)
