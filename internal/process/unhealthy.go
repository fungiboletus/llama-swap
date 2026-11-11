package process

import (
	"bytes"
	"io"
	"net/http"
	"sync"
	"time"
)

// unhealthyBodyPeek bounds how much of an error response is read to look for a
// configured bodyContains match.
const unhealthyBodyPeek = 4096

// checkUnhealthyResponse stops the process when an upstream error response
// matches one of the model's unhealthyResponses rules. The response itself is
// passed to the caller unchanged and the request is not retried; the process is
// stopped once that response has been delivered, so the next request starts a
// fresh one. Successful responses are never inspected.
func (p *ProcessCommand) checkUnhealthyResponse(resp *http.Response) {
	rules := p.config.UnhealthyResponses
	if len(rules) == 0 || resp.StatusCode < 400 {
		return
	}

	var matched bool
	var peeked []byte
	for _, rule := range rules {
		if rule.Status != resp.StatusCode {
			continue
		}
		if rule.BodyContains != "" {
			if peeked == nil {
				peeked = peekBody(resp)
			}
			if !bytes.Contains(peeked, []byte(rule.BodyContains)) {
				continue
			}
		}
		matched = true
		break
	}
	if !matched {
		return
	}

	p.proxyLogger.Warnf("<%s> upstream answered %d with a response marked unhealthy, stopping it so the next request starts fresh", p.id, resp.StatusCode)
	var once sync.Once
	resp.Body = &onCloseReader{ReadCloser: resp.Body, onClose: func() {
		once.Do(func() {
			go func() {
				if err := p.Stop(time.Duration(p.config.UnloadTimeout) * time.Second); err != nil {
					p.proxyLogger.Warnf("<%s> stopping unhealthy process failed: %v", p.id, err)
				}
			}()
		})
	}}
}

// peekBody reads up to unhealthyBodyPeek bytes of the response body and puts
// them back, so the caller still receives the whole body.
func peekBody(resp *http.Response) []byte {
	head, _ := io.ReadAll(io.LimitReader(resp.Body, unhealthyBodyPeek))
	resp.Body = &readCloser{Reader: io.MultiReader(bytes.NewReader(head), resp.Body), Closer: resp.Body}
	return head
}

type readCloser struct {
	io.Reader
	io.Closer
}

// onCloseReader runs onClose when the body is closed, which the reverse proxy
// does after copying the response to the client.
type onCloseReader struct {
	io.ReadCloser
	onClose func()
}

func (r *onCloseReader) Close() error {
	err := r.ReadCloser.Close()
	r.onClose()
	return err
}
