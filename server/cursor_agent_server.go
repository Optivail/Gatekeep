package main

// Port of backend/core/cursor_agent_server.py: the Cursor agent listener on
// GATEKEEP_AGENT_HOST:GATEKEEP_AGENT_PORT. It serves HTTP/1.1 and HTTP/2 with
// prior knowledge (h2c), and it relays agent.v1.AgentService/Run in both
// directions. The response is re-framed as Connect frames. Gzip-flagged frames
// are inflated, and an EndStream envelope is sent when the upstream sent none.
//
// Divergences from Python, all deliberate:
//   - A non-Run request on HTTP/2 gets a 404, not a RST_STREAM. The 404 is what
//     the spec asks for, and the client sees the same refusal either way.
//   - The probe body is exactly the 14 bytes of its Content-Length. Python sends
//     two more bytes after the body, which a keep-alive client would read as the
//     start of a new response.
//   - HTTP/1.1 responses carry a status line, which the Python relay never sent
//     (its send_response_start is never called). Upstream trailers are sent.
//     Python drops them and leaves out the terminating chunk.
//   - A HEAD request gets no body. Python writes the body for HEAD too.

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Optivail/Gatekeep/server/engines"
)

const (
	// agentRunSuffix is the one path the listener relays.
	agentRunSuffix = "agent.v1.AgentService/Run"
	// agentProbeText is the GET / answer. The listener's Python code writes it
	// with a capital G; agent_app.py uses lower case and is not run by main.py.
	agentProbeText = "Gatekeep-agent"
	// agentIdleTimeout closes an idle connection (120 s in Python's h2 loop).
	agentIdleTimeout = 120 * time.Second
	// agentReadHeaderTimeout bounds the request head.
	agentReadHeaderTimeout = 30 * time.Second
)

// errClientGone ends the response loop when the client stops reading.
var errClientGone = errors.New("client went away")

// newAgentServer returns the agent listener's HTTP server, which serves
// HTTP/1.1 and h2c. It is not serving yet.
func newAgentServer() *http.Server {
	var protos http.Protocols
	protos.SetHTTP1(true)
	protos.SetUnencryptedHTTP2(true)
	return &http.Server{
		Handler:           http.HandlerFunc(serveAgent),
		Protocols:         &protos,
		ReadHeaderTimeout: agentReadHeaderTimeout,
		IdleTimeout:       agentIdleTimeout,
	}
}

// startAgentListener binds the agent port and serves it in the background.
// The caller shuts the returned server down.
func startAgentListener(host string, port int) (*http.Server, error) {
	ln, err := net.Listen("tcp", net.JoinHostPort(host, strconv.Itoa(port)))
	if err != nil {
		return nil, err
	}
	srv := newAgentServer()
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error(fmt.Sprintf("[PROXY] agent listener stopped: %v", err))
		}
	}()
	logger.Info(fmt.Sprintf("[PROXY] Cursor agent listener on %s:%d (h1+h2)", host, port))
	return srv, nil
}

// agentRunTarget reports whether a request is an agent Run. HTTP/2 matches the
// path anywhere in the target, and any method is accepted, as the Python h2
// handler does. HTTP/1.1 needs a POST whose path ends with the Run path.
func agentRunTarget(method, target string, h2 bool) bool {
	if h2 {
		return strings.Contains(target, agentRunSuffix)
	}
	return method == http.MethodPost && strings.HasSuffix(strings.TrimRight(target, "/"), "/"+agentRunSuffix)
}

// serveAgent routes one request on the agent listener.
func serveAgent(w http.ResponseWriter, r *http.Request) {
	h2 := r.ProtoMajor == 2
	target := r.RequestURI
	if agentRunTarget(r.Method, target, h2) {
		serveAgentRun(w, r, strings.TrimLeft(target, "/"), h2)
		return
	}
	if !h2 && (r.Method == http.MethodGet || r.Method == http.MethodHead) && strings.TrimRight(target, "/") == "" {
		h := w.Header()
		h["Content-Type"] = nil
		h.Set("Content-Length", strconv.Itoa(len(agentProbeText)))
		w.WriteHeader(http.StatusOK)
		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte(agentProbeText))
		}
		return
	}
	if !h2 {
		logger.Warn(fmt.Sprintf("[PROXY] agent h1 reject %s %s", r.Method, truncRunes(target, 120)))
	}
	w.WriteHeader(http.StatusNotFound)
}

// serveAgentRun is _run_relay for one Run. The 200 with the Connect content type
// goes out first, as in Python, so the client can start its body before the
// upstream answers.
func serveAgentRun(w http.ResponseWriter, r *http.Request, path string, h2 bool) {
	rc := http.NewResponseController(w)
	_ = rc.EnableFullDuplex()
	w.Header().Set("Content-Type", "application/connect+proto")
	w.WriteHeader(http.StatusOK)
	_ = rc.Flush()
	logger.Debug(fmt.Sprintf("[PROXY] agent inbound h2=%v POST /%s", h2, path))

	settings, err := GetSettings()
	if err != nil {
		logger.Warn(fmt.Sprintf("[PROXY] agent relay error: get_settings: %v", err))
		return
	}
	crush := settingOn(settings, "compression_enabled")
	aggressive := crush && settingOn(settings, "aggressive_enabled")
	upstream := baseURLForPath("cursor", path)
	logger.Info(fmt.Sprintf("[PROXY] agent.v1 /%s -> %s (duplex relay crush=%v aggr=%v)", path, upstream, crush, aggressive))

	sink := newAgentQueue()
	turnDone := make(chan struct{})
	done := make(chan agentRelayResult, 1)
	args := agentRelayArgs{upstream: upstream, path: path, headers: requestHeaderMap(r), crush: crush, aggressive: aggressive}
	go func() {
		st, err := runAgentRelay(r.Body, sink, args, turnDone)
		if err != nil {
			logger.Warn(fmt.Sprintf("[PROXY] agent relay error: %v", err))
			sink.put(agentItem{end: true})
		}
		done <- agentRelayResult{stats: st, err: err}
	}()

	sess := &agentRespSession{w: w, rc: rc, framer: engines.NewConnectFrameBuffer(true)}
	loopErr := sess.run(sink)
	if loopErr != nil && !errors.Is(loopErr, errClientGone) {
		logger.Warn(fmt.Sprintf("[PROXY] agent response stream error: %v", loopErr))
	}
	if !sess.clientGone {
		sess.flushPending()
		sess.halfClose()
	}

	close(turnDone)
	res := <-done
	if sess.clientGone {
		return
	}
	if res.err != nil {
		logger.Warn(fmt.Sprintf("[PROXY] agent relay error: %v", res.err))
		return
	}
	agentUsage(path, res.stats.orig, res.stats.sent, aggressive, res.stats.proxyMS)
}

// agentRespSession is the client side of one Run: it re-frames the upstream
// bytes and writes them to the client as they arrive.
type agentRespSession struct {
	w          http.ResponseWriter
	rc         *http.ResponseController
	framer     *engines.ConnectFrameBuffer
	respBytes  int
	sawEnd     bool
	pending    []agentHeader
	clientGone bool
}

// write sends bytes to the client and flushes them. A failed write marks the
// client as gone.
func (s *agentRespSession) write(b []byte) error {
	if _, err := s.w.Write(b); err != nil {
		s.clientGone = true
		return errClientGone
	}
	_ = s.rc.Flush()
	return nil
}

// emit sends one complete Connect frame (_emit_to_client).
func (s *agentRespSession) emit(wire []byte) error {
	if engines.DebugConnectChunk(wire, "Upstream->Client") {
		s.sawEnd = true
	}
	s.respBytes += len(wire)
	return s.write(wire)
}

// feed re-frames one chunk of upstream bytes and emits every complete frame.
func (s *agentRespSession) feed(data []byte) error {
	frames, err := s.framer.Feed(data)
	if err != nil {
		return err
	}
	for _, wire := range frames {
		if err := s.emit(wire); err != nil {
			return err
		}
	}
	return nil
}

// run drains the queue until the upstream ends (the end marker), then drains
// anything queued after it. Trailers are kept for halfClose. The error is
// errClientGone, a re-framing error, or nil.
func (s *agentRespSession) run(sink *agentQueue) error {
	for {
		it := sink.get()
		if it.end {
			break
		}
		if it.trailers != nil {
			s.pending = it.trailers
			continue
		}
		if err := s.feed(it.data); err != nil {
			return err
		}
	}
	for {
		it, ok := sink.tryGet()
		if !ok {
			break
		}
		if it.end {
			continue
		}
		if it.trailers != nil {
			s.pending = it.trailers
			continue
		}
		if err := s.feed(it.data); err != nil {
			return err
		}
	}
	return nil
}

// flushPending sends the frames the framer still holds, then the raw tail of an
// incomplete frame (_flush_resp_to_client). A failure is logged and the stream
// ends.
func (s *agentRespSession) flushPending() {
	frames, err := s.framer.FlushComplete()
	if err != nil {
		logger.Warn(fmt.Sprintf("[PROXY] agent resp flush error: %v", err))
		return
	}
	for _, wire := range frames {
		if err := s.emit(wire); err != nil {
			return
		}
	}
	if tail := s.framer.TakeTail(); len(tail) > 0 {
		logger.Debug(fmt.Sprintf("[PROXY] agent resp reassembly tail %dB", len(tail)))
		_ = s.write(tail)
	}
}

// halfClose ends the client stream (_half_close_client). The upstream's
// trailers are sent when it gave some. Otherwise an EndStream envelope is added
// if the upstream sent none, and the handler return ends the stream.
func (s *agentRespSession) halfClose() {
	if s.pending != nil {
		for _, h := range s.pending {
			if strings.HasPrefix(h.name, ":") {
				continue
			}
			key := http.TrailerPrefix + http.CanonicalHeaderKey(h.name)
			s.w.Header().Add(key, h.value)
		}
		return
	}
	if !s.sawEnd {
		logger.Debug("[PROXY] agent injecting Connect EndStream envelope (0x02)")
		_ = s.write(connectEndStreamEnvelope)
		s.sawEnd = true
	}
}
