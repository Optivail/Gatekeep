package main

// Port of backend/core/cursor_agent.py: the duplex relay for Cursor's
// agent.v1.AgentService/Run, the upstream connection, warm_upstream, and the
// GetServerConfig forwarder of proxy.py (_forward_cursor_server_config).
//
// The relay runs one upstream HTTP/2 connection per Run, as the Python version
// does. The client's Connect body is framed, crushed (mild wirecrush only) and
// written to the upstream request body. The upstream's DATA is queued without
// limit and acknowledged as it arrives, so a slow client never holds the
// upstream back. The client side is read at the pace the upstream accepts, so
// a full upstream window slows the client down.
//
// Divergences from Python, all deliberate:
//   - A request body is streamed to the upstream as it arrives. Python first
//     reads the whole HTTP/1.1 body on the listener. The bytes sent are the same.
//   - On the main port, upstream trailers are dropped. Python hands them to the
//     Starlette stream, which raises and aborts the response.
//   - The upstream connection is torn down when the relay ends. Python's
//     relay waits for the client body to end first and can hang there.

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Optivail/Gatekeep/server/engines"
)

// defaultAgentUpstream is DEFAULT_AGENT_UPSTREAM.
const defaultAgentUpstream = "https://agentn.global.api5.cursor.sh"

// agentConnectTimeout bounds the TCP connect and the TLS handshake (15 s in Python).
const agentConnectTimeout = 15 * time.Second

// agentFeedGrace is how long the relay waits, after the response has ended, for
// the client body reader to stop. The handler must not touch the request body
// once it returns, so the wait keeps that from happening in the usual case. The
// wait is bounded, so a client that keeps its body open cannot hold a handler.
const agentFeedGrace = 2 * time.Second

// agentHeartbeatInterval is the quiet period after which an empty agent
// heartbeat is sent upstream while the client body is still open. It is a
// variable so that tests can shorten it.
var agentHeartbeatInterval = 5 * time.Second

// agentHopHeaders are the names the relay never sends upstream (_HOP).
var agentHopHeaders = map[string]bool{
	"host": true, "content-length": true, "connection": true,
	"transfer-encoding": true, "accept-encoding": true,
}

// agentHeartbeatFrame is a Connect frame whose payload is protobuf field 7 with
// no content: Cursor's agent heartbeat (_HEARTBEAT_FRAME).
var agentHeartbeatFrame = []byte{0x00, 0x00, 0x00, 0x00, 0x02, 0x3a, 0x00}

// connectEndStreamEnvelope is an empty Connect frame with the EndStream flag
// (_CONNECT_END_STREAM).
var connectEndStreamEnvelope = []byte{0x02, 0x00, 0x00, 0x00, 0x00}

// agentTLSBase is the TLS configuration of the agent upstream. Nil means the
// system roots, as ssl.create_default_context() uses. Tests replace it.
var agentTLSBase *tls.Config

// errAgentRelayDone is the error the upstream request body ends with once the
// caller has finished with the relay.
var errAgentRelayDone = errors.New("agent relay finished")

// agentProtocols selects HTTP/2 only, as the Python relay speaks.
func agentProtocols() *http.Protocols {
	var p http.Protocols
	p.SetHTTP2(true)
	return &p
}

// agentTLSConfig returns the client TLS configuration for host, with ALPN h2.
func agentTLSConfig(host string) *tls.Config {
	var cfg *tls.Config
	if agentTLSBase != nil {
		cfg = agentTLSBase.Clone()
	} else {
		cfg = &tls.Config{MinVersion: tls.VersionTLS12}
	}
	cfg.ServerName = host
	cfg.NextProtos = []string{"h2"}
	return cfg
}

// dialAgentTLS connects and completes the TLS handshake within the connect
// timeout. The returned connection is not tied to ctx after the handshake.
func dialAgentTLS(ctx context.Context, addr string, cfg *tls.Config) (net.Conn, error) {
	dctx, cancel := context.WithTimeout(ctx, agentConnectTimeout)
	defer cancel()
	d := &tls.Dialer{Config: cfg}
	return d.DialContext(dctx, "tcp", addr)
}

// agentEndpoint is urlparse(upstream): the lower-cased hostname, and the port
// (443 when none is given).
func agentEndpoint(upstream string) (host, port string) {
	u, err := url.Parse(upstream)
	if err != nil {
		return "", "443"
	}
	port = u.Port()
	if port == "" {
		port = "443"
	}
	return strings.ToLower(u.Hostname()), port
}

// resolveAgentUpstream is (upstream_base or "").strip() or DEFAULT_AGENT_UPSTREAM.
func resolveAgentUpstream(upstream string) string {
	if u := strings.TrimSpace(upstream); u != "" {
		return u
	}
	return defaultAgentUpstream
}

// agentUpstreamBase is base_url_for("cursor", path) for an agent.v1 path: the
// cached agent endpoint, or the default.
func agentUpstreamBase() string {
	if u := engines.GetAgentUpstream(); u != "" {
		return u
	}
	return defaultAgentUpstream
}

// baseURLForPath is base_url_for(provider, path). Only the cursor provider
// depends on the path.
func baseURLForPath(provider, path string) string {
	if provider == "cursor" && strings.HasPrefix(strings.ToLower(strings.TrimLeft(path, "/")), "agent.v1.") {
		return agentUpstreamBase()
	}
	return baseURLFor(provider)
}

// agentRequestURL builds the upstream request target for host and path. The
// path is sent as it was received, as the Python relay does; the query is kept.
func agentRequestURL(host, path string) *url.URL {
	p, q, _ := strings.Cut(strings.TrimLeft(path, "/"), "?")
	u := &url.URL{Scheme: "https", Host: host, RawQuery: q}
	unescaped, err := url.PathUnescape(p)
	if err != nil {
		unescaped = p
	}
	u.Path = "/" + unescaped
	u.RawPath = "/" + p
	return u
}

// agentUpstreamHeader builds the upstream request headers: every forwarded
// header except the hop-by-hop ones, te: trailers, and no User-Agent unless the
// client sent one (the Python relay sends no default).
func agentUpstreamHeader(h headerMap) http.Header {
	out := make(http.Header, len(h)+2)
	for name, v := range h {
		if agentHopHeaders[name] || strings.HasPrefix(name, ":") {
			continue
		}
		out.Set(name, v)
	}
	if _, ok := out["Te"]; !ok {
		out.Set("Te", "trailers")
	}
	if _, ok := out["User-Agent"]; !ok {
		// An empty User-Agent is omitted by the HTTP/2 encoder.
		out["User-Agent"] = []string{""}
	}
	return out
}

// agentItem is one entry of a response queue: a chunk of the upstream body,
// the upstream trailers, or the end of the stream. Python's bytes, the
// ("trailers", ...) tuple and None.
type agentItem struct {
	data     []byte
	trailers []agentHeader
	end      bool
}

// agentHeader is one trailer field.
type agentHeader struct{ name, value string }

// agentQueue is the unbounded queue between the upstream reader and the client
// writer. It is asyncio.Queue: it never blocks the producer.
type agentQueue struct {
	mu    sync.Mutex
	items []agentItem
	ready chan struct{}
}

func newAgentQueue() *agentQueue {
	return &agentQueue{ready: make(chan struct{}, 1)}
}

func (q *agentQueue) put(it agentItem) {
	q.mu.Lock()
	q.items = append(q.items, it)
	q.mu.Unlock()
	select {
	case q.ready <- struct{}{}:
	default:
	}
}

// tryGet returns the next item without blocking.
func (q *agentQueue) tryGet() (agentItem, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(q.items) == 0 {
		return agentItem{}, false
	}
	it := q.items[0]
	q.items[0] = agentItem{}
	q.items = q.items[1:]
	return it, true
}

// get blocks until an item is queued.
func (q *agentQueue) get() agentItem {
	for {
		if it, ok := q.tryGet(); ok {
			return it
		}
		<-q.ready
	}
}

// agentRelayArgs is the request a relay forwards.
type agentRelayArgs struct {
	upstream   string
	path       string
	headers    headerMap
	crush      bool
	aggressive bool
}

// agentStats are the byte counts of one Run, as _relay returns them.
type agentStats struct {
	orig, sent int
	proxyMS    int64
}

// agentRelayResult is what the relay goroutine hands back to its caller.
type agentRelayResult struct {
	stats agentStats
	err   error
}

// agentRelay holds the state of one Run while the client body is fed upstream.
type agentRelay struct {
	a        agentRelayArgs
	t0       time.Time
	pw       *io.PipeWriter
	seen     map[string]bool
	bodyDone chan struct{}
	feedEnd  chan struct{} // closed when feed returns

	wmu sync.Mutex // serializes writes to pw: frames and heartbeats

	smu   sync.Mutex
	stats agentStats
}

// runAgentRelay is _relay and _RelayConnection.relay. It opens one upstream
// connection, sends the client's Connect body upstream, and queues what the
// upstream sends onto sink. It returns once turnDone is closed, which the
// caller does after its own response loop ends. The upstream request is torn
// down then. An error means the upstream request could not be started.
func runAgentRelay(client io.Reader, sink *agentQueue, a agentRelayArgs, turnDone <-chan struct{}) (agentStats, error) {
	host, port := agentEndpoint(a.upstream)
	dialAddr := net.JoinHostPort(host, port)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	tr := &http.Transport{
		DisableCompression: true,
		Protocols:          agentProtocols(),
		DialTLSContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return dialAgentTLS(ctx, dialAddr, agentTLSConfig(host))
		},
	}
	defer tr.CloseIdleConnections()

	pr, pw := io.Pipe()
	rel := &agentRelay{a: a, t0: time.Now(), pw: pw, bodyDone: make(chan struct{}), feedEnd: make(chan struct{})}
	if a.aggressive {
		rel.seen = map[string]bool{}
	}
	stop := make(chan struct{})
	defer close(stop)
	go rel.feed(client)
	go rel.heartbeats(turnDone, stop)

	target := agentRequestURL(host, a.path)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://"+host+"/", pr)
	if err != nil {
		_ = pw.CloseWithError(err)
		return agentStats{}, err
	}
	// The URL is set as built, so the path and query are sent as received.
	req.URL = target
	req.Header = agentUpstreamHeader(a.headers)

	resp, err := tr.RoundTrip(req)
	if err != nil {
		_ = pw.CloseWithError(err)
		return agentStats{}, fmt.Errorf("agent upstream: %w", err)
	}
	go readAgentResponse(resp, sink)

	<-turnDone
	_ = pw.CloseWithError(errAgentRelayDone)
	select {
	case <-rel.feedEnd:
	case <-time.After(agentFeedGrace):
	}
	return rel.snapshot(), nil
}

// snapshot returns the counters at this moment.
func (rel *agentRelay) snapshot() agentStats {
	rel.smu.Lock()
	defer rel.smu.Unlock()
	return rel.stats
}

// feed reads the client body, frames it, crushes each frame and writes it
// upstream. At the end of the body it sends the END_STREAM. The tail of a
// partial frame is sent as it is, crushed when it is long enough.
func (rel *agentRelay) feed(client io.Reader) {
	defer close(rel.feedEnd)
	frames := engines.NewConnectFrameBuffer(false)
	buf := make([]byte, 64*1024)
	for {
		n, rerr := client.Read(buf)
		if n > 0 {
			rel.smu.Lock()
			rel.stats.orig += n
			rel.smu.Unlock()
			wire, err := frames.Feed(buf[:n])
			if err != nil {
				_ = rel.pw.CloseWithError(err)
				return
			}
			for _, frame := range wire {
				if !rel.sendFrame(frame) {
					return
				}
			}
		}
		if rerr != nil {
			break
		}
	}
	if tail := frames.TakeTail(); len(tail) > 0 {
		if !rel.sendFrame(tail) {
			return
		}
	}
	rel.smu.Lock()
	rel.stats.proxyMS = msSince(rel.t0)
	rel.smu.Unlock()
	close(rel.bodyDone)
	_ = rel.pw.Close()
}

// sendFrame crushes one frame and writes it upstream. It returns false when the
// upstream request has gone.
func (rel *agentRelay) sendFrame(frame []byte) bool {
	out := frame
	if rel.a.crush && len(frame) > 20 {
		crushed, saved, err := engines.CrushConnectBody(frame, rel.a.aggressive, rel.seen)
		if err != nil {
			_ = rel.pw.CloseWithError(err)
			return false
		}
		out = crushed
		if len(frame) > 2000 {
			pct := 0.0
			if len(frame) > 0 {
				pct = 100 * float64(saved) / float64(len(frame))
			}
			logger.Debug(fmt.Sprintf("[PROXY] agent crush frame %d->%d (%.0f%% saved)", len(frame), len(crushed), pct))
			if rel.a.aggressive && pct < 70 {
				engines.CaptureAgentFrame(frame, crushed, rel.a.aggressive)
			}
		}
	}
	rel.smu.Lock()
	rel.stats.sent += len(out)
	rel.smu.Unlock()
	return rel.write(out) == nil
}

// write sends bytes upstream. Frames and heartbeats are serialized, so a frame
// is never split by a heartbeat.
func (rel *agentRelay) write(b []byte) error {
	if len(b) == 0 {
		return nil
	}
	rel.wmu.Lock()
	defer rel.wmu.Unlock()
	_, err := rel.pw.Write(b)
	return err
}

// heartbeats sends an empty agent heartbeat every few seconds while the client
// body is still open. It stops when the body ends or the relay is done.
func (rel *agentRelay) heartbeats(turnDone, stop <-chan struct{}) {
	t := time.NewTicker(agentHeartbeatInterval)
	defer t.Stop()
	for {
		select {
		case <-rel.bodyDone:
			return
		case <-turnDone:
			return
		case <-stop:
			return
		case <-t.C:
		}
		select {
		case <-rel.bodyDone:
			return
		case <-turnDone:
			return
		default:
		}
		if rel.write(agentHeartbeatFrame) != nil {
			return
		}
	}
}

// readAgentResponse reads the upstream response body and queues it, in chunks
// as they arrive. Trailers come after the last chunk, then the end marker.
func readAgentResponse(resp *http.Response, sink *agentQueue) {
	defer resp.Body.Close()
	buf := make([]byte, 64*1024)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			sink.put(agentItem{data: append([]byte(nil), buf[:n]...)})
		}
		if err != nil {
			if err != io.EOF {
				logger.Warn(fmt.Sprintf("[PROXY] agent upstream read error: %v", err))
			}
			break
		}
	}
	if tr := sortedTrailers(resp.Trailer); len(tr) > 0 {
		sink.put(agentItem{trailers: tr})
	}
	sink.put(agentItem{end: true})
}

// sortedTrailers flattens the trailer map in name order. Go keeps no wire order,
// and HTTP/2 does not define one for trailers.
func sortedTrailers(h http.Header) []agentHeader {
	names := make([]string, 0, len(h))
	for name := range h {
		names = append(names, name)
	}
	sort.Strings(names)
	var out []agentHeader
	for _, name := range names {
		for _, v := range h[name] {
			out = append(out, agentHeader{name: name, value: v})
		}
	}
	return out
}

// agentUsage records one Run in request_logs, as the Python paths do with
// database.log_request(None, "cursor", ...).
func agentUsage(path string, orig, sent int, aggressive bool, latency int64) {
	saved := max(0, orig-sent)
	tag := "passthrough"
	if saved > 0 {
		tag = "wire_proto_crush"
		if aggressive {
			tag = "wire_proto_crush_aggressive"
		}
	}
	ot, st := max(1, orig/4), max(1, sent/4)
	sample := ""
	if saved > 0 {
		pct := 100 * float64(saved) / float64(orig)
		logger.Info(fmt.Sprintf("[OPTIMIZER] [cursor] agent %d -> %d | Saved: %d (%.1f%%) [%s]", ot, st, ot-st, pct, tag))
		sample = fmt.Sprintf("[agent %d->%dB]", orig, sent)
	} else {
		logger.Debug(fmt.Sprintf("[PROXY] agent passthrough /%s bytes=%d", path, orig))
	}
	if _, err := LogRequest(nil, "cursor", int64(ot), int64(st), 0, latency, tag, truncRunes(path, 200), sample, false); err != nil {
		logger.Error(fmt.Sprintf("[PROXY] /%s: usage log failed: %v", path, err))
	}
}

// forwardAgentBidi is forward_agent_bidi: the main port's Connect duplex for an
// agent.v1 POST. It answers 200 with the Connect content type at once, then
// streams the upstream to the client while the client body goes upstream.
func forwardAgentBidi(w http.ResponseWriter, r *http.Request, path string, headers headerMap, upstream string) {
	settings, err := GetSettings()
	if err != nil {
		logger.Error(fmt.Sprintf("[PROXY] /%s failed in get_settings: %v", path, err))
		writeServerError(w)
		return
	}
	crush := settingOn(settings, "compression_enabled")
	// Agent duplex: mild wirecrush only. The aggressive protobuf crush broke tool
	// frames in Python and is off here too.
	aggressive := false
	upstream = resolveAgentUpstream(upstream)
	logger.Info(fmt.Sprintf("[PROXY] agent.v1 /%s -> %s (connect duplex)", path, upstream))

	rc := http.NewResponseController(w)
	_ = rc.EnableFullDuplex()
	w.Header().Set("Content-Type", "application/connect+proto")
	w.WriteHeader(http.StatusOK)
	_ = rc.Flush()

	sink := newAgentQueue()
	turnDone := make(chan struct{})
	done := make(chan agentRelayResult, 1)
	args := agentRelayArgs{upstream: upstream, path: path, headers: headers, crush: crush, aggressive: aggressive}
	go func() {
		st, err := runAgentRelay(r.Body, sink, args, turnDone)
		if err != nil {
			logger.Warn(fmt.Sprintf("[PROXY] agent relay error /%s: %v", path, err))
			sink.put(agentItem{end: true})
		}
		done <- agentRelayResult{stats: st, err: err}
	}()

	clientGone := false
	for {
		it := sink.get()
		if it.end {
			break
		}
		if it.trailers != nil {
			// Python would hand these to Starlette and abort. They are dropped here.
			logger.Debug(fmt.Sprintf("[PROXY] agent upstream trailers dropped /%s", path))
			continue
		}
		if _, err := w.Write(it.data); err != nil {
			clientGone = true
			break
		}
		_ = rc.Flush()
	}
	close(turnDone)
	res := <-done
	if clientGone {
		logger.Debug(fmt.Sprintf("[PROXY] agent client went away /%s", path))
		return
	}
	if res.err != nil {
		// The 200 is already sent, as in Python where the stream raises. Aborting
		// the handler is the closest equivalent.
		panic(http.ErrAbortHandler)
	}
	agentUsage(path, res.stats.orig, res.stats.sent, aggressive, res.stats.proxyMS)
}

// warmState is the last host warmed up (_warm_host).
var warmState struct {
	mu   sync.Mutex
	host string
}

// warmUpstream opens and closes one TLS connection to the agent upstream at
// startup, so the first real Run does not pay the handshake. Errors are logged.
func warmUpstream(upstream string) {
	host, port := agentEndpoint(upstream)
	warmState.mu.Lock()
	warm := warmState.host == host
	warmState.mu.Unlock()
	if warm {
		return
	}
	conn, err := dialAgentTLS(context.Background(), net.JoinHostPort(host, port), agentTLSConfig(host))
	if err != nil {
		logger.Warn(fmt.Sprintf("[PROXY] agent upstream warm failed: %v", err))
		return
	}
	_ = conn.Close()
	warmState.mu.Lock()
	warmState.host = host
	warmState.mu.Unlock()
	logger.Debug(fmt.Sprintf("[PROXY] agent upstream h2 warm OK %s", host))
}

// forwardCursorServerConfig is _forward_cursor_server_config: it fetches a
// GetServerConfig response and rewrites the agent endpoints before the client
// caches them. Nothing is logged to request_logs, as in Python.
func (c *proxyCall) forwardCursorServerConfig(base string) {
	start := time.Now()
	agentOrigin := agentPublicURL()
	target := upstreamURL(base, c.path, c.r.URL.RawQuery)

	req, err := http.NewRequestWithContext(c.r.Context(), c.r.Method, target, bytes.NewReader(c.body))
	if err != nil {
		c.fail("build_request", err)
		return
	}
	req.Header = c.upstreamHeader()
	resp, err := upstreamClient.Do(req)
	if err != nil {
		c.fail("upstream_request", err)
		return
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		c.fail("upstream_read", err)
		return
	}

	newRaw, rewrites, err := engines.MaybeRewriteServerConfig(c.path, raw, agentOrigin)
	if err != nil {
		c.fail("server_config_rewrite", err)
		return
	}
	if len(rewrites) > 0 {
		logger.Debug(fmt.Sprintf("[PROXY] GetServerConfig rewrite: %s", strings.Join(rewrites, ", ")))
	} else {
		logger.Debug(fmt.Sprintf("[PROXY] GetServerConfig rewrite missed /%s (no agent URLs in body)", c.path))
	}
	logger.Debug(fmt.Sprintf("[PROXY] GetServerConfig done in %dms (%dB -> %dB)",
		msSince(start), len(raw), len(newRaw)))

	headers := responseHeaderMap(resp.Header)
	for _, k := range sortedKeys(headers) {
		c.w.Header().Set(k, headers[k])
	}
	if _, ok := headers["content-type"]; !ok {
		// Starlette adds no Content-Type when the upstream sent none. A nil entry
		// stops net/http from sniffing one.
		c.w.Header()["Content-Type"] = nil
	}
	c.w.WriteHeader(resp.StatusCode)
	_, _ = c.w.Write(newRaw)
}
