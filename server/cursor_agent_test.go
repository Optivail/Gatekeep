package main

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/binary"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/Optivail/Gatekeep/server/engines"
)

// connectEnvelope builds one Connect frame.
func connectEnvelope(flags byte, payload []byte) []byte {
	out := make([]byte, 5, 5+len(payload))
	out[0] = flags
	binary.BigEndian.PutUint32(out[1:5], uint32(len(payload)))
	return append(out, payload...)
}

// protoString encodes a string field (wire type 2).
func protoString(fn byte, s string) []byte {
	return append([]byte{fn<<3 | 2, byte(len(s))}, s...)
}

// bodyRecorder keeps the bytes the upstream received, across requests.
type bodyRecorder struct {
	mu    sync.Mutex
	body  bytes.Buffer
	paths []string
}

func (b *bodyRecorder) add(p []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.body.Write(p)
}

func (b *bodyRecorder) addPath(p string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.paths = append(b.paths, p)
}

func (b *bodyRecorder) bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]byte(nil), b.body.Bytes()...)
}

func (b *bodyRecorder) lastPath() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.paths) == 0 {
		return ""
	}
	return b.paths[len(b.paths)-1]
}

// newAgentUpstream starts a TLS server that speaks HTTP/2, as the agent
// upstream does. The relay is told to trust it, and the agent endpoint is
// pointed at it. Everything is restored when the test ends.
func newAgentUpstream(t *testing.T, h http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewUnstartedServer(h)
	srv.EnableHTTP2 = true
	srv.StartTLS()
	t.Cleanup(srv.Close)

	pool := x509.NewCertPool()
	pool.AddCert(srv.Certificate())
	prevTLS := agentTLSBase
	agentTLSBase = &tls.Config{RootCAs: pool}
	t.Cleanup(func() { agentTLSBase = prevTLS })

	engines.RememberAgentUpstream(srv.URL)
	t.Cleanup(engines.ForgetAgentUpstream)
	return srv
}

// echoRun records what the client sends and echoes it back once the request
// body has ended. It is the fake agent: a well-behaved Run.
func echoRun(rec *bodyRecorder) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rec.addPath(r.URL.RequestURI())
		var got bytes.Buffer
		buf := make([]byte, 4096)
		for {
			n, err := r.Body.Read(buf)
			if n > 0 {
				got.Write(buf[:n])
				rec.add(buf[:n])
			}
			if err != nil {
				break
			}
		}
		w.Header().Set("Content-Type", "application/connect+proto")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(got.Bytes())
	}
}

// startAgentTestListener serves the agent listener on a free loopback port and
// returns its host:port.
func startAgentTestListener(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	srv := newAgentServer()
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(func() { _ = srv.Close() })
	return ln.Addr().String()
}

// h2cClient speaks HTTP/2 with prior knowledge over cleartext TCP.
func h2cClient() *http.Client {
	var protos http.Protocols
	protos.SetUnencryptedHTTP2(true)
	return &http.Client{Transport: &http.Transport{Protocols: &protos}}
}

// withCompression sets the compression toggle for one test, so that the relay
// passes frames through unchanged.
func withCompression(t *testing.T, on int64) {
	t.Helper()
	if err := SetSetting("compression_enabled", on); err != nil {
		t.Fatalf("SetSetting: %v", err)
	}
}

// eventually polls cond until it holds or the deadline passes.
func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestAgentListenerProbeAndRejects(t *testing.T) {
	addr := startAgentTestListener(t)
	base := "http://" + addr

	resp, err := http.Get(base + "/")
	if err != nil {
		t.Fatalf("probe: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || string(body) != "Gatekeep-agent" {
		t.Fatalf("probe = %d %q, want 200 Gatekeep-agent", resp.StatusCode, body)
	}

	cases := []struct {
		method, target string
	}{
		{http.MethodPost, "/foo"},
		{http.MethodPost, "/agent.v1.AgentService/Other"},
		{http.MethodGet, "/agent.v1.AgentService/Run"},
		{http.MethodGet, "/foo"},
	}
	for _, tc := range cases {
		req, _ := http.NewRequest(tc.method, base+tc.target, nil)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("%s %s: %v", tc.method, tc.target, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusNotFound {
			t.Fatalf("%s %s = %d, want 404", tc.method, tc.target, resp.StatusCode)
		}
	}
}

func TestAgentListenerRunOverH2C(t *testing.T) {
	setupTestDB(t)
	withCompression(t, 0)
	rec := &bodyRecorder{}
	newAgentUpstream(t, echoRun(rec))
	addr := startAgentTestListener(t)

	in := append(connectEnvelope(0, []byte(`{"hello":1}`)), connectEnvelope(2, []byte("{}"))...)
	req, _ := http.NewRequest(http.MethodPost, "http://"+addr+"/agent.v1.AgentService/Run", bytes.NewReader(in))
	resp, err := h2cClient().Do(req)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "application/connect+proto" {
		t.Fatalf("status %d type %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	if !bytes.Equal(got, in) {
		t.Fatalf("client got %x, want %x", got, in)
	}
	if !bytes.Equal(rec.bytes(), in) {
		t.Fatalf("upstream got %x, want %x", rec.bytes(), in)
	}
	if p := rec.lastPath(); p != "/agent.v1.AgentService/Run" {
		t.Fatalf("upstream path %q", p)
	}
	eventually(t, "the usage row", func() bool {
		provider, _, _, _ := lastLog(t)
		return provider == "cursor"
	})
}

func TestAgentListenerRunOverH1(t *testing.T) {
	setupTestDB(t)
	withCompression(t, 0)
	rec := &bodyRecorder{}
	newAgentUpstream(t, echoRun(rec))
	addr := startAgentTestListener(t)

	in := append(connectEnvelope(0, []byte("h1 body")), connectEnvelope(2, []byte("{}"))...)
	pr, pw := io.Pipe()
	go func() {
		_, _ = pw.Write(in[:3])
		time.Sleep(20 * time.Millisecond)
		_, _ = pw.Write(in[3:])
		_ = pw.Close()
	}()
	req, _ := http.NewRequest(http.MethodPost, "http://"+addr+"/agent.v1.AgentService/Run", pr)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("run over h1: %v", err)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !bytes.Equal(got, in) {
		t.Fatalf("h1 status %d body %x, want %x", resp.StatusCode, got, in)
	}
}

func TestAgentListenerInjectsEndStream(t *testing.T) {
	setupTestDB(t)
	withCompression(t, 0)
	newAgentUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/connect+proto")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(connectEnvelope(0, []byte("data")))
	})
	addr := startAgentTestListener(t)

	req, _ := http.NewRequest(http.MethodPost, "http://"+addr+"/agent.v1.AgentService/Run", bytes.NewReader(nil))
	resp, err := h2cClient().Do(req)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	want := append(connectEnvelope(0, []byte("data")), 0x02, 0, 0, 0, 0)
	if !bytes.Equal(got, want) {
		t.Fatalf("client got %x, want %x (EndStream envelope added)", got, want)
	}
}

func TestAgentListenerForwardsTrailers(t *testing.T) {
	setupTestDB(t)
	withCompression(t, 0)
	newAgentUpstream(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/connect+proto")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(connectEnvelope(0, []byte("x")))
		w.Header().Set(http.TrailerPrefix+"Grpc-Status", "0")
	})
	addr := startAgentTestListener(t)

	req, _ := http.NewRequest(http.MethodPost, "http://"+addr+"/agent.v1.AgentService/Run", bytes.NewReader(nil))
	resp, err := h2cClient().Do(req)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !bytes.Equal(got, connectEnvelope(0, []byte("x"))) {
		t.Fatalf("body %x: an EndStream envelope must not be added when trailers are sent", got)
	}
	if v := resp.Trailer.Get("Grpc-Status"); v != "0" {
		t.Fatalf("trailer Grpc-Status = %q, want 0", v)
	}
}

func TestAgentListenerSendsHeartbeatWhileBodyOpen(t *testing.T) {
	setupTestDB(t)
	withCompression(t, 0)
	prev := agentHeartbeatInterval
	agentHeartbeatInterval = 10 * time.Millisecond
	t.Cleanup(func() { agentHeartbeatInterval = prev })

	rec := &bodyRecorder{}
	newAgentUpstream(t, echoRun(rec))
	addr := startAgentTestListener(t)

	pr, pw := io.Pipe()
	go func() {
		_, _ = pw.Write(connectEnvelope(0, []byte("first")))
		time.Sleep(150 * time.Millisecond)
		_, _ = pw.Write(connectEnvelope(2, []byte("{}")))
		_ = pw.Close()
	}()
	req, _ := http.NewRequest(http.MethodPost, "http://"+addr+"/agent.v1.AgentService/Run", pr)
	resp, err := h2cClient().Do(req)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if !bytes.Contains(rec.bytes(), agentHeartbeatFrame) {
		t.Fatalf("upstream got no heartbeat while the body was open: %x", rec.bytes())
	}
}

func TestAgentListenerUpstreamDownSendsEndStream(t *testing.T) {
	setupTestDB(t)
	withCompression(t, 0)
	// A TLS endpoint that refuses connections: the relay cannot start.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	closed := "https://" + ln.Addr().String()
	_ = ln.Close()
	engines.RememberAgentUpstream(closed)
	t.Cleanup(engines.ForgetAgentUpstream)
	addr := startAgentTestListener(t)

	req, _ := http.NewRequest(http.MethodPost, "http://"+addr+"/agent.v1.AgentService/Run", bytes.NewReader(nil))
	resp, err := h2cClient().Do(req)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || !bytes.Equal(got, []byte{0x02, 0, 0, 0, 0}) {
		t.Fatalf("status %d body %x, want 200 with the EndStream envelope", resp.StatusCode, got)
	}
}

func TestAgentRunMainPortForwardsAndLogs(t *testing.T) {
	setupTestDB(t)
	withCompression(t, 0)
	rec := &bodyRecorder{}
	newAgentUpstream(t, echoRun(rec))
	srv := httptest.NewServer(newHandler())
	t.Cleanup(srv.Close)

	// A non-Run agent.v1 method on the main port is forwarded as well.
	in := connectEnvelope(0, []byte("main port"))
	resp, err := http.Post(srv.URL+"/agent.v1.AgentService/Other", "application/connect+proto", bytes.NewReader(in))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK || resp.Header.Get("Content-Type") != "application/connect+proto" {
		t.Fatalf("status %d type %q", resp.StatusCode, resp.Header.Get("Content-Type"))
	}
	if !bytes.Equal(got, in) || !bytes.Equal(rec.bytes(), in) {
		t.Fatalf("client %x, upstream %x, want %x", got, rec.bytes(), in)
	}
	if p := rec.lastPath(); p != "/agent.v1.AgentService/Other" {
		t.Fatalf("upstream path %q", p)
	}
	provider, category, _, _ := lastLog(t)
	if provider != "cursor" || category != "passthrough" {
		t.Fatalf("usage row provider=%q category=%q", provider, category)
	}
}

func TestAgentRunMainPortAbortsWhenRelayFails(t *testing.T) {
	setupTestDB(t)
	withCompression(t, 0)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	closed := "https://" + ln.Addr().String()
	_ = ln.Close()
	engines.RememberAgentUpstream(closed)
	t.Cleanup(engines.ForgetAgentUpstream)
	srv := httptest.NewServer(newHandler())
	t.Cleanup(srv.Close)

	resp, err := http.Post(srv.URL+"/agent.v1.AgentService/Run", "application/connect+proto",
		bytes.NewReader(connectEnvelope(0, []byte("x"))))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	// The 200 is sent before the relay fails, so the body is cut short, as in Python.
	if _, err := io.ReadAll(resp.Body); err == nil {
		t.Fatal("a failed relay ended the response cleanly; it should abort the stream")
	}
	if n := countRows(t, "request_logs"); n != 0 {
		t.Fatalf("a failed relay wrote %d usage rows", n)
	}
}

func TestCursorServerConfigIsRewritten(t *testing.T) {
	setupTestDB(t)
	t.Setenv("GATEKEEP_AGENT_URL", "http://gk.test:9478/")
	engines.ForgetAgentUpstream()
	t.Cleanup(engines.ForgetAgentUpstream)

	agent := "https://agentn.global.api5.cursor.sh"
	// Field 1: the agent URL. Field 7: http2_config = 0, which must become 3.
	upstreamMsg := append(protoString(1, agent), 0x38, 0x00)
	up := newFakeUpstream(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/connect+proto")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(connectEnvelope(0, upstreamMsg))
	})
	t.Setenv("CURSOR_BASE_URL", up.URL)

	rec := callProxy("POST", "aiserver.v1.ServerConfigService/GetServerConfig", "", []byte("{}"),
		map[string]string{"content-type": "application/connect+proto"})
	want := connectEnvelope(0, append(protoString(1, "http://gk.test:9478"), 0x38, 0x03))
	if rec.Code != http.StatusOK || !bytes.Equal(rec.Body.Bytes(), want) {
		t.Fatalf("status %d body %x, want %x", rec.Code, rec.Body.Bytes(), want)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/connect+proto" {
		t.Fatalf("content type %q", ct)
	}
	if got := engines.GetAgentUpstream(); got != agent {
		t.Fatalf("cached agent upstream %q, want %q", got, agent)
	}
	if up.count() != 1 {
		t.Fatalf("upstream calls %d, want 1", up.count())
	}
}

func TestCursorWireRequestsAreForwarded(t *testing.T) {
	setupTestDB(t)
	up := newFakeUpstream(t, replyWith(200, "application/json", `{"ok":true}`))
	t.Setenv("CURSOR_BASE_URL", up.URL)

	rec := callProxy("POST", "v1/messages", "", []byte(anthropicBody),
		map[string]string{"x-gatekeep-provider": "cursor"})
	if rec.Code != http.StatusOK || rec.Body.String() != `{"ok":true}` {
		t.Fatalf("status %d body %q", rec.Code, rec.Body.String())
	}
	if call := up.last(t); call.RequestURI != "/v1/messages" {
		t.Fatalf("upstream URI %q", call.RequestURI)
	}
	if provider, _, _, _ := lastLog(t); provider != "cursor" {
		t.Fatalf("provider %q, want cursor", provider)
	}
	// A cursor request is never answered from the response cache.
	if n := countRows(t, "response_cache"); n != 0 {
		t.Fatalf("cursor response cached (%d rows)", n)
	}
}

func TestWarmUpstreamRecordsHost(t *testing.T) {
	srv := newAgentUpstream(t, func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	warmState.mu.Lock()
	prev := warmState.host
	warmState.host = ""
	warmState.mu.Unlock()
	t.Cleanup(func() {
		warmState.mu.Lock()
		warmState.host = prev
		warmState.mu.Unlock()
	})

	warmUpstream(srv.URL)
	warmState.mu.Lock()
	got := warmState.host
	warmState.mu.Unlock()
	if got != "127.0.0.1" {
		t.Fatalf("warm host %q, want 127.0.0.1", got)
	}
	// A second warm for the same host does not dial again: it returns at once.
	warmUpstream(srv.URL)
}

func TestAgentRequestURLKeepsRawPathAndQuery(t *testing.T) {
	u := agentRequestURL("agentn.global.api5.cursor.sh", "/agent.v1.AgentService/Run?x=1%2F&y=2")
	if got := u.RequestURI(); got != "/agent.v1.AgentService/Run?x=1%2F&y=2" {
		t.Fatalf("RequestURI %q", got)
	}
	if got := agentRequestURL("h", "agent.v1.A/B%20C").RequestURI(); got != "/agent.v1.A/B%20C" {
		t.Fatalf("escaped path %q", got)
	}
}

func TestAgentUpstreamHeaderFiltering(t *testing.T) {
	h := headerMap{
		"host":              "gk",
		"content-length":    "3",
		"connection":        "keep-alive",
		"accept-encoding":   "gzip",
		"transfer-encoding": "chunked",
		":authority":        "x",
		"authorization":     "Bearer t",
		"content-type":      "application/connect+proto",
	}
	out := agentUpstreamHeader(h)
	for _, k := range []string{"Host", "Content-Length", "Connection", "Accept-Encoding", "Transfer-Encoding", ":authority"} {
		if _, ok := out[k]; ok {
			t.Fatalf("hop header %s was forwarded", k)
		}
	}
	if out.Get("Authorization") != "Bearer t" || out.Get("Content-Type") != "application/connect+proto" {
		t.Fatalf("forwarded headers wrong: %v", out)
	}
	if out.Get("Te") != "trailers" {
		t.Fatalf("te = %q, want trailers", out.Get("Te"))
	}
	if v, ok := out["User-Agent"]; !ok || len(v) != 1 || v[0] != "" {
		t.Fatalf("User-Agent must be present but empty so no default is sent, got %v", v)
	}
}

func TestAgentEndpointAndBaseURL(t *testing.T) {
	if host, port := agentEndpoint(defaultAgentUpstream); host != "agentn.global.api5.cursor.sh" || port != "443" {
		t.Fatalf("endpoint %s:%s", host, port)
	}
	if host, port := agentEndpoint("https://Agent.Cursor.sh:8443/x"); host != "agent.cursor.sh" || port != "8443" {
		t.Fatalf("endpoint %s:%s", host, port)
	}

	engines.ForgetAgentUpstream()
	if got := baseURLForPath("cursor", "agent.v1.AgentService/Run"); got != defaultAgentUpstream {
		t.Fatalf("agent base without cache = %q", got)
	}
	engines.RememberAgentUpstream("https://agent-gcpp.cursor.sh")
	t.Cleanup(engines.ForgetAgentUpstream)
	if got := baseURLForPath("cursor", "/Agent.V1.AgentService/Run"); got != "https://agent-gcpp.cursor.sh" {
		t.Fatalf("agent base with cache = %q", got)
	}
	t.Setenv("CURSOR_BASE_URL", "https://cursor.example.test/")
	if got := baseURLForPath("cursor", "aiserver.v1.AiService/StreamChat"); got != "https://cursor.example.test" {
		t.Fatalf("cursor base = %q", got)
	}
}

func TestAgentQueueKeepsOrderAndBlocks(t *testing.T) {
	q := newAgentQueue()
	done := make(chan agentItem, 1)
	go func() { done <- q.get() }()
	time.Sleep(10 * time.Millisecond)
	q.put(agentItem{data: []byte("a")})
	q.put(agentItem{data: []byte("b")})
	if it := <-done; string(it.data) != "a" {
		t.Fatalf("first item %q", it.data)
	}
	if it, ok := q.tryGet(); !ok || string(it.data) != "b" {
		t.Fatalf("second item %q %v", it.data, ok)
	}
	if _, ok := q.tryGet(); ok {
		t.Fatal("queue should be empty")
	}
}
