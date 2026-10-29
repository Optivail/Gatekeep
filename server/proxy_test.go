package main

// Tests for the ports of backend/core/proxy.py and backend/core/providers.py
// (forward.go, providers.go, proxy_optimize.go). The upstream providers are
// httptest servers, selected through the *_BASE_URL environment variables. No
// test touches the network.

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"database/sql"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf16"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// upstreamCall is one request as the fake upstream saw it.
type upstreamCall struct {
	Method     string
	RequestURI string
	Header     http.Header
	Body       []byte
}

// fakeUpstream is an httptest server that records every request.
type fakeUpstream struct {
	*httptest.Server
	mu    sync.Mutex
	calls []upstreamCall
}

func newFakeUpstream(t *testing.T, respond func(w http.ResponseWriter, r *http.Request)) *fakeUpstream {
	t.Helper()
	f := &fakeUpstream{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		f.mu.Lock()
		f.calls = append(f.calls, upstreamCall{r.Method, r.RequestURI, r.Header.Clone(), body})
		f.mu.Unlock()
		respond(w, r)
	}))
	t.Cleanup(f.Close)
	return f
}

func (f *fakeUpstream) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.calls)
}

func (f *fakeUpstream) last(t *testing.T) upstreamCall {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.calls) == 0 {
		t.Fatal("upstream was not called")
	}
	return f.calls[len(f.calls)-1]
}

// replyWith returns a responder that sends a fixed status, type and body.
func replyWith(status int, contentType, body string) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

// callProxy runs forwardRequest on a recorder, the way the router does.
func callProxy(method, path, rawQuery string, body []byte, hdr map[string]string) *httptest.ResponseRecorder {
	target := "/" + path
	if rawQuery != "" {
		target += "?" + rawQuery
	}
	req := httptest.NewRequest(method, target, bytes.NewReader(body))
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	forwardRequest(rec, req, path)
	return rec
}

func countRows(t *testing.T, table string) int {
	t.Helper()
	var n int
	if err := sqlDB.QueryRow("SELECT COUNT(*) FROM " + table).Scan(&n); err != nil {
		t.Fatalf("count %s: %v", table, err)
	}
	return n
}

// lastLog returns the newest request_logs row's provider, category, cache flag
// and upstream input tokens.
func lastLog(t *testing.T) (provider, category string, cacheHit int64, upIn sql.NullInt64) {
	t.Helper()
	err := sqlDB.QueryRow(
		"SELECT provider, savings_category, cache_hit, upstream_input_tokens "+
			"FROM request_logs ORDER BY id DESC LIMIT 1",
	).Scan(&provider, &category, &cacheHit, &upIn)
	if err != nil {
		t.Fatalf("last log row: %v", err)
	}
	return
}

// anthropicBody is a small Messages request. Its optimized form is unchanged in
// meaning, which the tests check by parsing it again.
const anthropicBody = `{"model":"claude-haiku-4-5","max_tokens":64,"messages":[{"role":"user","content":"hello there"}]}`

func jsonEqual(t *testing.T, got []byte, want string) {
	t.Helper()
	var g, w any
	if err := json.Unmarshal(got, &g); err != nil {
		t.Fatalf("body is not JSON: %v (%q)", err, got)
	}
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("bad test fixture: %v", err)
	}
	gb, _ := json.Marshal(g)
	wb, _ := json.Marshal(w)
	if string(gb) != string(wb) {
		t.Fatalf("JSON differs:\n got %s\nwant %s", gb, wb)
	}
}

// ---- routing and the upstream request ------------------------------------------

func TestProxyAnthropicJSONRoundTrip(t *testing.T) {
	setupTestDB(t)
	up := newFakeUpstream(t, replyWith(200, "application/json",
		`{"id":"msg_1","usage":{"input_tokens":12,"output_tokens":3}}`))
	t.Setenv("ANTHROPIC_BASE_URL", up.URL+"/")

	rec := callProxy("POST", "v1/messages", "beta=1", []byte(anthropicBody), map[string]string{
		"x-api-key":           "test-key-test",
		"anthropic-version":   "2023-06-01",
		"x-gatekeep-provider": "anthropic",
	})

	if rec.Code != 200 {
		t.Fatalf("status %d, want 200", rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("Content-Type %q", got)
	}
	if got := rec.Body.String(); got != `{"id":"msg_1","usage":{"input_tokens":12,"output_tokens":3}}` {
		t.Fatalf("body %q", got)
	}
	c := up.last(t)
	if c.RequestURI != "/v1/messages?beta=1" {
		t.Fatalf("upstream URI %q", c.RequestURI)
	}
	if c.Header.Get("X-Api-Key") != "test-key-test" || c.Header.Get("Anthropic-Version") != "2023-06-01" {
		t.Fatalf("auth or version header not forwarded: %v", c.Header)
	}
	if c.Header.Get("X-Gatekeep-Provider") != "" {
		t.Fatal("x-gatekeep-provider reached the upstream")
	}
	if c.Header.Get("User-Agent") != pyHTTPXUserAgent {
		t.Fatalf("User-Agent %q, want the httpx default", c.Header.Get("User-Agent"))
	}
	jsonEqual(t, c.Body, anthropicBody)

	provider, category, _, upIn := lastLog(t)
	if provider != "anthropic" || category == "" {
		t.Fatalf("log row provider=%q category=%q", provider, category)
	}
	if !upIn.Valid || upIn.Int64 != 12 {
		t.Fatalf("upstream_input_tokens %v, want 12 from the response usage", upIn)
	}
}

func TestProxyAnthropicSSEStreamsUnchangedAndRecordsUsage(t *testing.T) {
	setupTestDB(t)
	stream := "event: message_start\n" +
		`data: {"type":"message_start","message":{"usage":{"input_tokens":20,"cache_read_input_tokens":5}}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","delta":{"text":"hi"}}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","usage":{"output_tokens":7}}` + "\n\n"
	up := newFakeUpstream(t, replyWith(200, "text/event-stream", stream))
	t.Setenv("ANTHROPIC_BASE_URL", up.URL)

	rec := callProxy("POST", "v1/messages", "", []byte(anthropicBody), map[string]string{"x-api-key": "test-key-s"})

	if rec.Body.String() != stream {
		t.Fatalf("streamed bytes differ:\n got %q\nwant %q", rec.Body.String(), stream)
	}
	if rec.Header().Get("Content-Type") != "text/event-stream" {
		t.Fatalf("Content-Type %q", rec.Header().Get("Content-Type"))
	}
	var up0, down, cache sql.NullInt64
	if err := sqlDB.QueryRow(
		"SELECT upstream_input_tokens, upstream_output_tokens, upstream_cache_read_tokens "+
			"FROM request_logs ORDER BY id DESC LIMIT 1",
	).Scan(&up0, &down, &cache); err != nil {
		t.Fatal(err)
	}
	if up0.Int64 != 20 || down.Int64 != 7 || cache.Int64 != 5 {
		t.Fatalf("usage in, out, cache_read = %v, %v, %v; want 20, 7, 5", up0, down, cache)
	}
}

func TestProxyOpenAIChatAndResponsesRoute(t *testing.T) {
	setupTestDB(t)
	up := newFakeUpstream(t, replyWith(200, "application/json", `{"ok":true}`))
	t.Setenv("OPENAI_BASE_URL", up.URL)

	chat := `{"model":"gpt-4o","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}]}`
	rec := callProxy("POST", "v1/chat/completions", "", []byte(chat), map[string]string{"authorization": "Bearer sk-test"})
	if rec.Code != 200 || up.last(t).RequestURI != "/v1/chat/completions" {
		t.Fatalf("chat: status %d upstream %q", rec.Code, up.last(t).RequestURI)
	}
	jsonEqual(t, up.last(t).Body, chat)

	responses := `{"model":"gpt-4.1","input":[{"role":"user","content":"hi"}]}`
	rec = callProxy("POST", "v1/responses", "", []byte(responses), map[string]string{"authorization": "Bearer sk-test"})
	if rec.Code != 200 || up.last(t).RequestURI != "/v1/responses" {
		t.Fatalf("responses: status %d upstream %q", rec.Code, up.last(t).RequestURI)
	}
	jsonEqual(t, up.last(t).Body, responses)
	if provider, _, _, _ := lastLog(t); provider != "openai" {
		t.Fatalf("provider %q, want openai", provider)
	}
}

func TestProxyQueryIsReencodedLikeHTTPX(t *testing.T) {
	setupTestDB(t)
	up := newFakeUpstream(t, replyWith(200, "application/json", `{}`))
	t.Setenv("ANTHROPIC_BASE_URL", up.URL)

	// parse_qsl keeps blank values and decodes "+"; urlencode writes "+" for a space.
	callProxy("GET", "v1/models/list", "a=1&b=x+y%21&flag&a=2", nil, map[string]string{"x-api-key": "test-key-q"})
	if got := up.last(t).RequestURI; got != "/v1/models/list?a=1&b=x+y%21&flag=&a=2" {
		t.Fatalf("upstream URI %q", got)
	}
}

// ---- gzip ----------------------------------------------------------------------

func gzipBytes(t *testing.T, s string) []byte {
	t.Helper()
	var b bytes.Buffer
	zw := gzip.NewWriter(&b)
	_, _ = zw.Write([]byte(s))
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func TestProxyGzipRequestBodyIsDecodedBeforeForwarding(t *testing.T) {
	setupTestDB(t)
	up := newFakeUpstream(t, replyWith(200, "application/json", `{"ok":1}`))
	t.Setenv("ANTHROPIC_BASE_URL", up.URL)

	rec := callProxy("POST", "v1/messages", "", gzipBytes(t, anthropicBody), map[string]string{
		"x-api-key": "test-key-gz", "Content-Encoding": "gzip",
	})
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	c := up.last(t)
	if c.Header.Get("Content-Encoding") != "" {
		t.Fatal("Content-Encoding reached the upstream after the body was decoded")
	}
	jsonEqual(t, c.Body, anthropicBody)
}

func TestProxyGzipResponseIsSentDecoded(t *testing.T) {
	setupTestDB(t)
	up := newFakeUpstream(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Encoding", "gzip")
		_, _ = w.Write(gzipBytes(t, `{"gzipped":true}`))
	})
	t.Setenv("ANTHROPIC_BASE_URL", up.URL)

	rec := callProxy("GET", "v1/models", "", nil, map[string]string{"x-api-key": "test-key-gzr"})
	if rec.Body.String() != `{"gzipped":true}` {
		t.Fatalf("body %q", rec.Body.String())
	}
	if rec.Header().Get("Content-Encoding") != "" {
		t.Fatal("Content-Encoding must not reach the client on a decoded body")
	}
}

// ---- error passthrough and invalid input ----------------------------------------

func TestProxyUpstreamErrorsPassThroughAndAreNotCached(t *testing.T) {
	setupTestDB(t)
	for _, tc := range []struct {
		status int
		ctype  string
		body   string
	}{
		{429, "application/json", `{"type":"error","error":{"type":"rate_limit_error"}}`},
		{500, "text/plain", "upstream exploded"},
	} {
		up := newFakeUpstream(t, replyWith(tc.status, tc.ctype, tc.body))
		t.Setenv("ANTHROPIC_BASE_URL", up.URL)
		for i := 0; i < 2; i++ {
			rec := callProxy("POST", "v1/messages", "", []byte(anthropicBody), map[string]string{"x-api-key": "test-key-err"})
			if rec.Code != tc.status || rec.Body.String() != tc.body {
				t.Fatalf("status %d body %q, want %d %q", rec.Code, rec.Body.String(), tc.status, tc.body)
			}
		}
		if up.count() != 2 {
			t.Fatalf("status %d: upstream called %d times, want 2 (errors are not cached)", tc.status, up.count())
		}
	}
	if n := countRows(t, "response_cache"); n != 0 {
		t.Fatalf("response_cache has %d rows after error responses", n)
	}
}

func TestProxyInvalidJSONAndEmptyBodyPassThroughUnchanged(t *testing.T) {
	setupTestDB(t)
	up := newFakeUpstream(t, replyWith(200, "application/json", `{}`))
	t.Setenv("ANTHROPIC_BASE_URL", up.URL)

	raw := []byte(`{"messages": [oops`)
	if rec := callProxy("POST", "v1/messages", "", raw, map[string]string{"x-api-key": "test-key-bad"}); rec.Code != 200 {
		t.Fatalf("invalid JSON: status %d", rec.Code)
	}
	if got := up.last(t).Body; !bytes.Equal(got, raw) {
		t.Fatalf("invalid JSON body changed: %q", got)
	}

	if rec := callProxy("POST", "v1/messages", "", nil, map[string]string{"x-api-key": "test-key-empty"}); rec.Code != 200 {
		t.Fatalf("empty body: status %d", rec.Code)
	}
	if got := up.last(t).Body; len(got) != 0 {
		t.Fatalf("empty body was changed to %q", got)
	}
}

// ---- response cache -------------------------------------------------------------

func TestProxyCacheHitKeyedOnRequestAndCredential(t *testing.T) {
	setupTestDB(t)
	up := newFakeUpstream(t, replyWith(200, "application/json", `{"answer":"cached-me"}`))
	t.Setenv("ANTHROPIC_BASE_URL", up.URL)

	first := callProxy("POST", "v1/messages", "", []byte(anthropicBody), map[string]string{"x-api-key": "test-key-a"})
	if first.Code != 200 || up.count() != 1 {
		t.Fatalf("first: status %d, upstream calls %d", first.Code, up.count())
	}

	second := callProxy("POST", "v1/messages", "", []byte(anthropicBody), map[string]string{"x-api-key": "test-key-a"})
	if up.count() != 1 {
		t.Fatalf("identical request reached the upstream again (calls=%d)", up.count())
	}
	if second.Code != 200 || second.Body.String() != first.Body.String() {
		t.Fatalf("replay differs: status %d body %q", second.Code, second.Body.String())
	}
	if second.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("replay Content-Type %q", second.Header().Get("Content-Type"))
	}
	if _, category, hit, _ := lastLog(t); category != "cache_hit" || hit != 1 {
		t.Fatalf("hit log category=%q cache_hit=%d", category, hit)
	}

	// Another caller with the same body must not get this response.
	other := callProxy("POST", "v1/messages", "", []byte(anthropicBody), map[string]string{"x-api-key": "test-key-b"})
	if other.Code != 200 || up.count() != 2 {
		t.Fatalf("different credential: upstream calls %d, want 2", up.count())
	}
}

func TestProxyChatGPTIsNeverCached(t *testing.T) {
	setupTestDB(t)
	up := newFakeUpstream(t, replyWith(200, "application/json", `{"codex":true}`))
	t.Setenv("CHATGPT_BASE_URL", up.URL)

	body := `{"model":"gpt-5","input":[{"role":"user","content":"hi"}],"instructions":"x"}`
	for i := 0; i < 2; i++ {
		callProxy("POST", "backend-api/codex/responses", "", []byte(body), map[string]string{"authorization": "Bearer chatgpt"})
	}
	if up.count() != 2 {
		t.Fatalf("chatgpt request cached: upstream calls %d, want 2", up.count())
	}
	if provider, _, _, _ := lastLog(t); provider != "chatgpt" {
		t.Fatalf("provider %q, want chatgpt", provider)
	}
}

// ---- NOT PORTED and ErrUnsupported ----------------------------------------------

func TestProxyCursorIsNotPorted(t *testing.T) {
	setupTestDB(t)
	up := newFakeUpstream(t, replyWith(200, "application/json", `{}`))
	t.Setenv("CURSOR_BASE_URL", up.URL)

	cases := []struct {
		path string
		hdr  map[string]string
	}{
		{"v1/messages", map[string]string{"x-gatekeep-provider": "cursor"}},
		{"agent.v1.AgentService/Run", nil},
		{"v1/messages", map[string]string{"content-type": "application/connect+proto"}},
	}
	for _, tc := range cases {
		rec := callProxy("POST", tc.path, "", []byte(anthropicBody), tc.hdr)
		if rec.Code != http.StatusNotImplemented {
			t.Fatalf("%s: status %d, want 501", tc.path, rec.Code)
		}
		var detail map[string]string
		if err := json.Unmarshal(rec.Body.Bytes(), &detail); err != nil || !strings.Contains(detail["detail"], "NOT PORTED") {
			t.Fatalf("%s: body %q is not the NOT PORTED JSON", tc.path, rec.Body.String())
		}
	}
	if up.count() != 0 {
		t.Fatalf("a cursor request reached an upstream (calls=%d)", up.count())
	}
}

// Both ErrUnsupported cases must forward the ORIGINAL bytes, as the policy says.
func TestProxyErrUnsupportedForwardsOriginalBody(t *testing.T) {
	setupTestDB(t)
	up := newFakeUpstream(t, replyWith(200, "application/json", `{"ok":true}`))
	t.Setenv("ANTHROPIC_BASE_URL", up.URL)

	t.Run("zstd request body", func(t *testing.T) {
		raw := []byte{0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x01, 0x02, 0x03}
		rec := callProxy("POST", "v1/messages", "", raw, map[string]string{
			"x-api-key": "test-key-zstd", "Content-Encoding": "zstd",
		})
		if rec.Code != 200 {
			t.Fatalf("status %d", rec.Code)
		}
		c := up.last(t)
		if !bytes.Equal(c.Body, raw) {
			t.Fatalf("body changed: %x", c.Body)
		}
		if c.Header.Get("Content-Encoding") != "zstd" {
			t.Fatalf("Content-Encoding %q, want zstd kept", c.Header.Get("Content-Encoding"))
		}
		if _, category, _, _ := lastLog(t); category != "unoptimized" {
			t.Fatalf("category %q, want unoptimized", category)
		}
	})

	t.Run("UTF-16 JSON wire body", func(t *testing.T) {
		// Starts with "{", so the wire crush takes its JSON path and declines the codec.
		u16 := utf16Bytes(`{"messages":[{"role":"user","content":"` + strings.Repeat("x", 220) + `"}]}`)
		rec := callProxy("POST", "v1/messages", "", u16, map[string]string{"x-api-key": "test-key-u16"})
		if rec.Code != 200 {
			t.Fatalf("status %d", rec.Code)
		}
		if got := up.last(t).Body; !bytes.Equal(got, u16) {
			t.Fatal("UTF-16 body was rewritten; the original must be forwarded")
		}
		if _, category, _, _ := lastLog(t); category != "unoptimized" {
			t.Fatalf("category %q, want unoptimized", category)
		}
	})
}

func utf16Bytes(s string) []byte {
	var out []byte
	for _, u := range utf16.Encode([]rune(s)) {
		out = append(out, byte(u), byte(u>>8))
	}
	return out
}

// ---- streaming -------------------------------------------------------------------

// TestProxyStreamsEachChunkBeforeTheUpstreamFinishes checks that the relay
// flushes per chunk. The upstream holds its second event until the client has
// read the first.
func TestProxyStreamsEachChunkBeforeTheUpstreamFinishes(t *testing.T) {
	setupTestDB(t)
	release := make(chan struct{})
	var once sync.Once
	releaseOnce := func() { once.Do(func() { close(release) }) }

	up := newFakeUpstream(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: first\ndata: {}\n\n"))
		w.(http.Flusher).Flush()
		<-release
		_, _ = w.Write([]byte("event: second\ndata: {}\n\n"))
	})
	// Cleanups run last-in first-out: releasing the upstream must come before
	// the upstream server closes, or Close waits forever.
	t.Cleanup(releaseOnce)
	t.Setenv("ANTHROPIC_BASE_URL", up.URL)

	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		forwardRequest(w, r, strings.TrimPrefix(r.URL.Path, "/"))
	}))
	defer proxy.Close()
	defer releaseOnce()

	req, _ := http.NewRequest("POST", proxy.URL+"/v1/messages", strings.NewReader(anthropicBody))
	req.Header.Set("x-api-key", "test-key-stream")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	br := bufio.NewReader(resp.Body)
	first := make(chan string, 1)
	go func() {
		line, _ := br.ReadString('\n')
		first <- line
	}()
	select {
	case line := <-first:
		if line != "event: first\n" {
			t.Fatalf("first line %q", line)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the first event was not flushed while the upstream was still open")
	}
	releaseOnce()
	rest, _ := io.ReadAll(br)
	if !strings.Contains(string(rest), "event: second") {
		t.Fatalf("second event missing: %q", rest)
	}
}

// ---- provider resolution and URLs ------------------------------------------------

func TestResolveProviderTable(t *testing.T) {
	cases := []struct {
		name  string
		hdr   headerMap
		path  string
		model string
		want  string
	}{
		{"x-api-key sk-ant", headerMap{"x-api-key": "test-key-1"}, "v1/messages", "", "anthropic"},
		{"bearer on chat", headerMap{"authorization": "Bearer x"}, "v1/chat/completions", "", "openai"},
		{"goog key header", headerMap{"x-goog-api-key": "k"}, "v1beta/models/m", "", "google"},
		{"codex backend path", headerMap{}, "backend-api/codex/responses", "", "chatgpt"},
		{"override is case-insensitive", headerMap{"x-gatekeep-provider": " Mistral "}, "v1/chat/completions", "", "mistral"},
		{"mistral model hint", headerMap{}, "v1/chat/completions", "mistral-large", "mistral"},
		{"model with vendor prefix", headerMap{}, "v1/chat/completions", "vendor/codestral-2", "mistral"},
		{"connect content type is cursor", headerMap{"content-type": "application/connect+json"}, "x", "", "cursor"},
		{"anthropic-version beats bearer", headerMap{"anthropic-version": "2023-06-01", "authorization": "x"}, "v1/messages", "", "anthropic"},
		{"models with bearer", headerMap{"authorization": "x"}, "v1/models", "", "openai"},
		{"devin org header", headerMap{"x-devin-org": "o"}, "something", "", "devin"},
		{"unknown path defaults to anthropic", headerMap{}, "no/such/path", "", "anthropic"},
	}
	for _, tc := range cases {
		if got := resolveProvider(tc.hdr, tc.path, tc.model); got != tc.want {
			t.Errorf("%s: got %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestBaseURLFromEnvironmentIsTrimmed(t *testing.T) {
	t.Setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9///")
	if got := baseURLFor("anthropic"); got != "http://127.0.0.1:9" {
		t.Fatalf("got %q", got)
	}
	if got := baseURLFor("devin"); got != "https://api.devin.ai" {
		t.Fatalf("default devin base %q", got)
	}
}

func TestShouldOptimizePayloadAndShapes(t *testing.T) {
	obj := func(s string) *pyjson.Object {
		v, err := pyjson.Loads([]byte(s))
		if err != nil {
			t.Fatal(err)
		}
		return v.(*pyjson.Object)
	}
	if !shouldOptimizePayload(obj(`{"input":[]}`), "v1/responses") {
		t.Error("input payload should be optimized")
	}
	if shouldOptimizePayload(obj(`{"messages":[]}`), "v1/messages/count_tokens") {
		t.Error("count_tokens must not be optimized")
	}
	if shouldOptimizePayload(obj(`{"messages":[]}`), "api/v1/freebuff") {
		t.Error("freebuff must not be optimized")
	}
	anth, err := isAnthropicMessagesPayload(obj(`{"messages":[{"role":"user","content":[{"type":"tool_result"}]}]}`))
	if err != nil || !anth {
		t.Errorf("tool_result block should mark Anthropic: %v %v", anth, err)
	}
	if _, err := isAnthropicMessagesPayload(obj(`{"messages":5}`)); err == nil {
		t.Error("a non-iterable messages value should raise, as Python's TypeError does")
	}
	chat, _ := isOpenAIChatPayload(obj(`{"messages":[{"role":"user","content":"x"}]}`))
	if !chat {
		t.Error("user-role messages are an OpenAI chat payload")
	}
}

func TestUpstreamURLEscapingMatchesHTTPX(t *testing.T) {
	if got := escapeUpstreamPath("v1/a b/é?#%41"); got != "v1/a%20b/%C3%A9%3F%23%41" {
		t.Fatalf("escapeUpstreamPath = %q", got)
	}
	if got := encodeQuery(parseQuery("q=a+b&empty=&x")); got != "q=a+b&empty=&x=" {
		t.Fatalf("encodeQuery = %q", got)
	}
	if got := truncRunes("ééé", 2); got != "éé" {
		t.Fatalf("truncRunes = %q", got)
	}
}
