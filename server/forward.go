package main

// Port of backend/core/proxy.py forward_request: the LLM proxy. A request is
// routed (providers.go), decoded, optionally rewritten by the engines
// (proxy_optimize.go), answered from the response cache when the exact request
// was seen before, and otherwise streamed from the upstream provider.
//
// NOT PORTED: the Cursor agent (core/cursor_agent*.py, backend/agent_app.py),
// the GetServerConfig rewrite, and the cursor branches of the wire crush.
// A request that resolves to provider "cursor" gets a 501 with a JSON detail.
//
// Known divergences from Python, all deliberate:
//   - ErrUnsupported (an engine declines the input): the original body is
//     forwarded unchanged, one warn line is logged, and the log row has the
//     category "unoptimized". Python would rewrite the body.
//   - A response whose stream breaks, or whose client disconnects, is not
//     cached. Python's finally block caches the partial body.
//   - Upstream responses are decoded only for gzip. Go's transport advertises
//     gzip alone; httpx advertises gzip, deflate and zstd. Any other
//     Content-Encoding is forwarded raw and logged.
//   - Transfer-Encoding from the client is not forwarded; Go frames the body.
//   - Header names reach the upstream and the client in Go's canonical case.
//     Python sends lower-case names. HTTP names are case-insensitive, so this
//     is not visible to a client.
//   - Where the model is not a string, the cache-hit sample and the "anthropic"
//     model check use the JSON text (pyStrOf). Python uses repr(). The text
//     differs, but the substring test gives the same answer.

import (
	"bytes"
	"crypto/tls"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/Optivail/Gatekeep/server/engines"
	"github.com/Optivail/Gatekeep/server/pyjson"
)

// pyHTTPXUserAgent is the User-Agent httpx sends when the client sent none,
// as the Python proxy does. The version matches the oracle venv.
const pyHTTPXUserAgent = "python-httpx/0.28.1"

// textEventStream is the default media type of a StreamingResponse with no
// upstream Content-Type. Starlette adds the charset to every text/ type.
const textEventStream = "text/event-stream; charset=utf-8"

var (
	// strippedRequestHeaders and strippedResponseHeaders are the Python sets of
	// the same names. transfer-encoding is also dropped on the request side,
	// because Go frames the body itself.
	strippedRequestHeaders  = map[string]bool{"host": true, "content-length": true, "connection": true, "accept-encoding": true, "transfer-encoding": true}
	strippedResponseHeaders = map[string]bool{"content-length": true, "connection": true, "transfer-encoding": true}
)

// upstreamClient mirrors _client (HTTP/2 allowed). chatgptClient mirrors
// _chatgpt_client, which is HTTP/1.1 only because the ChatGPT Codex streams
// hang on h2. Both follow no redirects and have no overall timeout, as Python
// has timeout=None. Only connect and TLS handshake are bounded.
var (
	upstreamClient = newUpstreamClient(true)
	chatgptClient  = newUpstreamClient(false)
)

func newUpstreamClient(allowHTTP2 bool) *http.Client {
	dialer := &net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}
	tr := &http.Transport{
		// httpx trusts the environment (HTTP(S)_PROXY, NO_PROXY) by default.
		Proxy:               http.ProxyFromEnvironment,
		DialContext:         dialer.DialContext,
		ForceAttemptHTTP2:   allowHTTP2,
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 20,
		IdleConnTimeout:     90 * time.Second,
		TLSHandshakeTimeout: 30 * time.Second,
	}
	if !allowHTTP2 {
		// A non-nil empty map turns HTTP/2 off for this transport.
		tr.TLSNextProto = map[string]func(string, *tls.Conn) http.RoundTripper{}
	}
	return &http.Client{
		Transport: tr,
		// follow_redirects=False in Python: the 3xx goes back to the client.
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
}

// proxyCall holds one request's state across the branches.
type proxyCall struct {
	w        http.ResponseWriter
	r        *http.Request
	path     string
	provider string
	raw      headerMap // every request header, for routing and the credential
	headers  headerMap // the headers forwarded upstream
	body     []byte    // the body to forward, after decoding and rewriting
	payload  *pyjson.Object
	// modelHint is the "model" string of the body. It is "" when there is none.
	modelHint string

	logID       int64
	shouldCache bool
	requestHash string
	unoptimized bool
}

// forwardRequest is forward_request. path is the request path without its
// leading slash, as FastAPI's {path:path} route gives it.
func forwardRequest(w http.ResponseWriter, r *http.Request, path string) {
	raw := requestHeaderMap(r)
	if resolveProvider(raw, path, "") == "cursor" {
		refuseCursor(w, path)
		return
	}
	c := &proxyCall{w: w, r: r, path: path, raw: raw, headers: forwardableHeaders(raw)}

	body, err := readRequestBody(r)
	if err != nil {
		logger.Warn(fmt.Sprintf("[PROXY] /%s: could not read the request body: %v", path, err))
		w.WriteHeader(http.StatusBadRequest)
		return
	}

	// Codex 0.144+ and others send zstd or gzip JSON. The crush needs plaintext.
	undecodable := false
	if len(body) > 0 {
		hobj := headersObject(c.headers)
		decoded, ok, err := engines.DecodeRequestBody(body, hobj)
		c.headers = headersFromObject(hobj)
		switch {
		case isUnsupported(err):
			undecodable = true
			logger.Warn(fmt.Sprintf("[OPTIMIZER] NOT OPTIMIZED /%s: request body encoding cannot be decoded in Go (zstd); forwarded unmodified", path))
		case err != nil:
			c.fail("decode_request_body", err)
			return
		default:
			body = decoded
			if ok {
				logger.Info(fmt.Sprintf("[PROXY] decoded body /%s -> %dB", path, len(body)))
			}
		}
	}
	c.body = body

	// The body is parsed once. Python parses it twice to the same value: once
	// for model_hint, and once for the optimize branch.
	if r.Method == http.MethodPost && len(body) > 0 && !undecodable {
		v, err := pyjson.Loads(body)
		if err != nil && strings.Contains(err.Error(), "maximum recursion depth") {
			c.fail("json.loads", err)
			return
		}
		if err == nil {
			if o, ok := v.(*pyjson.Object); ok {
				c.payload = o
				if m, ok := objValue(o, "model").(string); ok {
					c.modelHint = m
				}
			}
		}
	}

	c.provider = resolveProvider(raw, path, c.modelHint)
	start := time.Now()

	switch {
	case undecodable:
		if c.record(nil, 0, 0, 0, msSince(start), "unoptimized", fmt.Sprintf("[undecoded body %dB]", len(body)), "", false) {
			return
		}
	case r.Method == http.MethodPost && len(body) > 0:
		if c.payload != nil && shouldOptimizePayload(c.payload, path) {
			if c.optimizeJSON(start) {
				return
			}
		} else if c.wirePath(start) {
			return
		}
	default:
		// GET and HEAD probes, and any request without a body. Nothing is measured.
		if c.record(nil, 0, 0, 0, 0, "passthrough", truncRunes(r.Method+" /"+path, 200), "", false) {
			return
		}
	}

	c.forward()
}

// refuseCursor answers a Cursor request with 501. The Cursor agent is not ported.
func refuseCursor(w http.ResponseWriter, path string) {
	logger.Warn(fmt.Sprintf("[PROXY] NOT PORTED: cursor request /%s refused with 501 (cursor is not in the Go server)", path))
	writeDetail(w, http.StatusNotImplemented, "cursor proxying is NOT PORTED to the Go server yet; use the Python backend for Cursor")
}

// fail logs an unexpected error and answers 500, which is what Starlette does
// for an exception the handler did not catch.
func (c *proxyCall) fail(step string, err error) {
	logger.Error(fmt.Sprintf("[PROXY] /%s failed in %s: %v", c.path, step, err))
	writeServerError(c.w)
}

// credential returns a request header as Starlette's headers.get does: the
// first value, decoded as latin-1.
func (c *proxyCall) credential(name string) string {
	return latin1(c.r.Header.Get(name))
}

// optimizeJSON is the optimize branch. It returns true when it has already
// written the response (a cache replay or an error).
func (c *proxyCall) optimizeJSON(start time.Time) bool {
	settings, err := GetSettings()
	if err != nil {
		c.fail("get_settings", err)
		return true
	}
	f := optFlags{
		compression:    settingOn(settings, "compression_enabled"),
		antiYap:        settingOn(settings, "anti_yap_enabled"),
		historyPruning: settingOn(settings, "history_pruning_enabled"),
		logTruncation:  settingOn(settings, "log_truncation_enabled"),
		aggressive:     settingOn(settings, "aggressive_enabled"),
	}

	// Exact-hash cache only. Skip it for chatgpt (replay loops). Cursor never
	// reaches this branch.
	c.shouldCache = f.compression && c.provider != "chatgpt"
	if c.shouldCache {
		// The key is over the ORIGINAL request, before any rewrite.
		hash, err := engines.ComputeRequestHash(c.payload, c.provider, engines.CredentialFingerprint(c.credential))
		if err != nil {
			c.fail("compute_request_hash", err)
			return true
		}
		c.requestHash = hash
		modelVal := objValue(c.payload, "model")
		cached, err := GetCachedResponse(hash)
		if err != nil {
			c.fail("get_cached_response", err)
			return true
		}
		if cached != nil {
			logger.Info("[OPTIMIZER] Cache HIT (cache_hit) - 0ms upstream, replay")
			sample := truncRunes(pyStrOf(modelVal), 200)
			if _, err := LogRequest(modelVal, c.provider, 0, 0, 0, msSince(start), "cache_hit", sample, "", true); err != nil {
				c.fail("log_request", err)
				return true
			}
			c.replay(cached)
			return true
		}
	}

	res, err := crushJSONConversation(c.payload, c.provider, c.body, f)
	if err != nil {
		if !isUnsupported(err) {
			c.fail("crush_json_conversation", err)
			return true
		}
		logger.Warn(fmt.Sprintf("[OPTIMIZER] NOT OPTIMIZED /%s (%v): forwarded the original body unmodified", c.path, err))
		c.unoptimized = true
		res.body = c.body
		res.optimizedTokens = res.originalTokens
		res.optimizedSample = res.originalSample
		res.cats = nil
	}
	c.body = res.body

	saved := max(0, res.originalTokens-res.optimizedTokens)
	pct := 0.0
	if res.originalTokens != 0 {
		pct = float64(saved) / float64(res.originalTokens) * 100
	}
	logger.Info(fmt.Sprintf("[OPTIMIZER] [%s] Input Tokens: %d -> %d | Saved: %d tokens (%.1f%%)",
		c.provider, res.originalTokens, res.optimizedTokens, saved, pct))

	est := 0
	for _, cat := range res.cats {
		if cat == "anti_yap" {
			est = engines.EstimatedAntiYapOutputTokensSaved
		}
	}
	category := "unchanged"
	if len(res.cats) > 0 {
		category = strings.Join(res.cats, ",")
	}
	if c.unoptimized {
		category = "unoptimized"
	}
	return c.record(objValue(c.payload, "model"), res.originalTokens, res.optimizedTokens, est,
		msSince(start), category, res.originalSample, res.optimizedSample, false)
}

// wirePath is the branch for a POST body that is not a conversation: the Connect,
// protobuf and codex bodies, and any other JSON or binary body. It returns true
// when the response has already been written.
func (c *proxyCall) wirePath(start time.Time) bool {
	settings, err := GetSettings()
	if err != nil {
		c.fail("get_settings", err)
		return true
	}
	compression := settingOn(settings, "compression_enabled")
	aggressive := settingOn(settings, "aggressive_enabled")

	codex := strings.HasPrefix(strings.ToLower(strings.TrimLeft(c.path, "/")), "backend-api/codex/")
	var origTokens int
	var origSample string
	if codex {
		// The Codex bodies skip the tokenizer: it froze the event loop in Python.
		origTokens = max(1, len(c.body)/4)
		origSample = fmt.Sprintf("[codex passthrough %dB]", len(c.body))
	} else {
		var payload any
		if c.payload != nil {
			payload = c.payload
		}
		origTokens, origSample, err = engines.MeasureWireTokens(c.body, payload)
		if err != nil {
			c.fail("measure_wire_tokens", err)
			return true
		}
	}

	ctype := latin1(c.r.Header.Get("Content-Type"))
	res, err := crushWire(c.body, c.path, ctype, origTokens, origSample, compression && !codex, compression && aggressive)
	if err != nil {
		if !isUnsupported(err) {
			c.fail("wire_crush", err)
			return true
		}
		logger.Warn(fmt.Sprintf("[OPTIMIZER] NOT OPTIMIZED /%s (%v): forwarded the original body unmodified", c.path, err))
		c.unoptimized = true
	}
	c.body = res.body

	saved := max(0, res.originalTokens-res.optimizedTokens)
	tag := "passthrough"
	if len(res.cats) > 0 && saved > 0 {
		tag = strings.Join(res.cats, ",")
	}
	if c.unoptimized {
		tag = "unoptimized"
	}
	if saved > 0 {
		pct := float64(saved) / float64(res.originalTokens) * 100
		logger.Info(fmt.Sprintf("[OPTIMIZER] [%s] wire %d -> %d | Saved: %d (%.1f%%) [%s]",
			c.provider, res.originalTokens, res.optimizedTokens, saved, pct, tag))
	} else {
		logger.Debug(fmt.Sprintf("[PROXY] passthrough provider=%s path=/%s wire_tokens=%d",
			c.provider, c.path, res.originalTokens))
	}

	sample := res.originalSample
	if sample == "" {
		sample = truncRunes(c.path, 200)
	}
	optimizedSample := ""
	if saved > 0 {
		optimizedSample = res.optimizedSample
	}
	return c.record(wireModel(c.modelHint, c.payload), res.originalTokens, res.optimizedTokens, 0,
		msSince(start), tag, sample, optimizedSample, false)
}

// wireModel is the model value logged by the wire branch: model_hint, else the
// payload's model when it is a string, else nil.
func wireModel(modelHint string, payload *pyjson.Object) any {
	if modelHint != "" {
		return modelHint
	}
	if payload != nil {
		if s, ok := objValue(payload, "model").(string); ok {
			return s
		}
	}
	return nil
}

// record writes one request_logs row. It returns true when the request has
// failed and the response is written.
func (c *proxyCall) record(model any, orig, opt, est int, latency int64, category, origSample, optSample string, cacheHit bool) bool {
	id, err := LogRequest(model, c.provider, int64(orig), int64(opt), int64(est), latency, category, origSample, optSample, cacheHit)
	if err != nil {
		c.fail("log_request", err)
		return true
	}
	c.logID = id
	return false
}

// forward sends the request upstream and relays the response.
func (c *proxyCall) forward() {
	base := baseURLFor(c.provider)
	target := upstreamURL(base, c.path, c.r.URL.RawQuery)
	logger.Info(fmt.Sprintf("[PROXY] %s /%s -> %s (%s)", c.r.Method, c.path, c.provider, base))

	client := upstreamClient
	if c.provider == "chatgpt" {
		client = chatgptClient
	}
	req, err := http.NewRequestWithContext(c.r.Context(), c.r.Method, target, bytes.NewReader(c.body))
	if err != nil {
		c.fail("build_request", err)
		return
	}
	req.Header = c.upstreamHeader()

	resp, err := client.Do(req)
	if err != nil {
		c.fail("upstream_request", err)
		return
	}
	defer resp.Body.Close()
	c.relay(resp)
}

// upstreamHeader builds the outgoing headers. httpx's defaults fill the gaps:
// Accept: */* and a User-Agent when the client sent none.
func (c *proxyCall) upstreamHeader() http.Header {
	h := make(http.Header, len(c.headers)+2)
	for name, v := range c.headers {
		h.Set(name, v)
	}
	if _, ok := h["Accept"]; !ok {
		h.Set("Accept", "*/*")
	}
	if _, ok := h["User-Agent"]; !ok {
		h.Set("User-Agent", pyHTTPXUserAgent)
	}
	return h
}

// relay streams the upstream body to the client, flushing each chunk. When
// the request is cacheable or logged, it keeps the bytes for the cache and the
// usage update. Those are stored only after a complete 2xx stream.
func (c *proxyCall) relay(resp *http.Response) {
	headers := responseHeaderMap(resp.Header)
	if ce := resp.Header.Get("Content-Encoding"); ce != "" {
		logger.Warn(fmt.Sprintf("[PROXY] /%s: upstream Content-Encoding %q is not decoded in Go; forwarded raw", c.path, ce))
	}
	writeResponse(c.w, resp.StatusCode, headers)

	collect := c.shouldCache || c.logID != 0
	var full bytes.Buffer
	complete := false
	rc := http.NewResponseController(c.w)
	buf := make([]byte, 32*1024)
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if collect {
				full.Write(buf[:n])
			}
			if _, werr := c.w.Write(buf[:n]); werr != nil {
				// The client went away. Nothing is cached from a truncated body.
				return
			}
			_ = rc.Flush()
		}
		if rerr == io.EOF {
			complete = true
			break
		}
		if rerr != nil {
			logger.Warn(fmt.Sprintf("[PROXY] /%s: upstream stream ended early: %v", c.path, rerr))
			return
		}
	}
	if collect && complete && resp.StatusCode < 300 {
		c.finish(resp.StatusCode, headers, full.Bytes())
	}
}

// finish stores the cache entry and the upstream usage figures. This is the
// work Python does in the stream's finally block.
func (c *proxyCall) finish(status int, headers map[string]string, full []byte) {
	if c.shouldCache && c.requestHash != "" {
		hj, err := pyjson.Dumps(headersObject(headers), pyjson.Default)
		if err == nil {
			err = StoreCachedResponse(c.requestHash, int64(status), string(hj), full)
		}
		if err != nil {
			logger.Error(fmt.Sprintf("[PROXY] /%s: cache store failed: %v", c.path, err))
		}
	}
	if c.logID != 0 {
		usage, err := engines.ExtractUpstreamUsage(full)
		if err == nil {
			err = UpdateUpstreamUsage(c.logID, usage)
		}
		if err != nil {
			logger.Error(fmt.Sprintf("[PROXY] /%s: usage update failed: %v", c.path, err))
		}
	}
}

// replay answers from a cache row. It always writes the response, or a 500
// when the row cannot be read.
func (c *proxyCall) replay(cached *CachedResponse) {
	hv, err := pyjson.Loads([]byte(cached.Headers))
	ho, ok := hv.(*pyjson.Object)
	if err != nil || !ok {
		c.fail("cached_headers", fmt.Errorf("stored headers are not a JSON object: %v", err))
		return
	}
	headers := make(map[string]string, ho.Len())
	for _, k := range ho.Keys() {
		v, _ := ho.Get(k)
		s, ok := v.(string)
		if !ok {
			c.fail("cached_headers", fmt.Errorf("stored header %q is not a string", k))
			return
		}
		headers[k] = s
	}
	if cached.StatusCode < 100 || cached.StatusCode > 999 {
		c.fail("cached_status", fmt.Errorf("stored status %d is not valid", cached.StatusCode))
		return
	}
	writeResponse(c.w, int(cached.StatusCode), headers)
	_, _ = c.w.Write(cached.Body)
}

// writeResponse sets the response headers, the default Content-Type, and the
// status, then flushes so the client gets the headers before the body.
func writeResponse(w http.ResponseWriter, status int, headers map[string]string) {
	for _, k := range sortedKeys(headers) {
		w.Header().Set(k, headers[k])
	}
	if _, ok := headers["content-type"]; !ok {
		w.Header().Set("Content-Type", textEventStream)
	}
	w.WriteHeader(status)
	_ = http.NewResponseController(w).Flush()
}

// responseHeaderMap is the Python response_headers: lower-cased names, values
// that repeat joined with ", " (httpx's items()), and the stripped names gone.
// Content-Encoding is removed because the body is sent decoded.
func responseHeaderMap(h http.Header) map[string]string {
	out := make(map[string]string, len(h))
	for name, vals := range h {
		k := strings.ToLower(name)
		if strippedResponseHeaders[k] || k == "content-encoding" {
			continue
		}
		out[k] = strings.Join(vals, ", ")
	}
	return out
}

// requestHeaderMap is the Python _header_map: lower-cased names, the last value
// wins, values decoded as latin-1 as Starlette does.
func requestHeaderMap(r *http.Request) headerMap {
	h := make(headerMap, len(r.Header))
	for name, vals := range r.Header {
		if len(vals) > 0 {
			h[strings.ToLower(name)] = latin1(vals[len(vals)-1])
		}
	}
	return h
}

// forwardableHeaders is the header dict sent upstream: the stripped names and
// the x-gatekeep-provider routing hint are removed.
func forwardableHeaders(raw headerMap) headerMap {
	out := make(headerMap, len(raw))
	for k, v := range raw {
		if strippedRequestHeaders[k] || k == "x-gatekeep-provider" {
			continue
		}
		out[k] = v
	}
	return out
}

// headersObject turns a header map into a pyjson object for the engines. Keys
// are sorted, so the object is deterministic.
func headersObject(m map[string]string) *pyjson.Object {
	o := pyjson.NewObject()
	for _, k := range sortedKeys(m) {
		o.Set(k, m[k])
	}
	return o
}

// headersFromObject reads back a header object after the engines changed it.
func headersFromObject(o *pyjson.Object) headerMap {
	out := make(headerMap, o.Len())
	for _, k := range o.Keys() {
		v, _ := o.Get(k)
		if s, ok := v.(string); ok {
			out[k] = s
		}
	}
	return out
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// settingOn reads a toggle as bool(settings.get(key, 1)): a missing key is on.
func settingOn(settings *pyjson.Object, key string) bool {
	v, ok := settings.Get(key)
	if !ok {
		return true
	}
	return pyTruthy(v)
}

// msSince is round((perf_counter() - start) * 1000). Python rounds half to even.
func msSince(start time.Time) int64 {
	return int64(math.RoundToEven(float64(time.Since(start)) / float64(time.Millisecond)))
}

func readRequestBody(r *http.Request) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}
	return io.ReadAll(r.Body)
}

// latin1 reinterprets header bytes as latin-1 code points, as Starlette does.
// The result is a Go string whose UTF-8 bytes are what httpx would send.
func latin1(s string) string {
	ascii := true
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			ascii = false
			break
		}
	}
	if ascii {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		b.WriteRune(rune(s[i]))
	}
	return b.String()
}

// upstreamURL builds base + "/" + path, plus the re-encoded query, as
// proxy.py does with an f-string and httpx params.
func upstreamURL(base, path, rawQuery string) string {
	u := base + "/" + escapeUpstreamPath(path)
	if q := encodeQuery(parseQuery(latin1(rawQuery))); q != "" {
		u += "?" + q
	}
	return u
}

// escapeUpstreamPath percent-encodes what httpx would encode in a path. The
// sub-delimiters and existing %XX escapes are kept. "?" and "#" are encoded
// here; httpx would read them as query and fragment (a known divergence).
func escapeUpstreamPath(p string) string {
	const hexDigits = "0123456789ABCDEF"
	var b strings.Builder
	for i := 0; i < len(p); i++ {
		c := p[i]
		switch {
		case isUnreserved(c), strings.IndexByte("!$&'()*+,;=:@/", c) >= 0:
			b.WriteByte(c)
		case c == '%' && i+2 < len(p) && isHexDigit(p[i+1]) && isHexDigit(p[i+2]):
			b.WriteByte(c)
		default:
			b.WriteByte('%')
			b.WriteByte(hexDigits[c>>4])
			b.WriteByte(hexDigits[c&0x0f])
		}
	}
	return b.String()
}

func isUnreserved(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
		c == '-' || c == '.' || c == '_' || c == '~'
}

func isHexDigit(c byte) bool {
	return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
}

// parseQuery mirrors parse_qsl(keep_blank_values=True): split on "&", keep the
// order and blank values, and decode "+" and %XX. An empty segment is dropped.
func parseQuery(q string) [][2]string {
	var pairs [][2]string
	for _, part := range strings.Split(q, "&") {
		if part == "" {
			continue
		}
		name, value, _ := strings.Cut(part, "=")
		pairs = append(pairs, [2]string{
			pyUnquote(strings.ReplaceAll(name, "+", " ")),
			pyUnquote(strings.ReplaceAll(value, "+", " ")),
		})
	}
	return pairs
}

// pyUnquote mirrors urllib.parse.unquote: a bad %XX is kept as text, and the
// bytes are decoded as UTF-8 with replacement.
func pyUnquote(s string) string {
	if !strings.Contains(s, "%") {
		return s
	}
	buf := make([]byte, 0, len(s))
	for i := 0; i < len(s); i++ {
		if s[i] == '%' && i+2 < len(s) && isHexDigit(s[i+1]) && isHexDigit(s[i+2]) {
			buf = append(buf, unhex(s[i+1])<<4|unhex(s[i+2]))
			i += 2
			continue
		}
		buf = append(buf, s[i])
	}
	return strings.ToValidUTF8(string(buf), "�")
}

func unhex(c byte) byte {
	switch {
	case c >= '0' && c <= '9':
		return c - '0'
	case c >= 'a' && c <= 'f':
		return c - 'a' + 10
	}
	return c - 'A' + 10
}

// encodeQuery mirrors urlencode(quote_plus) over the pairs, in order.
func encodeQuery(pairs [][2]string) string {
	parts := make([]string, 0, len(pairs))
	for _, p := range pairs {
		parts = append(parts, url.QueryEscape(p[0])+"="+url.QueryEscape(p[1]))
	}
	return strings.Join(parts, "&")
}
