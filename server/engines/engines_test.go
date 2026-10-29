// Mirrors backend/engines/optimizer.py (cache key, file delta, pruning, cache
// alignment, body codec) and the security checks in backend/tests/test_security.py.
package engines

import (
	"encoding/hex"
	"errors"
	"strings"
	"testing"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// mustObj parses a JSON object literal with the same loader the proxy uses.
func mustObj(t *testing.T, s string) *pyjson.Object {
	t.Helper()
	v, err := pyjson.Loads([]byte(s))
	if err != nil {
		t.Fatalf("bad test JSON: %v", err)
	}
	o, ok := v.(*pyjson.Object)
	if !ok {
		t.Fatalf("test JSON is not an object")
	}
	return o
}

// compactJSON returns the canonical bytes of v (json.dumps with compact
// separators and ensure_ascii=False) as a string.
func compactJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := canonicalJSON(v)
	if err != nil {
		t.Fatalf("dumps: %v", err)
	}
	return string(b)
}

// credentialFor builds the fingerprint of one x-api-key header, as the proxy
// does for a request that carries only that header.
func credentialFor(key string) string {
	return CredentialFingerprint(func(name string) string {
		if name == "x-api-key" {
			return key
		}
		return ""
	})
}

// --- security (mirrors backend/tests/test_security.py) ----------------------

const securityPayload = `{"model":"m","messages":[{"role":"user","content":"hi"}]}`

func cacheKey(t *testing.T, payload *pyjson.Object, key string) string {
	t.Helper()
	h, err := ComputeRequestHash(payload, "anthropic", credentialFor(key))
	if err != nil {
		t.Fatalf("ComputeRequestHash: %v", err)
	}
	return h
}

func TestCacheKeyDependsOnCallerCredential(t *testing.T) {
	p := mustObj(t, securityPayload)
	if cacheKey(t, p, "key-a") == cacheKey(t, p, "key-b") {
		t.Fatal("two callers with different credentials share a cache key")
	}
}

func TestCacheKeyDependsOnEveryRequestField(t *testing.T) {
	base := cacheKey(t, mustObj(t, securityPayload), "k")
	for _, extra := range []string{`{"stream":true}`, `{"max_tokens":5}`, `{"tools":[{"name":"t"}]}`} {
		p := mustObj(t, securityPayload)
		e := mustObj(t, extra)
		for _, k := range e.Keys() {
			v, _ := e.Get(k)
			p.Set(k, v)
		}
		if cacheKey(t, p, "k") == base {
			t.Fatalf("cache key ignores %s", extra)
		}
	}
}

func TestCacheKeyIsStable(t *testing.T) {
	a := cacheKey(t, mustObj(t, securityPayload), "k")
	b := cacheKey(t, mustObj(t, securityPayload), "k")
	if a != b {
		t.Fatalf("cache key not stable: %s vs %s", a, b)
	}
}

// Python golden values, computed with the backend oracle.
func TestCacheKeyGolden(t *testing.T) {
	const want = "7ce1e2999e432aacd6c47f157c5ec6a463c6c1879091f28dea431b98621460e1"
	if got := cacheKey(t, mustObj(t, securityPayload), "key-a"); got != want {
		t.Fatalf("cache key = %s, want %s", got, want)
	}
}

func TestCredentialFingerprintGolden(t *testing.T) {
	cases := []struct {
		name string
		get  func(string) string
		want string
	}{
		{"whitespace is stripped", func(n string) string {
			if n == "authorization" {
				return "  Bearer x \t"
			}
			return ""
		}, "0c48cd47bb7f1fe44fc4e4b48b1a3da1"},
		{"no credential headers", func(string) string { return "" }, "459f03424f94974d16ab9c2ef079e7aa"},
	}
	for _, c := range cases {
		if got := CredentialFingerprint(c.get); got != c.want {
			t.Errorf("%s: fingerprint = %s, want %s", c.name, got, c.want)
		}
	}
}

func TestCredentialFingerprintSeparatesHeaderNames(t *testing.T) {
	// The same value under two different header names must not collide.
	a := CredentialFingerprint(func(n string) string {
		if n == "authorization" {
			return "same"
		}
		return ""
	})
	b := CredentialFingerprint(func(n string) string {
		if n == "x-api-key" {
			return "same"
		}
		return ""
	})
	if a == b {
		t.Fatal("authorization and x-api-key with the same value share a fingerprint")
	}
}

// --- optimizer -------------------------------------------------------------

func TestOptimizeTextGolden(t *testing.T) {
	in := "Certainly! So the file is a_b.c. Could you please check `the code` now.\n\n\n\nend   x"
	const want = "file is a_b.c. check `the code` now.\n\nend x"
	if got := OptimizeText(in); got != want {
		t.Fatalf("OptimizeText = %q, want %q", got, want)
	}
}

func TestOptimizeTextCodeBlockGolden(t *testing.T) {
	in := "```python\nx = 1   \n\n\n\ny = 2\n```\nThe   end"
	const want = "```python\nx = 1\n\ny = 2\n```\n end"
	if got := OptimizeText(in); got != want {
		t.Fatalf("OptimizeText = %q, want %q", got, want)
	}
}

func TestOptimizeTextValueRejectsNonString(t *testing.T) {
	if _, err := optimizeTextValue(int64(5)); err == nil {
		t.Fatal("optimize_text(5) must raise, as Python's regex split does")
	}
	if v, err := optimizeTextValue(nil); err != nil || v != nil {
		t.Fatalf("optimize_text(None) = %v, %v; want None, nil", v, err)
	}
}

func TestInjectAntiYap(t *testing.T) {
	if got := InjectAntiYap("sys"); got != "sys\n\n"+antiYap {
		t.Fatalf("string system: got %q", got)
	}
	if got := InjectAntiYap(nil); got != antiYap {
		t.Fatalf("None system: got %q", got)
	}
	if got := InjectAntiYap(int64(5)); got != int64(5) {
		t.Fatalf("non-str system must pass through, got %v", got)
	}
}

func TestCountTokens(t *testing.T) {
	if n, err := CountTokens("hello world"); err != nil || n != 2 {
		t.Fatalf("CountTokens = %d, %v; want 2, nil", n, err)
	}
	if n, err := CountTokens(""); err != nil || n != 0 {
		t.Fatalf("CountTokens(\"\") = %d, %v; want 0, nil", n, err)
	}
	// Python raises ValueError for a special token in the text.
	if _, err := CountTokens("a <|endoftext|> b"); err == nil {
		t.Fatal("special token must raise, as tiktoken does")
	}
}

func TestMeasureWireTokensBinary(t *testing.T) {
	n, sample, err := MeasureWireTokens([]byte{0xff, 0xfe}, nil)
	if err != nil || n != 1 || sample != "[binary 2B ~1 tok]" {
		t.Fatalf("MeasureWireTokens = %d, %q, %v", n, sample, err)
	}
	if n, sample, err := MeasureWireTokens(nil, nil); err != nil || n != 0 || sample != "" {
		t.Fatalf("empty body = %d, %q, %v", n, sample, err)
	}
}

func TestExtractUpstreamUsageSSE(t *testing.T) {
	body := "event: message_start\n" +
		`data: {"type":"message_start","message":{"usage":{"input_tokens":12,"cache_creation":{"ephemeral_5m_input_tokens":3,"ephemeral_1h_input_tokens":4}}}}` +
		"\n\n" + `data: {"type":"message_delta","usage":{"output_tokens":7}}` + "\n\ndata: [DONE]\n"
	u, err := ExtractUpstreamUsage([]byte(body))
	if err != nil {
		t.Fatalf("ExtractUpstreamUsage: %v", err)
	}
	const want = `{"input_tokens":12,"cache_creation_input_tokens":7,"output_tokens":7}`
	if got := compactJSON(t, u); got != want {
		t.Fatalf("usage = %s, want %s", got, want)
	}
}

func TestExtractUpstreamUsageBOMIsNotJSON(t *testing.T) {
	// json.loads(str) rejects a leading BOM, so the plain-JSON path yields nothing.
	u, err := ExtractUpstreamUsage([]byte("\xef\xbb\xbf" + `{"usage":{"input_tokens":1}}`))
	if err != nil || u.Len() != 0 {
		t.Fatalf("usage = %v, %v; want empty", u, err)
	}
}

func TestFileDeltaTrackingCollapsesRereads(t *testing.T) {
	read := func(id, content string) string {
		return `{"role":"assistant","content":[{"type":"tool_use","id":"` + id +
			`","name":"read_file","input":{"file_path":"f.txt"}}]},` +
			`{"role":"user","content":[{"type":"tool_result","tool_use_id":"` + id +
			`","content":"` + content + `"}]}`
	}
	p := mustObj(t, `{"messages":[{"role":"user","content":"start"},`+
		read("1", "one\\ntwo\\nthree\\n")+","+read("2", "one\\ntwo\\nthree\\n")+","+
		read("3", "one\\nTWO\\nthree\\n")+`]}`)
	out, err := ApplyFileDeltaTracking(p, "s1")
	if err != nil {
		t.Fatalf("ApplyFileDeltaTracking: %v", err)
	}
	const want = `{"messages":[{"role":"user","content":"start"},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"1","name":"read_file","input":{"file_path":"f.txt"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"1","content":"one\ntwo\nthree\n"}]},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"2","name":"read_file","input":{"file_path":"f.txt"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"2","content":"[FILE UNCHANGED: f.txt]"}]},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"3","name":"read_file","input":{"file_path":"f.txt"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"3","content":"--- f.txt\n+++ f.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three"}]}]}`
	if got := compactJSON(t, out); got != want {
		t.Fatalf("file delta output:\n got %s\nwant %s", got, want)
	}
}

func TestDeriveSessionKeyGolden(t *testing.T) {
	p := mustObj(t, `{"system":"x","messages":[{"role":"user","content":"y"}]}`)
	if got := DeriveSessionKey(p); got != "ae8df3c38021db66" {
		t.Fatalf("DeriveSessionKey = %s", got)
	}
}

func TestNewestToolResultHeadTailCap(t *testing.T) {
	// 2003 characters: the cap keeps 800 at each end and reports 403 omitted.
	long := strings.Repeat("A", 1000) + "MID" + strings.Repeat("B", 1000)
	block := objectOf("type", "tool_result", "tool_use_id", "1", "content", long)
	p := mustObj(t, `{"messages":[{"role":"assistant","content":"x"},{"role":"user","content":[]}]}`)
	last := objGet(p, "messages").([]any)[1].(*pyjson.Object)
	last.Set("content", []any{block})
	if _, err := ApplyNewestToolResultCompaction(p); err != nil {
		t.Fatalf("ApplyNewestToolResultCompaction: %v", err)
	}
	want := strings.Repeat("A", 800) + "\n... [403 chars truncated] ...\n" + strings.Repeat("B", 800)
	if got := objGet(block, "content").(string); got != want {
		t.Fatalf("tail cap: got %d chars, want %d", len(got), len(want))
	}
}

func TestPruneMiddlePythonFenceStub(t *testing.T) {
	// Expected value from CPython 3.12: _prune_middle_content on this block.
	in := "```python\ndef f():\n    pass\n```"
	want := "```python\ndef f():\n    ...\n```"
	got, err := pruneMiddleContent(in, true)
	if err != nil || got != want {
		t.Fatalf("pruneMiddleContent = %q, %v; want %q", got, err, want)
	}
}

func TestPruneMiddlePythonFenceDeclinesDeepNesting(t *testing.T) {
	// 200 nested dict displays: CPython's ast.unparse fails on them and the
	// distiller would return the code unchanged, so the port declines.
	code := "def f(a=" + strings.Repeat("{1:", 150) + "1" + strings.Repeat("}", 150) + "):\n    pass"
	_, err := pruneMiddleContent("```python\n"+code+"\n```", true)
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("pruneMiddleContent error = %v, want ErrUnsupported", err)
	}
}

func TestPruneHistoryShortListUnchanged(t *testing.T) {
	msgs := []any{objectOf("role", "user", "content", "a")}
	out, err := PruneHistory(msgs, true, true)
	if err != nil {
		t.Fatalf("PruneHistory: %v", err)
	}
	if got := out.([]any); len(got) != 1 {
		t.Fatalf("short history changed length to %d", len(got))
	}
}

func TestCacheAlignmentSkipsWhenClientMarked(t *testing.T) {
	p := mustObj(t, `{"system":"`+strings.Repeat("S", 4000)+`","messages":[{"role":"user","content":[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}}]}]}`)
	before := compactJSON(t, p)
	if _, err := ApplyAnthropicCacheAlignment(p, "anthropic"); err != nil {
		t.Fatalf("ApplyAnthropicCacheAlignment: %v", err)
	}
	if after := compactJSON(t, p); after != before {
		t.Fatal("cache alignment changed a payload the client already marked")
	}
}

func TestCacheAlignmentOnlyForAnthropic(t *testing.T) {
	p := mustObj(t, `{"system":"`+strings.Repeat("S", 4000)+`"}`)
	before := compactJSON(t, p)
	if _, err := ApplyAnthropicCacheAlignment(p, "openai"); err != nil {
		t.Fatalf("ApplyAnthropicCacheAlignment: %v", err)
	}
	if compactJSON(t, p) != before {
		t.Fatal("cache alignment changed a non-Anthropic payload")
	}
}

func TestClampMaxTokens(t *testing.T) {
	short := mustObj(t, `{"max_tokens":8192,"messages":[{"role":"user","content":" hi there "}]}`)
	ClampMaxTokens(short)
	if got := compactJSON(t, short); got != `{"max_tokens":2048,"messages":[{"role":"user","content":" hi there "}]}` {
		t.Fatalf("short query: %s", got)
	}
	code := mustObj(t, `{"max_tokens":8192,"messages":[{"role":"user","content":"`+"```x```"+`"}]}`)
	ClampMaxTokens(code)
	if v, _ := code.Get("max_tokens"); v != int64(8192) {
		t.Fatalf("code query must keep max_tokens, got %v", v)
	}
}

func TestMinifyToolsGolden(t *testing.T) {
	in := mustObj(t, `{"name":"t","description":"Does a thing. Long tail here.","input_schema":{"type":"object","properties":{"a":{"type":"string","description":"First one! Second."}}}}`)
	out := MinifyToolsDeterministic([]any{in})
	const want = `[{"name":"t","description":"Does a thing.","input_schema":{"type":"object","properties":{"a":{"type":"string","description":"First one! Second."}}}}]`
	if got := compactJSON(t, out); got != want {
		t.Fatalf("minify = %s, want %s", got, want)
	}
}

// --- body codec -------------------------------------------------------------

func TestDecodeRequestBodyGzip(t *testing.T) {
	plain := []byte(`{"model":"m"}`)
	headers := mustObj(t, `{"Content-Encoding":"gzip","x":"1"}`)
	out, ok, err := DecodeRequestBody(gzipCompress(plain), headers)
	if err != nil || !ok || string(out) != string(plain) {
		t.Fatalf("gzip decode = %q, %v, %v", out, ok, err)
	}
	if _, present := headers.Get("Content-Encoding"); present {
		t.Fatal("Content-Encoding must be removed after a successful decode")
	}
}

func TestDecodeRequestBodyZstd(t *testing.T) {
	// Frame produced by python-zstandard for b"payload" (CPython 3.12 oracle).
	frame, _ := hex.DecodeString("28b52ffd20073900007061796c6f6164")
	headers := mustObj(t, `{"content-encoding":"zstd"}`)
	out, ok, err := DecodeRequestBody(frame, headers)
	if err != nil || !ok || string(out) != "payload" {
		t.Fatalf("zstd decode = %q, %v, %v", out, ok, err)
	}
	if _, present := headers.Get("content-encoding"); present {
		t.Fatal("Content-Encoding must be removed after a successful decode")
	}
}

func TestDecodeRequestBodyZstdFailureReturnsBodyUnchanged(t *testing.T) {
	// Truncated frame: python-zstandard raises, so the body is kept as is.
	frame, _ := hex.DecodeString("28b52ffd20073900007061796c6f6164")
	body := frame[:len(frame)-2]
	headers := mustObj(t, `{"content-encoding":"zstd"}`)
	out, ok, err := DecodeRequestBody(body, headers)
	if err != nil || ok || string(out) != string(body) {
		t.Fatalf("truncated zstd = %q, %v, %v; want the body unchanged", out, ok, err)
	}
}
