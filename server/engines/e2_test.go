package engines

import (
	"bytes"
	"compress/gzip"
	"errors"
	"strings"
	"testing"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

func e2Obj(t *testing.T, src string) *pyjson.Object {
	t.Helper()
	v, err := pyjson.Loads([]byte(src))
	if err != nil {
		t.Fatalf("parse %q: %v", src, err)
	}
	o, ok := v.(*pyjson.Object)
	if !ok {
		t.Fatalf("%q is not an object", src)
	}
	return o
}

func TestReadVarintOverflowKeepsParsing(t *testing.T) {
	// A 2-tagged key whose value needs 71 bits: Python keeps the big int, so the
	// parse succeeds, and the field number cannot be re-encoded exactly.
	buf := append([]byte{0x82}, bytes.Repeat([]byte{0x80}, 9)...)
	buf = append(buf, 0x01, 0x03, 'a', 'b', 'c')
	fields, ok := parseFields(buf)
	if !ok || len(fields) != 1 {
		t.Fatalf("parseFields ok=%v fields=%d, want one field", ok, len(fields))
	}
	if fields[0].fn != bigField || fields[0].wt != 2 {
		t.Fatalf("field = fn %d wt %d, want bigField wt 2", fields[0].fn, fields[0].wt)
	}
	if _, enc := appendField(nil, fields[0].fn, fields[0].wt, fields[0].val); enc {
		t.Fatal("appendField encoded a bigField key")
	}
}

func TestParseFieldsRejectsGroupsAndFieldZero(t *testing.T) {
	for _, buf := range [][]byte{{0x0B}, {0x0C}, {0x00}, {0x03}} {
		if _, ok := parseFields(buf); ok {
			t.Errorf("parseFields(%x) accepted, want ValueError", buf)
		}
	}
}

func TestEncodeUTF8ReplaceSurrogate(t *testing.T) {
	in := "a" + string([]byte{0xED, 0xA0, 0x80}) + "b"
	if got := string(encodeUTF8Replace(in)); got != "a?b" {
		t.Fatalf("encodeUTF8Replace = %q, want %q", got, "a?b")
	}
	if _, err := encodeUTF8Strict(in); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("encodeUTF8Strict err = %v, want ErrUnsupported", err)
	}
}

func TestCrushJSONArraySampling(t *testing.T) {
	// 20 items: head 0..2, tail 17..19, middle 3..16 (14 items), step 14//4=3,
	// so the sample is middle[0,3,6,9] = 3,6,9,12.
	var items []string
	for i := 0; i < 20; i++ {
		items = append(items, itoa(i))
	}
	got, err := crushJSONArray("[" + strings.Join(items, ",") + "]")
	if err != nil {
		t.Fatal(err)
	}
	want := `{"_n":20,"_head":[0,1,2],"_sample":[3,6,9,12],"_tail":[17,18,19]}`
	if got != want {
		t.Fatalf("crushJSONArray = %s\nwant %s", got, want)
	}
}

func TestDropEmptyTextBlocksNonStrRaises(t *testing.T) {
	// Python calls .strip() on a non-str text inside a tool_result list.
	p := e2Obj(t, `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"x",
		"content":[{"type":"text","text":5}]}]}]}`)
	if err := DropEmptyTextBlocks(p); err == nil {
		t.Fatal("DropEmptyTextBlocks accepted a non-str nested text, want error")
	}
}

func TestDropEmptyTextBlocksSystemNonStrKept(t *testing.T) {
	// The top-level text block calls str() on its value, so 5 is kept.
	p := e2Obj(t, `{"system":[{"type":"text","text":5}],"messages":[]}`)
	if err := DropEmptyTextBlocks(p); err != nil {
		t.Fatal(err)
	}
	sys, _ := objGet(p, "system").([]any)
	if len(sys) != 1 {
		t.Fatalf("system blocks = %d, want 1", len(sys))
	}
}

func TestStaleToolCompactionKeepsNewestRead(t *testing.T) {
	long := strings.Repeat("line of file content\n", 10)
	p := e2Obj(t, `{"messages":[
		{"role":"assistant","content":[{"type":"tool_use","id":"a","name":"Read","input":{"file_path":"/p/x.py"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":`+jsonQuote(long)+`}]},
		{"role":"assistant","content":[{"type":"tool_use","id":"b","name":"Read","input":{"file_path":"/p/x.py"}}]},
		{"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":`+jsonQuote(long)+`}]}]}`)
	changed, err := ApplyStaleToolCompaction(p)
	if err != nil || !changed {
		t.Fatalf("changed=%v err=%v, want changed", changed, err)
	}
	msgs := objGet(p, "messages").([]any)
	first := contentList(msgs[1].(*pyjson.Object))[0].(*pyjson.Object)
	last := contentList(msgs[3].(*pyjson.Object))[0].(*pyjson.Object)
	if ContentText(objGet(first, "content")) != staleStub {
		t.Errorf("older read = %q, want %q", ContentText(objGet(first, "content")), staleStub)
	}
	if ContentText(objGet(last, "content")) == staleStub {
		t.Error("newest read was stubbed")
	}
}

func TestApplyAggressiveOpenAIDedup(t *testing.T) {
	body := strings.Repeat("same tool output line\n", 6)
	p := e2Obj(t, `{"messages":[
		{"role":"tool","tool_call_id":"1","content":`+jsonQuote(body)+`},
		{"role":"tool","tool_call_id":"2","content":`+jsonQuote(body)+`}]}`)
	cats, err := ApplyAggressiveOpenAI(p)
	if err != nil {
		t.Fatal(err)
	}
	if !containsStr(cats, "tool_dedup") {
		t.Fatalf("cats = %v, want tool_dedup", cats)
	}
	msgs := objGet(p, "messages").([]any)
	if got := ContentText(objGet(msgs[1].(*pyjson.Object), "content")); got != dupStub {
		t.Fatalf("second tool content = %q, want %q", got, dupStub)
	}
}

func TestGzipChangedFrameDeclines(t *testing.T) {
	// A skill entry longer than the limit shrinks to its id, so the body changes.
	// The frame is gzip-flagged, and re-compressing it would not be reproducible.
	inner := append(pldTest(1, []byte("skill-id")), pldTest(3, []byte(strings.Repeat("d", 100)))...)
	msg := pldTest(29, inner)
	var zb bytes.Buffer
	zw, _ := gzip.NewWriterLevel(&zb, 6)
	_, _ = zw.Write(msg)
	_ = zw.Close()
	payload := zb.Bytes()
	frame := append([]byte{0x01, 0, 0, 0, 0}, payload...)
	frame[1], frame[2], frame[3], frame[4] = byte(len(payload)>>24), byte(len(payload)>>16), byte(len(payload)>>8), byte(len(payload))
	// A changed gzip frame is re-compressed (decoded-level parity; the gzip bytes
	// cannot match Python's, which embed the time). Its payload must decode to
	// the same crushed message as the plain-frame case below.
	gzOut, _, err := CrushConnectBody(frame, false, nil)
	if err != nil {
		t.Fatalf("changed gzip frame: %v", err)
	}
	gzFrames, ok := IterFrames(gzOut)
	if !ok || len(gzFrames) != 1 || gzFrames[0].Flags&0x01 == 0 {
		t.Fatalf("recompressed frame: ok=%v frames=%v", ok, gzFrames)
	}
	gzPayload, err := gzipDecompress(gzFrames[0].Payload)
	if err != nil {
		t.Fatalf("gunzip recompressed payload: %v", err)
	}

	// The same message as a plain frame crushes without declining.
	plain := append([]byte{0x00, 0, 0, 0, 0}, msg...)
	n := len(msg)
	plain[1], plain[2], plain[3], plain[4] = byte(n>>24), byte(n>>16), byte(n>>8), byte(n)
	out, saved, err := CrushConnectBody(plain, false, nil)
	if err != nil {
		t.Fatal(err)
	}
	if saved <= 0 || len(out) >= len(plain) {
		t.Fatalf("plain frame saved=%d len %d->%d, want a crush", saved, len(plain), len(out))
	}
	plainFrames, ok := IterFrames(out)
	if !ok || len(plainFrames) != 1 {
		t.Fatalf("plain crushed frame: ok=%v", ok)
	}
	if !bytes.Equal(gzPayload, plainFrames[0].Payload) {
		t.Fatalf("decoded payload differs: gzip frame %q, plain frame %q", gzPayload, plainFrames[0].Payload)
	}
}

// pldTest encodes a length-delimited field for the unit tests.
func pldTest(fn int, b []byte) []byte {
	out, _ := appendField(nil, uint64(fn), 2, b)
	return out
}

// jsonQuote returns s as a JSON string literal.
func jsonQuote(s string) string {
	out := []byte{'"'}
	for _, r := range s {
		switch r {
		case '"', '\\':
			out = append(out, '\\', byte(r))
		case '\n':
			out = append(out, '\\', 'n')
		default:
			out = append(out, string(r)...)
		}
	}
	return string(append(out, '"'))
}

func TestNonUTF8JSONCodecDetection(t *testing.T) {
	cases := map[string]bool{
		`{"a":1}`:               false,
		"\xef\xbb\xbf{\"a\":1}": false,
		"\xff\xfe{\x00":         true,
		"\x00{\x00\x00":         true,
		"ab":                    false,
		"\x00\x7b":              true,
	}
	for in, want := range cases {
		if got := nonUTF8JSONCodec([]byte(in)); got != want {
			t.Errorf("nonUTF8JSONCodec(%q) = %v, want %v", in, got, want)
		}
	}
}
