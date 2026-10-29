// Mirrors backend/engines/wirecrush.py: crushes Cursor and Codex Connect and
// protobuf wire bodies, and JSON bodies on the same path. It keeps field
// structure and crushes only the large text leaves, so the request stays valid.
//
// Python ints are unbounded. A protobuf field key that needs more than 64 bits
// is kept as bigField, which can be parsed but not re-encoded exactly. Any
// output that would re-encode such a key declines with ErrUnsupported.
//
// A changed gzip-flagged Connect frame also declines. Python re-compresses it
// with gzip.compress, whose header carries the current time, so the bytes are
// not reproducible even against Python itself.
package engines

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

const (
	wireMinStr       = 80
	maxProtoDepth    = 32
	wireAggrMinStr   = 60
	wireAggrMaxStr   = 100
	wireAggrDedupMin = 150
	// bigField marks a field whose key does not fit in 64 bits. No real field
	// number reaches it (the largest is 2^61-1).
	bigField uint64 = math.MaxUint64
)

// wireCapture holds the offline debug captures that Python keeps in module
// globals. The mutex guards the counters, which Python does not lock.
var wireCapture struct {
	mu            sync.Mutex
	dir           string // "" keeps the counters but writes nothing
	dumped        int
	agentCaptures int
}

// SetWireCaptureDir enables the debug captures in dir. Python writes them
// under bench/runs. An empty dir disables the writes.
func SetWireCaptureDir(dir string) {
	wireCapture.mu.Lock()
	defer wireCapture.mu.Unlock()
	wireCapture.dir = dir
}

// protoField is one parsed field. Varint values keep their raw bytes so they
// re-encode exactly as Python does.
type protoField struct {
	fn  uint64
	wt  int
	val []byte
}

// readVarint mirrors _read_varint. It returns the low 64 bits, whether the
// value needs more than 64 bits, and the next index. ok is false where Python
// raises ValueError (a varint longer than 11 bytes or truncated).
func readVarint(buf []byte, i int) (n uint64, overflow bool, next int, ok bool) {
	shift := 0
	for i < len(buf) {
		b := buf[i]
		i++
		chunk := uint64(b & 0x7f)
		switch {
		case shift < 64:
			n |= chunk << shift
			if shift > 57 && chunk>>(64-shift) != 0 {
				overflow = true
			}
		case chunk != 0:
			overflow = true
		}
		if b&0x80 == 0 {
			return n, overflow, i, true
		}
		shift += 7
		if shift > 70 {
			break
		}
	}
	return 0, false, 0, false
}

// writeVarint mirrors _write_varint.
func writeVarint(n uint64) []byte {
	var out []byte
	for n > 0x7f {
		out = append(out, byte(n&0x7f)|0x80)
		n >>= 7
	}
	return append(out, byte(n&0x7f))
}

// parseFields mirrors _parse_fields. ok is false where Python raises ValueError.
func parseFields(buf []byte) ([]protoField, bool) {
	var fields []protoField
	i := 0
	n := len(buf)
	for i < n {
		key, keyBig, next, ok := readVarint(buf, i)
		if !ok {
			return nil, false
		}
		i = next
		wt := int(key & 7)
		fn := key >> 3
		if keyBig {
			fn = bigField
		}
		if !keyBig && fn == 0 {
			return nil, false
		}
		switch wt {
		case 0:
			_, _, i2, ok := readVarint(buf, i)
			if !ok {
				return nil, false
			}
			fields = append(fields, protoField{fn, 0, buf[i:i2]})
			i = i2
		case 1:
			if i+8 > n {
				return nil, false
			}
			fields = append(fields, protoField{fn, 1, buf[i : i+8]})
			i += 8
		case 2:
			ln, lnBig, i2, ok := readVarint(buf, i)
			if !ok {
				return nil, false
			}
			// Python compares i + ln > n. An overlong length exceeds n too.
			if lnBig || ln > uint64(n-i2) {
				return nil, false
			}
			i = i2
			fields = append(fields, protoField{fn, 2, buf[i : i+int(ln)]})
			i += int(ln)
		case 5:
			if i+4 > n {
				return nil, false
			}
			fields = append(fields, protoField{fn, 5, buf[i : i+4]})
			i += 4
		default:
			// Groups (3, 4) and wire types 6, 7 are rejected, as Python does.
			return nil, false
		}
	}
	return fields, true
}

// appendField mirrors _encode_field. ok is false when fn is bigField, which
// cannot be re-encoded exactly.
func appendField(dst []byte, fn uint64, wt int, val []byte) ([]byte, bool) {
	if fn == bigField {
		return dst, false
	}
	dst = append(dst, writeVarint(fn<<3|uint64(wt))...)
	if wt == 2 {
		dst = append(dst, writeVarint(uint64(len(val)))...)
	}
	return append(dst, val...), true
}

// printableCount counts the characters that _PRINTABLE_RE matches (tab, LF,
// CR, and printable ASCII) plus every code point above 127.
func printableCount(s string) int {
	n := 0
	for _, r := range s {
		if r == '\t' || r == '\n' || r == '\r' || (r >= 0x20 && r <= 0x7e) || r > 127 {
			n++
		}
	}
	return n
}

// mostlyUTF8Text mirrors _mostly_utf8_text.
func mostlyUTF8Text(raw []byte) bool {
	if len(raw) == 0 || !utf8.Valid(raw) {
		return false
	}
	s := string(raw)
	if strings.Contains(s, "\x00") {
		return false
	}
	return float64(printableCount(s))/float64(utf8.RuneCountInString(s)) > 0.85
}

// looksLikeMessage mirrors _looks_like_message: the bytes parse as a message
// and are not text that only false-parses as fixed or varint fields.
func looksLikeMessage(raw []byte) bool {
	if len(raw) < 2 {
		return false
	}
	fields, ok := parseFields(raw)
	if !ok || len(fields) == 0 {
		return false
	}
	anyWT2 := false
	for _, f := range fields {
		if f.fn == 0 {
			return false
		}
		if f.wt == 2 {
			anyWT2 = true
		}
	}
	if !anyWT2 && mostlyUTF8Text(raw) {
		return false
	}
	return true
}

// shouldCrushContext mirrors _should_crush_context.
func shouldCrushContext(s string, aggressive bool) bool {
	if aggressive && pyLen(s) >= wireAggrMinStr {
		return true
	}
	if strings.Contains(s, "tool output line") || strings.Contains(s, "Certainly!") {
		return true
	}
	if pyLen(s) < wireMinStr {
		return false
	}
	if strings.Contains(s, "Context dump") || strings.Contains(s, "[...truncated...]") {
		return true
	}
	if strings.Count(s, "\n") >= 10 && pyLen(s) > 500 {
		return true
	}
	return pyLen(s) > 1200
}

// looksLikeModelCatalog mirrors _looks_like_model_catalog.
func looksLikeModelCatalog(s string) bool {
	if pyLen(s) < 400 {
		return false
	}
	markers := []string{".fast", ".thinking", "composer-", "claude-", "gpt-", "grok-", "gemini-"}
	hits := 0
	for _, m := range markers {
		if strings.Contains(s, m) {
			hits++
		}
	}
	return hits >= 4 && strings.Count(s, "\n")+strings.Count(s, ".") > 20
}

// crushFatPromptBlob mirrors _crush_fat_prompt_blob. ok is false where Python
// returns None.
func crushFatPromptBlob(s string) (string, bool) {
	if !strings.Contains(s, "tool output line") &&
		(!strings.Contains(s, "Certainly!") || strings.Count(s, "Certainly!") < 3) {
		return "", false
	}
	if strings.Contains(s, "Reply with exactly:") {
		head := pyStrip(strings.SplitN(s, "Context dump", 2)[0])
		return head + "\n\nContext dump (ignore):\n[crushed]", true
	}
	return "[crushed context]", true
}

// looksLikeJSON mirrors _looks_like_json.
func looksLikeJSON(s string) bool {
	t := pyLStrip(s)
	return strings.HasPrefix(t, "{") || strings.HasPrefix(t, "[")
}

// sanitizeControls replaces control characters that would break a JSON
// string if the text were embedded again.
func sanitizeControls(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		r, n := pyDecodeRune(s, i)
		if r >= 32 || r == '\n' || r == '\t' || r == '\r' {
			b.WriteString(s[i : i+n])
		} else {
			b.WriteByte(' ')
		}
		i += n
	}
	return b.String()
}

// safeTruncate mirrors _safe_truncate: a head and a tail with a count of the
// omitted characters between them.
func safeTruncate(s string, maxChars int) string {
	s = sanitizeControls(s)
	if pyLen(s) <= maxChars {
		return s
	}
	head := maxChars / 2
	tail := maxChars - head
	return pyPrefix(s, head) + " …[" + itoa(pyLen(s)-maxChars) + " chars]… " + pySuffix(s, tail)
}

// crushJSONText mirrors _crush_json_text. Text that is not valid JSON is
// truncated instead.
func crushJSONText(s string, aggressive bool) (string, error) {
	data, ok, err := jsonLoadsValue(s)
	if err != nil {
		return "", err
	}
	if !ok {
		limit := 1500
		if aggressive {
			limit = wireAggrMaxStr
		}
		return safeTruncate(s, limit), nil
	}
	crushed, err := CrushJSONValue(data, aggressive)
	if err != nil {
		return "", err
	}
	b, err := dumpsCompact(crushed)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// crushText mirrors _crush_text: the text crush for one str leaf.
func crushText(s string, aggressive bool) (string, error) {
	if IsErrorPayload(s, "", false) {
		return PreserveDiagnosticText(s, StacktraceCeiling), nil
	}
	if looksLikeJSON(s) {
		return crushJSONText(s, aggressive)
	}
	t := OptimizeText(s)
	if aggressive {
		var err error
		t, err = CrushText(t)
		if err != nil {
			return "", err
		}
		if pyLen(t) > wireAggrMaxStr {
			t = safeTruncate(t, wireAggrMaxStr)
		}
		return sanitizeControls(t), nil
	}
	switch n := pyLen(t); {
	case n > 2500:
		t = safeTruncate(t, 1800)
	case n > 900:
		t = safeTruncate(t, 850)
	}
	return sanitizeControls(t), nil
}

// crushTextBytes mirrors _crush_text_bytes. It returns the new bytes and true
// when the leaf got shorter. In aggressive mode with a seen set, a repeated
// body becomes a short duplicate stub.
func crushTextBytes(val []byte, aggressive bool, seen map[string]bool) ([]byte, bool, error) {
	if len(val) < wireAggrMinStr || !utf8.Valid(val) {
		return nil, false, nil
	}
	s := string(val)
	if IsErrorPayload(s, "", false) {
		nb, err := encodeUTF8Strict(PreserveDiagnosticText(s, StacktraceCeiling))
		if err != nil {
			return nil, false, err
		}
		return nb, len(nb) < len(val), nil
	}
	if !aggressive && !shouldCrushContext(s, false) {
		return nil, false, nil
	}
	if aggressive && seen != nil && pyLen(s) >= wireAggrDedupMin {
		digest := sha256Hex(val)[:12]
		if seen[digest] {
			var stub []byte
			if looksLikeJSON(s) {
				stub = []byte(`{"_dup":"` + digest + `","_n":` + itoa(pyLen(s)) + `}`)
			} else {
				stub = []byte("[dup " + digest + " len=" + itoa(pyLen(s)) + "]")
			}
			if len(stub) < len(val) {
				return stub, true, nil
			}
			return nil, false, nil
		}
		seen[digest] = true
	}
	text, err := crushText(s, aggressive)
	if err != nil {
		return nil, false, err
	}
	nb, err := encodeUTF8Strict(text)
	if err != nil {
		return nil, false, err
	}
	if len(nb) < len(val) {
		return nb, true, nil
	}
	return nil, false, nil
}

// appendPath returns path + (fn,) without aliasing path.
func appendPath(path []uint64, fn uint64) []uint64 {
	out := make([]uint64, len(path)+1)
	copy(out, path)
	out[len(path)] = fn
	return out
}

// pathEnds reports whether path[-2:] == (a, b).
func pathEnds(path []uint64, a, b uint64) bool {
	return len(path) >= 2 && path[len(path)-2] == a && path[len(path)-1] == b
}

// containsField reports whether v is in path.
func containsField(path []uint64, v uint64) bool {
	for _, p := range path {
		if p == v {
			return true
		}
	}
	return false
}

// aggressiveSweep mirrors _aggressive_sweep: a second pass that crushes any
// large UTF-8 leaf the field-aware pass left behind. A nested message is
// swept recursively, and the message itself is not collapsed.
func aggressiveSweep(data []byte, fieldPath []uint64, aggressive bool, seen map[string]bool, depth int) ([]byte, int, error) {
	if depth > maxProtoDepth {
		return data, 0, nil
	}
	fields, ok := parseFields(data)
	if !ok {
		return data, 0, nil
	}
	var out []byte
	saved := 0
	changed := false
	unencodable := false
	for _, f := range fields {
		val := f.val
		if f.wt == 2 {
			if looksLikeMessage(val) {
				nv, ns, err := aggressiveSweep(val, appendPath(fieldPath, f.fn), aggressive, seen, depth+1)
				if err != nil {
					return nil, 0, err
				}
				val = nv
				saved += ns
				if ns != 0 {
					changed = true
				}
			} else {
				// Python tests _is_text_blob and then calls the same crush in
				// both branches, so one call covers them.
				crushed, ok, err := crushTextBytes(val, true, seen)
				if err != nil {
					return nil, 0, err
				}
				if ok {
					saved += len(val) - len(crushed)
					val = crushed
					changed = true
				}
			}
		}
		var enc bool
		out, enc = appendField(out, f.fn, f.wt, val)
		if !enc {
			unencodable = true
		}
	}
	if !changed {
		return data, saved, nil
	}
	if unencodable {
		return nil, 0, ErrUnsupported
	}
	return out, saved, nil
}

// parseLineWrapper mirrors _parse_line_wrapper: a {1: varint, 2: string}
// message, returned as (lineno bytes, text bytes).
func parseLineWrapper(fv []byte) (lineno, text []byte, ok bool) {
	if !looksLikeMessage(fv) {
		return nil, nil, false
	}
	nested, ok := parseFields(fv)
	if !ok {
		return nil, nil, false
	}
	hasText := false
	for _, f := range nested {
		switch {
		case f.fn == 1 && f.wt == 0:
			lineno = f.val
		case f.fn == 2 && f.wt == 2:
			if bytes.IndexByte(f.val, 0) >= 0 || !utf8.Valid(f.val) {
				return nil, nil, false
			}
			text = f.val
			hasText = true
		case f.wt == 0:
			continue
		default:
			return nil, nil, false
		}
	}
	if !hasText || len(lineno) == 0 {
		return nil, nil, false
	}
	return lineno, text, true
}

// encodeLineWrapper mirrors _encode_line_wrapper.
func encodeLineWrapper(lineno, text []byte) []byte {
	out, _ := appendField(nil, 1, 0, lineno)
	out, _ = appendField(out, 2, 2, text)
	return out
}

// tryCollapseLineMessage mirrors _try_collapse_line_message. It is the file
// crush for a re-read stub. The first read of a long file caps every line,
// keeping the structure. Dropping the line wrappers would hang tool use, so
// they stay. A file under FileReadLineCap lines is never capped.
func tryCollapseLineMessage(val []byte, aggressive bool, seen map[string]bool) ([]byte, bool, error) {
	if !aggressive || len(val) < 800 {
		return nil, false, nil
	}
	fields, ok := parseFields(val)
	if !ok {
		return nil, false, nil
	}

	var filename []byte
	lineFn := uint64(2)
	type lineWrap struct{ lineno, text []byte }
	var wrappers []lineWrap
	var other []protoField
	for _, f := range fields {
		if f.wt != 2 {
			other = append(other, f)
			continue
		}
		if f.fn == 1 && !looksLikeMessage(f.val) && len(f.val) > 0 && len(f.val) < 260 && len(wrappers) == 0 {
			if utf8.Valid(f.val) {
				filename = f.val
				continue
			}
		}
		ln, tx, ok := parseLineWrapper(f.val)
		if !ok {
			other = append(other, f)
			continue
		}
		wrappers = append(wrappers, lineWrap{ln, tx})
		lineFn = f.fn
	}

	if len(wrappers) < 24 || len(wrappers) < FileReadLineCap {
		return nil, false, nil
	}
	texts := make([]string, len(wrappers))
	for i, w := range wrappers {
		texts[i] = string(w.text)
	}
	joined := strings.Join(texts, "\n")
	if pyLen(joined) < 600 {
		return nil, false, nil
	}

	keyMaterial := make([]byte, 0, len(filename)+1+len(joined))
	keyMaterial = append(keyMaterial, filename...)
	keyMaterial = append(keyMaterial, 0)
	keyMaterial = append(keyMaterial, joined...)
	digest := sha256Hex(keyMaterial)[:12]
	key := "file:" + digest
	pending := "~" + key

	prefix := func() ([]byte, bool) {
		var out []byte
		enc := true
		if len(filename) > 0 {
			var ok bool
			out, ok = appendField(out, 1, 2, filename)
			enc = enc && ok
		}
		for _, f := range other {
			var ok bool
			out, ok = appendField(out, f.fn, f.wt, f.val)
			enc = enc && ok
		}
		return out, enc
	}

	if seen != nil && seen[key] {
		// The same file was already crushed: one duplicate stub for the body.
		out, enc := prefix()
		stub := []byte("[dup file " + digest + " lines=" + itoa(len(wrappers)) + " len=" + itoa(pyLen(joined)) + "]")
		var ok bool
		out, ok = appendField(out, lineFn, 2, encodeLineWrapper(wrappers[0].lineno, stub))
		enc = enc && ok
		if len(out) < len(val)-50 {
			if !enc {
				return nil, false, ErrUnsupported
			}
			return out, true, nil
		}
		return nil, false, nil
	}
	if seen != nil {
		seen[pending] = true
	}

	n := len(wrappers)
	out, enc := prefix()
	changed := false
	for i, w := range wrappers {
		var capN int
		switch {
		case i < 40:
			capN = 72
		case i >= n-20:
			capN = 56
		default:
			capN = 32
		}
		nw := w.text
		s := string(w.text)
		if pyLen(s) > capN {
			nw = []byte(safeTruncate(s, capN))
			changed = true
		}
		var ok bool
		out, ok = appendField(out, lineFn, 2, encodeLineWrapper(w.lineno, nw))
		enc = enc && ok
	}
	if changed && len(out) < len(val)-80 {
		if !enc {
			return nil, false, ErrUnsupported
		}
		return out, true, nil
	}
	return nil, false, nil
}

// CommitFileSeen mirrors commit_file_seen: after a Connect body is fully
// crushed, the provisional "~file:" marks become real file keys.
func CommitFileSeen(seen map[string]bool) {
	if len(seen) == 0 {
		return
	}
	keys := make([]string, 0, len(seen))
	for k := range seen {
		keys = append(keys, k)
	}
	for _, x := range keys {
		if strings.HasPrefix(x, "~file:") {
			delete(seen, x)
			seen[x[1:]] = true
		}
	}
}

// minimalSkillEntry mirrors _minimal_skill_entry: keep a short skill id and
// drop the MCP prose bundled in the request context.
func minimalSkillEntry(val []byte) []byte {
	fields, ok := parseFields(val)
	if !ok {
		return val
	}
	name := []byte("[skill]")
	for _, f := range fields {
		if f.fn == 1 && f.wt == 2 && len(f.val) > 0 {
			name = f.val[:min(28, len(f.val))]
			break
		}
	}
	out, _ := appendField(nil, 1, 2, name)
	return out
}

// crushAgentRulesBlock mirrors _crush_agent_rules_block.
func crushAgentRulesBlock(s string, aggressive bool) string {
	limit := 200
	if aggressive {
		limit = 80
	}
	if pyLen(s) <= limit {
		return s
	}
	return "[ctx]"
}

// CrushProtobufMessage mirrors crush_protobuf_message at the top level. It
// returns the message and the bytes saved inside it. seen may be nil.
func CrushProtobufMessage(data []byte, aggressive bool, seen map[string]bool) ([]byte, int, error) {
	return crushProto(data, nil, aggressive, seen, 0)
}

// crushProto mirrors crush_protobuf_message with its field path and depth.
// The branch order matches the Python elif chain: once a field matches a
// branch, later branches are not tried, even if the crush does nothing.
func crushProto(data []byte, fieldPath []uint64, aggressive bool, seen map[string]bool, depth int) ([]byte, int, error) {
	if depth > maxProtoDepth {
		return data, 0, nil
	}
	if aggressive && seen == nil {
		seen = map[string]bool{}
	}
	fields, ok := parseFields(data)
	if !ok {
		return data, 0, nil
	}

	skillLimit, mcpLimit, mcpNestedLimit, skillDescLimit, rulesLimit := 80, 500, 200, 80, 200
	if aggressive {
		skillLimit, mcpLimit, mcpNestedLimit, skillDescLimit, rulesLimit = 40, 50, 80, 40, 80
	}

	var out []byte
	saved := 0
	changed := false
	unencodable := false
	for _, f := range fields {
		fn, wt, val := f.fn, f.wt, f.val
		fpath := appendPath(fieldPath, fn)
		if wt == 2 {
			switch {
			case fn == 29 && len(val) > skillLimit:
				nv := minimalSkillEntry(val)
				if len(nv) < len(val) {
					saved += len(val) - len(nv)
					val = nv
					changed = true
				}
			case fn == 34 && len(val) > mcpLimit:
				stub, _ := appendField(nil, 1, 0, []byte{1})
				saved += len(val) - len(stub)
				val = stub
				changed = true
			case pathEnds(fpath, 34, 2) && len(val) > mcpNestedLimit:
				nv := []byte("\n\x04[mcp]")
				if len(nv) < len(val) {
					saved += len(val) - len(nv)
					val = nv
					changed = true
				}
			case pathEnds(fpath, 29, 3) && len(val) > skillDescLimit:
				// _crush_agent_skill_desc ignores its input, so the decode is moot.
				nv := []byte("[skill]")
				if len(nv) < len(val) {
					saved += len(val) - len(nv)
					val = nv
					changed = true
				}
			case pathEnds(fpath, 14, 2) && len(val) > rulesLimit:
				if utf8.Valid(val) {
					nv := []byte(crushAgentRulesBlock(string(val), aggressive))
					if len(nv) < len(val) {
						saved += len(val) - len(nv)
						val = nv
						changed = true
					}
				}
			case looksLikeMessage(val):
				collapsed, ok, err := tryCollapseLineMessage(val, aggressive, seen)
				if err != nil {
					return nil, 0, err
				}
				if ok {
					saved += len(val) - len(collapsed)
					val = collapsed
					changed = true
				} else {
					nested, ns, err := crushProto(val, fpath, aggressive, seen, depth+1)
					if err != nil {
						return nil, 0, err
					}
					if !bytes.Equal(nested, val) {
						val = nested
						saved += ns
						changed = true
					}
				}
			default:
				if utf8.Valid(val) {
					s := string(val)
					var nw []byte
					have := false
					if fat, isFat := crushFatPromptBlob(s); isFat {
						nw, have = []byte(fat), true
					} else if looksLikeModelCatalog(s) {
						nw, have = []byte("[models]"), true
					} else if shouldCrushContext(s, aggressive) {
						x, isNew, err := crushTextBytes(val, aggressive, seen)
						if err != nil {
							return nil, 0, err
						}
						nw, have = x, isNew
					} else if pyLen(s) > 300 && containsField(fpath, 29) {
						nw, have = []byte("[skill]"), true
					}
					if have && len(nw) < len(val) {
						saved += len(val) - len(nw)
						val = nw
						changed = true
					}
				}
			}
		}
		var enc bool
		out, enc = appendField(out, fn, wt, val)
		if !enc {
			unencodable = true
		}
	}

	result := data
	if changed {
		if unencodable {
			return nil, 0, ErrUnsupported
		}
		result = out
	}
	if aggressive && len(result) > 500 {
		swept, sweepSaved, err := aggressiveSweep(result, fieldPath, aggressive, seen, depth)
		if err != nil {
			return nil, 0, err
		}
		if sweepSaved != 0 {
			result = swept
			saved += sweepSaved
		}
	}
	return result, saved, nil
}

// mapFramePayloads mirrors map_frame_payloads with skip_end_stream=True. It
// applies fn to each frame payload, inflating gzip-flagged frames first. If the
// body changes and a gzip frame was re-compressed, it declines (see the package
// comment). A body that is not framed is passed to fn whole.
func mapFramePayloads(body []byte, fn func([]byte) ([]byte, error)) ([]byte, bool, error) {
	frames, ok := IterFrames(body)
	if !ok {
		nw, err := fn(body)
		if err != nil {
			return nil, false, err
		}
		return nw, !bytes.Equal(nw, body), nil
	}
	out := make([]Frame, 0, len(frames))
	changed := false
	for _, f := range frames {
		if f.Flags&0x02 != 0 {
			out = append(out, f)
			continue
		}
		raw := f.Payload
		compressed := f.Flags&0x01 != 0
		if compressed {
			d, err := gzipDecompress(f.Payload)
			switch {
			case err == nil:
				raw = d
			case isGzipOSError(err):
				// Python keeps the payload as is and drops the compressed flag.
				compressed = false
			default:
				return nil, false, err
			}
		}
		mapped, err := fn(raw)
		if err != nil {
			return nil, false, err
		}
		if !bytes.Equal(mapped, raw) {
			changed = true
		}
		if compressed {
			// Re-compressed at level 6 like Python. Byte-identical output is not
			// possible (Python's gzip header carries the current time), so parity
			// holds at the decoded payload level; see connect_frames.go.
			out = append(out, Frame{Flags: f.Flags | 0x01, Payload: gzipCompress(mapped)})
		} else {
			out = append(out, Frame{Flags: f.Flags, Payload: mapped})
		}
	}
	if !changed {
		return body, false, nil
	}
	enc, err := EncodeFrames(out)
	if err != nil {
		return nil, false, err
	}
	return enc, true, nil
}

// CrushConnectBody mirrors crush_connect_body: a framed or raw Connect body.
// It returns the body and the protobuf bytes saved. seen may be nil.
func CrushConnectBody(body []byte, aggressive bool, seen map[string]bool) ([]byte, int, error) {
	if aggressive && seen == nil {
		seen = map[string]bool{}
	}
	mapped, changed, err := mapFramePayloads(body, func(raw []byte) ([]byte, error) {
		crushed, _, err := CrushProtobufMessage(raw, aggressive, seen)
		return crushed, err
	})
	if err != nil {
		return nil, 0, err
	}
	if changed {
		CommitFileSeen(seen)
		return mapped, max(0, len(body)-len(mapped)), nil
	}
	crushed, saved, err := CrushProtobufMessage(body, aggressive, seen)
	if err != nil {
		return nil, 0, err
	}
	CommitFileSeen(seen)
	if !bytes.Equal(crushed, body) {
		return crushed, saved, nil
	}
	return crushed, 0, nil
}

// DrainCrushFrames mirrors drain_crush_frames: it pulls the complete Connect
// frames out of buf, crushes each one, and returns the rest of buf.
func DrainCrushFrames(buf []byte, aggressive bool, seen map[string]bool) ([]byte, [][]byte, error) {
	var out [][]byte
	i := 0
	n := len(buf)
	for i+5 <= n {
		ln := int(binary.BigEndian.Uint32(buf[i+1 : i+5]))
		end := i + 5 + ln
		if end > n {
			break
		}
		crushed, _, err := CrushConnectBody(buf[i:end], aggressive, seen)
		if err != nil {
			return nil, nil, err
		}
		out = append(out, crushed)
		i = end
	}
	return buf[i:], out, nil
}

// CrushJSONValue mirrors crush_json_value: it crushes the long strings in a
// decoded JSON value and keeps the structure.
func CrushJSONValue(node any, aggressive bool) (any, error) {
	switch v := node.(type) {
	case string:
		minLen := wireMinStr
		if aggressive {
			minLen = wireAggrMinStr
		}
		if pyLen(v) < minLen {
			return v, nil
		}
		s, err := crushText(v, aggressive)
		if err != nil {
			return nil, err
		}
		return s, nil
	case []any:
		out := make([]any, len(v))
		for i, x := range v {
			c, err := CrushJSONValue(x, aggressive)
			if err != nil {
				return nil, err
			}
			out[i] = c
		}
		return out, nil
	case *pyjson.Object:
		out := pyjson.NewObject()
		for _, k := range v.Keys() {
			c, err := CrushJSONValue(objGet(v, k), aggressive)
			if err != nil {
				return nil, err
			}
			out.Set(k, c)
		}
		return out, nil
	}
	return node, nil
}

// nonUTF8JSONCodec reports whether CPython's json.loads(bytes) would decode
// with a UTF-16 or UTF-32 codec (detect_encoding). The port does not decode
// those, so it declines.
func nonUTF8JSONCodec(b []byte) bool {
	switch {
	case bytes.HasPrefix(b, []byte{0, 0, 0xFE, 0xFF}), bytes.HasPrefix(b, []byte{0xFF, 0xFE, 0, 0}):
		return true
	case bytes.HasPrefix(b, []byte{0xFE, 0xFF}), bytes.HasPrefix(b, []byte{0xFF, 0xFE}):
		return true
	case bytes.HasPrefix(b, []byte{0xEF, 0xBB, 0xBF}):
		return false
	}
	if len(b) >= 4 || len(b) == 2 {
		return b[0] == 0 || b[1] == 0
	}
	return false
}

// jsonLoadsBytes mirrors json.loads(bytes). ok is false where Python raises
// ValueError (including invalid UTF-8).
func jsonLoadsBytes(body []byte) (any, bool, error) {
	if nonUTF8JSONCodec(body) {
		return nil, false, ErrUnsupported
	}
	v, err := pyjson.Loads(body)
	if err == nil {
		return v, true, nil
	}
	if strings.Contains(err.Error(), "maximum recursion depth") {
		return nil, false, ErrUnsupported
	}
	return nil, false, nil
}

// CrushJSONBody mirrors crush_json_body. ok is false where Python returns None
// (not valid JSON, or not an object or array).
func CrushJSONBody(body []byte, aggressive bool) ([]byte, bool, error) {
	obj, ok, err := jsonLoadsBytes(body)
	if err != nil || !ok {
		return nil, false, err
	}
	switch obj.(type) {
	case *pyjson.Object, []any:
	default:
		return nil, false, nil
	}
	crushed, err := CrushJSONValue(obj, aggressive)
	if err != nil {
		return nil, false, err
	}
	b, err := dumpsCompact(crushed)
	if err != nil {
		return nil, false, err
	}
	return b, true, nil
}

// isJSONStart reports whether body starts with '{' or '[' after ASCII
// whitespace, as body.lstrip()[:1] does.
func isJSONStart(body []byte) bool {
	t := bytes.TrimLeft(body, " \t\n\r\v\f")
	return len(t) > 0 && (t[0] == '{' || t[0] == '[')
}

// CrushWireBody mirrors crush_wire_body. It returns the new body, the
// category tags, and the bytes saved.
func CrushWireBody(body []byte, path, contentType string, aggressive bool) ([]byte, []string, int, error) {
	if len(body) < 200 {
		return body, nil, 0, nil
	}
	ctype := pyLower(contentType)
	pathL := pyLower(strings.TrimLeft(path, "/"))

	if strings.Contains(ctype, "connect+proto") || strings.Contains(ctype, "application/grpc") ||
		strings.Contains(ctype, "protobuf") || strings.HasPrefix(pathL, "aiserver.") || strings.HasPrefix(pathL, "agent.v1.") {
		nw, saved, err := CrushConnectBody(body, aggressive, nil)
		if err != nil {
			return nil, nil, 0, err
		}
		if saved > 0 || (!bytes.Equal(nw, body) && len(nw) < len(body)) {
			tag := "wire_proto_crush"
			if aggressive {
				tag = "wire_proto_crush_aggressive"
			}
			return nw, []string{tag}, max(saved, len(body)-len(nw)), nil
		}
		dumpMiss(body, path, contentType)
		return body, nil, 0, nil
	}

	if strings.Contains(ctype, "connect+json") || isJSONStart(body) {
		nw, ok, err := CrushJSONBody(body, aggressive)
		if err != nil {
			return nil, nil, 0, err
		}
		if ok && len(nw) < len(body) {
			return nw, []string{"wire_json_crush"}, len(body) - len(nw), nil
		}
		if len(body) > 800 {
			dumpMiss(body, path, contentType)
		}
	}
	return body, nil, 0, nil
}

// CaptureAgentFrame mirrors capture_agent_frame. It saves a frame whose crush
// saved under 70 percent, at most eight per process, when a capture dir is set.
func CaptureAgentFrame(raw, crushed []byte, aggressive bool) {
	if len(raw) < 2000 {
		return
	}
	saved := len(raw) - len(crushed)
	pct := float64(saved) / float64(len(raw)) * 100
	if pct >= 70 {
		return
	}
	wireCapture.mu.Lock()
	if wireCapture.agentCaptures >= 8 {
		wireCapture.mu.Unlock()
		return
	}
	wireCapture.agentCaptures++
	n := wireCapture.agentCaptures
	dir := wireCapture.dir
	wireCapture.mu.Unlock()
	if dir == "" {
		return
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	tag := fmt.Sprintf("%d-%d", len(raw), int(pct))
	base := filepath.Join(dir, fmt.Sprintf("%d-%s", n, tag))
	if err := os.WriteFile(base+"-raw.bin", raw[:min(len(raw), 500000)], 0o644); err != nil {
		return
	}
	if err := os.WriteFile(base+"-crushed.bin", crushed[:min(len(crushed), 500000)], 0o644); err != nil {
		return
	}
	pyAggr := "False"
	if aggressive {
		pyAggr = "True"
	}
	meta := fmt.Sprintf("raw=%d crushed=%d saved=%d pct=%.1f aggr=%s\n", len(raw), len(crushed), saved, pct, pyAggr)
	_ = os.WriteFile(base+".meta.txt", []byte(meta), 0o644)
}

// dumpMiss mirrors _dump_miss: up to five unhandled bodies are saved for
// analysis, except analytics and trackevents paths.
func dumpMiss(body []byte, path, contentType string) {
	wireCapture.mu.Lock()
	if wireCapture.dumped >= 5 || len(body) < 400 {
		wireCapture.mu.Unlock()
		return
	}
	lp := pyLower(path)
	if strings.Contains(lp, "analytics") || strings.Contains(lp, "trackevents") {
		wireCapture.mu.Unlock()
		return
	}
	wireCapture.dumped++
	n := wireCapture.dumped
	dir := wireCapture.dir
	wireCapture.mu.Unlock()
	if dir == "" {
		return
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	if err := os.WriteFile(filepath.Join(dir, fmt.Sprintf("wire-miss-%d.bin", n)), body[:min(len(body), 400000)], 0o644); err != nil {
		return
	}
	head := body[:min(len(body), 48)]
	meta := fmt.Sprintf("path=%s\nctype=%s\nlen=%d\nhead=%s\n", path, contentType, len(body), hex.EncodeToString(head))
	_ = os.WriteFile(filepath.Join(dir, fmt.Sprintf("wire-miss-%d.meta.txt", n)), []byte(meta), 0o644)
}
