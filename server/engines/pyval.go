// Package engines ports the Gatekeep Python engines (backend/engines/*.py) to Go.
//
// pyval.go holds the CPython value semantics the ports depend on: truthiness,
// str() and repr(), code-point length and slicing (Python str indexes code
// points, Go strings index bytes), str.strip/lower/splitlines, == and dict-key
// equality across int/float/bool, and iteration. Lone surrogates arrive as
// WTF-8 from pyjson, so every helper here treats a 3-byte surrogate as one
// code point.
package engines

import (
	"fmt"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// pyError stands in for a Python exception. Callers only need to know that
// Python would have raised, so the message is informational.
func pyError(kind, msg string) error {
	return fmt.Errorf("%s: %s", kind, msg)
}

// pyTruthy mirrors bool(v) for JSON values.
func pyTruthy(v any) bool {
	switch t := v.(type) {
	case nil:
		return false
	case bool:
		return t
	case int:
		return t != 0
	case int64:
		return t != 0
	case *big.Int:
		return t.Sign() != 0
	case float64:
		return t != 0 // NaN is truthy, as in Python
	case string:
		return t != ""
	case []any:
		return len(t) > 0
	case *pyjson.Object:
		return t.Len() > 0
	}
	return true
}

// pyIsInt mirrors isinstance(v, int). bool is a subclass of int in Python.
func pyIsInt(v any) bool {
	switch v.(type) {
	case bool, int, int64, *big.Int:
		return true
	}
	return false
}

// pyIsNumber mirrors isinstance(v, (int, float)).
func pyIsNumber(v any) bool {
	if _, ok := v.(float64); ok {
		return true
	}
	return pyIsInt(v)
}

// pyIntOf converts int-like values (bool included) to *big.Int.
func pyIntOf(v any) (*big.Int, bool) {
	switch t := v.(type) {
	case bool:
		if t {
			return big.NewInt(1), true
		}
		return big.NewInt(0), true
	case int:
		return big.NewInt(int64(t)), true
	case int64:
		return big.NewInt(t), true
	case *big.Int:
		return t, true
	}
	return nil, false
}

// pyIntConv mirrors int(v) for int-like and float values. NaN and infinities
// raise in Python.
func pyIntConv(v any) (*big.Int, error) {
	if f, ok := v.(float64); ok {
		if math.IsNaN(f) {
			return nil, pyError("ValueError", "cannot convert float NaN to integer")
		}
		if math.IsInf(f, 0) {
			return nil, pyError("OverflowError", "cannot convert float infinity to integer")
		}
		bf := new(big.Float).SetFloat64(f)
		n, _ := bf.Int(nil) // truncates toward zero, like int()
		return n, nil
	}
	n, ok := pyIntOf(v)
	if !ok {
		return nil, pyError("TypeError", "int() argument")
	}
	return n, nil
}

// pyFloatRepr mirrors repr(float). Unlike json.dumps, repr uses nan/inf.
func pyFloatRepr(f float64) string {
	switch {
	case math.IsNaN(f):
		return "nan"
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	}
	return pyjson.FloatRepr(f)
}

// pyStr mirrors str(v) for JSON values.
func pyStr(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return pyRepr(v)
}

// pyRepr mirrors repr(v) for JSON values (dict and list use element reprs).
func pyRepr(v any) string {
	switch t := v.(type) {
	case nil:
		return "None"
	case bool:
		if t {
			return "True"
		}
		return "False"
	case int:
		return strconv.Itoa(t)
	case int64:
		return strconv.FormatInt(t, 10)
	case *big.Int:
		return t.String()
	case float64:
		return pyFloatRepr(t)
	case string:
		return pyStrRepr(t)
	case []any:
		parts := make([]string, len(t))
		for i, x := range t {
			parts[i] = pyRepr(x)
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case *pyjson.Object:
		var b strings.Builder
		b.WriteByte('{')
		for i, k := range t.Keys() {
			if i > 0 {
				b.WriteString(", ")
			}
			val, _ := t.Get(k)
			b.WriteString(pyStrRepr(k))
			b.WriteString(": ")
			b.WriteString(pyRepr(val))
		}
		b.WriteByte('}')
		return b.String()
	}
	return ""
}

// pyStrRepr mirrors repr(str): quote choice, escapes, and \x/\u/\U escapes
// for non-printable code points.
func pyStrRepr(s string) string {
	quote := byte('\'')
	if strings.IndexByte(s, '\'') >= 0 && strings.IndexByte(s, '"') < 0 {
		quote = '"'
	}
	var b strings.Builder
	b.WriteByte(quote)
	for i := 0; i < len(s); {
		r, n := pyDecodeRune(s, i)
		raw := s[i : i+n]
		i += n
		switch {
		case r == rune(quote) || r == '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r < ' ' || r == 0x7f:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r < 0x7f:
			b.WriteRune(r)
		case pyIsPrintable(r):
			b.WriteString(raw)
		case r <= 0xff:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r <= 0xffff:
			fmt.Fprintf(&b, `\u%04x`, r)
		default:
			fmt.Fprintf(&b, `\U%08x`, r)
		}
	}
	b.WriteByte(quote)
	return b.String()
}

// pyIsPrintable mirrors str.isprintable for one code point: everything except
// the Other (C*) and Separator (Z*) categories, with ' ' as the only space.
func pyIsPrintable(r rune) bool {
	if r == ' ' {
		return true
	}
	return unicode.IsGraphic(r) && !unicode.Is(unicode.Zs, r)
}

// pyDecodeRune returns the code point at s[i] and its byte width. A WTF-8
// encoded surrogate (ED A0..BF xx) decodes to one code point of 3 bytes.
func pyDecodeRune(s string, i int) (rune, int) {
	c := s[i]
	if c < 0x80 {
		return rune(c), 1
	}
	if isSurrogateAt(s, i) {
		r := rune(c&0x0f)<<12 | rune(s[i+1]&0x3f)<<6 | rune(s[i+2]&0x3f)
		return r, 3
	}
	return utf8.DecodeRuneInString(s[i:])
}

// isSurrogateAt reports whether s[i:] starts with a WTF-8 lone surrogate.
func isSurrogateAt(s string, i int) bool {
	return s[i] == 0xED && i+2 < len(s) && s[i+1] >= 0xA0 && s[i+1] <= 0xBF && s[i+2] >= 0x80 && s[i+2] <= 0xBF
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= 0x80 {
			return false
		}
	}
	return true
}

// pyRuneOffsets returns the byte offset of every code point plus len(s).
func pyRuneOffsets(s string) []int {
	offs := make([]int, 0, len(s)+1)
	for i := 0; i < len(s); {
		offs = append(offs, i)
		_, n := pyDecodeRune(s, i)
		i += n
	}
	return append(offs, len(s))
}

// pyLen mirrors len(str) in code points.
func pyLen(s string) int {
	if isASCII(s) {
		return len(s)
	}
	return len(pyRuneOffsets(s)) - 1
}

// pyPrefix mirrors s[:n] in code points, including negative n.
func pyPrefix(s string, n int) string {
	if isASCII(s) {
		if n < 0 {
			n += len(s)
			if n < 0 {
				n = 0
			}
		}
		if n >= len(s) {
			return s
		}
		return s[:n]
	}
	offs := pyRuneOffsets(s)
	count := len(offs) - 1
	if n < 0 {
		n += count
		if n < 0 {
			n = 0
		}
	}
	if n >= count {
		return s
	}
	return s[:offs[n]]
}

// pySuffix mirrors s[-n:] in code points. Note the sign: pySuffix(s, 3) is
// the last three code points, and pySuffix(s, -2) is s[2:] (Python takes
// -(-2) = 2), which the error-guard head/tail split can produce.
func pySuffix(s string, n int) string {
	if n == 0 {
		return s
	}
	offs := pyRuneOffsets(s)
	count := len(offs) - 1
	var start int
	if n > 0 {
		start = count - n
		if start < 0 {
			start = 0
		}
	} else {
		start = -n
		if start > count {
			start = count
		}
	}
	return s[offs[start]:]
}

// pyFloorDiv mirrors // for integers (rounds toward negative infinity).
func pyFloorDiv(a, b int) int {
	q := a / b
	if (a%b != 0) && ((a < 0) != (b < 0)) {
		q--
	}
	return q
}

// pyIsSpace mirrors str.isspace() for one code point. It differs from
// unicode.IsSpace: \x1c-\x1f count as whitespace in Python.
func pyIsSpace(r rune) bool {
	switch {
	case r <= 0x7f:
		return r == ' ' || (r >= 0x09 && r <= 0x0d) || (r >= 0x1c && r <= 0x1f)
	case r == 0x85, r == 0xa0, r == 0x1680, r >= 0x2000 && r <= 0x200a,
		r == 0x2028, r == 0x2029, r == 0x202f, r == 0x205f, r == 0x3000:
		return true
	}
	return false
}

// pyStrip mirrors str.strip() with no arguments.
func pyStrip(s string) string {
	return strings.TrimFunc(s, pyIsSpace)
}

// pyRStrip mirrors str.rstrip() with no arguments.
func pyRStrip(s string) string {
	return strings.TrimRightFunc(s, pyIsSpace)
}

// pyLStrip mirrors str.lstrip() with no arguments.
func pyLStrip(s string) string {
	return strings.TrimLeftFunc(s, pyIsSpace)
}

// pyLower mirrors str.lower(). U+0130 lowers to two code points in Python,
// which matters for substring tests such as "exception" in s.lower(). Final
// sigma is not mirrored: it only ever yields a non-ASCII code point, so no
// ASCII keyword test can see the difference.
func pyLower(s string) string {
	if isASCII(s) {
		return strings.ToLower(s)
	}
	var b strings.Builder
	for i := 0; i < len(s); {
		r, n := pyDecodeRune(s, i)
		switch {
		case r == 0x130:
			b.WriteString("i̇")
		case r >= 0xD800 && r <= 0xDFFF:
			b.WriteString(s[i : i+n])
		default:
			b.WriteRune(unicode.ToLower(r))
		}
		i += n
	}
	return b.String()
}

// pySplitLines mirrors str.splitlines() (no keepends). The Python separators
// are wider than "\n": \r, \r\n, \v, \f, \x1c-\x1e, \x85,   and  .
func pySplitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); {
		r, n := pyDecodeRune(s, i)
		sep := 0
		switch r {
		case '\n', '\v', '\f', 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029:
			sep = n
		case '\r':
			sep = n
			if i+1 < len(s) && s[i+1] == '\n' {
				sep = 2
			}
		}
		if sep > 0 {
			out = append(out, s[start:i])
			i += sep
			start = i
			continue
		}
		i += n
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

// pyDecodeIgnore mirrors bytes.decode("utf-8", errors="ignore").
func pyDecodeIgnore(b []byte) string {
	return strings.ToValidUTF8(string(b), "")
}

// pyEncodeIgnore mirrors str.encode("utf-8", errors="ignore"): lone
// surrogates are dropped, everything else is emitted as UTF-8.
func pyEncodeIgnore(s string) []byte {
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); {
		if isSurrogateAt(s, i) {
			i += 3
			continue
		}
		out = append(out, s[i])
		i++
	}
	return out
}

// pyIterItems mirrors `for x in v`: lists yield items, dicts their keys,
// strings their code points. Other values raise TypeError.
func pyIterItems(v any) ([]any, error) {
	switch t := v.(type) {
	case []any:
		return t, nil
	case *pyjson.Object:
		keys := t.Keys()
		out := make([]any, len(keys))
		for i, k := range keys {
			out[i] = k
		}
		return out, nil
	case string:
		var out []any
		for i := 0; i < len(t); {
			_, n := pyDecodeRune(t, i)
			out = append(out, t[i:i+n])
			i += n
		}
		return out, nil
	}
	return nil, pyError("TypeError", "object is not iterable")
}

// pyIterOr mirrors `for x in v or []`.
func pyIterOr(v any) ([]any, error) {
	if !pyTruthy(v) {
		return nil, nil
	}
	return pyIterItems(v)
}

// objGet returns the value for key, or nil like dict.get(key).
func objGet(o *pyjson.Object, key string) any {
	v, _ := o.Get(key)
	return v
}

// lowerOrEmpty mirrors (v or "").lower(): falsy values become "", strings
// are lowered, and any other truthy value raises AttributeError.
func lowerOrEmpty(v any) (string, error) {
	if !pyTruthy(v) {
		return "", nil
	}
	s, ok := v.(string)
	if !ok {
		return "", pyError("AttributeError", "object has no attribute 'lower'")
	}
	return pyLower(s), nil
}

// strOrEmpty mirrors (v or "") where the result must be a str (e.g. before
// .strip()). Truthy non-strings raise AttributeError.
func strOrEmpty(v any) (string, error) {
	if !pyTruthy(v) {
		return "", nil
	}
	s, ok := v.(string)
	if !ok {
		return "", pyError("AttributeError", "object has no attribute 'strip'")
	}
	return s, nil
}

// pyEqual mirrors == for JSON values.
func pyEqual(a, b any) bool {
	if pyIsNumber(a) && pyIsNumber(b) {
		c, ok := pyNumCmp(a, b)
		return ok && c == 0
	}
	switch x := a.(type) {
	case nil:
		return b == nil
	case string:
		y, ok := b.(string)
		return ok && x == y
	case []any:
		y, ok := b.([]any)
		if !ok || len(x) != len(y) {
			return false
		}
		for i := range x {
			if !pyEqual(x[i], y[i]) {
				return false
			}
		}
		return true
	case *pyjson.Object:
		y, ok := b.(*pyjson.Object)
		if !ok || x.Len() != y.Len() {
			return false
		}
		for _, k := range x.Keys() {
			yv, found := y.Get(k)
			if !found || !pyEqual(objGet(x, k), yv) {
				return false
			}
		}
		return true
	}
	return false
}

// pyNumCmp compares two numeric JSON values exactly. ok is false when either
// side is NaN (Python's comparisons then yield False, not an error).
func pyNumCmp(a, b any) (int, bool) {
	fa, aF := a.(float64)
	fb, bF := b.(float64)
	if (aF && math.IsNaN(fa)) || (bF && math.IsNaN(fb)) {
		return 0, false
	}
	if (aF && math.IsInf(fa, 0)) || (bF && math.IsInf(fb, 0)) {
		sign := func(v any, f float64, isF bool) int {
			if isF && math.IsInf(f, 1) {
				return 1
			}
			if isF && math.IsInf(f, -1) {
				return -1
			}
			return 0 // finite: compared below via the other side
		}
		sa, sb := sign(a, fa, aF), sign(b, fb, bF)
		switch {
		case sa != 0 && sb != 0:
			return sa - sb, true
		case sa != 0:
			return sa, true
		default:
			return -sb, true
		}
	}
	ra, rb := toRat(a), toRat(b)
	return ra.Cmp(rb), true
}

// toRat converts a finite numeric JSON value to an exact rational.
func toRat(v any) *big.Rat {
	if f, ok := v.(float64); ok {
		r := new(big.Rat)
		r.SetFloat64(f)
		return r
	}
	n, _ := pyIntOf(v)
	return new(big.Rat).SetInt(n)
}

// pyKey mirrors dict-key identity. Numbers compare by value (1, 1.0 and True
// are one key). ok is false for unhashable values (list, dict).
func pyKey(v any) (string, bool) {
	switch t := v.(type) {
	case nil:
		return "N", true
	case string:
		return "S" + t, true
	case float64:
		if !math.IsNaN(t) && !math.IsInf(t, 0) && t == math.Trunc(t) {
			n, _ := pyIntConv(t)
			return "I" + n.String(), true
		}
		return "F" + pyFloatRepr(t), true
	case []any, *pyjson.Object:
		return "", false
	}
	n, _ := pyIntOf(v)
	return "I" + n.String(), true
}

// pyLess mirrors a < b for JSON values. Incomparable pairs raise TypeError.
func pyLess(a, b any) (bool, error) {
	if pyIsNumber(a) && pyIsNumber(b) {
		c, ok := pyNumCmp(a, b)
		return ok && c < 0, nil
	}
	if x, ok := a.(string); ok {
		if y, ok := b.(string); ok {
			return x < y, nil
		}
	}
	if x, ok := a.([]any); ok {
		if y, ok := b.([]any); ok {
			for i := 0; i < len(x) && i < len(y); i++ {
				if !pyEqual(x[i], y[i]) {
					return pyLess(x[i], y[i])
				}
			}
			return len(x) < len(y), nil
		}
	}
	return false, pyError("TypeError", "'<' not supported between instances")
}

// pySortByKey mirrors sorted(items, key=...) for JSON-valued keys. Python
// raises as soon as a comparison is between incomparable types, so any
// incomparable pair is an error here.
func pySortByKey[T any](items []T, key func(T) any) error {
	keys := make([]any, len(items))
	for i, it := range items {
		keys[i] = key(it)
	}
	for i := range keys {
		for j := i + 1; j < len(keys); j++ {
			if _, err := pyLess(keys[i], keys[j]); err != nil {
				return err
			}
			if _, err := pyLess(keys[j], keys[i]); err != nil {
				return err
			}
		}
	}
	idx := make([]int, len(items))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(p, q int) bool {
		less, _ := pyLess(keys[idx[p]], keys[idx[q]])
		return less
	})
	sorted := make([]T, len(items))
	for i, k := range idx {
		sorted[i] = items[k]
	}
	copy(items, sorted)
	return nil
}

// copyObject mirrors dict(d): a shallow copy that keeps key order.
func copyObject(o *pyjson.Object) *pyjson.Object {
	n := pyjson.NewObject()
	for _, k := range o.Keys() {
		n.Set(k, objGet(o, k))
	}
	return n
}

// objectOf builds a dict from alternating key/value pairs, in order.
func objectOf(kv ...any) *pyjson.Object {
	o := pyjson.NewObject()
	for i := 0; i+1 < len(kv); i += 2 {
		o.Set(kv[i].(string), kv[i+1])
	}
	return o
}
