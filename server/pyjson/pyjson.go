// Package pyjson reproduces the parts of CPython's json module that Gatekeep's
// engines depend on: insertion-ordered objects, the exact accept/reject rules
// of json.loads, and the exact bytes of json.dumps (separators, ensure_ascii,
// sort_keys, float repr). Byte-identical engine output is the whole point, so
// anything Go's encoding/json would normalise differently lives here instead.
//
// Value model (what Loads returns and Dumps accepts):
//
//	nil, bool, string, int64, *big.Int, float64, []any, *Object
//
// Strings may carry lone surrogates (Python str can, JSON "\ud800" produces
// one). They are stored as WTF-8. Dumps with EnsureASCII escapes them like
// Python does; without it Python would fail at the later .encode("utf-8"), so
// we return ErrLoneSurrogate instead.
package pyjson

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// ErrLoneSurrogate is returned by Dumps when a string holds a lone surrogate
// and EnsureASCII is off. CPython's dumps succeeds there, but every caller then
// calls .encode("utf-8"), which raises. Surfacing the error keeps that failure.
var ErrLoneSurrogate = errors.New("pyjson: lone surrogate is not valid UTF-8")

// Object is an insertion-ordered dict with string keys.
// Setting an existing key keeps its position, like a Python dict.
type Object struct {
	keys []string
	vals map[string]any
}

// NewObject returns an empty Object.
func NewObject() *Object {
	return &Object{vals: map[string]any{}}
}

// Get returns the value for k and whether it exists.
func (o *Object) Get(k string) (any, bool) {
	v, ok := o.vals[k]
	return v, ok
}

// Set assigns k. A new key is appended; an existing key keeps its position.
func (o *Object) Set(k string, v any) {
	if _, ok := o.vals[k]; !ok {
		o.keys = append(o.keys, k)
	}
	o.vals[k] = v
}

// Delete removes k, keeping the order of the remaining keys. It reports
// whether k was present.
func (o *Object) Delete(k string) bool {
	if _, ok := o.vals[k]; !ok {
		return false
	}
	delete(o.vals, k)
	for i, key := range o.keys {
		if key == k {
			o.keys = append(o.keys[:i], o.keys[i+1:]...)
			break
		}
	}
	return true
}

// Keys returns the keys in insertion order. The slice is a copy.
func (o *Object) Keys() []string {
	return append([]string(nil), o.keys...)
}

// Len returns the number of keys.
func (o *Object) Len() int { return len(o.keys) }

// Loads parses JSON with CPython json.loads semantics. Input is bytes, as
// the proxy receives it: a UTF-8 BOM is stripped, invalid UTF-8 is rejected,
// NaN/Infinity/-Infinity are accepted, and duplicate keys keep the last value
// at the first position.
func Loads(data []byte) (any, error) {
	if len(data) >= 3 && data[0] == 0xEF && data[1] == 0xBB && data[2] == 0xBF {
		data = data[3:]
	}
	if !validUTF8Surrogatepass(data) {
		return nil, errors.New("pyjson: invalid UTF-8")
	}
	p := &parser{s: string(data)}
	p.skipWS()
	v, err := p.value(0)
	if err != nil {
		return nil, err
	}
	p.skipWS()
	if p.i != len(p.s) {
		return nil, fmt.Errorf("pyjson: extra data at %d", p.i)
	}
	return v, nil
}

// Options controls Dumps. The zero value is not CPython's default: use
// Default, Compact, or set the fields explicitly.
type Options struct {
	ItemSep     string // "," or ", "
	KeySep      string // ":" or ": "
	EnsureASCII bool
	SortKeys    bool
}

// Default matches json.dumps(v) with no arguments.
var Default = Options{ItemSep: ", ", KeySep: ": ", EnsureASCII: true}

// Compact matches json.dumps(v, ensure_ascii=False, separators=(",", ":")).
var Compact = Options{ItemSep: ",", KeySep: ":"}

// Dumps serialises v with CPython json.dumps semantics.
func Dumps(v any, opts Options) ([]byte, error) {
	var b strings.Builder
	if err := encode(&b, v, opts, 0); err != nil {
		return nil, err
	}
	return []byte(b.String()), nil
}

// FloatRepr returns Python's repr(float) for finite and non-finite values.
// json.dumps uses NaN, Infinity and -Infinity for the non-finite ones.
func FloatRepr(f float64) string {
	switch {
	case math.IsNaN(f):
		return "NaN"
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	}
	// Shortest round-trip digits, as Python's repr uses.
	s := strconv.FormatFloat(f, 'e', -1, 64)
	sign := ""
	if s[0] == '-' {
		sign = "-"
		s = s[1:]
	}
	mant, expStr, _ := strings.Cut(s, "e")
	exp, _ := strconv.Atoi(expStr)
	digits := strings.Replace(mant, ".", "", 1)
	if exp < -4 || exp >= 16 {
		out := digits[:1]
		if len(digits) > 1 {
			out += "." + digits[1:]
		}
		es := "+"
		if exp < 0 {
			es = "-"
			exp = -exp
		}
		return fmt.Sprintf("%s%se%s%02d", sign, out, es, exp)
	}
	var out string
	switch {
	case exp < 0:
		out = "0." + strings.Repeat("0", -exp-1) + digits
	case len(digits) > exp+1:
		out = digits[:exp+1] + "." + digits[exp+1:]
	default:
		out = digits + strings.Repeat("0", exp+1-len(digits)) + ".0"
	}
	return sign + out
}

func encode(b *strings.Builder, v any, o Options, depth int) error {
	if depth > 1000 {
		return errors.New("pyjson: maximum recursion depth exceeded")
	}
	switch t := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if t {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case string:
		return writeString(b, t, o.EnsureASCII)
	case int:
		b.WriteString(strconv.Itoa(t))
	case int64:
		b.WriteString(strconv.FormatInt(t, 10))
	case *big.Int:
		b.WriteString(t.String())
	case float64:
		b.WriteString(FloatRepr(t))
	case []any:
		b.WriteByte('[')
		for i, item := range t {
			if i > 0 {
				b.WriteString(o.ItemSep)
			}
			if err := encode(b, item, o, depth+1); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case *Object:
		keys := t.Keys()
		if o.SortKeys {
			sort.Strings(keys)
		}
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteString(o.ItemSep)
			}
			if err := writeString(b, k, o.EnsureASCII); err != nil {
				return err
			}
			b.WriteString(o.KeySep)
			val, _ := t.Get(k)
			if err := encode(b, val, o, depth+1); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		return fmt.Errorf("pyjson: unsupported type %T", v)
	}
	return nil
}

// writeString emits a JSON string with CPython's escaping rules.
func writeString(b *strings.Builder, s string, ensureASCII bool) error {
	b.WriteByte('"')
	for i := 0; i < len(s); {
		c := s[i]
		if c < utf8.RuneSelf {
			switch c {
			case '"':
				b.WriteString(`\"`)
			case '\\':
				b.WriteString(`\\`)
			case '\n':
				b.WriteString(`\n`)
			case '\r':
				b.WriteString(`\r`)
			case '\t':
				b.WriteString(`\t`)
			case '\b':
				b.WriteString(`\b`)
			case '\f':
				b.WriteString(`\f`)
			default:
				// CPython's ensure_ascii escapes everything outside 0x20..0x7e,
				// DEL included.
				if c < 0x20 || (ensureASCII && c == 0x7f) {
					fmt.Fprintf(b, `\u%04x`, c)
				} else {
					b.WriteByte(c)
				}
			}
			i++
			continue
		}
		r, size := decodeWTF8(s[i:])
		if size == 0 {
			return ErrLoneSurrogate
		}
		i += size
		if r < 0 {
			return ErrLoneSurrogate
		}
		if !ensureASCII {
			if r >= 0xD800 && r <= 0xDFFF {
				return ErrLoneSurrogate
			}
			b.WriteRune(r)
			continue
		}
		if r > 0xFFFF {
			r -= 0x10000
			fmt.Fprintf(b, `\u%04x\u%04x`, 0xD800+(r>>10), 0xDC00+(r&0x3FF))
		} else {
			fmt.Fprintf(b, `\u%04x`, r)
		}
	}
	b.WriteByte('"')
	return nil
}

// decodeWTF8 decodes one code point from s. It accepts encoded surrogates
// (ED A0..BF xx), which utf8.DecodeRuneInString rejects. It returns size 0 on
// malformed input.
func decodeWTF8(s string) (rune, int) {
	if len(s) >= 3 && s[0] == 0xED && s[1] >= 0xA0 && s[1] <= 0xBF && s[2] >= 0x80 && s[2] <= 0xBF {
		r := rune(s[0]&0x0F)<<12 | rune(s[1]&0x3F)<<6 | rune(s[2]&0x3F)
		return r, 3
	}
	r, size := utf8.DecodeRuneInString(s)
	if r == utf8.RuneError && size <= 1 {
		return -1, 1
	}
	return r, size
}

// validUTF8Surrogatepass mirrors bytes.decode("utf-8", "surrogatepass"):
// valid UTF-8, plus encoded surrogates.
func validUTF8Surrogatepass(p []byte) bool {
	s := string(p)
	for i := 0; i < len(s); {
		if s[i] < utf8.RuneSelf {
			i++
			continue
		}
		r, size := decodeWTF8(s[i:])
		if size == 0 || (r == utf8.RuneError && size == 1) {
			return false
		}
		i += size
	}
	return true
}
