package pyjson

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
	"unicode/utf8"
)

// maxDepth mirrors the nesting at which CPython raises RecursionError.
const maxDepth = 1000

type parser struct {
	s string
	i int
}

func (p *parser) fail(msg string) error {
	return fmt.Errorf("pyjson: %s at %d", msg, p.i)
}

func (p *parser) skipWS() {
	for p.i < len(p.s) {
		switch p.s[p.i] {
		case ' ', '\t', '\n', '\r':
			p.i++
		default:
			return
		}
	}
}

func (p *parser) value(depth int) (any, error) {
	if depth > maxDepth {
		return nil, p.fail("maximum recursion depth exceeded")
	}
	if p.i >= len(p.s) {
		return nil, p.fail("expecting value")
	}
	switch c := p.s[p.i]; {
	case c == '"':
		p.i++
		return p.str()
	case c == '{':
		p.i++
		return p.object(depth + 1)
	case c == '[':
		p.i++
		return p.array(depth + 1)
	case c == 'n' && strings.HasPrefix(p.s[p.i:], "null"):
		p.i += 4
		return nil, nil
	case c == 't' && strings.HasPrefix(p.s[p.i:], "true"):
		p.i += 4
		return true, nil
	case c == 'f' && strings.HasPrefix(p.s[p.i:], "false"):
		p.i += 5
		return false, nil
	case c == 'N' && strings.HasPrefix(p.s[p.i:], "NaN"):
		p.i += 3
		return math.NaN(), nil
	case c == 'I' && strings.HasPrefix(p.s[p.i:], "Infinity"):
		p.i += 8
		return math.Inf(1), nil
	case c == '-' && strings.HasPrefix(p.s[p.i:], "-Infinity"):
		p.i += 9
		return math.Inf(-1), nil
	case c == '-' || (c >= '0' && c <= '9'):
		return p.number()
	}
	return nil, p.fail("expecting value")
}

// number follows CPython's NUMBER_RE: -?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?
// Ints stay exact (int64 or *big.Int); fractions and exponents become float64.
func (p *parser) number() (any, error) {
	start := p.i
	if p.s[p.i] == '-' {
		p.i++
	}
	if p.i >= len(p.s) {
		p.i = start
		return nil, p.fail("expecting value")
	}
	if p.s[p.i] == '0' {
		p.i++
	} else if p.s[p.i] >= '1' && p.s[p.i] <= '9' {
		for p.i < len(p.s) && isDigit(p.s[p.i]) {
			p.i++
		}
	} else {
		p.i = start
		return nil, p.fail("expecting value")
	}
	isFloat := false
	if p.i+1 < len(p.s) && p.s[p.i] == '.' && isDigit(p.s[p.i+1]) {
		isFloat = true
		p.i++
		for p.i < len(p.s) && isDigit(p.s[p.i]) {
			p.i++
		}
	}
	if p.i < len(p.s) && (p.s[p.i] == 'e' || p.s[p.i] == 'E') {
		j := p.i + 1
		if j < len(p.s) && (p.s[j] == '+' || p.s[j] == '-') {
			j++
		}
		if j < len(p.s) && isDigit(p.s[j]) {
			isFloat = true
			for j < len(p.s) && isDigit(p.s[j]) {
				j++
			}
			p.i = j
		}
	}
	lit := p.s[start:p.i]
	if isFloat {
		f, err := strconv.ParseFloat(lit, 64)
		if err != nil && !errors.Is(err, strconv.ErrRange) {
			return nil, p.fail("bad number")
		}
		return f, nil
	}
	if n, err := strconv.ParseInt(lit, 10, 64); err == nil {
		return n, nil
	}
	b, ok := new(big.Int).SetString(lit, 10)
	if !ok {
		return nil, p.fail("bad number")
	}
	return b, nil
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

// str parses a string body after the opening quote. Control characters are
// rejected (strict=True). Lone surrogates from \u escapes are kept as WTF-8.
func (p *parser) str() (string, error) {
	var b strings.Builder
	for {
		if p.i >= len(p.s) {
			return "", p.fail("unterminated string")
		}
		c := p.s[p.i]
		switch {
		case c == '"':
			p.i++
			return b.String(), nil
		case c == '\\':
			p.i++
			if p.i >= len(p.s) {
				return "", p.fail("unterminated string")
			}
			e := p.s[p.i]
			p.i++
			switch e {
			case '"', '\\', '/':
				b.WriteByte(e)
			case 'b':
				b.WriteByte('\b')
			case 'f':
				b.WriteByte('\f')
			case 'n':
				b.WriteByte('\n')
			case 'r':
				b.WriteByte('\r')
			case 't':
				b.WriteByte('\t')
			case 'u':
				r, err := p.hex4()
				if err != nil {
					return "", err
				}
				if r >= 0xD800 && r <= 0xDBFF && strings.HasPrefix(p.s[p.i:], `\u`) {
					save := p.i
					p.i += 2
					lo, err := p.hex4()
					if err == nil && lo >= 0xDC00 && lo <= 0xDFFF {
						r = 0x10000 + (r-0xD800)<<10 + (lo - 0xDC00)
					} else {
						p.i = save
					}
				}
				writeCodePoint(&b, r)
			default:
				p.i--
				return "", p.fail("invalid \\escape")
			}
		case c < 0x20:
			return "", p.fail("invalid control character")
		default:
			b.WriteByte(c)
			p.i++
		}
	}
}

func (p *parser) hex4() (rune, error) {
	if p.i+4 > len(p.s) {
		return 0, p.fail("invalid \\uXXXX escape")
	}
	n, err := strconv.ParseUint(p.s[p.i:p.i+4], 16, 32)
	if err != nil || strings.ContainsAny(p.s[p.i:p.i+4], "+-_") {
		return 0, p.fail("invalid \\uXXXX escape")
	}
	p.i += 4
	return rune(n), nil
}

// writeCodePoint writes r as UTF-8, or as WTF-8 when r is a lone surrogate.
func writeCodePoint(b *strings.Builder, r rune) {
	if r >= 0xD800 && r <= 0xDFFF {
		b.WriteByte(0xE0 | byte(r>>12))
		b.WriteByte(0x80 | byte((r>>6)&0x3F))
		b.WriteByte(0x80 | byte(r&0x3F))
		return
	}
	var buf [utf8.UTFMax]byte
	n := utf8.EncodeRune(buf[:], r)
	b.Write(buf[:n])
}

func (p *parser) object(depth int) (*Object, error) {
	if depth > maxDepth {
		return nil, p.fail("maximum recursion depth exceeded")
	}
	obj := NewObject()
	p.skipWS()
	if p.i < len(p.s) && p.s[p.i] == '}' {
		p.i++
		return obj, nil
	}
	for {
		if p.i >= len(p.s) || p.s[p.i] != '"' {
			return nil, p.fail("expecting property name enclosed in double quotes")
		}
		p.i++
		key, err := p.str()
		if err != nil {
			return nil, err
		}
		p.skipWS()
		if p.i >= len(p.s) || p.s[p.i] != ':' {
			return nil, p.fail("expecting ':' delimiter")
		}
		p.i++
		p.skipWS()
		val, err := p.value(depth)
		if err != nil {
			return nil, err
		}
		obj.Set(key, val)
		p.skipWS()
		if p.i < len(p.s) && p.s[p.i] == '}' {
			p.i++
			return obj, nil
		}
		if p.i >= len(p.s) || p.s[p.i] != ',' {
			return nil, p.fail("expecting ',' delimiter")
		}
		p.i++
		p.skipWS()
	}
}

func (p *parser) array(depth int) ([]any, error) {
	if depth > maxDepth {
		return nil, p.fail("maximum recursion depth exceeded")
	}
	out := []any{}
	p.skipWS()
	if p.i < len(p.s) && p.s[p.i] == ']' {
		p.i++
		return out, nil
	}
	for {
		v, err := p.value(depth)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
		p.skipWS()
		if p.i < len(p.s) && p.s[p.i] == ']' {
			p.i++
			return out, nil
		}
		if p.i >= len(p.s) || p.s[p.i] != ',' {
			return nil, p.fail("expecting ',' delimiter")
		}
		p.i++
		p.skipWS()
	}
}
