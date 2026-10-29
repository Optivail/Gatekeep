package pyre

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// nodeKind is the type of a parsed pattern node.
type nodeKind uint8

const (
	nEmpty  nodeKind = iota
	nSet             // matches one code point from set
	nCat             // subs in sequence
	nAlt             // subs, first alternative that matches wins
	nRep             // subs[0] repeated min..max times (max < 0: unbounded)
	nCap             // capturing group number group, body subs[0]
	nAssert          // zero-width assertion
	nLook            // lookahead or lookbehind (ahead, neg) over subs[0]
)

// assertKind names the zero-width assertions. The Python meaning is noted
// on each; the emitters translate them per backend.
type assertKind uint8

const (
	aStrStart  assertKind = iota // \A, and ^ without re.M
	aStrEnd                      // \Z: the very end of the string
	aLineStart                   // ^ with re.M: start, or right after a \n
	aLineEnd                     // $ with re.M: before any \n, or the end
	aDollar                      // $ without re.M: end, or before one final \n
	aWordB                       // \b, Unicode word boundary
	aNotWordB                    // \B
)

type node struct {
	kind     nodeKind
	set      charSet
	subs     []*node
	min, max int
	lazy     bool
	group    int
	assert   assertKind
	ahead    bool
	neg      bool
}

// parser reads a Python pattern and builds a node tree. It accepts only the
// syntax the engines use, plus the close neighbours of it, and panics on
// everything else so that an unsupported construct cannot silently differ.
type parser struct {
	pat   string
	src   []rune
	pos   int
	flags Flags
	ncap  int
}

func (p *parser) fail(format string, args ...any) {
	panic(fmt.Sprintf("pyre: %s at position %d in pattern %q", fmt.Sprintf(format, args...), p.pos, p.pat))
}

func (p *parser) eof() bool { return p.pos >= len(p.src) }

func (p *parser) peek() rune {
	if p.eof() {
		return -1
	}
	return p.src[p.pos]
}

func (p *parser) peekAt(off int) rune {
	if p.pos+off >= len(p.src) {
		return -1
	}
	return p.src[p.pos+off]
}

func (p *parser) next() rune {
	c := p.src[p.pos]
	p.pos++
	return c
}

func parse(pat string, flags Flags) (*node, *parser) {
	p := &parser{pat: pat, src: []rune(pat), flags: flags}
	p.parseGlobalFlags()
	root := p.parseAlt()
	if !p.eof() {
		// parseAlt only stops early at an unmatched ')'.
		p.fail("unbalanced parenthesis")
	}
	return root, p
}

// parseGlobalFlags accepts (?aiLmsux) groups only at the very start of the
// pattern, as CPython 3.11+ requires. Only i, m, s and the no-op u are used.
func (p *parser) parseGlobalFlags() {
	for p.peek() == '(' && p.peekAt(1) == '?' {
		end := p.pos + 2
		for end < len(p.src) && p.src[end] != ')' {
			if !strings.ContainsRune("aiLmsux", p.src[end]) {
				return // not a flag group; leave it for parseGroup
			}
			end++
		}
		if end >= len(p.src) || end == p.pos+2 {
			return
		}
		for _, f := range p.src[p.pos+2 : end] {
			switch f {
			case 'i':
				p.flags |= IgnoreCase
			case 'm':
				p.flags |= Multiline
			case 's':
				p.flags |= DotAll
			case 'u':
			default:
				p.fail("inline flag %q is not supported", f)
			}
		}
		p.pos = end + 1
	}
}

func (p *parser) parseAlt() *node {
	branches := []*node{p.parseCat()}
	for p.peek() == '|' {
		p.pos++
		branches = append(branches, p.parseCat())
	}
	if len(branches) == 1 {
		return branches[0]
	}
	return &node{kind: nAlt, subs: branches}
}

func (p *parser) parseCat() *node {
	var items []*node
	for !p.eof() && p.peek() != '|' && p.peek() != ')' {
		items = append(items, p.parseQuant(p.parseAtom()))
	}
	switch len(items) {
	case 0:
		return &node{kind: nEmpty}
	case 1:
		return items[0]
	}
	return &node{kind: nCat, subs: items}
}

func (p *parser) parseAtom() *node {
	c := p.next()
	switch c {
	case '(':
		return p.parseGroup()
	case '[':
		return p.parseClass()
	case '.':
		if p.flags&DotAll != 0 {
			return &node{kind: nSet, set: charSet{{0, maxRune}}}
		}
		return &node{kind: nSet, set: complement(charSet{{'\n', '\n'}})}
	case '^':
		if p.flags&Multiline != 0 {
			return &node{kind: nAssert, assert: aLineStart}
		}
		return &node{kind: nAssert, assert: aStrStart}
	case '$':
		if p.flags&Multiline != 0 {
			return &node{kind: nAssert, assert: aLineEnd}
		}
		return &node{kind: nAssert, assert: aDollar}
	case '\\':
		return p.parseEscape()
	case '*', '+', '?':
		p.pos--
		p.fail("nothing to repeat")
	case '{':
		if _, _, _, ok := p.braces(p.pos - 1); ok {
			p.fail("nothing to repeat")
		}
	case ')':
		p.fail("unbalanced parenthesis")
	}
	return p.literal(c)
}

func (p *parser) literal(c rune) *node {
	cs := charSet{{c, c}}
	if p.flags&IgnoreCase != 0 {
		cs = foldSet(cs)
	}
	return &node{kind: nSet, set: cs}
}

// braces parses a {m,n} quantifier that starts at index start (the '{').
// It follows sre_parse: "{}" and malformed braces are literal text, not an
// error. The returned bounds use max < 0 for "no upper bound".
func (p *parser) braces(start int) (min, max, end int, ok bool) {
	i := start + 1
	read := func() string {
		j := i
		for j < len(p.src) && p.src[j] >= '0' && p.src[j] <= '9' {
			j++
		}
		s := string(p.src[i:j])
		i = j
		return s
	}
	if i < len(p.src) && p.src[i] == '}' {
		return 0, 0, 0, false
	}
	lo := read()
	hi := lo
	if i < len(p.src) && p.src[i] == ',' {
		i++
		hi = read()
	}
	if i >= len(p.src) || p.src[i] != '}' {
		return 0, 0, 0, false
	}
	min, max = 0, -1
	if lo != "" {
		v, err := strconv.Atoi(lo)
		if err != nil || v > 4294967295 {
			panic(fmt.Sprintf("pyre: repeat count too large in pattern %q", p.pat))
		}
		min = v
	}
	if hi != "" {
		v, err := strconv.Atoi(hi)
		if err != nil || v > 4294967295 {
			panic(fmt.Sprintf("pyre: repeat count too large in pattern %q", p.pat))
		}
		max = v
	}
	if max >= 0 && max < min {
		panic(fmt.Sprintf("pyre: min repeat greater than max repeat in pattern %q", p.pat))
	}
	return min, max, i + 1, true
}

func (p *parser) parseQuant(atom *node) *node {
	var min, max int
	switch p.peek() {
	case '*':
		min, max = 0, -1
		p.pos++
	case '+':
		min, max = 1, -1
		p.pos++
	case '?':
		min, max = 0, 1
		p.pos++
	case '{':
		m, x, end, ok := p.braces(p.pos)
		if !ok {
			return atom
		}
		min, max = m, x
		p.pos = end
	default:
		return atom
	}
	if atom.kind == nAssert || atom.kind == nLook {
		p.fail("nothing to repeat")
	}
	lazy := false
	if p.peek() == '?' {
		lazy = true
		p.pos++
	} else if p.peek() == '+' {
		p.fail("possessive quantifiers are not supported")
	}
	if c := p.peek(); c == '*' || c == '+' || c == '?' || (c == '{' && p.isBraceQuant()) {
		p.fail("multiple repeat")
	}
	return &node{kind: nRep, subs: []*node{atom}, min: min, max: max, lazy: lazy}
}

func (p *parser) isBraceQuant() bool {
	_, _, _, ok := p.braces(p.pos)
	return ok
}

func (p *parser) parseGroup() *node {
	if p.peek() == '?' {
		p.pos++
		c := p.peek()
		switch {
		case c == ':':
			p.pos++
			return p.groupBody(&node{kind: nCat})
		case c == '#':
			for !p.eof() && p.peek() != ')' {
				p.pos++
			}
			p.expect(')')
			return &node{kind: nEmpty}
		case c == '=' || c == '!':
			p.pos++
			return p.groupBody(&node{kind: nLook, ahead: true, neg: c == '!'})
		case c == '<' && (p.peekAt(1) == '=' || p.peekAt(1) == '!'):
			neg := p.peekAt(1) == '!'
			p.pos += 2
			return p.groupBody(&node{kind: nLook, ahead: false, neg: neg})
		case c == 'P' && p.peekAt(1) == '<':
			p.pos += 2
			start := p.pos
			for !p.eof() && p.peek() != '>' {
				p.pos++
			}
			if p.eof() || p.pos == start {
				p.fail("bad group name")
			}
			p.pos++
			p.ncap++
			return p.groupBody(&node{kind: nCap, group: p.ncap})
		case c == 'P':
			p.fail("named backreferences are not supported")
		default:
			p.fail("group syntax %q is not supported", c)
		}
	}
	p.ncap++
	return p.groupBody(&node{kind: nCap, group: p.ncap})
}

// groupBody parses the alternation inside a group whose kind is already set
// and consumes the closing parenthesis.
func (p *parser) groupBody(g *node) *node {
	body := p.parseAlt()
	p.expect(')')
	if g.kind == nCat {
		// Non-capturing group: keep the body as is, no extra node needed.
		return body
	}
	g.subs = []*node{body}
	return g
}

func (p *parser) expect(c rune) {
	if p.eof() || p.peek() != c {
		p.fail("missing %q", c)
	}
	p.pos++
}

// categories returns the set for \d \D \s \S \w \W, or nil if c is not one.
func (p *parser) category(c rune) charSet {
	t := loadTables()
	switch c {
	case 'd':
		return t.digit
	case 'D':
		return complement(t.digit)
	case 's':
		return t.space
	case 'S':
		return complement(t.space)
	case 'w':
		return t.word
	case 'W':
		return complement(t.word)
	}
	return nil
}

// parseEscape handles a backslash outside a character class.
func (p *parser) parseEscape() *node {
	if p.eof() {
		p.fail("bad escape (end of pattern)")
	}
	c := p.peek()
	switch c {
	case 'A':
		p.pos++
		return &node{kind: nAssert, assert: aStrStart}
	case 'Z':
		p.pos++
		return &node{kind: nAssert, assert: aStrEnd}
	case 'b':
		p.pos++
		return &node{kind: nAssert, assert: aWordB}
	case 'B':
		p.pos++
		return &node{kind: nAssert, assert: aNotWordB}
	}
	if cs := p.category(c); cs != nil {
		p.pos++
		return &node{kind: nSet, set: cs}
	}
	if (c >= '1' && c <= '9') || c == 'N' {
		p.fail("backreferences and named characters are not supported")
	}
	r := p.escapeRune(false)
	return p.literal(r)
}

// escapeRune reads the rune for a single-character escape. The backslash has
// been consumed in the sense that p.pos points at the escape letter or symbol.
// inClass selects the class rules: \b is backspace there.
func (p *parser) escapeRune(inClass bool) rune {
	c := p.next()
	switch c {
	case 'a':
		return 7
	case 'b':
		if inClass {
			return 8
		}
	case 'f':
		return 12
	case 'n':
		return '\n'
	case 'r':
		return '\r'
	case 't':
		return '\t'
	case 'v':
		return 11
	case 'x':
		return p.hexEscape(2)
	case 'u':
		return p.hexEscape(4)
	case 'U':
		return p.hexEscape(8)
	case '0':
		p.fail("octal escapes are not supported")
	}
	if c < 128 && unicode.IsLetter(c) || c < 128 && unicode.IsDigit(c) {
		p.pos--
		p.fail("bad escape \\%c", c)
	}
	return c
}

func (p *parser) hexEscape(n int) rune {
	if p.pos+n > len(p.src) {
		p.fail("incomplete escape")
	}
	v, err := strconv.ParseUint(string(p.src[p.pos:p.pos+n]), 16, 32)
	if err != nil || v > maxRune {
		p.fail("bad hex escape")
	}
	p.pos += n
	return rune(v)
}

// parseClass reads a bracketed set after the '['.
func (p *parser) parseClass() *node {
	neg := false
	if p.peek() == '^' {
		neg = true
		p.pos++
	}
	var explicit charSet
	var cats charSet
	first := true
	for {
		if p.eof() {
			p.fail("unterminated character set")
		}
		if p.peek() == ']' && !first {
			p.pos++
			break
		}
		first = false
		lo, cat, isCat := p.classItem()
		if isCat {
			cats = union(cats, cat)
			// CPython rejects a range that starts at a class, such as [\\d-z].
			if p.peek() == '-' && p.peekAt(1) != ']' && p.peekAt(1) != -1 {
				p.fail("bad character range")
			}
			continue
		}
		hi := lo
		if p.peek() == '-' && p.peekAt(1) != ']' && p.peekAt(1) != -1 {
			p.pos++
			h, _, isCat := p.classItem()
			if isCat {
				p.fail("bad character range")
			}
			hi = h
			if hi < lo {
				p.fail("bad character range")
			}
		}
		explicit = append(explicit, span{lo, hi})
	}
	explicit = explicit.normalize()
	if p.flags&IgnoreCase != 0 {
		explicit = foldSet(explicit)
	}
	set := union(explicit, cats)
	if neg {
		set = complement(set)
	}
	return &node{kind: nSet, set: set}
}

// classItem reads one class member: a single code point, or a category
// escape such as \d (isCat true, set returned).
func (p *parser) classItem() (rune, charSet, bool) {
	c := p.next()
	if c != '\\' {
		return c, nil, false
	}
	if p.eof() {
		p.fail("bad escape (end of pattern)")
	}
	if cs := p.category(p.peek()); cs != nil {
		p.pos++
		return 0, cs, true
	}
	switch p.peek() {
	case 'A', 'Z', 'B':
		p.fail("bad escape in character class")
	}
	return p.escapeRune(true), nil, false
}
