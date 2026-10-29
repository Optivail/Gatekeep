package pyre

import (
	"fmt"
	"strings"
	"sync"
)

// The same parsed tree is rendered for two engines:
//
//   - RE2 (package regexp) when the pattern has no lookaround, no \b and no
//     non-multiline $. RE2 has no lookaround, and its \b is ASCII-only.
//   - regexp2 (a .NET-style backtracking engine) for everything else.
//
// Classes are always written as explicit code point ranges. That keeps \w,
// \d, \s and IGNORECASE exactly CPython's, whatever the engine's own tables
// say.

// needsBacktrack reports whether RE2 cannot express the tree exactly.
func needsBacktrack(n *node) bool {
	switch n.kind {
	case nLook:
		return true
	case nAssert:
		return n.assert == aDollar || n.assert == aWordB || n.assert == aNotWordB
	}
	for _, s := range n.subs {
		if needsBacktrack(s) {
			return true
		}
	}
	return false
}

// needsBacktrackASCII is needsBacktrack for input known to be ASCII: \b and
// \B are then expressible in RE2 as well.
func needsBacktrackASCII(n *node) bool {
	switch n.kind {
	case nLook:
		return true
	case nAssert:
		return n.assert == aDollar
	}
	for _, s := range n.subs {
		if needsBacktrackASCII(s) {
			return true
		}
	}
	return false
}

// fixedWidth returns the width in code points of every match of n, or -1 if
// the width varies. Lookbehind bodies must be fixed-width in CPython.
func fixedWidth(n *node) int {
	switch n.kind {
	case nEmpty, nAssert, nLook:
		return 0
	case nSet:
		return 1
	case nCap:
		return fixedWidth(n.subs[0])
	case nCat:
		total := 0
		for _, s := range n.subs {
			w := fixedWidth(s)
			if w < 0 {
				return -1
			}
			total += w
		}
		return total
	case nAlt:
		w := fixedWidth(n.subs[0])
		for _, s := range n.subs[1:] {
			if fixedWidth(s) != w {
				return -1
			}
		}
		return w
	case nRep:
		w := fixedWidth(n.subs[0])
		if w >= 0 && n.min == n.max {
			return w * n.min
		}
	}
	return -1
}

// validateLookbehind panics on a variable-width lookbehind, as CPython does.
func validateLookbehind(pat string, n *node) {
	if n.kind == nLook && !n.ahead && fixedWidth(n.subs[0]) < 0 {
		panic(fmt.Sprintf("pyre: look-behind requires fixed-width pattern in %q", pat))
	}
	for _, s := range n.subs {
		validateLookbehind(pat, s)
	}
}

// writer renders nodes for one engine.
type writer struct {
	sb  strings.Builder
	re2 bool
	// asciiWord makes \b and \B RE2's ASCII boundaries. They equal Python's
	// Unicode boundaries only on ASCII input, so the caller must check that.
	asciiWord bool
}

var (
	wordBTOnce sync.Once
	wordBT     string
)

// wordClassBT is the \w class as a regexp2 string. \b and \B use it inside
// lookarounds, since regexp2's own \w and \b follow .NET's Unicode rules.
func wordClassBT() string {
	wordBTOnce.Do(func() {
		var w writer
		w.set(loadTables().word)
		wordBT = w.sb.String()
	})
	return wordBT
}

func (w *writer) rune(r rune) {
	if w.re2 {
		fmt.Fprintf(&w.sb, `\x{%X}`, r)
		return
	}
	if r < 0x10000 {
		fmt.Fprintf(&w.sb, `\u%04X`, r)
		return
	}
	// regexp2 reads the pattern as runes, so astral code points can be literal.
	w.sb.WriteRune(r)
}

// set writes a class from explicit spans. An empty set matches nothing.
func (w *writer) set(cs charSet) {
	if len(cs) == 0 {
		if w.re2 {
			w.sb.WriteString(`[^\x{0}-\x{10FFFF}]`)
		} else {
			w.sb.WriteString(`(?!)`)
		}
		return
	}
	if len(cs) == 1 && cs[0].lo == cs[0].hi {
		w.rune(cs[0].lo)
		return
	}
	w.sb.WriteByte('[')
	for _, s := range cs {
		w.rune(s.lo)
		if s.hi != s.lo {
			w.sb.WriteByte('-')
			w.rune(s.hi)
		}
	}
	w.sb.WriteByte(']')
}

const btNewline = `\u000A`

func (w *writer) assert(a assertKind) {
	if w.re2 {
		switch a {
		case aStrStart:
			w.sb.WriteString(`\A`)
		case aStrEnd:
			w.sb.WriteString(`\z`)
		case aLineStart:
			w.sb.WriteString(`(?m:^)`)
		case aLineEnd:
			w.sb.WriteString(`(?m:$)`)
		case aWordB:
			if !w.asciiWord {
				panic("pyre: assertion not expressible in RE2")
			}
			w.sb.WriteString(`\b`)
		case aNotWordB:
			if !w.asciiWord {
				panic("pyre: assertion not expressible in RE2")
			}
			w.sb.WriteString(`\B`)
		default:
			panic("pyre: assertion not expressible in RE2")
		}
		return
	}
	word := wordClassBT()
	switch a {
	case aStrStart:
		w.sb.WriteString(`\A`)
	case aStrEnd:
		w.sb.WriteString(`\z`)
	case aLineStart:
		// Python's (?m)^ also matches at the very end after a final newline.
		w.sb.WriteString(`(?:\A|(?<=` + btNewline + `))`)
	case aLineEnd:
		w.sb.WriteString(`(?=` + btNewline + `|\z)`)
	case aDollar:
		// Without re.M, $ is the end or just before one final newline.
		w.sb.WriteString(`(?=` + btNewline + `?\z)`)
	case aWordB:
		w.sb.WriteString(`(?:(?<=` + word + `)(?!` + word + `)|(?<!` + word + `)(?=` + word + `))`)
	case aNotWordB:
		w.sb.WriteString(`(?:(?<=` + word + `)(?=` + word + `)|(?<!` + word + `)(?!` + word + `))`)
	}
}

func (w *writer) node(n *node) {
	switch n.kind {
	case nEmpty:
		w.sb.WriteString(`(?:)`)
	case nSet:
		w.set(n.set)
	case nCat:
		for _, s := range n.subs {
			w.node(s)
		}
	case nAlt:
		w.sb.WriteString(`(?:`)
		for i, s := range n.subs {
			if i > 0 {
				w.sb.WriteByte('|')
			}
			w.node(s)
		}
		w.sb.WriteByte(')')
	case nRep:
		sub := n.subs[0]
		wrap := sub.kind != nSet && sub.kind != nCap
		if wrap {
			w.sb.WriteString(`(?:`)
		}
		w.node(sub)
		if wrap {
			w.sb.WriteByte(')')
		}
		w.quant(n)
	case nCap:
		w.sb.WriteByte('(')
		w.node(n.subs[0])
		w.sb.WriteByte(')')
	case nAssert:
		w.assert(n.assert)
	case nLook:
		switch {
		case n.ahead && !n.neg:
			w.sb.WriteString(`(?=`)
		case n.ahead && n.neg:
			w.sb.WriteString(`(?!`)
		case !n.ahead && !n.neg:
			w.sb.WriteString(`(?<=`)
		default:
			w.sb.WriteString(`(?<!`)
		}
		w.sb.WriteString(`(?:`)
		w.node(n.subs[0])
		w.sb.WriteString(`))`)
	}
}

func (w *writer) quant(n *node) {
	switch {
	case n.min == 0 && n.max < 0:
		w.sb.WriteByte('*')
	case n.min == 1 && n.max < 0:
		w.sb.WriteByte('+')
	case n.min == 0 && n.max == 1:
		w.sb.WriteByte('?')
	case n.min == n.max:
		fmt.Fprintf(&w.sb, "{%d}", n.min)
	case n.max < 0:
		fmt.Fprintf(&w.sb, "{%d,}", n.min)
	default:
		fmt.Fprintf(&w.sb, "{%d,%d}", n.min, n.max)
	}
	if n.lazy {
		w.sb.WriteByte('?')
	}
}

// emitRE2 renders the tree for Go's regexp. Callers must check
// needsBacktrack first.
func emitRE2(n *node) string {
	w := &writer{re2: true}
	w.node(n)
	return w.sb.String()
}

// emitRE2ASCII renders the tree for RE2 on ASCII input only. Callers must
// check needsBacktrackASCII first and the input for ASCII.
func emitRE2ASCII(n *node) string {
	w := &writer{re2: true, asciiWord: true}
	w.node(n)
	return w.sb.String()
}

// emitBT renders the tree for regexp2.
func emitBT(n *node) string {
	w := &writer{}
	w.node(n)
	return w.sb.String()
}
