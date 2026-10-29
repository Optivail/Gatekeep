// Package pyre implements the part of CPython 3's re module (str patterns)
// that Gatekeep's engines use, with CPython's match results.
//
// Behaviour that is reproduced exactly:
//   - \w, \d, \s, \b and IGNORECASE follow CPython's Unicode tables, measured
//     by server/parity/gen_unicode.py (see charset.go), not Go's or .NET's.
//   - Empty matches follow CPython 3.7+: after an empty match at p the scan
//     resumes at p, but an empty match at p is not allowed again; a non-empty
//     match at p is tried first. So re.sub('x*', '-', 'abxd') is '-a-b--d-'.
//   - $ without re.M matches at the end or before one final newline.
//
// Offsets are byte offsets into the Go string. Python lengths are code points.
// Patterns outside the supported subset panic in MustCompile.
package pyre

import (
	"regexp"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/dlclark/regexp2"
)

// Flags mirror the re module flags that the engines use.
type Flags int

const (
	IgnoreCase Flags = 1 << iota // re.I
	Multiline                    // re.M
	DotAll                       // re.S
)

// Regexp is a compiled Python-compatible pattern.
type Regexp struct {
	pat   string
	btSrc string // regexp2 source, reused for the non-empty variant
	ncap  int

	// re2 runs the pattern when RE2 is exact for it. nil otherwise.
	re2 *regexp.Regexp
	// reASCII is the RE2 form that also handles \b and \B. It runs only on
	// ASCII input. nil when the pattern has no such assertions or RE2 is not exact.
	reASCII *regexp.Regexp
	// nonNull is set when the pattern can never match the empty string. For
	// such patterns CPython's scan and RE2's all-matches iteration agree, so
	// the scan can run on RE2.
	nonNull bool
	// run is set for [set]{min,} patterns. They skip RE2 (see runs.go).
	run *runPattern
	// prefix is the lower-case ASCII literal every match starts with, or "".
	// See prefix.go.
	prefix string
	// bt runs every search; it is the reference engine for all patterns.
	bt *regexp2.Regexp

	// neOnce guards neBT, which matches only a non-empty match that starts
	// exactly at the search position. CPython uses it after an empty match.
	neOnce sync.Once
	neBT   *regexp2.Regexp
}

// MustCompile compiles a Python re pattern and panics on an invalid or
// unsupported pattern. Patterns are constants, so a panic is a programmer error.
func MustCompile(pattern string, flags ...Flags) *Regexp {
	var f Flags
	for _, x := range flags {
		f |= x
	}
	root, p := parse(pattern, f)
	validateLookbehind(pattern, root)

	btSrc := emitBT(root)
	re := &Regexp{pat: pattern, btSrc: btSrc, ncap: p.ncap}
	bt, err := regexp2.Compile(btSrc, regexp2.None)
	if err != nil {
		panic("pyre: cannot compile " + pattern + ": " + err.Error())
	}
	re.bt = bt
	re.nonNull = !canBeEmpty(root)
	re.run = runFor(root)
	re.prefix = asciiPrefix(literalPrefix(root))
	if !needsBacktrack(root) {
		r2, err := regexp.Compile(emitRE2(root))
		if err != nil {
			panic("pyre: RE2 rejects " + pattern + ": " + err.Error())
		}
		re.re2 = r2
	} else if !needsBacktrackASCII(root) {
		r2, err := regexp.Compile(emitRE2ASCII(root))
		if err != nil {
			panic("pyre: RE2 rejects " + pattern + ": " + err.Error())
		}
		re.reASCII = r2
	}
	return re
}

// nonEmptyAtStart returns the regexp2 variant that matches only a non-empty
// match anchored at the search start. \G is the search start in regexp2.
func (re *Regexp) nonEmptyAtStart() *regexp2.Regexp {
	re.neOnce.Do(func() {
		ne, err := regexp2.Compile(`\G(?:`+re.btSrc+`)(?<!\G)`, regexp2.None)
		if err != nil {
			panic("pyre: " + err.Error())
		}
		re.neBT = ne
	})
	return re.neBT
}

// subject is the input as code points, plus the byte offset of each one.
type subject struct {
	runes []rune
	offs  []int // offs[i] is the byte offset of runes[i]; nil for ASCII input
}

func newSubject(s string) *subject {
	sub := &subject{}
	ascii := true
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			ascii = false
			break
		}
	}
	sub.runes = make([]rune, 0, len(s))
	if !ascii {
		sub.offs = make([]int, 0, len(s))
	}
	for i := 0; i < len(s); {
		r, w := decodeAt(s, i)
		sub.runes = append(sub.runes, r)
		if sub.offs != nil {
			sub.offs = append(sub.offs, i)
		}
		i += w
	}
	if sub.offs != nil {
		sub.offs = append(sub.offs, len(s))
	}
	return sub
}

// byteOff converts a code point index into a byte offset.
func (sub *subject) byteOff(i int) int {
	if sub.offs == nil {
		return i
	}
	return sub.offs[i]
}

// search returns the leftmost match whose start is at or after code point pos.
// Context before pos is visible to lookbehind and \b, as in CPython.
func (re *Regexp) search(sub *subject, pos int) (lo, hi int, ok bool) {
	m, err := re.bt.FindRunesMatchStartingAt(sub.runes, pos)
	if err != nil {
		panic("pyre: regexp2: " + err.Error())
	}
	if m == nil {
		return 0, 0, false
	}
	return m.Index, m.Index + m.Length, true
}

// nextMatch implements the scan step. If mustAdvance is set, the previous
// match was empty at pos, and CPython then requires the next match to be
// non-empty if it starts at pos. Otherwise the next match starts later.
func (re *Regexp) nextMatch(sub *subject, pos int, mustAdvance bool) (int, int, bool) {
	if pos > len(sub.runes) {
		return 0, 0, false
	}
	if mustAdvance {
		m, err := re.nonEmptyAtStart().FindRunesMatchStartingAt(sub.runes, pos)
		if err != nil {
			panic("pyre: regexp2: " + err.Error())
		}
		if m != nil {
			return m.Index, m.Index + m.Length, true
		}
		pos++
		if pos > len(sub.runes) {
			return 0, 0, false
		}
	}
	lo, hi, ok := re.search(sub, pos)
	return lo, hi, ok
}

// scan calls yield for each match in order, at most limit times (limit < 0
// means no limit), with the empty-match rules above.
func (re *Regexp) scan(sub *subject, limit int, yield func(lo, hi int)) {
	pos, mustAdvance, n := 0, false, 0
	for limit < 0 || n < limit {
		lo, hi, ok := re.nextMatch(sub, pos, mustAdvance)
		if !ok {
			return
		}
		n++
		yield(lo, hi)
		pos = hi
		mustAdvance = lo == hi
	}
}

// MatchString reports whether re.search(p, s) is not None.
func (re *Regexp) MatchString(s string) bool {
	if re.re2 != nil && !hasSurrogateBytes(s) {
		return re.re2.MatchString(s)
	}
	return re.FindIndex(s) != nil
}

// FindIndex is re.search: [start, end] byte offsets, nil if no match.
func (re *Regexp) FindIndex(s string) []int {
	if re.re2 != nil && !hasSurrogateBytes(s) {
		return re.re2.FindStringIndex(s)
	}
	sub := newSubject(s)
	lo, hi, ok := re.search(sub, 0)
	if !ok {
		return nil
	}
	return []int{sub.byteOff(lo), sub.byteOff(hi)}
}

// FindSubmatchIndex is re.search with groups: [s0,e0,s1,e1,...], -1 for unmatched groups.
func (re *Regexp) FindSubmatchIndex(s string) []int {
	if re.re2 != nil && !hasSurrogateBytes(s) {
		return re.re2.FindStringSubmatchIndex(s)
	}
	sub := newSubject(s)
	m, err := re.bt.FindRunesMatchStartingAt(sub.runes, 0)
	if err != nil {
		panic("pyre: regexp2: " + err.Error())
	}
	if m == nil {
		return nil
	}
	out := make([]int, 2*(re.ncap+1))
	out[0], out[1] = sub.byteOff(m.Index), sub.byteOff(m.Index+m.Length)
	for g := 1; g <= re.ncap; g++ {
		grp := m.GroupByNumber(g)
		if grp == nil || len(grp.Captures) == 0 {
			out[2*g], out[2*g+1] = -1, -1
			continue
		}
		// The last capture is the value CPython reports for a repeated group.
		c := grp.Captures[len(grp.Captures)-1]
		out[2*g] = sub.byteOff(c.Index)
		out[2*g+1] = sub.byteOff(c.Index + c.Length)
	}
	return out
}

// FindAllIndex is re.finditer/findall. It returns [start, end] byte offsets
// of the whole match. n < 0 returns all matches; n >= 0 returns at most n.
func (re *Regexp) FindAllIndex(s string, n int) [][]int {
	if locs, ok := re.allRE2(s, n); ok {
		return locs
	}
	sub := newSubject(s)
	var out [][]int
	re.scan(sub, n, func(lo, hi int) {
		out = append(out, []int{sub.byteOff(lo), sub.byteOff(hi)})
	})
	return out
}

// Sub is re.sub(p, repl, s) with a literal replacement.
func (re *Regexp) Sub(repl, s string) string { return re.SubN(repl, s, 0) }

// SubN is re.sub(p, repl, s, count=count) with a literal replacement. count 0
// means all matches. Like CPython, a negative count replaces nothing.
func (re *Regexp) SubN(repl, s string, count int) string {
	limit := count
	if count == 0 {
		limit = -1
	} else if count < 0 {
		return s
	}
	var b strings.Builder
	prev := 0
	if locs, ok := re.allRE2(s, limit); ok {
		for _, loc := range locs {
			b.WriteString(s[prev:loc[0]])
			b.WriteString(repl)
			prev = loc[1]
		}
		b.WriteString(s[prev:])
		return b.String()
	}
	sub := newSubject(s)
	re.scan(sub, limit, func(lo, hi int) {
		b.WriteString(s[prev:sub.byteOff(lo)])
		b.WriteString(repl)
		prev = sub.byteOff(hi)
	})
	b.WriteString(s[prev:])
	return b.String()
}

// Split is re.split(p, s, maxsplit=maxsplit) for patterns without capture
// groups. maxsplit 0 means no limit. A negative maxsplit splits nothing,
// as in CPython.
func (re *Regexp) Split(s string, maxsplit int) []string {
	if re.ncap > 0 {
		panic("pyre: Split does not support capture groups in " + re.pat)
	}
	limit := maxsplit
	if maxsplit == 0 {
		limit = -1
	} else if maxsplit < 0 {
		limit = 0
	}
	var out []string
	prev := 0
	if locs, ok := re.allRE2(s, limit); ok {
		for _, loc := range locs {
			out = append(out, s[prev:loc[0]])
			prev = loc[1]
		}
		return append(out, s[prev:])
	}
	sub := newSubject(s)
	re.scan(sub, limit, func(lo, hi int) {
		out = append(out, s[prev:sub.byteOff(lo)])
		prev = sub.byteOff(hi)
	})
	return append(out, s[prev:])
}

// allRE2 returns every match of re in s as [start, end] byte offsets, the
// way scan would find them, when RE2 gives the same answer. That holds when
// the pattern is exact in RE2 (re2 is set), it cannot match the empty string
// (nonNull), and the input is valid UTF-8 without WTF-8 surrogates. RE2's
// all-matches iteration then sees the same leftmost-first matches and the
// same context before each search position. n is scan's limit: negative
// means no limit. ok is false when the scan must run on regexp2.
func (re *Regexp) allRE2(s string, n int) (locs [][]int, ok bool) {
	if !re.nonNull || !utf8.ValidString(s) || hasSurrogateBytes(s) {
		return nil, false
	}
	if re.prefix != "" && isASCII(s) && !containsFoldASCII(s, re.prefix) {
		return nil, true
	}
	if re.run != nil {
		if n == 0 {
			return nil, true
		}
		return re.run.matches(s, n), true
	}
	if re.re2 != nil {
		return re.re2.FindAllStringIndex(s, n), true
	}
	if re.reASCII != nil && isASCII(s) {
		return re.reASCII.FindAllStringIndex(s, n), true
	}
	return nil, false
}

// isASCII reports whether s has no byte >= 0x80.
func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

// canBeEmpty reports whether n can match the empty string. Assertions and
// lookarounds count as empty, which is the safe answer.
func canBeEmpty(n *node) bool {
	switch n.kind {
	case nSet:
		return false
	case nCat:
		for _, s := range n.subs {
			if !canBeEmpty(s) {
				return false
			}
		}
		return true
	case nAlt:
		for _, s := range n.subs {
			if canBeEmpty(s) {
				return true
			}
		}
		return false
	case nRep:
		return n.min == 0 || canBeEmpty(n.subs[0])
	case nCap:
		return canBeEmpty(n.subs[0])
	}
	return true
}

// decodeAt returns the code point at byte offset i and its width. It reads
// WTF-8 surrogates (ED A0..BF xx) as one code point, as pyjson produces them.
// Other invalid bytes decode as U+FFFD with width 1, as []rune does.
func decodeAt(s string, i int) (rune, int) {
	if s[i] == 0xED && i+2 < len(s) && s[i+1] >= 0xA0 && s[i+1] <= 0xBF && s[i+2] >= 0x80 && s[i+2] <= 0xBF {
		return 0xD000 | rune(s[i+1]&0x3F)<<6 | rune(s[i+2]&0x3F), 3
	}
	return utf8.DecodeRuneInString(s[i:])
}

// hasSurrogateBytes reports whether s contains a WTF-8 encoded surrogate.
// RE2 reads such bytes as invalid UTF-8, so those inputs go to regexp2.
func hasSurrogateBytes(s string) bool {
	for i := 0; i+1 < len(s); i++ {
		if s[i] == 0xED && s[i+1] >= 0xA0 {
			return true
		}
	}
	return false
}
