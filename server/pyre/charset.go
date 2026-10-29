package pyre

import (
	"embed"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// The tables below are measured by server/parity/gen_unicode.py with CPython,
// not taken from Go's unicode package. Go's Unicode version differs from
// CPython's, and only CPython's answers define what a Python pattern matches.
//
//go:embed testdata/unicode_classes.txt testdata/casefold_classes.txt
var tableFS embed.FS

const maxRune = 0x10FFFF

// span is an inclusive range of code points.
type span struct{ lo, hi rune }

// charSet is a sorted, disjoint, non-adjacent list of spans.
type charSet []span

func (cs charSet) normalize() charSet {
	if len(cs) == 0 {
		return cs
	}
	sort.Slice(cs, func(i, j int) bool { return cs[i].lo < cs[j].lo })
	out := cs[:1]
	for _, s := range cs[1:] {
		last := &out[len(out)-1]
		if s.lo <= last.hi+1 {
			if s.hi > last.hi {
				last.hi = s.hi
			}
			continue
		}
		out = append(out, s)
	}
	return out
}

func union(a, b charSet) charSet {
	c := make(charSet, 0, len(a)+len(b))
	c = append(c, a...)
	c = append(c, b...)
	return c.normalize()
}

// complement returns the code points in [0, maxRune] that are not in cs. Surrogates
// are included: Python matches a lone surrogate with ., [^a] and the like, and
// pyjson keeps such strings as WTF-8, which this package decodes as one code point.
func complement(cs charSet) charSet {
	var out charSet
	next := rune(0)
	for _, s := range cs {
		if s.lo > next {
			out = append(out, span{next, s.lo - 1})
		}
		next = s.hi + 1
	}
	if next <= maxRune {
		out = append(out, span{next, maxRune})
	}
	return out
}

// tables holds the CPython classes and the IGNORECASE classes.
type tables struct {
	word, digit, space charSet
	// fold maps every cased code point to the other members of its
	// IGNORECASE class. foldKeys is the sorted list of those code points.
	fold     map[rune][]rune
	foldKeys []rune
}

var (
	tablesOnce sync.Once
	tab        *tables
	tabErr     error
)

func loadTables() *tables {
	tablesOnce.Do(func() {
		tab, tabErr = parseTables()
	})
	if tabErr != nil {
		panic("pyre: bad embedded table: " + tabErr.Error())
	}
	return tab
}

func parseHex(f string) (rune, error) {
	v, err := strconv.ParseUint(f, 16, 32)
	if err != nil || v > maxRune {
		return 0, fmt.Errorf("bad code point %q", f)
	}
	return rune(v), nil
}

func parseTables() (*tables, error) {
	raw, err := tableFS.ReadFile("testdata/unicode_classes.txt")
	if err != nil {
		return nil, err
	}
	t := &tables{fold: map[rune][]rune{}}
	for _, line := range strings.Split(string(raw), "\n") {
		if line == "" || line[0] == '#' {
			continue
		}
		name, rest, _ := strings.Cut(line, " ")
		var cs charSet
		for _, f := range strings.Fields(rest) {
			lo, hi, ok := strings.Cut(f, "-")
			if !ok {
				return nil, fmt.Errorf("bad range %q", f)
			}
			l, err1 := parseHex(lo)
			h, err2 := parseHex(hi)
			if err1 != nil || err2 != nil {
				return nil, fmt.Errorf("bad range %q", f)
			}
			cs = append(cs, span{l, h})
		}
		switch name {
		case "W":
			t.word = cs.normalize()
		case "D":
			t.digit = cs.normalize()
		case "S":
			t.space = cs.normalize()
		}
	}
	if t.word == nil || t.digit == nil || t.space == nil {
		return nil, fmt.Errorf("missing class in unicode_classes.txt")
	}

	raw, err = tableFS.ReadFile("testdata/casefold_classes.txt")
	if err != nil {
		return nil, err
	}
	for _, line := range strings.Split(string(raw), "\n") {
		if line == "" || line[0] == '#' {
			continue
		}
		var group []rune
		for _, f := range strings.Fields(line) {
			c, err := parseHex(f)
			if err != nil {
				return nil, err
			}
			group = append(group, c)
		}
		for _, c := range group {
			others := make([]rune, 0, len(group)-1)
			for _, o := range group {
				if o != c {
					others = append(others, o)
				}
			}
			t.fold[c] = others
			t.foldKeys = append(t.foldKeys, c)
		}
	}
	sort.Slice(t.foldKeys, func(i, j int) bool { return t.foldKeys[i] < t.foldKeys[j] })
	return t, nil
}

// foldSet returns cs plus every IGNORECASE partner of its members. This is
// how CPython's re.I treats literals and classes: a literal c matches every
// member of c's class, and a class matches a character when one of its
// members is in that character's class.
func foldSet(cs charSet) charSet {
	t := loadTables()
	var extra charSet
	for _, s := range cs {
		i := sort.Search(len(t.foldKeys), func(i int) bool { return t.foldKeys[i] >= s.lo })
		for ; i < len(t.foldKeys) && t.foldKeys[i] <= s.hi; i++ {
			for _, o := range t.fold[t.foldKeys[i]] {
				extra = append(extra, span{o, o})
			}
		}
	}
	if len(extra) == 0 {
		return cs
	}
	return union(cs, extra)
}
