package pyre

import (
	"math/rand"
	"reflect"
	"testing"
)

// The RE2 all-matches path must give the same spans as the regexp2 scan it
// replaces. This test runs both on random input for patterns that take the
// fast path.
func TestAllRE2MatchesScan(t *testing.T) {
	cases := []struct {
		pat   string
		flags []Flags
	}{
		{`\n{3,}`, nil},
		{`[ \t]+`, nil},
		{"(```.*?```)", []Flags{DotAll}},
		{"(`[^`\n]+`)", nil},
		{`x+`, nil},
		{`a|ab`, nil},
		{`(ab|a)c?`, nil},
		{`(?i)[a-c]+`, nil},
		{`[^\s]+`, nil},
		{`\d+`, nil},
		{`[a-c]{2,}`, nil},
		{`(?i)\b(a|an|the)\b ?`, nil},
		{`(?im)^[ \t]*(so|well)\b[,:]?\s*`, nil},
		{`\bx\B`, nil},
		{`(?i)i hope (this|that) helps!?\.?`, nil},
		{`(?i)feel free to (ask|reach out) if you (have|need) [^.\n]*\.?`, nil},
		{"(```.*?```)", []Flags{DotAll}},
	}
	alphabet := []string{"a", "b", "c", "x", "A", "B", " ", "\t", "\n", "\r", "`", "```", "1", "é", " ", "—", "\U0001F600", "�"}
	ascii := []string{"a", "b", "c", "x", "A", "B", " ", "\t", "\n", "\r", "`", "```", "1", "the", "The", "an", "so", "well", ",", "_", "-", "x9", "I hope ", "i hope that helps", "Feel Free To reach out if you need ", "```"}
	rng := rand.New(rand.NewSource(7))
	for _, c := range cases {
		re := MustCompile(c.pat, c.flags...)
		if (re.re2 == nil && re.run == nil && re.reASCII == nil) || !re.nonNull {
			t.Fatalf("%q: expected the fast path (re2=%v run=%v ascii=%v nonNull=%v)", c.pat, re.re2 != nil, re.run != nil, re.reASCII != nil, re.nonNull)
		}
		for i := 0; i < 3000; i++ {
			// Odd rounds draw from the ASCII alphabet only, so the ASCII-
			// specific RE2 form is compared with the scan too.
			pool := alphabet
			if i%2 == 1 {
				pool = ascii
			}
			s := ""
			for k := rng.Intn(30); k > 0; k-- {
				s += pool[rng.Intn(len(pool))]
			}
			for _, n := range []int{-1, 0, 1, 2} {
				want := slowAll(re, s, n)
				got := re.FindAllIndex(s, n)
				if !reflect.DeepEqual(normalize(got), normalize(want)) {
					t.Fatalf("%q n=%d on %q: fast %v slow %v", c.pat, n, s, got, want)
				}
			}
			// SubN and Split use the same spans, so compare them too.
			if g, w := re.Sub("<>", s), slowSub(re, s, "<>"); g != w {
				t.Fatalf("%q Sub on %q: fast %q slow %q", c.pat, s, g, w)
			}
			if !re.isGroupedSplit() {
				if g, w := re.Split(s, 0), slowSplit(re, s); !reflect.DeepEqual(g, w) {
					t.Fatalf("%q Split on %q: fast %q slow %q", c.pat, s, g, w)
				}
			}
		}
	}
}

// isGroupedSplit reports whether Split would panic for this pattern.
func (re *Regexp) isGroupedSplit() bool { return re.ncap > 0 }

func normalize(x [][]int) [][]int {
	if len(x) == 0 {
		return nil
	}
	return x
}

func slowAll(re *Regexp, s string, n int) [][]int {
	sub := newSubject(s)
	var out [][]int
	re.scan(sub, n, func(lo, hi int) {
		out = append(out, []int{sub.byteOff(lo), sub.byteOff(hi)})
	})
	return out
}

func slowSub(re *Regexp, s, repl string) string {
	sub := newSubject(s)
	out := ""
	prev := 0
	re.scan(sub, -1, func(lo, hi int) {
		out += s[prev:sub.byteOff(lo)] + repl
		prev = sub.byteOff(hi)
	})
	return out + s[prev:]
}

func slowSplit(re *Regexp, s string) []string {
	sub := newSubject(s)
	var out []string
	prev := 0
	re.scan(sub, -1, func(lo, hi int) {
		out = append(out, s[prev:sub.byteOff(lo)])
		prev = sub.byteOff(hi)
	})
	return append(out, s[prev:])
}
