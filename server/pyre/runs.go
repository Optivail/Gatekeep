package pyre

import (
	"sort"
	"unicode/utf8"
)

// runPattern describes a pattern of the form [set]{min,} with min >= 1,
// greedy, with no groups. Its matches are exactly the maximal runs of set
// members of length at least min: a match starts at the first member of a
// run, is greedy, and nothing follows it that could force a shorter match.
// A run shorter than min has no match at any of its positions.
type runPattern struct {
	set charSet
	min int
	// ascii[c] is set.has(c) for c < 128, so the common bytes skip the search.
	ascii [128]bool
}

// runFor returns the run form of root, or nil when root is not that shape.
func runFor(root *node) *runPattern {
	if root.kind != nRep || root.lazy || root.max >= 0 || root.min < 1 {
		return nil
	}
	sub := root.subs[0]
	if sub.kind != nSet {
		return nil
	}
	rp := &runPattern{set: sub.set, min: root.min}
	for c := range rp.ascii {
		rp.ascii[c] = sub.set.has(rune(c))
	}
	return rp
}

// has reports whether r is in the set, by binary search over the spans.
func (cs charSet) has(r rune) bool {
	i := sort.Search(len(cs), func(i int) bool { return cs[i].hi >= r })
	return i < len(cs) && cs[i].lo <= r
}

// matches returns the spans of s that the run pattern matches, as byte
// offsets, at most n of them when n >= 0. s must be valid UTF-8.
func (rp *runPattern) matches(s string, n int) [][]int {
	var out [][]int
	start, count := -1, 0
	flush := func(end int) bool {
		if start >= 0 && count >= rp.min {
			out = append(out, []int{start, end})
			if n >= 0 && len(out) >= n {
				return false
			}
		}
		start, count = -1, 0
		return true
	}
	for i := 0; i < len(s); {
		r, w := rune(s[i]), 1
		if r >= utf8.RuneSelf {
			r, w = utf8.DecodeRuneInString(s[i:])
		}
		var in bool
		if r < utf8.RuneSelf {
			in = rp.ascii[r]
		} else {
			in = rp.set.has(r)
		}
		if in {
			if start < 0 {
				start = i
			}
			count++
		} else if !flush(i) {
			return out
		}
		i += w
	}
	flush(len(s))
	return out
}
