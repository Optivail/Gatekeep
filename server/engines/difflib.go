// Port of the parts of CPython's difflib that optimizer.py reaches:
// SequenceMatcher(None, a, b) with autojunk, and unified_diff(..., n=3,
// lineterm=""). The output must match Python line for line, so the matching
// heuristics (autojunk popular-element pruning, longest-block tie-breaks) are
// copied from difflib.py rather than replaced with a faster diff.
package engines

import (
	"sort"
	"strconv"
)

type matchBlock struct{ a, b, size int }

type diffOp struct {
	tag            string
	i1, i2, j1, j2 int
}

type seqMatcher struct {
	a, b []string
	b2j  map[string][]int
}

// newSeqMatcher indexes b and applies autojunk: for b of 200+ lines, any line
// occurring more than n/100+1 times is dropped from the index (it can still
// be extended over, but never starts a match).
func newSeqMatcher(a, b []string) *seqMatcher {
	m := &seqMatcher{a: a, b: b, b2j: map[string][]int{}}
	for i, elt := range b {
		m.b2j[elt] = append(m.b2j[elt], i)
	}
	if n := len(b); n >= 200 {
		ntest := n/100 + 1
		for elt, idxs := range m.b2j {
			if len(idxs) > ntest {
				delete(m.b2j, elt)
			}
		}
	}
	return m
}

// findLongestMatch returns the earliest-starting longest match in the given
// windows. With no junk elements, difflib's junk-extension loops are no-ops,
// so only the non-junk extension is kept.
func (m *seqMatcher) findLongestMatch(alo, ahi, blo, bhi int) (int, int, int) {
	besti, bestj, bestsize := alo, blo, 0
	j2len := map[int]int{}
	for i := alo; i < ahi; i++ {
		newj2len := map[int]int{}
		for _, j := range m.b2j[m.a[i]] {
			if j < blo {
				continue
			}
			if j >= bhi {
				break
			}
			k := j2len[j-1] + 1
			newj2len[j] = k
			if k > bestsize {
				besti, bestj, bestsize = i-k+1, j-k+1, k
			}
		}
		j2len = newj2len
	}
	for besti > alo && bestj > blo && m.a[besti-1] == m.b[bestj-1] {
		besti, bestj, bestsize = besti-1, bestj-1, bestsize+1
	}
	for besti+bestsize < ahi && bestj+bestsize < bhi && m.a[besti+bestsize] == m.b[bestj+bestsize] {
		bestsize++
	}
	return besti, bestj, bestsize
}

// matchingBlocks mirrors get_matching_blocks, including the final sentinel
// block (la, lb, 0) and the merging of adjacent blocks.
func (m *seqMatcher) matchingBlocks() []matchBlock {
	la, lb := len(m.a), len(m.b)
	queue := [][4]int{{0, la, 0, lb}}
	var found []matchBlock
	for len(queue) > 0 {
		q := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		i, j, k := m.findLongestMatch(q[0], q[1], q[2], q[3])
		if k > 0 {
			found = append(found, matchBlock{i, j, k})
			if q[0] < i && q[2] < j {
				queue = append(queue, [4]int{q[0], i, q[2], j})
			}
			if i+k < q[1] && j+k < q[3] {
				queue = append(queue, [4]int{i + k, q[1], j + k, q[3]})
			}
		}
	}
	sort.Slice(found, func(p, q int) bool {
		x, y := found[p], found[q]
		if x.a != y.a {
			return x.a < y.a
		}
		if x.b != y.b {
			return x.b < y.b
		}
		return x.size < y.size
	})
	var i1, j1, k1 int
	var nonAdj []matchBlock
	for _, mb := range found {
		if k1 > 0 && i1+k1 == mb.a && j1+k1 == mb.b {
			k1 += mb.size
			continue
		}
		if k1 > 0 {
			nonAdj = append(nonAdj, matchBlock{i1, j1, k1})
		}
		i1, j1, k1 = mb.a, mb.b, mb.size
	}
	if k1 > 0 {
		nonAdj = append(nonAdj, matchBlock{i1, j1, k1})
	}
	return append(nonAdj, matchBlock{la, lb, 0})
}

// opcodes mirrors get_opcodes.
func (m *seqMatcher) opcodes() []diffOp {
	var out []diffOp
	i, j := 0, 0
	for _, mb := range m.matchingBlocks() {
		tag := ""
		switch {
		case i < mb.a && j < mb.b:
			tag = "replace"
		case i < mb.a:
			tag = "delete"
		case j < mb.b:
			tag = "insert"
		}
		if tag != "" {
			out = append(out, diffOp{tag, i, mb.a, j, mb.b})
		}
		i, j = mb.a+mb.size, mb.b+mb.size
		if mb.size > 0 {
			out = append(out, diffOp{"equal", mb.a, i, mb.b, j})
		}
	}
	return out
}

// groupedOpcodes mirrors get_grouped_opcodes(n): hunks with n lines of
// context, split where an equal run is longer than 2n.
func (m *seqMatcher) groupedOpcodes(n int) [][]diffOp {
	codes := m.opcodes()
	if len(codes) == 0 {
		codes = []diffOp{{"equal", 0, 1, 0, 1}}
	}
	if codes[0].tag == "equal" {
		c := codes[0]
		codes[0] = diffOp{"equal", maxInt(c.i1, c.i2-n), c.i2, maxInt(c.j1, c.j2-n), c.j2}
	}
	if last := len(codes) - 1; codes[last].tag == "equal" {
		c := codes[last]
		codes[last] = diffOp{"equal", c.i1, minInt(c.i2, c.i1+n), c.j1, minInt(c.j2, c.j1+n)}
	}
	nn := n + n
	var groups [][]diffOp
	var group []diffOp
	for _, c := range codes {
		i1, i2, j1, j2 := c.i1, c.i2, c.j1, c.j2
		if c.tag == "equal" && i2-i1 > nn {
			group = append(group, diffOp{"equal", i1, minInt(i2, i1+n), j1, minInt(j2, j1+n)})
			groups = append(groups, group)
			group = nil
			i1, j1 = maxInt(i1, i2-n), maxInt(j1, j2-n)
		}
		group = append(group, diffOp{c.tag, i1, i2, j1, j2})
	}
	if len(group) > 0 && (len(group) != 1 || group[0].tag != "equal") {
		groups = append(groups, group)
	}
	return groups
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// formatRangeUnified mirrors difflib._format_range_unified.
func formatRangeUnified(start, stop int) string {
	beginning := start + 1
	length := stop - start
	if length == 1 {
		return strconv.Itoa(beginning)
	}
	if length == 0 {
		beginning--
	}
	return strconv.Itoa(beginning) + "," + strconv.Itoa(length)
}

// unifiedDiffLines mirrors list(difflib.unified_diff(a, b, fromfile, tofile,
// n=3, lineterm="")). Dates are empty, so the header has no tab suffix.
func unifiedDiffLines(a, b []string, fromfile, tofile string) []string {
	const n = 3
	var out []string
	started := false
	for _, group := range newSeqMatcher(a, b).groupedOpcodes(n) {
		if !started {
			started = true
			out = append(out, "--- "+fromfile, "+++ "+tofile)
		}
		first, last := group[0], group[len(group)-1]
		out = append(out, "@@ -"+formatRangeUnified(first.i1, last.i2)+
			" +"+formatRangeUnified(first.j1, last.j2)+" @@")
		for _, op := range group {
			if op.tag == "equal" {
				for _, line := range a[op.i1:op.i2] {
					out = append(out, " "+line)
				}
				continue
			}
			if op.tag == "replace" || op.tag == "delete" {
				for _, line := range a[op.i1:op.i2] {
					out = append(out, "-"+line)
				}
			}
			if op.tag == "replace" || op.tag == "insert" {
				for _, line := range b[op.j1:op.j2] {
					out = append(out, "+"+line)
				}
			}
		}
	}
	return out
}
