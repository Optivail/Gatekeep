package pytok

import (
	"math"
	"sync"
	"unicode"

	load "github.com/pkoukk/tiktoken-go-loader"
)

// This file replaces the regexp2 pre-tokenizer that tiktoken-go runs for
// every call. regexp2 is a backtracking engine and was most of the CPU time
// of the proxy. The scanner below is a hand-written matcher for the exact
// cl100k_base pattern that tiktoken-go v0.1.8 uses:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// The alternatives are tried in order at each position, as the regex engine
// does, and each one is resolved with the backtracking result it would give.
// \s is unicode.IsSpace, which is what regexp2 uses; \p{L} and \p{N} are the
// unicode.Letter and unicode.Number tables. The fallback in Encode covers
// the one case where case folding could make the two differ, and the
// differential test in pretok_test.go compares against tiktoken on random
// input.

const cl100kBpeURL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

var (
	ranksOnce sync.Once
	ranks     map[string]int
	ranksErr  error
)

// cl100kRanks returns the cl100k_base merge ranks from the embedded data.
// The same offline loader that tiktoken uses reads them.
func cl100kRanks() (map[string]int, error) {
	ranksOnce.Do(func() {
		ranks, ranksErr = load.NewOfflineLoader().LoadTiktokenBpe(cl100kBpeURL)
	})
	return ranks, ranksErr
}

// encodeFast is encode(text) with the ordinary encoder, for text that holds
// no special token and no surrogates. ok is false when the text needs the
// regexp2 path instead.
func encodeFast(text string) ([]int, bool) {
	rk, err := cl100kRanks()
	if err != nil {
		return nil, false
	}
	rs := []rune(text)
	out := make([]int, 0, len(rs)/3+4)
	for i := 0; i < len(rs); {
		n, ok := pretokLen(rs, i)
		if !ok {
			return nil, false
		}
		piece := string(rs[i : i+n])
		if tok, found := rk[piece]; found {
			out = append(out, tok)
		} else {
			out = append(out, bytePairEncode([]byte(piece), rk)...)
		}
		i += n
	}
	return out, true
}

// isLetter is \p{L}.
func isLetter(r rune) bool { return unicode.IsLetter(r) }

// isNumber is \p{N}.
func isNumber(r rune) bool { return unicode.IsNumber(r) }

// isSpace is \s.
func isSpace(r rune) bool { return unicode.IsSpace(r) }

// classOther is [^\s\p{L}\p{N}].
func classOther(r rune) bool { return !isSpace(r) && !isLetter(r) && !isNumber(r) }

// classNotNL is [^\r\n\p{L}\p{N}].
func classNotNL(r rune) bool {
	return r != '\r' && r != '\n' && !isLetter(r) && !isNumber(r)
}

// pretokLen returns the length in runes of the pre-token that starts at i,
// following the alternatives of the cl100k pattern in order. ok is false
// when the case is not modelled here (a non-ASCII rune right after an
// apostrophe, where (?i) could fold a letter).
func pretokLen(rs []rune, i int) (int, bool) {
	n := len(rs)
	c := rs[i]

	// 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if c == '\'' && i+1 < n {
		d := rs[i+1]
		if d >= 0x80 {
			return 0, false
		}
		switch d | 0x20 { // ASCII lower case
		case 's', 't', 'm', 'd':
			return 2, true
		case 'r', 'v', 'l':
			if i+2 < n {
				e := rs[i+2]
				if e < 0x80 {
					switch d | 0x20 {
					case 'r', 'v':
						if e|0x20 == 'e' {
							return 3, true
						}
					case 'l':
						if e|0x20 == 'l' {
							return 3, true
						}
					}
				}
			}
		}
	}

	// 2. [^\r\n\p{L}\p{N}]?\p{L}+
	if classNotNL(c) {
		if i+1 < n && isLetter(rs[i+1]) {
			j := i + 2
			for j < n && isLetter(rs[j]) {
				j++
			}
			return j - i, true
		}
	} else if isLetter(c) {
		j := i + 1
		for j < n && isLetter(rs[j]) {
			j++
		}
		return j - i, true
	}

	// 3. \p{N}{1,3}
	if isNumber(c) {
		j := i + 1
		for j < n && j-i < 3 && isNumber(rs[j]) {
			j++
		}
		return j - i, true
	}

	// 4. ?[^\s\p{L}\p{N}]+[\r\n]*
	start := -1
	if c == ' ' && i+1 < n && classOther(rs[i+1]) {
		start = i + 1
	} else if classOther(c) {
		start = i
	}
	if start >= 0 {
		j := start
		for j < n && classOther(rs[j]) {
			j++
		}
		for j < n && (rs[j] == '\r' || rs[j] == '\n') {
			j++
		}
		return j - i, true
	}

	// The remaining alternatives all start with whitespace.
	if !isSpace(c) {
		// Not reachable: every rune is a letter, a number, a space or
		// classOther. Report it so the caller falls back.
		return 0, false
	}
	w := i
	for w < n && isSpace(rs[w]) {
		w++
	}

	// 5. \s*[\r\n]+ : ends at the last newline inside the whitespace run.
	last := -1
	for k := i; k < w; k++ {
		if rs[k] == '\r' || rs[k] == '\n' {
			last = k
		}
	}
	if last >= 0 {
		return last - i + 1, true
	}

	// 6. \s+(?!\S) gives one rune back when more whitespace precedes a
	// non-space. 7. \s+ takes the whole run otherwise.
	l := w - i
	if w < n && l >= 2 {
		return l - 1, true
	}
	return l, true
}

// bytePairEncode is tiktoken-go's bytePairEncode: the BPE merge of piece
// with the given ranks. It is kept step for step with the original so that
// tie-breaks and the missing-rank value (0 for a single rune) are the same.
func bytePairEncode(piece []byte, ranks map[string]int) []int {
	if len(piece) == 1 {
		return []int{ranks[string(piece)]}
	}
	parts := make([][2]int, len(piece)+1)
	for i := range parts {
		parts[i][0], parts[i][1] = i, math.MaxInt
	}
	getRank := func(startIdx, skip int) int {
		if startIdx+skip+2 < len(parts) {
			b := piece[parts[startIdx][0]:parts[startIdx+skip+2][0]]
			if rank, ok := ranks[string(b)]; ok {
				return rank
			}
		}
		return -1
	}
	for i := 0; i < len(parts)-2; i++ {
		if rank := getRank(i, 0); rank >= 0 {
			parts[i][1] = rank
		}
	}
	for len(parts) > 1 {
		minRank, minIdx := math.MaxInt, -1
		for i := 0; i < len(parts)-1; i++ {
			if parts[i][1] < minRank {
				minRank, minIdx = parts[i][1], i
			}
		}
		if minRank >= math.MaxInt {
			break
		}
		i := minIdx
		if rank := getRank(i, 1); rank >= 0 {
			parts[i][1] = rank
		} else {
			parts[i][1] = math.MaxInt
		}
		if i > 0 {
			if rk := getRank(i-1, 1); rk >= 0 {
				parts[i-1][1] = rk
			} else {
				parts[i-1][1] = math.MaxInt
			}
		}
		parts = append(parts[:i+1], parts[i+2:]...)
	}
	out := make([]int, len(parts)-1)
	for i := range out {
		out[i] = ranks[string(piece[parts[i][0]:parts[i+1][0]])]
	}
	return out
}
