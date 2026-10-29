package engines

import "strings"

// Hand-written matchers for the three caveman patterns in optimizer.go. They
// reproduce, for ASCII-only input, exactly what Python's re.sub("", ...) does
// with these patterns (CPython semantics, see pyre):
//
//	_FILLER_PREAMBLE = (?im)^[ \t]*(so|well|basically|essentially|actually|just|simply)\b[,:]?\s*
//	_POLITE_AUX      = (?i)\b(could you please|would you please|can you please|could you|would you|can you)\b\s*
//	_DETERMINERS     = (?i)\b(a|an|the)\b ?
//
// Why ASCII input is enough: the matched bytes are ASCII letters, spaces and
// punctuation, and \w, \s and \b on an ASCII string are the ASCII sets (CPython
// measured: \w = [0-9A-Za-z_], \s = \t \n \v \f \r \x1c-\x1f and space). Case
// folding of an ASCII letter maps only to its ASCII counterpart for these
// words (the non-ASCII folds of s and k do not occur in ASCII input). Inputs
// with any non-ASCII byte go to the pyre regexes, so they are unchanged.
// caveman_ascii_test.go checks the equality against the regexes.

// cavemanWordsASCII runs the three substitutions in the Python order.
func cavemanWordsASCII(text string) string {
	return stripDeterminersASCII(stripPoliteAuxASCII(stripFillerPreambleASCII(text)))
}

func isWordByte(c byte) bool {
	return c == '_' || ('0' <= c && c <= '9') || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z')
}

// isSpaceByte is \s for ASCII text.
func isSpaceByte(c byte) bool {
	return c == ' ' || (9 <= c && c <= 13) || (28 <= c && c <= 31)
}

func lowerByte(c byte) byte {
	if 'A' <= c && c <= 'Z' {
		return c + ('a' - 'A')
	}
	return c
}

// wordRunEnd returns the end of the run of \w bytes that starts at i.
func wordRunEnd(s string, i int) int {
	j := i
	for j < len(s) && isWordByte(s[j]) {
		j++
	}
	return j
}

// hasPrefixFoldASCII reports whether s[i:] starts with the lower-case ASCII
// word w, ignoring ASCII case.
func hasPrefixFoldASCII(s string, i int, w string) bool {
	if len(s)-i < len(w) {
		return false
	}
	for k := 0; k < len(w); k++ {
		if lowerByte(s[i+k]) != w[k] {
			return false
		}
	}
	return true
}

// equalFoldASCII reports whether s equals the lower-case ASCII word w, ignoring ASCII case.
func equalFoldASCII(s, w string) bool {
	return len(s) == len(w) && hasPrefixFoldASCII(s, 0, w)
}

// stripper collects the kept text around removed matches.
type stripper struct {
	src  string
	out  strings.Builder
	last int // src[last:] is not yet copied to out
	hit  bool
}

// remove drops src[start:end] from the output.
func (st *stripper) remove(start, end int) {
	st.out.WriteString(st.src[st.last:start])
	st.last = end
	st.hit = true
}

// result returns the substituted text, or the input itself when nothing matched.
func (st *stripper) result() string {
	if !st.hit {
		return st.src
	}
	st.out.WriteString(st.src[st.last:])
	return st.out.String()
}

// stripFillerPreambleASCII mirrors _FILLER_PREAMBLE.sub("", text).
func stripFillerPreambleASCII(text string) string {
	st := &stripper{src: text}
	s := text
	i := 0
	for i < len(s) {
		// ^ with re.M: start of the string or just after a newline.
		if i > 0 && s[i-1] != '\n' {
			i++
			continue
		}
		p := i
		for p < len(s) && (s[p] == ' ' || s[p] == '\t') {
			p++
		}
		q := wordRunEnd(s, p)
		if q == p || !isFillerWord(s[p:q]) {
			i++
			continue
		}
		k := q
		if k < len(s) && (s[k] == ',' || s[k] == ':') {
			k++
		}
		for k < len(s) && isSpaceByte(s[k]) {
			k++
		}
		st.remove(i, k)
		i = k
	}
	return st.result()
}

var fillerWords = [...]string{"so", "well", "basically", "essentially", "actually", "just", "simply"}

func isFillerWord(w string) bool {
	for _, f := range fillerWords {
		if equalFoldASCII(w, f) {
			return true
		}
	}
	return false
}

// politeAuxPhrases is in the alternation order of _POLITE_AUX.
var politeAuxPhrases = [...]string{"could you please", "would you please", "can you please", "could you", "would you", "can you"}

// stripPoliteAuxASCII mirrors _POLITE_AUX.sub("", text).
func stripPoliteAuxASCII(text string) string {
	st := &stripper{src: text}
	s := text
	i := 0
	for i < len(s) {
		// \b before the first letter of every alternative.
		if !isWordByte(s[i]) || (i > 0 && isWordByte(s[i-1])) {
			i++
			continue
		}
		end := -1
		for _, phrase := range politeAuxPhrases {
			if hasPrefixFoldASCII(s, i, phrase) {
				e := i + len(phrase)
				if e == len(s) || !isWordByte(s[e]) {
					end = e
					break
				}
			}
		}
		if end < 0 {
			i++
			continue
		}
		for end < len(s) && isSpaceByte(s[end]) {
			end++
		}
		st.remove(i, end)
		i = end
	}
	return st.result()
}

// stripDeterminersASCII mirrors _DETERMINERS.sub("", text). A match is a whole
// word run equal to a, an or the, plus one optional space.
func stripDeterminersASCII(text string) string {
	st := &stripper{src: text}
	s := text
	i := 0
	for i < len(s) {
		if !isWordByte(s[i]) || (i > 0 && isWordByte(s[i-1])) {
			i++
			continue
		}
		j := wordRunEnd(s, i)
		w := s[i:j]
		if equalFoldASCII(w, "a") || equalFoldASCII(w, "an") || equalFoldASCII(w, "the") {
			end := j
			if end < len(s) && s[end] == ' ' {
				end++
			}
			st.remove(i, end)
			i = end
			continue
		}
		i = j
	}
	return st.result()
}

// fillerTriggers[i] lists lower-case words such that every match of
// fillerPatterns[i] contains one of them (case-insensitively). The pattern
// order matches fillerPatterns. For ASCII text that has none of the words,
// the pattern cannot match, so its Sub is skipped; the result is unchanged.
var fillerTriggers = [...][]string{
	{"certainly", "sure", "of course", "great question", "absolutely"},
	{"i hope "},
	{"please let me know if you "},
	{"be happy to help"},
	{"feel free to "},
	{"thank you for "},
}

// stripFillers applies fillerPatterns in order, exactly as repeated Sub calls
// would. The trigger check runs on a lower-case copy that is rebuilt whenever
// a substitution shortens the text.
func stripFillers(text string) string {
	ascii := isASCII(text)
	var lower string
	if ascii {
		lower = strings.ToLower(text)
	}
	for i, p := range fillerPatterns {
		if ascii && !containsAnyWord(lower, fillerTriggers[i]) {
			continue
		}
		next := p.Sub("", text)
		if len(next) != len(text) { // a match was removed, so the text shrank
			text = next
			if ascii {
				lower = strings.ToLower(text)
			}
		}
	}
	return text
}

func containsAnyWord(s string, words []string) bool {
	for _, w := range words {
		if strings.Contains(s, w) {
			return true
		}
	}
	return false
}
