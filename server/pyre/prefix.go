package pyre

// A required prefix is the ASCII literal that every match must start with,
// compared without case. Go's regexp scans every byte for a pattern such as
// (?i)i hope or (```.*?```) that has no literal in front of the scan, so a
// text without the literal is cheaper to rule out with a plain search. The
// check only applies to ASCII input, where case folding cannot relate a
// subject byte to a different pattern letter, so a missing literal means the
// pattern has no match there.

// literalPrefix returns the runes every match of n starts with, as far as the
// tree shows them: single-character sets in a leading sequence, through
// groups and past zero-width assertions. It stops at anything else.
func literalPrefix(n *node) []rune {
	switch n.kind {
	case nCap:
		return literalPrefix(n.subs[0])
	case nSet:
		if r, ok := singleRune(n.set); ok {
			return []rune{r}
		}
	case nCat:
		var out []rune
		for _, s := range n.subs {
			if s.kind == nAssert || s.kind == nEmpty {
				continue
			}
			if s.kind == nSet {
				if r, ok := singleRune(s.set); ok {
					out = append(out, r)
					continue
				}
				break
			}
			out = append(out, literalPrefix(s)...)
			break
		}
		return out
	}
	return nil
}

// singleRune reports the one rune that cs holds, or the lower-case ASCII
// letter of a case pair {X, x}, which is what (?i) produces for a letter.
func singleRune(cs charSet) (rune, bool) {
	switch {
	case len(cs) == 1 && cs[0].lo == cs[0].hi:
		return cs[0].lo, true
	case len(cs) == 2 && cs[0].lo == cs[0].hi && cs[1].lo == cs[1].hi:
		a, b := cs[0].lo, cs[1].lo
		if a > b {
			a, b = b, a
		}
		if a >= 'A' && a <= 'Z' && b == a+32 {
			return b, true
		}
	}
	return 0, false
}

// asciiPrefix returns the lower-case form of the prefix when every rune of it
// is ASCII and the prefix is not empty. Otherwise it returns "".
func asciiPrefix(rs []rune) string {
	if len(rs) == 0 {
		return ""
	}
	b := make([]byte, len(rs))
	for i, r := range rs {
		if r >= 0x80 {
			return ""
		}
		if r >= 'A' && r <= 'Z' {
			r += 32
		}
		b[i] = byte(r)
	}
	return string(b)
}

// containsFoldASCII reports whether the lower-case ASCII string p occurs in
// s, ignoring ASCII case. s must be ASCII.
func containsFoldASCII(s, p string) bool {
	n := len(p)
	for i := 0; i+n <= len(s); i++ {
		if lowerASCII(s[i]) != p[0] {
			continue
		}
		j := 1
		for j < n && lowerASCII(s[i+j]) == p[j] {
			j++
		}
		if j == n {
			return true
		}
	}
	return false
}

func lowerASCII(c byte) byte {
	if c >= 'A' && c <= 'Z' {
		return c + 32
	}
	return c
}
