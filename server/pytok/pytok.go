// Package pytok counts and encodes text the way tiktoken's Python package does
// for the cl100k_base encoding. Its encoder is tiktoken-go, loaded from the
// offline BPE data embedded in tiktoken-go-loader, so nothing is downloaded.
//
// Python behaviour reproduced here:
//   - encode() raises ValueError when the text contains one of the special
//     tokens, because disallowed_special defaults to "all". Here that is an
//     error wrapping ErrDisallowedSpecial.
//   - Lone surrogates in a str are not encodable as UTF-8. tiktoken's
//     fallback then joins surrogate pairs and replaces lone ones with U+FFFD,
//     and pytok does the same.
//   - count_tokens-style helpers return 0 for "" without encoding, as the
//     engines' "if not text: return 0" does.
package pytok

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/pkoukk/tiktoken-go"
	load "github.com/pkoukk/tiktoken-go-loader"
)

// ErrDisallowedSpecial is returned when the text contains a special token.
// Python raises ValueError in the same case.
var ErrDisallowedSpecial = errors.New("text contains a disallowed special token")

// specialTokens are the special tokens of cl100k_base. The encoding's own
// set is not exposed by tiktoken-go, so the list is fixed here; the test
// checks it against the oracle fixture.
var specialTokens = []string{
	"<|endoftext|>",
	"<|fim_prefix|>",
	"<|fim_middle|>",
	"<|fim_suffix|>",
	"<|endofprompt|>",
}

var (
	encOnce sync.Once
	enc     *tiktoken.Tiktoken
	encErr  error
)

func encoding() (*tiktoken.Tiktoken, error) {
	encOnce.Do(func() {
		// The default loader would download the BPE file. The offline loader
		// reads the copy embedded in the module instead.
		tiktoken.SetBpeLoader(load.NewOfflineLoader())
		enc, encErr = tiktoken.GetEncoding("cl100k_base")
	})
	return enc, encErr
}

// findSpecial returns the special token that appears first in text, or "".
func findSpecial(text string) string {
	best, at := "", -1
	for _, tok := range specialTokens {
		if i := strings.Index(text, tok); i >= 0 && (at < 0 || i < at) {
			best, at = tok, i
		}
	}
	return best
}

// normalizeSurrogates rewrites WTF-8 surrogates the way tiktoken does: a high
// surrogate followed by a low one becomes one code point, and every other
// surrogate becomes U+FFFD. Strings without surrogates are returned as is.
func normalizeSurrogates(s string) string {
	if !hasSurrogates(s) {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); {
		u, ok := surrogateAt(s, i)
		if !ok {
			b.WriteByte(s[i])
			i++
			continue
		}
		if u >= 0xD800 && u <= 0xDBFF {
			if lo, ok := surrogateAt(s, i+3); ok && lo >= 0xDC00 && lo <= 0xDFFF {
				b.WriteRune(0x10000 + (u-0xD800)<<10 + (lo - 0xDC00))
				i += 6
				continue
			}
		}
		b.WriteRune(0xFFFD)
		i += 3
	}
	return b.String()
}

// hasSurrogates reports whether s holds a WTF-8 encoded surrogate (ED A0..BF xx).
func hasSurrogates(s string) bool {
	for i := 0; i+1 < len(s); i++ {
		if s[i] == 0xED && s[i+1] >= 0xA0 {
			return true
		}
	}
	return false
}

// surrogateAt decodes a WTF-8 surrogate that starts at byte i.
func surrogateAt(s string, i int) (rune, bool) {
	if i+2 < len(s) && s[i] == 0xED && s[i+1] >= 0xA0 && s[i+1] <= 0xBF && s[i+2] >= 0x80 && s[i+2] <= 0xBF {
		return 0xD000 | rune(s[i+1]&0x3F)<<6 | rune(s[i+2]&0x3F), true
	}
	return 0, false
}

// Encode returns the cl100k_base token ids of text, as tiktoken's
// encode(text) does with the default special-token handling. It returns an
// error wrapping ErrDisallowedSpecial when text contains a special token.
func Encode(text string) ([]int, error) {
	if tok := findSpecial(text); tok != "" {
		return nil, fmt.Errorf("%w: %q", ErrDisallowedSpecial, tok)
	}
	text = normalizeSurrogates(text)
	// No special token is present, so the ordinary encoder is exactly
	// encode(text) with disallowed_special="all". encodeFast reproduces that
	// encoder without regexp2; the tiktoken path below is the fallback.
	if ids, ok := encodeFast(text); ok {
		if ids == nil {
			ids = []int{}
		}
		return ids, nil
	}
	e, err := encoding()
	if err != nil {
		return nil, fmt.Errorf("pytok: loading cl100k_base: %w", err)
	}
	ids := e.EncodeOrdinary(text)
	if ids == nil {
		ids = []int{}
	}
	return ids, nil
}

// CountTokens returns len(Encode(text)), with Python's shortcut: an empty
// text has 0 tokens and is not encoded.
func CountTokens(text string) (int, error) {
	if text == "" {
		return 0, nil
	}
	ids, err := Encode(text)
	if err != nil {
		return 0, err
	}
	return len(ids), nil
}

// idsHash is the fingerprint the oracle fixture stores: the first 16 hex
// digits of SHA-256 over the ids joined by spaces.
func idsHash(ids []int) string {
	var b strings.Builder
	for i, id := range ids {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprint(&b, id)
	}
	sum := sha256.Sum256([]byte(b.String()))
	return fmt.Sprintf("%x", sum[:8])
}
