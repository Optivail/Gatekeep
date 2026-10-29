package pytok

import (
	"math/rand"
	"strings"
	"testing"
)

// fragments cover every class the cl100k pattern branches on: contractions,
// letters, digits, whitespace and newlines, punctuation, combining marks,
// and the characters that are whitespace only in Unicode.
var fragments = []string{
	"'", "'s", "'S", "'t", "'re", "'RE", "'ve", "'m", "'ll", "'d", "'x", "'ſ",
	"s", "t", "r", "e", "v", "l", "m", "d", "a", "Z", "hello", "world", "x_y",
	" ", "  ", "\t", "\n", "\r", "\r\n", "\n\n", "\v", "\f", " ", "\u0085",
	" ", "　", "1", "9", "123", "1234", "١", "²", "中",
	"é", "—", "!", "?", ".,", "--", "```", "`", "(", ")", "{}", "̀",
	"\U0001F600", "�", "\xff", "\xed\xa0\x80", "é", "-", "=>", "\t\t\n", " \n ",
}

func randomText(rng *rand.Rand) string {
	n := rng.Intn(24)
	var b strings.Builder
	for i := 0; i < n; i++ {
		b.WriteString(fragments[rng.Intn(len(fragments))])
	}
	return b.String()
}

// TestFastMatchesTiktoken compares the scanner and the BPE port with the
// tiktoken-go encoder on random text. Text the scanner declines is skipped
// and counted; it must stay rare.
func TestFastMatchesTiktoken(t *testing.T) {
	e, err := encoding()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(20261029))
	compared, declined := 0, 0
	for i := 0; i < 40000; i++ {
		text := normalizeSurrogates(randomText(rng))
		if text == "" {
			continue
		}
		want := e.EncodeOrdinary(text)
		got, ok := encodeFast(text)
		if !ok {
			declined++
			continue
		}
		compared++
		if len(got) != len(want) {
			t.Fatalf("len mismatch for %q: got %v want %v", text, got, want)
		}
		for k := range got {
			if got[k] != want[k] {
				t.Fatalf("token mismatch for %q: got %v want %v", text, got, want)
			}
		}
	}
	if compared < 30000 {
		t.Fatalf("too many declined cases: compared %d declined %d", compared, declined)
	}
	t.Logf("compared %d, declined %d", compared, declined)
}

// TestFastDeclinesOnlyFoldCase checks the one case the scanner hands back to
// the regexp2 path: an apostrophe followed by a non-ASCII rune.
func TestFastDeclinesOnlyFoldCase(t *testing.T) {
	if _, ok := encodeFast("itſ"); !ok {
		t.Fatal("plain non-ASCII text must not be declined")
	}
	if _, ok := encodeFast("'ſ"); ok {
		t.Fatal("apostrophe before non-ASCII must be declined")
	}
}
