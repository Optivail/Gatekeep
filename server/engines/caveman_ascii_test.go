package engines

import (
	"math/rand"
	"strings"
	"testing"
)

// cavemanRegex is the reference: the three pyre patterns in the Python order.
func cavemanRegex(s string) string {
	s = reFillerPreamble.Sub("", s)
	s = rePoliteAux.Sub("", s)
	s = reDeterminers.Sub("", s)
	return s
}

var cavemanTokens = []string{
	"so", "So", "SO", "well", "Well", "basically", "BASICALLY", "essentially", "actually", "just", "Just",
	"simply", "could", "Could", "would", "WOULD", "can", "you", "You", "please", "PLEASE", "please,",
	"a", "A", "an", "An", "AN", "the", "The", "THE", "theme", "ant", "x_a", "_the", "the_", "the1", "1the",
	"a.", "can't", "could've", "k", "s", "Ss", "ca", "sob", "soon", "justice", "simplyx", "cann",
	",", ":", ".", "!", "?", "`", "```", "`x`", "\n", "\r\n", "\r", "\t", " ", "  ", "\x1c", "\x1f",
	"\v", "\f", "\x00", "\x7f", "-", "'", "\"", "/", "\\", "_", "0", "9",
	"could you", "could  you", "can you please", "would you please ", "could you please.",
}

// cavemanNonASCII holds bytes that must take the regex fallback.
var cavemanNonASCII = []string{"ſo", "ſimply", "\u212a", "İ", "é", "\u00a0", "\u0085", "\u3000", "ｔｈｅ", "—", "the\u00a0the", "a\u00e9"}

func randomCavemanText(rng *rand.Rand, tokens []string, n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		b.WriteString(tokens[rng.Intn(len(tokens))])
		if rng.Intn(3) == 0 {
			b.WriteByte(byte(rng.Intn(128)))
		}
	}
	return b.String()
}

func checkCaveman(t *testing.T, s string) {
	t.Helper()
	want := cavemanRegex(s)
	if got := cavemanWordsASCII(s); got != want {
		t.Fatalf("ASCII matcher differs for %q:\n got %q\nwant %q", s, got, want)
	}
	if got := cavemanWords(s); got != want {
		t.Fatalf("cavemanWords differs for %q:\n got %q\nwant %q", s, got, want)
	}
}

func TestCavemanASCIIExhaustive(t *testing.T) {
	alphabet := []byte{'a', 'A', 'n', 't', 'h', 'e', ' ', '\n', '\t', ',', 's', 'x', '1'}
	var rec func(prefix []byte, depth int)
	count := 0
	rec = func(prefix []byte, depth int) {
		checkCaveman(t, string(prefix))
		count++
		if depth == 0 {
			return
		}
		for _, c := range alphabet {
			rec(append(prefix, c), depth-1)
		}
	}
	rec(nil, 5)
	t.Logf("exhaustive strings: %d", count)
}

func TestCavemanASCIIRandomTokens(t *testing.T) {
	rng := rand.New(rand.NewSource(20261029))
	for i := 0; i < 20000; i++ {
		checkCaveman(t, randomCavemanText(rng, cavemanTokens, rng.Intn(40)))
	}
}

func TestCavemanASCIIRandomBytes(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	alphabet := []byte("aAnNtThHeEsSoOwWjJuUlLpPyYcCdD ,:.\n\t\r_0\x1c\x1d\x1e\x1f\x0b\x0c\x00`'")
	for i := 0; i < 20000; i++ {
		n := rng.Intn(120)
		b := make([]byte, n)
		for k := range b {
			b[k] = alphabet[rng.Intn(len(alphabet))]
		}
		checkCaveman(t, string(b))
	}
}

// Non-ASCII input takes the regex path in cavemanWords; it must still match.
func TestCavemanNonASCIIFallback(t *testing.T) {
	check := func(s string) {
		t.Helper()
		if got, want := cavemanWords(s), cavemanRegex(s); got != want {
			t.Fatalf("cavemanWords differs for %q:\n got %q\nwant %q", s, got, want)
		}
	}
	rng := rand.New(rand.NewSource(99))
	tokens := append(append([]string(nil), cavemanTokens...), cavemanNonASCII...)
	for i := 0; i < 5000; i++ {
		check(randomCavemanText(rng, tokens, rng.Intn(40)))
	}
	for _, s := range cavemanNonASCII {
		check(s)
		check("So " + s + " the " + s)
	}
}

// stripFillersRef applies every filler pattern, as cleanProse did before the trigger check.
func stripFillersRef(text string) string {
	for _, p := range fillerPatterns {
		text = p.Sub("", text)
	}
	return text
}

var fillerTokens = []string{
	"Certainly!", "certainly", "SURE THING!", "sure!", "Sure", "of course!", "Of Course!", "great question!",
	"Absolutely!", "absolutely", "I hope this helps.", "i hope that helps!", "I HOPE THIS HELPS",
	"Please let me know if you have any questions.", "please let me know if you need further help",
	"I'd be happy to help you with that!", "id be happy to help", "I'd be happy to help",
	"Feel free to ask if you have questions.", "feel free to reach out if you need anything",
	"Thank you for your patience!", "thank you for asking.", "THANK YOU FOR ASKING",
	"\n", "\n\n", " ", "  ", "\t", "x", "the", "\r", ".", "!", ":", "`", "```", "data", "sure ", "a",
	"é", "ſure!", "\u00a0",
}

func TestStripFillersMatchesAllPatterns(t *testing.T) {
	rng := rand.New(rand.NewSource(20261029))
	for i := 0; i < 20000; i++ {
		s := randomCavemanText(rng, fillerTokens, rng.Intn(30))
		if got, want := stripFillers(s), stripFillersRef(s); got != want {
			t.Fatalf("stripFillers differs for %q:\n got %q\nwant %q", s, got, want)
		}
	}
}
