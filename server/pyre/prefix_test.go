package pyre

import (
	"math/rand"
	"strings"
	"testing"
)

// containsFoldASCII must agree with a plain search on the lower-cased text.
func TestContainsFoldASCII(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	alpha := []byte("aAbB iI`\n\tzZ")
	for i := 0; i < 20000; i++ {
		s := make([]byte, rng.Intn(20))
		for k := range s {
			s[k] = alpha[rng.Intn(len(alpha))]
		}
		p := make([]byte, 1+rng.Intn(3))
		for k := range p {
			p[k] = lowerASCII(alpha[rng.Intn(len(alpha))])
		}
		want := strings.Contains(strings.ToLower(string(s)), string(p))
		if got := containsFoldASCII(string(s), string(p)); got != want {
			t.Fatalf("containsFoldASCII(%q, %q) = %v, want %v", s, p, got, want)
		}
	}
}
