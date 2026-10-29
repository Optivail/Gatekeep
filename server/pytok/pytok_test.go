package pytok

import (
	"bufio"
	"encoding/json"
	"errors"
	"os"
	"reflect"
	"strings"
	"testing"
)

// oracleRow is one line of testdata/cl100k_cases.jsonl, written by
// server/parity/gen_pytok_cases.py from tiktoken's Python encoder.
type oracleRow struct {
	Cat  string `json:"cat"`
	Text string `json:"text"`
	N    int    `json:"n"`
	Hash string `json:"hash"`
	Err  string `json:"err"`
}

func loadOracle(t *testing.T) []oracleRow {
	t.Helper()
	f, err := os.Open("testdata/cl100k_cases.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var rows []oracleRow
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<22)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		var r oracleRow
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			t.Fatalf("bad fixture line: %v", err)
		}
		rows = append(rows, r)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return rows
}

// TestOracleFixture checks every fixture row against tiktoken's Python output:
// the token count, the hash of the ids, and the special-token refusal.
func TestOracleFixture(t *testing.T) {
	rows := loadOracle(t)
	if len(rows) < 5000 {
		t.Fatalf("fixture has %d rows, want at least 5000", len(rows))
	}
	special := 0
	for i, r := range rows {
		if r.Err == "special" {
			special++
			if _, err := Encode(r.Text); !errors.Is(err, ErrDisallowedSpecial) {
				t.Errorf("row %d (%s): Encode error = %v, want ErrDisallowedSpecial", i, r.Cat, err)
			}
			if _, err := CountTokens(r.Text); !errors.Is(err, ErrDisallowedSpecial) {
				t.Errorf("row %d (%s): CountTokens error = %v, want ErrDisallowedSpecial", i, r.Cat, err)
			}
			continue
		}
		n, err := CountTokens(r.Text)
		if err != nil || n != r.N {
			t.Errorf("row %d (%s, %d bytes): CountTokens = %d, %v; Python %d", i, r.Cat, len(r.Text), n, err, r.N)
		}
		ids, err := Encode(r.Text)
		if err != nil {
			t.Errorf("row %d (%s): Encode error %v", i, r.Cat, err)
			continue
		}
		if len(ids) != r.N || idsHash(ids) != r.Hash {
			t.Errorf("row %d (%s, %d bytes): ids differ from Python (len %d, want %d)", i, r.Cat, len(r.Text), len(ids), r.N)
		}
	}
	t.Logf("fixture rows: %d, special-token refusals: %d", len(rows), special)
}

// TestPythonValues pins exact ids that were printed by tiktoken in Python 3.12
// (tiktoken 0.14.0), including the edge cases the design notes call out.
func TestPythonValues(t *testing.T) {
	cases := []struct {
		name string
		text string
		want []int
	}{
		{"empty", "", []int{}},
		{"partial special", "<|endoftext|", []int{27, 91, 8862, 728, 428, 91}},
		{"near special", "<|endoft ext|>", []int{27, 91, 8862, 728, 1327, 91, 29}},
		{"e-acute precomposed", "\u00e9", []int{978}},
		{"e plus combining acute", "e\u0301", []int{68, 54939}},
		{"max code point", "\U0010ffff", []int{176, 237, 123, 123}},
		{"crlf", "\r\n", []int{319}},
		{"zwj sequence", "\u200d\U0001F468\u200d\U0001F469", []int{378, 235, 9468, 239, 101, 378, 235, 9468, 239, 102}},
		{"lone surrogate", "a\xed\xa0\x80b", []int{64, 5809, 65}},
		{"surrogate pair", "\xed\xa0\x80\xed\xb0\x80", []int{172, 238, 222, 222}},
		{"pair in text", "x\xed\xa0\xbd\xed\xb8\x80y", []int{87, 76460, 222, 88}},
	}
	for _, c := range cases {
		got, err := Encode(c.text)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("%s: Encode = %v, Python %v", c.name, got, c.want)
		}
	}
}

// TestSpecialTokens checks that each special token is refused, as
// encode() with default arguments raises ValueError in Python.
func TestSpecialTokens(t *testing.T) {
	for _, tok := range specialTokens {
		for _, text := range []string{tok, "a" + tok, tok + " b", "x\n" + tok + "\ny"} {
			_, err := Encode(text)
			if !errors.Is(err, ErrDisallowedSpecial) {
				t.Errorf("Encode(%q) error = %v, want ErrDisallowedSpecial", text, err)
			}
			if _, err := CountTokens(text); err == nil {
				t.Errorf("CountTokens(%q) returned no error", text)
			}
		}
	}
	// Text that only looks like a special token is encoded normally.
	for _, text := range []string{"<|endoftext|", "endoftext|>", "<|endofprompt>"} {
		if _, err := Encode(text); err != nil {
			t.Errorf("Encode(%q): %v", text, err)
		}
	}
}

func TestCountEmptyShortcut(t *testing.T) {
	n, err := CountTokens("")
	if n != 0 || err != nil {
		t.Errorf(`CountTokens("") = %d, %v; want 0, nil`, n, err)
	}
	ids, err := Encode("")
	if err != nil || len(ids) != 0 {
		t.Errorf(`Encode("") = %v, %v`, ids, err)
	}
}

func TestEncodeCountAgree(t *testing.T) {
	for _, s := range []string{"hello world", strings.Repeat("token ", 500), "def f(x):\n\treturn x\n"} {
		ids, err := Encode(s)
		n, err2 := CountTokens(s)
		if err != nil || err2 != nil || n != len(ids) {
			t.Errorf("%q: Encode %d, CountTokens %d (%v, %v)", s, len(ids), n, err, err2)
		}
	}
}
