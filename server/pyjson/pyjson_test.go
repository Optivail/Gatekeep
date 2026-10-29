package pyjson

import (
	"errors"
	"math"
	"testing"
)

// Expected bytes were produced by CPython 3 json.loads/json.dumps.
func TestDumpsMatchesCPython(t *testing.T) {
	const doc = `{"b":1,"a":[1.0,1e16,1e-5,0.0001,-0.0,1.5e300,123456789.125,2.5e-7,100.0,1e22,3.14159]}`
	v, err := Loads([]byte(doc))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		opts Options
		want string
	}{
		{Compact, `{"b":1,"a":[1.0,1e+16,1e-05,0.0001,-0.0,1.5e+300,123456789.125,2.5e-07,100.0,1e+22,3.14159]}`},
		{Default, `{"b": 1, "a": [1.0, 1e+16, 1e-05, 0.0001, -0.0, 1.5e+300, 123456789.125, 2.5e-07, 100.0, 1e+22, 3.14159]}`},
	}
	for _, c := range cases {
		got, err := Dumps(v, c.opts)
		if err != nil || string(got) != c.want {
			t.Errorf("Dumps(%+v) = %q, %v; want %q", c.opts, got, err, c.want)
		}
	}
}

func TestLoadsDuplicateKeysKeepFirstPosition(t *testing.T) {
	v, err := Loads([]byte(`{"a":1,"a":2,"b":3}`))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := Dumps(v, Compact)
	if string(got) != `{"a":2,"b":3}` {
		t.Errorf("got %s", got)
	}
}

func TestStringEscaping(t *testing.T) {
	in := `["😀","é","\u0001\n\t\"\\/\b\f\r\u007f"]`
	v, err := Loads([]byte(in))
	if err != nil {
		t.Fatal(err)
	}
	compact, _ := Dumps(v, Compact)
	if want := "[\"😀\",\"é\",\"\\u0001\\n\\t\\\"\\\\/\\b\\f\\r\x7f\"]"; string(compact) != want {
		t.Errorf("compact = %q, want %q", compact, want)
	}
	def, _ := Dumps(v, Default)
	if want := `["\ud83d\ude00", "\u00e9", "\u0001\n\t\"\\/\b\f\r\u007f"]`; string(def) != want {
		t.Errorf("default = %q, want %q", def, want)
	}
}

func TestNonFiniteAndBigInts(t *testing.T) {
	v, err := Loads([]byte(`[NaN, Infinity, -Infinity, 12345678901234567890123, -0, 1E2, 1e400, 5e-324, 0.1, 1e15, 9007199254740993]`))
	if err != nil {
		t.Fatal(err)
	}
	got, _ := Dumps(v, Default)
	want := `[NaN, Infinity, -Infinity, 12345678901234567890123, 0, 100.0, Infinity, 5e-324, 0.1, 1000000000000000.0, 9007199254740993]`
	if string(got) != want {
		t.Errorf("got %s\nwant %s", got, want)
	}
}

func TestLoneSurrogate(t *testing.T) {
	v, err := Loads([]byte(`["\ud800x"]`))
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := Dumps(v, Default); string(got) != `["\ud800x"]` {
		t.Errorf("default dumps = %s", got)
	}
	if _, err := Dumps(v, Compact); !errors.Is(err, ErrLoneSurrogate) {
		t.Errorf("compact dumps err = %v, want ErrLoneSurrogate", err)
	}
}

func TestLoadsRejects(t *testing.T) {
	for _, bad := range []string{"01", "[1,]", `{"a" 1}`, `"\x"`, "\"a\x01\"", "1 2", "[1.]", "-", "[1e]", `{"a":1,}`} {
		if _, err := Loads([]byte(bad)); err == nil {
			t.Errorf("Loads(%q) accepted, want error", bad)
		}
	}
	if _, err := Loads([]byte("\xef\xbb\xbf[1]")); err != nil {
		t.Errorf("UTF-8 BOM should be stripped: %v", err)
	}
}

func TestFloatReprNonFinite(t *testing.T) {
	if FloatRepr(math.Inf(-1)) != "-Infinity" || FloatRepr(math.NaN()) != "NaN" {
		t.Error("non-finite repr mismatch")
	}
}
