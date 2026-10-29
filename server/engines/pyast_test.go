package engines

import (
	"errors"
	"strings"
	"testing"
)

// Expected values are CPython 3.12 results for _distill_python_signatures
// (backend/engines/optimizer.py), computed with the Python oracle.
func TestDistillPythonSignaturesMatchesCPython(t *testing.T) {
	cases := []struct{ code, want string }{
		{
			"def ｆｏｏ(ﬁle: ｉｆ = 1, _x=2):\n    return 1\n\nclass ＧＯ:\n    def ﬀ(self) -> None: ...\n",
			"def foo(file: if=1, _x=2):\n    ...\n\nclass GO:\n\n    def ff(self) -> None:\n        ...",
		},
		{
			"def f(a='\\N{SNOWMAN}', b='\\N{LINE FEED}'):\n    pass\n",
			"def f(a='☃', b='\\n'):\n    ...",
		},
		{
			"def f(a=1,\r\n  b=2):\r\n    pass\r\n",
			"def f(a=1, b=2):\n    ...",
		},
		{
			"def _private():\n    pass\n\n@dec\nclass K(Base, metaclass=M):\n    @property\n    def p(self) -> int: return 1\n    def _h(self): pass\n    class Inner: pass\n",
			"class K(Base, metaclass=M):\n\n    def p(self) -> int:\n        ...",
		},
		{
			"async def go[T: int](x: T, *args: *Ts, **kw) -> list[T]:\n    return []\n",
			"async def go[T: int](x: T, *args: *Ts, **kw) -> list[T]:\n    ...",
		},
		{
			"def bad(:\n    pass\n",
			"def bad(:\n    pass\n",
		},
		{
			// Unknown name and named sequence: both SyntaxError in CPython, so
			// the code comes back unchanged.
			"def f(a='\\N{NOT A REAL NAME}'):\n    pass\n",
			"def f(a='\\N{NOT A REAL NAME}'):\n    pass\n",
		},
		{
			"def f(a='\\N{LATIN CAPITAL LETTER A WITH MACRON AND GRAVE}'):\n    pass\n",
			"def f(a='\\N{LATIN CAPITAL LETTER A WITH MACRON AND GRAVE}'):\n    pass\n",
		},
		{
			"def f(a='\\N{latin small letter e with acute}', b=b'\\N{SNOWMAN}'):\n    pass\n",
			"def f(a='é', b=b'\\\\N{SNOWMAN}'):\n    ...",
		},
	}
	for _, c := range cases {
		got, err := distillPythonSignatures(c.code)
		if err != nil {
			t.Fatalf("distill(%q): %v", c.code, err)
		}
		if got != c.want {
			t.Errorf("distill(%q)\n got %q\nwant %q", c.code, got, c.want)
		}
	}
}

func TestDistillPythonSignaturesDeclinesDeepLine(t *testing.T) {
	// CPython's ast.unparse hits its recursion limit on this line, and the
	// error is caught, so Python returns the code. The port declines instead.
	code := "def f(a=" + strings.Repeat("{1:", 150) + "1" + strings.Repeat("}", 150) + "):\n    pass\n"
	if _, err := distillPythonSignatures(code); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("deep line error = %v, want ErrUnsupported", err)
	}
}

func TestPyNameLookupLiteralRules(t *testing.T) {
	// Each row is CPython 3.12's \N{...} result in a string literal.
	cases := []struct {
		name string
		want rune
		ok   bool
	}{
		{"LATIN SMALL LETTER E WITH ACUTE", 0xE9, true},
		{"latin small letter a", 'a', true},
		{"LINE FEED", 0x0A, true},
		{"HANGUL SYLLABLE GA", 0xAC00, true},
		{"hangul syllable ga", 0, false},
		{"CJK UNIFIED IDEOGRAPH-4E00", 0x4E00, true},
		{"CJK UNIFIED IDEOGRAPH-04E00", 0x4E00, true},
		{"CJK UNIFIED IDEOGRAPH-4e00", 0, false},
		{"CJK UNIFIED IDEOGRAPH-4E0", 0, false},
		{"CJK UNIFIED IDEOGRAPH-A000", 0, false},
		{"KHITAN SMALL SCRIPT CHARACTER-18B00", 0x18B00, true},
		{"LINE FEED (LF)", 0, false},
		{"LATIN CAPITAL LETTER A WITH MACRON AND GRAVE", 0, false},
		{"NOT A REAL NAME", 0, false},
		{"", 0, false},
	}
	for _, c := range cases {
		got, ok, err := pyNameLookup(c.name)
		if err != nil {
			t.Fatalf("pyNameLookup(%q): %v", c.name, err)
		}
		if ok != c.ok || (ok && got != c.want) {
			t.Errorf("pyNameLookup(%q) = %U, %v; want %U, %v", c.name, got, ok, c.want, c.ok)
		}
	}
}
