package engines

// Parity test: replays a corpus written by server/parity/engine_cases.py and
// compares every step with the Python result. The test is skipped unless
// GK_PARITY_CORPUS names the JSONL file.
//
// Canonical form (must match engine_cases.py):
//   - a str result is itself, or "b64:" + base64 of its WTF-8 bytes when it
//     holds a surrogate code point;
//   - bytes become "b64:" + base64;
//   - any other value is pyjson.Dumps with separators "," and ":", using
//     ensure_ascii=False, or ensure_ascii=True when the first form would
//     carry a surrogate;
//   - a returned error means want "" and err true.
//
// A step that returns ErrUnsupported is not compared. It is counted as
// unsupported, and the rest of its case is skipped. Python would succeed
// there, so the Go port must hand that request to the Python backend.

import (
	"bufio"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

const parityMaxShown = 20

// parityCanonical returns the canonical string of a Go result value.
func parityCanonical(v any) (string, error) {
	switch t := v.(type) {
	case string:
		if hasSurrogateWTF8(t) {
			return "b64:" + base64.StdEncoding.EncodeToString([]byte(t)), nil
		}
		return t, nil
	case []byte:
		return "b64:" + base64.StdEncoding.EncodeToString(t), nil
	}
	b, err := pyjson.Dumps(v, pyjson.Compact)
	if errors.Is(err, pyjson.ErrLoneSurrogate) {
		b, err = pyjson.Dumps(v, pyjson.Options{ItemSep: ",", KeySep: ":", EnsureASCII: true})
	}
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// hasSurrogateWTF8 reports whether s holds an encoded surrogate code point
// (ED A0..BF xx in WTF-8).
func hasSurrogateWTF8(s string) bool {
	for i := 0; i+1 < len(s); i++ {
		if s[i] == 0xED && s[i+1] >= 0xA0 && s[i+1] <= 0xBF {
			return true
		}
	}
	return false
}

// firstDiffIndex returns the byte index where a and b first differ, or the
// shorter length when one is a prefix of the other.
func firstDiffIndex(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// window returns s around byte index i, for a readable diff line.
func window(s string, i int) string {
	lo := i - 40
	if lo < 0 {
		lo = 0
	}
	hi := i + 40
	if hi > len(s) {
		hi = len(s)
	}
	return fmt.Sprintf("%q", s[lo:hi])
}

type parityDiff struct {
	caseID, step int
	fn           string
	want, got    string
	wantErr      bool
	gotErr       bool
	index        int
	detail       string
}

func (d parityDiff) String() string {
	if d.wantErr != d.gotErr {
		return fmt.Sprintf("case %d step %d %s: err want=%v got=%v (want %s, got %s)",
			d.caseID, d.step, d.fn, d.wantErr, d.gotErr, window(d.want, 0), window(d.got, 0))
	}
	return fmt.Sprintf("case %d step %d %s: byte %d: want %s got %s %s",
		d.caseID, d.step, d.fn, d.index, window(d.want, d.index), window(d.got, d.index), d.detail)
}

func TestParity(t *testing.T) {
	path := os.Getenv("GK_PARITY_CORPUS")
	if path == "" {
		t.Skip("GK_PARITY_CORPUS is not set; generate a corpus with server/parity/engine_cases.py")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open corpus: %v", err)
	}
	defer f.Close()

	var (
		cases, steps, compared, unsupported int
		diffs                               []parityDiff
		perFn                               = map[string]int{}
	)
	rd := bufio.NewReaderSize(f, 1<<20)
	for {
		line, rerr := rd.ReadBytes('\n')
		if len(strings.TrimSpace(string(line))) > 0 {
			cases++
			n, c, u, ds := runParityCase(t, line, perFn)
			steps += n
			compared += c
			unsupported += u
			diffs = append(diffs, ds...)
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			t.Fatalf("read corpus: %v", rerr)
		}
	}

	t.Logf("parity: cases=%d steps=%d compared=%d unsupported=%d diffs=%d",
		cases, steps, compared, unsupported, len(diffs))
	fns := make([]string, 0, len(perFn))
	for fn := range perFn {
		fns = append(fns, fn)
	}
	sort.Strings(fns)
	for _, fn := range fns {
		t.Logf("parity: %-46s steps=%d", fn, perFn[fn])
	}
	for i, d := range diffs {
		if i == parityMaxShown {
			t.Logf("parity: ... %d more diffs not shown", len(diffs)-parityMaxShown)
			break
		}
		t.Logf("parity diff %d: %s", i+1, d)
	}
	if len(diffs) > 0 {
		t.Fatalf("parity: %d diffs", len(diffs))
	}
}

// runParityCase replays one JSONL line. It returns the number of steps, the
// number compared, the number unsupported, and any diffs.
func runParityCase(t *testing.T, line []byte, perFn map[string]int) (int, int, int, []parityDiff) {
	t.Helper()
	raw, err := pyjson.Loads(trimLine(line))
	if err != nil {
		t.Fatalf("parse corpus line: %v", err)
	}
	obj, ok := raw.(*pyjson.Object)
	if !ok {
		t.Fatalf("corpus line is not an object")
	}
	caseID, _ := objGet(obj, "id").(int64)
	stepList, _ := objGet(obj, "steps").([]any)
	wantList, _ := objGet(obj, "want").([]any)
	errList, _ := objGet(obj, "err").([]any)

	for _, reset := range parityResets {
		reset()
	}

	var diffs []parityDiff
	compared, unsupported := 0, 0
	for i, st := range stepList {
		so := st.(*pyjson.Object)
		fn := objGet(so, "fn").(string)
		args := objGet(so, "args").(*pyjson.Object)
		perFn[fn]++
		wantStr, _ := wantList[i].(string)
		wantErr, _ := errList[i].(bool)

		adapter, ok := parityAdapters[fn]
		if !ok {
			diffs = append(diffs, parityDiff{caseID: int(caseID), step: i, fn: fn, detail: "no Go adapter"})
			continue
		}
		val, gerr := adapter(args)
		if errors.Is(gerr, ErrUnsupported) {
			// The Go port declined this input, so it never returns a wrong
			// value for it. The caller sends it to Python, which decides.
			unsupported++
			break // later steps of this case would run on unknown state
		}
		gotStr := ""
		if gerr == nil {
			gotStr, gerr = parityCanonical(val)
		}
		compared++
		gotErr := gerr != nil
		if gotErr != wantErr || gotStr != wantStr {
			d := parityDiff{caseID: int(caseID), step: i, fn: fn, want: wantStr, got: gotStr,
				wantErr: wantErr, gotErr: gotErr}
			if gotErr && gerr != nil {
				d.detail = "(" + gerr.Error() + ")"
			}
			d.index = firstDiffIndex(wantStr, gotStr)
			diffs = append(diffs, d)
		}
	}
	return len(stepList), compared, unsupported, diffs
}

// trimLine drops the trailing newline and spaces that the JSON reader ignores.
func trimLine(b []byte) []byte {
	return []byte(strings.TrimSpace(string(b)))
}
