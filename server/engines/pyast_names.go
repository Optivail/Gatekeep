// Unicode character names for \N{name} escapes in Python string literals.
//
// CPython 3.12 (Unicode 15.0.0) resolves \N{...} through its own name
// database. The Go standard library has none, so the table is embedded from
// pyast_names.txt.gz, which server/parity/gen_pyast_names.py builds and checks
// against CPython. The rules below follow CPython's literal path:
//
//   - names are matched with ASCII letters uppercased, so "latin small letter
//     a" resolves like "LATIN SMALL LETTER A";
//   - the HANGUL SYLLABLE and CJK UNIFIED IDEOGRAPH- prefixes are matched
//     exactly, so a lowercase prefix is rejected;
//   - CJK UNIFIED IDEOGRAPH names take exactly four or five uppercase hex
//     digits and must fall inside an ideograph range;
//   - named sequences (several code points) are rejected, as in a literal.
package engines

import (
	"bufio"
	"bytes"
	"compress/gzip"
	_ "embed"
	"io"
	"strconv"
	"strings"
	"sync"
)

//go:embed pyast_names.txt.gz
var pyNamesGz []byte

type pyNameRange struct{ lo, hi rune }

type pyNameTable struct {
	names  map[string]rune
	ranges []pyNameRange
}

var (
	pyNamesOnce sync.Once
	pyNames     *pyNameTable
	pyNamesErr  error
)

const (
	pyHangulPrefix = "HANGUL SYLLABLE "
	pyCJKPrefix    = "CJK UNIFIED IDEOGRAPH-"
)

// loadPyNames decodes the embedded table once. Each record is "N<TAB>HEX<TAB>NAME"
// or "R<TAB>LO<TAB>HI".
func loadPyNames() (*pyNameTable, error) {
	pyNamesOnce.Do(func() {
		zr, err := gzip.NewReader(bytes.NewReader(pyNamesGz))
		if err != nil {
			pyNamesErr = err
			return
		}
		data, err := io.ReadAll(zr)
		if err != nil {
			pyNamesErr = err
			return
		}
		t := &pyNameTable{names: make(map[string]rune, 47000)}
		sc := bufio.NewScanner(bytes.NewReader(data))
		for sc.Scan() {
			f := strings.SplitN(sc.Text(), "\t", 3)
			if len(f) != 3 {
				pyNamesErr = io.ErrUnexpectedEOF
				return
			}
			a, errA := strconv.ParseUint(f[1], 16, 32)
			if errA != nil {
				pyNamesErr = errA
				return
			}
			switch f[0] {
			case "N":
				t.names[f[2]] = rune(a)
			case "R":
				b, errB := strconv.ParseUint(f[2], 16, 32)
				if errB != nil {
					pyNamesErr = errB
					return
				}
				t.ranges = append(t.ranges, pyNameRange{rune(a), rune(b)})
			default:
				pyNamesErr = io.ErrUnexpectedEOF
				return
			}
		}
		if err := sc.Err(); err != nil {
			pyNamesErr = err
			return
		}
		pyNames = t
	})
	return pyNames, pyNamesErr
}

// pyNameLookup mirrors a \N{raw} escape in a non-raw str literal. ok is false
// where CPython raises SyntaxError. err reports a broken embedded table, which
// the caller must treat as a decline.
func pyNameLookup(raw string) (r rune, ok bool, err error) {
	t, err := loadPyNames()
	if err != nil {
		return 0, false, err
	}
	upper := asciiUpper(raw)
	if strings.HasPrefix(upper, pyHangulPrefix) && !strings.HasPrefix(raw, pyHangulPrefix) {
		return 0, false, nil
	}
	if strings.HasPrefix(raw, pyCJKPrefix) {
		return cjkUnifiedLookup(t, raw[len(pyCJKPrefix):])
	}
	if strings.HasPrefix(upper, pyCJKPrefix) {
		return 0, false, nil
	}
	if strings.HasPrefix(raw, pyHangulPrefix) {
		r, ok = t.names[raw]
		return r, ok, nil
	}
	r, ok = t.names[upper]
	return r, ok, nil
}

// cjkUnifiedLookup resolves the hex part of a CJK UNIFIED IDEOGRAPH name.
func cjkUnifiedLookup(t *pyNameTable, hex string) (rune, bool, error) {
	if len(hex) != 4 && len(hex) != 5 {
		return 0, false, nil
	}
	var v rune
	for i := 0; i < len(hex); i++ {
		c := hex[i]
		switch {
		case c >= '0' && c <= '9':
			v = v*16 + rune(c-'0')
		case c >= 'A' && c <= 'F':
			v = v*16 + rune(c-'A'+10)
		default:
			return 0, false, nil
		}
	}
	for _, rg := range t.ranges {
		if v >= rg.lo && v <= rg.hi {
			return v, true, nil
		}
	}
	return 0, false, nil
}

// asciiUpper uppercases ASCII letters only, as CPython's name matching does.
func asciiUpper(s string) string {
	b := []byte(s)
	for i, c := range b {
		if c >= 'a' && c <= 'z' {
			b[i] = c - 32
		}
	}
	return string(b)
}
