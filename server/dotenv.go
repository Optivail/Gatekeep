package main

// A small .env loader that stands in for python-dotenv's load_dotenv().
//
// It looks for ".env" in the current directory and then in each parent,
// and stops at the first file found. It never overrides a variable that is
// already set, as python-dotenv does by default. Supported syntax: KEY=VALUE
// lines, an optional "export " prefix, '#' comments, single quotes (literal),
// double quotes (with \n \r \t \" \\ escapes), and an inline " #" comment on
// unquoted values. Multi-line quoted values are not supported.

import (
	"bufio"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var dotenvKeyRE = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_.-]*$`)

// loadDotenv applies the first .env file it finds. A missing file is not an
// error; an unreadable one is ignored, as it would be in Python's loader.
func loadDotenv() {
	dir, err := os.Getwd()
	if err != nil {
		return
	}
	for {
		candidate := filepath.Join(dir, ".env")
		if f, err := os.Open(candidate); err == nil {
			applyDotenv(f)
			f.Close()
			return
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return
		}
		dir = parent
	}
}

func applyDotenv(f *os.File) {
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)
	for sc.Scan() {
		key, value, ok := parseDotenvLine(sc.Text())
		if !ok {
			continue
		}
		if _, set := os.LookupEnv(key); !set {
			os.Setenv(key, value)
		}
	}
}

// parseDotenvLine returns the key and value of one .env line. ok is false for
// blank lines, comments, and lines without a valid KEY=.
func parseDotenvLine(line string) (key, value string, ok bool) {
	s := strings.TrimSpace(line)
	if s == "" || strings.HasPrefix(s, "#") {
		return "", "", false
	}
	s = strings.TrimPrefix(s, "export ")
	eq := strings.Index(s, "=")
	if eq < 0 {
		return "", "", false
	}
	key = strings.TrimSpace(s[:eq])
	if !dotenvKeyRE.MatchString(key) {
		return "", "", false
	}
	rest := strings.TrimLeft(s[eq+1:], " \t")
	if rest == "" {
		return key, "", true
	}
	switch rest[0] {
	case '\'':
		if end := strings.Index(rest[1:], "'"); end >= 0 {
			return key, rest[1 : 1+end], true
		}
		return key, rest[1:], true
	case '"':
		return key, unescapeDoubleQuoted(rest[1:]), true
	}
	if i := strings.Index(rest, " #"); i >= 0 {
		rest = rest[:i]
	}
	if i := strings.Index(rest, "\t#"); i >= 0 {
		rest = rest[:i]
	}
	if strings.HasPrefix(rest, "#") {
		rest = ""
	}
	return key, strings.TrimSpace(rest), true
}

// unescapeDoubleQuoted reads up to the closing quote and expands escapes.
// Anything after the closing quote is ignored.
func unescapeDoubleQuoted(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '"' {
			break
		}
		if c == '\\' && i+1 < len(s) {
			i++
			switch s[i] {
			case 'n':
				b.WriteByte('\n')
			case 'r':
				b.WriteByte('\r')
			case 't':
				b.WriteByte('\t')
			case '"', '\\':
				b.WriteByte(s[i])
			default:
				b.WriteByte('\\')
				b.WriteByte(s[i])
			}
			continue
		}
		b.WriteByte(c)
	}
	return b.String()
}
