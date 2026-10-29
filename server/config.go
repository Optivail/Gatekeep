package main

// Deployment settings, read from the environment the way backend/config.py
// reads them. Defaults and precedence match Python exactly; see loadDotenv
// for the .env handling.

import (
	"crypto/subtle"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Set once by loadConfig at startup. Tests overwrite them directly.
var (
	cfgHost      = "0.0.0.0"
	cfgPort      = 9477
	cfgAgentHost = "0.0.0.0"
	cfgAgentPort = 9478
	cfgAPIKey    = ""
)

// loadConfig loads .env, then reads the environment. It returns an error
// for a non-integer port, which Python would also fail on at import time.
func loadConfig() error {
	loadDotenv()

	cfgHost = envOr("GATEKEEP_HOST", envOr("HOST", "0.0.0.0"))
	port, err := parsePyInt(envOr("GATEKEEP_PORT", envOr("PORT", "9477")))
	if err != nil {
		return fmt.Errorf("GATEKEEP_PORT: %w", err)
	}
	cfgPort = port
	cfgAgentHost = envOr("GATEKEEP_AGENT_HOST", cfgHost)
	agentPort, err := parsePyInt(envOr("GATEKEEP_AGENT_PORT", "9478"))
	if err != nil {
		return fmt.Errorf("GATEKEEP_AGENT_PORT: %w", err)
	}
	cfgAgentPort = agentPort
	cfgAPIKey = strings.TrimSpace(os.Getenv("GATEKEEP_API_KEY"))
	return nil
}

// envOr mirrors os.getenv(name, default): a variable that is set but empty
// still wins over the default.
func envOr(name, def string) string {
	if v, ok := os.LookupEnv(name); ok {
		return v
	}
	return def
}

// parsePyInt accepts what Python's int() accepts for the simple cases:
// surrounding whitespace and an optional sign.
func parsePyInt(s string) (int, error) {
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0, fmt.Errorf("invalid integer %q", s)
	}
	return n, nil
}

// pyOrString returns the first non-empty value, like Python's `a or b`
// applied to strings.
func pyOrString(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// urlHost maps wildcard bind addresses to loopback for client-facing URLs.
func urlHost(host string) string {
	switch host {
	case "0.0.0.0", "::", "[::]", "":
		return "127.0.0.1"
	}
	return host
}

// agentPublicURL is config.agent_public_url(): the origin that GetServerConfig
// responses are rewritten to, and that the agent listener is reached at.
func agentPublicURL() string {
	explicit := strings.TrimRight(strings.TrimSpace(os.Getenv("GATEKEEP_AGENT_URL")), "/")
	if explicit != "" {
		return explicit
	}
	base := strings.TrimRight(strings.TrimSpace(pyOrString(os.Getenv("GATEKEEP_PUBLIC_URL"), os.Getenv("GATEKEEP_BASE_URL"))), "/")
	if base != "" {
		scheme, host := pyURLSplit(base)
		if host == "" {
			host = urlHost(cfgHost)
		}
		if scheme == "" {
			scheme = "http"
		}
		return fmt.Sprintf("%s://%s:%d", scheme, host, cfgAgentPort)
	}
	return fmt.Sprintf("http://%s:%d", urlHost(cfgAgentHost), cfgAgentPort)
}

// pyURLSplit returns the scheme and hostname that urllib.parse.urlparse
// reports for a URL. Hostname is lowercased and has no brackets or port.
// Only the common shapes are handled: scheme://[user@]host[:port][/...].
func pyURLSplit(raw string) (scheme, host string) {
	rest := raw
	if i := strings.Index(raw, ":"); i > 0 && isURLScheme(raw[:i]) {
		scheme = strings.ToLower(raw[:i])
		rest = raw[i+1:]
	}
	if !strings.HasPrefix(rest, "//") {
		return scheme, ""
	}
	rest = rest[2:]
	if i := strings.IndexAny(rest, "/?#"); i >= 0 {
		rest = rest[:i]
	}
	if i := strings.LastIndex(rest, "@"); i >= 0 {
		rest = rest[i+1:]
	}
	if strings.HasPrefix(rest, "[") {
		if i := strings.Index(rest, "]"); i >= 0 {
			return scheme, strings.ToLower(rest[1:i])
		}
		return scheme, ""
	}
	if i := strings.Index(rest, ":"); i >= 0 {
		rest = rest[:i]
	}
	return scheme, strings.ToLower(rest)
}

func isURLScheme(s string) bool {
	if s == "" {
		return false
	}
	for i, c := range s {
		letter := (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
		other := (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.'
		if i == 0 && !letter {
			return false
		}
		if !letter && !other {
			return false
		}
	}
	return true
}

// isAuthorized is config.is_authorized. Both credentials are compared in
// constant time. Only the length of the presented value can leak, which is
// the same guarantee hmac.compare_digest gives.
func isAuthorized(r *http.Request) bool {
	if cfgAPIKey == "" {
		return true
	}
	key := []byte(cfgAPIKey)
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	if len(auth) >= 7 && strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		if constantTimeEq([]byte(strings.TrimSpace(auth[7:])), key) {
			return true
		}
	}
	return constantTimeEq([]byte(strings.TrimSpace(r.Header.Get("X-Gatekeep-Key"))), key)
}

// constantTimeEq compares two secrets without leaking where they differ.
func constantTimeEq(a, b []byte) bool {
	return subtle.ConstantTimeCompare(a, b) == 1
}

// dataDirFromEnv resolves GATEKEEP_DATA_DIR. When it is unset, the default
// is the repo's backend/ directory, the same place database.py uses, so a Go
// server and the Python one share one proxy_data.db. Without a repo the
// current directory is used.
func dataDirFromEnv() string {
	if v, ok := os.LookupEnv("GATEKEEP_DATA_DIR"); ok {
		if v == "" {
			v = "."
		}
		return absPath(v)
	}
	if backend := filepath.Join(repoDir, "backend"); isDir(backend) {
		return backend
	}
	return absPath(".")
}

func absPath(p string) string {
	if a, err := filepath.Abs(p); err == nil {
		return a
	}
	return p
}

func isDir(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

// listenAddr is the bind address for the HTTP server.
func listenAddr() string {
	return net.JoinHostPort(cfgHost, strconv.Itoa(cfgPort))
}
