package main

// Logging. Python's gklog.py uses the root logger with message-only output at
// INFO, or DEBUG when GATEKEEP_DEBUG is truthy. slog keeps that shape: the
// timestamp is dropped so the lines read like the Python ones.

import (
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"
)

var logger = slog.New(slog.NewTextHandler(os.Stderr, nil))

// setupLogging installs the process logger. Call it after loadConfig.
func setupLogging() {
	debug := isTruthy(os.Getenv("GATEKEEP_DEBUG"))
	level := slog.LevelInfo
	if debug {
		level = slog.LevelDebug
	}
	logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: level,
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			if len(groups) == 0 && a.Key == slog.TimeKey {
				return slog.Attr{}
			}
			return a
		},
	}))
}

// isTruthy matches the env-flag checks in the Python code: "1", "true" or
// "yes", case-insensitive, with surrounding whitespace ignored.
func isTruthy(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes":
		return true
	}
	return false
}

// suppressedAccessPaths are the dashboard polling endpoints. main.py hides
// them from the access log with EndpointFilter. The match is a substring
// test on the request URI, as in Python.
var suppressedAccessPaths = []string{
	"/api/stats",
	"/api/settings",
	"/api/logs",
	"/api/prices",
	"/api/server-info",
	"/dashboard",
	"/favicon.ico",
}

// statusRecorder captures the status code for the access log.
type statusRecorder struct {
	http.ResponseWriter
	status int
}

// Unwrap lets http.ResponseController reach the real writer, so that Flush and
// EnableFullDuplex work for streaming responses.
func (s *statusRecorder) Unwrap() http.ResponseWriter {
	return s.ResponseWriter
}

func (s *statusRecorder) WriteHeader(code int) {
	s.status = code
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusRecorder) Write(b []byte) (int, error) {
	if s.status == 0 {
		s.status = http.StatusOK
	}
	return s.ResponseWriter.Write(b)
}

// accessLog writes one line per request, except the suppressed paths.
func accessLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec := &statusRecorder{ResponseWriter: w}
		start := time.Now()
		next.ServeHTTP(rec, r)
		uri := r.RequestURI
		for _, p := range suppressedAccessPaths {
			if strings.Contains(uri, p) {
				return
			}
		}
		logger.Info("access", "client", r.RemoteAddr, "request", r.Method+" "+uri, "status", rec.status, "ms", time.Since(start).Milliseconds())
	})
}
