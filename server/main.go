// Command server is the Go port of the Gatekeep backend. It serves the
// dashboard API, the static dashboard, the LLM proxy (forward.go) and, on its
// own port, the Cursor agent listener (cursor_agent_server.go).
package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

func main() {
	if err := loadConfig(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	setupLogging()

	repoDir = detectRepoDir()
	if dist := filepath.Join(repoDir, "frontend", "dist"); isDir(dist) {
		frontendDistDir = dist
	}

	// Startup order follows main.py: database, seed prices, config log,
	// agent listener, then auto refresh in the background.
	if err := InitDB(dataDirFromEnv()); err != nil {
		logger.Error("[PROXY] database init failed", "err", err)
		os.Exit(1)
	}
	if err := ensureSeedPrices(); err != nil {
		logger.Error("[PROXY] seed prices failed", "err", err)
		os.Exit(1)
	}
	logger.Info(fmt.Sprintf("[PROXY] Gatekeep listening config host=%s port=%d", cfgHost, cfgPort))

	// The agent listener is a separate port. A failure to bind it is logged and
	// the dashboard and the proxy keep running, as in Python.
	var agentSrv *http.Server
	if agentDisabled(os.Getenv("GATEKEEP_DISABLE_AGENT")) {
		logger.Debug("[PROXY] Cursor agent listener disabled (GATEKEEP_DISABLE_AGENT)")
	} else {
		srv, err := startAgentListener(cfgAgentHost, cfgAgentPort)
		if err != nil {
			logger.Error(fmt.Sprintf("[PROXY] cursor agent listener %s:%d failed to start: %v", cfgAgentHost, cfgAgentPort, err))
		} else {
			agentSrv = srv
			logger.Info(fmt.Sprintf("[PROXY] Agent URL rewrite target: %s (listener %s:%d)",
				agentPublicURL(), cfgAgentHost, cfgAgentPort))
			go warmUpstream(defaultAgentUpstream)
		}
	}

	// Python's lifespan awaits this before serving, so the listener opens only
	// after the price table is current. Running it in the background made the
	// first dashboard reads see stale prices. The download is bounded by the
	// same HTTP timeout as the Python reference.
	if res := maybeAutoRefresh(); res != nil {
		if e, ok := res.Get("error"); ok {
			logger.Warn(fmt.Sprintf("[PRICING] auto-refresh failed: %v", e))
		} else {
			models, _ := res.Get("models")
			updated, _ := res.Get("updated_at")
			logger.Info(fmt.Sprintf("[PRICING] auto-refreshed %v models @ %v", models, updated))
		}
	}

	ln, err := net.Listen("tcp", listenAddr())
	if err != nil {
		logger.Error("[PROXY] listen failed", "addr", listenAddr(), "err", err)
		os.Exit(1)
	}
	srv := &http.Server{Handler: newHandler(), ReadHeaderTimeout: 30 * time.Second}
	errCh := make(chan error, 1)
	go func() { errCh <- srv.Serve(ln) }()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	select {
	case err := <-errCh:
		if err != nil && err != http.ErrServerClosed {
			logger.Error("[PROXY] server stopped", "err", err)
			os.Exit(1)
		}
	case <-ctx.Done():
		logger.Info("[PROXY] shutting down")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			logger.Error("[PROXY] shutdown error", "err", err)
		}
		// Python cancels the agent task, which drops its streams at once.
		if agentSrv != nil {
			if err := agentSrv.Close(); err != nil {
				logger.Error("[PROXY] agent listener close error", "err", err)
			}
		}
	}
}

// agentDisabled uses the same exact-match values as main.py: "1", "true" or
// "yes" after trimming, and case-sensitive.
func agentDisabled(v string) bool {
	switch strings.TrimSpace(v) {
	case "1", "true", "yes":
		return true
	}
	return false
}

// detectRepoDir finds the repo root. It walks up from the executable's
// directory and from the working directory, looking for server/go.mod, which
// only this repo has. Without a match it falls back to the working directory.
func detectRepoDir() string {
	starts := []string{}
	if exe, err := os.Executable(); err == nil {
		starts = append(starts, filepath.Dir(exe))
	}
	if wd, err := os.Getwd(); err == nil {
		starts = append(starts, wd)
	}
	for _, s := range starts {
		d := s
		for {
			if _, err := os.Stat(filepath.Join(d, "server", "go.mod")); err == nil {
				return d
			}
			parent := filepath.Dir(d)
			if parent == d {
				break
			}
			d = parent
		}
	}
	wd, _ := os.Getwd()
	return wd
}
