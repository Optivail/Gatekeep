package main

// HTTP surface, a port of the non-proxy routes in backend/main.py: the API-key
// middleware, the dashboard routes, the static dashboard mount, and the
// catch-all that hands everything else to the proxy.
//
// Routing is done by hand instead of with http.ServeMux. ServeMux cleans
// paths and answers method mismatches with its own 405, while Starlette
// sends those requests on to the catch-all. Matching Starlette needs the
// manual version.

import (
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// frontendDistDir is the built dashboard. When it is empty the /dashboard/
// mount is skipped, as main.py does when frontend/dist is missing. repoDir is
// the repo root, which holds install/ and frontend/. main sets both at
// startup and tests override them.
var (
	frontendDistDir string
	repoDir         = "."
)

// dashboardAPI are the exact paths that skip the API-key check. Read-only
// dashboard data is open, as in _DASHBOARD_API. /api/data/clear and
// /api/prices/refresh are deliberately NOT here: they change state, so they
// need the key when one is set. This is a documented fix (KNOWN in the HTTP
// parity harness), because the Python reference leaves them open.
var dashboardAPI = []string{
	"/api/stats",
	"/api/settings",
	"/api/logs",
	"/api/prices",
	"/api/server-info",
}

// catchAllMethods are the methods FastAPI's catch-all route accepts.
var catchAllMethods = map[string]bool{
	http.MethodGet: true, http.MethodPost: true, http.MethodPut: true,
	http.MethodPatch: true, http.MethodDelete: true, http.MethodHead: true,
	http.MethodOptions: true,
}

// newHandler builds the full stack: access log, then API-key check, then routes.
func newHandler() http.Handler {
	return accessLog(apiKeyMiddleware(http.HandlerFunc(route)))
}

// apiKeyMiddleware is gatekeep_api_key_middleware. With no key configured
// every request passes. Otherwise the exempt paths pass, and the rest need
// a valid key.
func apiKeyMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if cfgAPIKey == "" {
			next.ServeHTTP(w, r)
			return
		}
		p := r.URL.Path
		exempt := strings.HasPrefix(p, "/dashboard") ||
			strings.HasPrefix(p, "/install/") ||
			strings.HasPrefix(p, "/api/settings/") ||
			inStrings(dashboardAPI, p)
		if exempt || isAuthorized(r) {
			next.ServeHTTP(w, r)
			return
		}
		writeDetail(w, http.StatusUnauthorized, "Unauthorized")
	})
}

func route(w http.ResponseWriter, r *http.Request) {
	p := r.URL.Path
	m := r.Method
	switch {
	case p == "/" && m == http.MethodGet:
		rootRedirect(w, r)
	case p == "/api/server-info" && m == http.MethodGet:
		serverInfo(w, r)
	case p == "/install/wire-agents.ps1" && m == http.MethodGet:
		wireAgentsScript(w, r)
	case p == "/api/settings" && m == http.MethodGet:
		apiGetSettings(w, r)
	case strings.HasPrefix(p, "/api/settings/") && m == http.MethodPost && validSettingKey(p[len("/api/settings/"):]):
		apiUpdateSetting(w, r, p[len("/api/settings/"):])
	case p == "/api/stats" && m == http.MethodGet:
		apiStats(w, r)
	case p == "/api/logs" && m == http.MethodGet:
		apiLogs(w, r)
	case p == "/api/prices" && m == http.MethodGet:
		apiPrices(w, r)
	case p == "/api/prices/refresh" && m == http.MethodPost:
		apiRefreshPrices(w, r)
	case p == "/api/data/clear" && m == http.MethodPost:
		apiClearData(w, r)
	case p == "/dashboard" && m == http.MethodGet:
		redirect(w, "/dashboard/", http.StatusTemporaryRedirect)
	case frontendDistDir != "" && strings.HasPrefix(p, "/dashboard/"):
		serveDashboard(w, r)
	default:
		if !catchAllMethods[m] {
			writeDetail(w, http.StatusMethodNotAllowed, "Method Not Allowed")
			return
		}
		forwardRequest(w, r, strings.TrimPrefix(p, "/"))
	}
}

// validSettingKey matches Starlette's {key} parameter, which is one
// non-empty path segment.
func validSettingKey(k string) bool {
	return k != "" && !strings.Contains(k, "/")
}

// ---- JSON helpers -------------------------------------------------------------

// writeJSON serialises like Starlette's JSONResponse: compact separators, raw
// UTF-8, and allow_nan=False. A value that cannot be serialised gives a
// plain 500, as an exception in a Starlette endpoint would.
func writeJSON(w http.ResponseWriter, status int, v any) {
	if hasNonFinite(v) {
		writeServerError(w)
		return
	}
	body, err := pyjson.Dumps(v, pyjson.Compact)
	if err != nil {
		writeServerError(w)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

func writeDetail(w http.ResponseWriter, status int, detail any) {
	o := pyjson.NewObject()
	o.Set("detail", detail)
	writeJSON(w, status, o)
}

func writeServerError(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusInternalServerError)
	_, _ = io.WriteString(w, "Internal Server Error")
}

// hasNonFinite reports whether v holds a NaN or infinite float anywhere.
// Python's json.dumps(allow_nan=False) rejects those.
func hasNonFinite(v any) bool {
	switch t := v.(type) {
	case float64:
		return t != t || t > 1.7976931348623157e308 || t < -1.7976931348623157e308
	case []any:
		for _, x := range t {
			if hasNonFinite(x) {
				return true
			}
		}
	case *pyjson.Object:
		for _, k := range t.Keys() {
			x, _ := t.Get(k)
			if hasNonFinite(x) {
				return true
			}
		}
	}
	return false
}

func inStrings(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func redirect(w http.ResponseWriter, location string, status int) {
	w.Header().Set("Location", location)
	w.WriteHeader(status)
}

// requestBaseURL is Starlette's request.base_url without the trailing slash.
// It is built from the Host header. Without one, the listen address is used.
func requestBaseURL(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	host := r.Host
	if host == "" {
		host = fmt.Sprintf("%s:%d", urlHost(cfgHost), cfgPort)
	}
	return scheme + "://" + host
}

// ---- query parameters -----------------------------------------------------------

// paramErrors collects FastAPI-style 422 entries. Python reports every bad
// parameter at once, and so does this.
type paramErrors []any

func (pe *paramErrors) add(kind, name, msg, input string) {
	e := pyjson.NewObject()
	e.Set("type", kind)
	e.Set("loc", []any{"query", name})
	e.Set("msg", msg)
	e.Set("input", input)
	*pe = append(*pe, e)
}

// queryLast returns the last value for name. Starlette's query dict keeps the
// last duplicate, so that is the one used here too.
func queryLast(r *http.Request, name string) (string, bool) {
	vals := r.URL.Query()[name]
	if len(vals) == 0 {
		return "", false
	}
	return vals[len(vals)-1], true
}

func intParam(r *http.Request, name string, def int64, errs *paramErrors) int64 {
	raw, ok := queryLast(r, name)
	if !ok {
		return def
	}
	n, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64)
	if err != nil {
		var ne *strconv.NumError
		if errors.As(err, &ne) && ne.Err == strconv.ErrRange {
			return n // already clamped to the int64 bound
		}
		errs.add("int_parsing", name, "Input should be a valid integer, unable to parse string as an integer", raw)
		return def
	}
	return n
}

func boolParam(r *http.Request, name string, def bool, errs *paramErrors) bool {
	raw, ok := queryLast(r, name)
	if !ok {
		return def
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "on", "t", "true", "y", "yes":
		return true
	case "0", "off", "f", "false", "n", "no":
		return false
	}
	errs.add("bool_parsing", name, "Input should be a valid boolean, unable to interpret input", raw)
	return def
}

func stringParam(r *http.Request, name, def string) string {
	if raw, ok := queryLast(r, name); ok {
		return raw
	}
	return def
}

// ---- routes ------------------------------------------------------------------------

func rootRedirect(w http.ResponseWriter, r *http.Request) {
	if strings.Contains(strings.ToLower(r.Header.Get("Accept")), "text/html") {
		redirect(w, "/dashboard/", http.StatusTemporaryRedirect)
		return
	}
	o := pyjson.NewObject()
	o.Set("service", "gatekeep")
	o.Set("dashboard", "/dashboard/")
	writeJSON(w, http.StatusOK, o)
}

// serverInfo is GET /api/server-info. It gives the dashboard the public URLs
// and ready-to-paste wire commands. Explicit env values win over the request
// host.
func serverInfo(w http.ResponseWriter, r *http.Request) {
	explicit := strings.TrimSpace(pyOrString(os.Getenv("GATEKEEP_PUBLIC_URL"), os.Getenv("GATEKEEP_BASE_URL")))
	var base string
	if explicit != "" {
		base = strings.TrimRight(explicit, "/")
	} else {
		base = strings.TrimRight(requestBaseURL(r), "/")
	}

	agent := strings.TrimRight(strings.TrimSpace(os.Getenv("GATEKEEP_AGENT_URL")), "/")
	if agent == "" {
		scheme, host := pyURLSplit(base)
		if host == "" {
			host = "127.0.0.1"
		}
		if scheme == "" {
			scheme = "http"
		}
		agent = fmt.Sprintf("%s://%s:%d", scheme, host, cfgAgentPort)
	}

	wireLocal := fmt.Sprintf(`powershell -File install/wire-agents.ps1 -GatekeepUrl "%s" -AgentUrl "%s"`, base, agent)
	wirePS := fmt.Sprintf(
		"$g='%s'; $a='%s'; $f=Join-Path $env:TEMP 'gk-wire.ps1'; irm \"%s/install/wire-agents.ps1\" -OutFile $f; & $f -GatekeepUrl $g -AgentUrl $a",
		base, agent, base,
	)
	wireCmd := fmt.Sprintf(
		`powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $g='%s'; $a='%s'; $f=Join-Path $env:TEMP 'gk-wire.ps1'; irm ($g+'/install/wire-agents.ps1') -OutFile $f; & $f -GatekeepUrl $g -AgentUrl $a }"`,
		base, agent,
	)

	o := pyjson.NewObject()
	o.Set("gatekeep_url", base)
	o.Set("agent_url", agent)
	o.Set("dashboard_url", base+"/dashboard/")
	o.Set("api_key_required", cfgAPIKey != "")
	o.Set("wire_command", wirePS)
	o.Set("wire_command_cmd", wireCmd)
	o.Set("wire_command_local", wireLocal)
	o.Set("wire_script_url", base+"/install/wire-agents.ps1")
	writeJSON(w, http.StatusOK, o)
}

// wireAgentsScript serves install/wire-agents.ps1 as a download.
func wireAgentsScript(w http.ResponseWriter, r *http.Request) {
	p := filepath.Join(repoDir, "install", "wire-agents.ps1")
	f, err := os.Open(p)
	if err != nil {
		writeDetail(w, http.StatusNotFound, "wire-agents.ps1 not found")
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil || !st.Mode().IsRegular() {
		writeDetail(w, http.StatusNotFound, "wire-agents.ps1 not found")
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.Header().Set("Content-Disposition", `attachment; filename="wire-agents.ps1"`)
	http.ServeContent(w, r, "wire-agents.ps1", st.ModTime(), f)
}

func apiGetSettings(w http.ResponseWriter, r *http.Request) {
	s, err := GetSettings()
	if err != nil {
		writeServerError(w)
		return
	}
	writeJSON(w, http.StatusOK, s)
}

// apiUpdateSetting is POST /api/settings/{key}. It reproduces how Python
// handles the body: a missing "value" gives 400, and a body that is not an
// object fails in one of two ways (see the switch below). Malformed JSON
// gives 500, as an unhandled exception would.
func apiUpdateSetting(w http.ResponseWriter, r *http.Request, key string) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeServerError(w)
		return
	}
	body, err := pyjson.Loads(raw)
	if err != nil {
		writeServerError(w)
		return
	}
	var value any
	switch b := body.(type) {
	case *pyjson.Object:
		v, ok := b.Get("value")
		if !ok {
			writeDetail(w, http.StatusBadRequest, "Missing 'value' in request body")
			return
		}
		value = v
	case []any:
		// `"value" in list` is a membership test, and then body["value"]
		// raises TypeError on a list.
		for _, x := range b {
			if s, ok := x.(string); ok && s == "value" {
				writeServerError(w)
				return
			}
		}
		writeDetail(w, http.StatusBadRequest, "Missing 'value' in request body")
		return
	case string:
		// Substring test on a string, then string indexing raises TypeError.
		if strings.Contains(b, "value") {
			writeServerError(w)
			return
		}
		writeDetail(w, http.StatusBadRequest, "Missing 'value' in request body")
		return
	default:
		// `in` on None, bool, or a number raises TypeError.
		writeServerError(w)
		return
	}

	var stored int64
	if pyTruthy(value) {
		stored = 1
	}
	if err := SetSetting(key, stored); err != nil {
		writeServerError(w)
		return
	}
	// Aggressive mode replaces the mild history and log passes. Forcing both
	// on keeps the dashboard from showing them as off.
	if key == "aggressive_enabled" && stored == 1 {
		if err := SetSetting("history_pruning_enabled", 1); err != nil {
			writeServerError(w)
			return
		}
		if err := SetSetting("log_truncation_enabled", 1); err != nil {
			writeServerError(w)
			return
		}
	}
	settings, err := GetSettings()
	if err != nil {
		writeServerError(w)
		return
	}
	o := pyjson.NewObject()
	o.Set("key", key)
	o.Set("value", stored)
	o.Set("settings", settings)
	writeJSON(w, http.StatusOK, o)
}

func apiStats(w http.ResponseWriter, r *http.Request) {
	var errs paramErrors
	includeNoise := boolParam(r, "include_noise", false, &errs)
	rng := stringParam(r, "range", "24h")
	if len(errs) > 0 {
		writeDetail(w, http.StatusUnprocessableEntity, []any(errs))
		return
	}
	o, err := GetStats(includeNoise, rng)
	if err != nil {
		writeServerError(w)
		return
	}
	writeJSON(w, http.StatusOK, o)
}

func apiLogs(w http.ResponseWriter, r *http.Request) {
	var errs paramErrors
	limit := intParam(r, "limit", 20, &errs)
	offset := intParam(r, "offset", 0, &errs)
	includeNoise := boolParam(r, "include_noise", false, &errs)
	rng := stringParam(r, "range", "24h")
	if len(errs) > 0 {
		writeDetail(w, http.StatusUnprocessableEntity, []any(errs))
		return
	}
	o, err := GetLogs(limit, offset, includeNoise, rng)
	if err != nil {
		writeServerError(w)
		return
	}
	writeJSON(w, http.StatusOK, o)
}

func apiPrices(w http.ResponseWriter, r *http.Request) {
	o, err := pricingStatus()
	if err != nil {
		writeServerError(w)
		return
	}
	writeJSON(w, http.StatusOK, o)
}

// apiRefreshPrices returns 502 on any failure, with the error text as the
// detail, as main.py does.
func apiRefreshPrices(w http.ResponseWriter, r *http.Request) {
	o, err := refreshPricesFromLiteLLM()
	if err != nil {
		writeDetail(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, o)
}

func apiClearData(w http.ResponseWriter, r *http.Request) {
	o, err := ClearRequestData()
	if err != nil {
		writeServerError(w)
		return
	}
	writeJSON(w, http.StatusOK, o)
}

// ---- static dashboard ----------------------------------------------------------------

// serveDashboard reproduces Starlette's StaticFiles with html=True, under the
// _NoCacheStaticFiles wrapper. GET and HEAD only, with directory index
// fallback, a redirect to add the trailing slash, optional 404.html, and
// realpath containment so that symlinks cannot leave the dist directory.
func serveDashboard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		writeDetail(w, http.StatusMethodNotAllowed, "Method Not Allowed")
		return
	}
	rootReal, err := filepath.EvalSymlinks(frontendDistDir)
	if err != nil {
		writeDetail(w, http.StatusNotFound, "Not Found")
		return
	}
	rel := path.Clean("/" + strings.TrimPrefix(r.URL.Path, "/dashboard/"))
	full := filepath.Join(rootReal, filepath.FromSlash(rel))

	real, ok := resolveWithin(rootReal, full)
	if ok {
		st, err := os.Stat(real)
		if err == nil && st.Mode().IsRegular() {
			serveDashboardFile(w, r, real, http.StatusOK)
			return
		}
		if err == nil && st.IsDir() {
			idx, ok := resolveWithin(rootReal, filepath.Join(real, "index.html"))
			if ok {
				if ist, err := os.Stat(idx); err == nil && ist.Mode().IsRegular() {
					if !strings.HasSuffix(r.URL.Path, "/") {
						loc := requestBaseURL(r) + r.URL.Path + "/"
						if r.URL.RawQuery != "" {
							loc += "?" + r.URL.RawQuery
						}
						redirect(w, loc, http.StatusTemporaryRedirect)
						return
					}
					serveDashboardFile(w, r, idx, http.StatusOK)
					return
				}
			}
		}
	}
	if nf, ok := resolveWithin(rootReal, filepath.Join(rootReal, "404.html")); ok {
		if st, err := os.Stat(nf); err == nil && st.Mode().IsRegular() {
			serveDashboardFile(w, r, nf, http.StatusNotFound)
			return
		}
	}
	writeDetail(w, http.StatusNotFound, "Not Found")
}

// resolveWithin returns the real path of p when it lies inside root. ok is
// false when p does not exist or resolves outside root.
func resolveWithin(root, p string) (string, bool) {
	real, err := filepath.EvalSymlinks(p)
	if err != nil {
		return "", false
	}
	if real != root && !strings.HasPrefix(real, root+string(filepath.Separator)) {
		return "", false
	}
	return real, true
}

// serveDashboardFile sends one file. Content types come from the extension
// and fall back to text/plain, as Starlette's FileResponse does. text/*
// responses get charset=utf-8. HTML is never cached.
func serveDashboardFile(w http.ResponseWriter, r *http.Request, p string, status int) {
	ct := mime.TypeByExtension(strings.ToLower(filepath.Ext(p)))
	if ct == "" {
		ct = "text/plain"
	}
	if strings.HasPrefix(ct, "text/") && !strings.Contains(ct, "charset=") {
		ct += "; charset=utf-8"
	}
	w.Header().Set("Content-Type", ct)
	if strings.HasSuffix(p, ".html") {
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
	}
	f, err := os.Open(p)
	if err != nil {
		writeDetail(w, http.StatusNotFound, "Not Found")
		return
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		writeServerError(w)
		return
	}
	if status == http.StatusOK {
		http.ServeContent(w, r, "", st.ModTime(), f)
		return
	}
	w.WriteHeader(status)
	if r.Method != http.MethodHead {
		_, _ = io.Copy(w, f)
	}
}
