package main

// Tests for the Go port. They need no network and no Python. The
// test_api_key_check case from backend/tests/test_security.py is ported
// here, along with schema, settings, route, and pricing checks.

import (
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// setupTestDB points the package at a fresh database in a temp directory.
func setupTestDB(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := InitDB(dir); err != nil {
		t.Fatalf("InitDB: %v", err)
	}
	if err := ensureSeedPrices(); err != nil {
		t.Fatalf("ensureSeedPrices: %v", err)
	}
	t.Cleanup(func() {
		if sqlDB != nil {
			sqlDB.Close()
			sqlDB = nil
		}
	})
	return dir
}

// withAPIKey sets the API key for one test and restores it afterwards.
func withAPIKey(t *testing.T, key string) {
	t.Helper()
	old := cfgAPIKey
	cfgAPIKey = key
	t.Cleanup(func() { cfgAPIKey = old })
}

func do(t *testing.T, method, target string, body string, headers map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	var r io.Reader
	if body != "" {
		r = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, target, r)
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	newHandler().ServeHTTP(rec, req)
	return rec
}

// ---- security (ported from test_security.py) -----------------------------------------

func TestAPIKeyCheck(t *testing.T) {
	withAPIKey(t, "s3cret")
	cases := []struct {
		name    string
		headers map[string]string
		want    bool
	}{
		{"bearer ok", map[string]string{"Authorization": "Bearer s3cret"}, true},
		{"bearer scheme is case-insensitive", map[string]string{"Authorization": "bearer s3cret"}, true},
		{"bearer token is stripped", map[string]string{"Authorization": "Bearer   s3cret  "}, true},
		{"x-gatekeep-key ok", map[string]string{"X-Gatekeep-Key": "s3cret"}, true},
		{"bearer wrong", map[string]string{"Authorization": "Bearer wrong"}, false},
		{"basic scheme rejected", map[string]string{"Authorization": "Basic s3cret"}, false},
		{"prefix of key rejected", map[string]string{"Authorization": "Bearer s3c"}, false},
		{"no headers", map[string]string{}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			for k, v := range c.headers {
				req.Header.Set(k, v)
			}
			if got := isAuthorized(req); got != c.want {
				t.Fatalf("isAuthorized = %v, want %v", got, c.want)
			}
		})
	}
}

func TestAPIKeyUnsetPassesEverything(t *testing.T) {
	withAPIKey(t, "")
	setupTestDB(t)
	rec := do(t, http.MethodGet, "/", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want 200 when no key is configured", rec.Code)
	}
}

func TestAPIKeyMiddlewareExemptions(t *testing.T) {
	withAPIKey(t, "s3cret")
	setupTestDB(t)
	// Exempt paths must not get a 401 even without a key.
	for _, p := range []string{"/api/stats", "/api/logs", "/api/settings", "/api/prices",
		"/api/server-info", "/api/settings/x", "/install/wire-agents.ps1", "/dashboard"} {
		method := http.MethodGet
		if p == "/api/settings/x" {
			method = http.MethodPost
		}
		rec := do(t, method, p, "", nil)
		if rec.Code == http.StatusUnauthorized {
			t.Errorf("%s %s was rejected, but it is exempt", method, p)
		}
	}
	// State-changing dashboard actions need the key even though they sit under /api/.
	for _, p := range []string{"/api/data/clear", "/api/prices/refresh"} {
		if rec := do(t, http.MethodPost, p, "", nil); rec.Code != http.StatusUnauthorized {
			t.Errorf("POST %s without key: status %d, want 401", p, rec.Code)
		}
	}
	// Anything else needs the key, and the 401 body matches FastAPI's.
	rec := do(t, http.MethodGet, "/", "", nil)
	if rec.Code != http.StatusUnauthorized || rec.Body.String() != `{"detail":"Unauthorized"}` {
		t.Fatalf("unauthenticated / gave %d %q", rec.Code, rec.Body.String())
	}
	rec = do(t, http.MethodGet, "/", "", map[string]string{"Authorization": "Bearer s3cret"})
	if rec.Code != http.StatusOK {
		t.Fatalf("authenticated / gave %d", rec.Code)
	}
}

// ---- schema and round trip -----------------------------------------------------

func TestSchemaColumns(t *testing.T) {
	setupTestDB(t)
	want := map[string]string{
		"id": "INTEGER", "timestamp": "DATETIME", "model": "TEXT", "provider": "TEXT",
		"original_input_tokens": "INTEGER", "optimized_input_tokens": "INTEGER",
		"input_tokens_saved": "INTEGER", "estimated_output_tokens_saved": "INTEGER",
		"latency_ms": "INTEGER", "savings_category": "TEXT", "original_payload_sample": "TEXT",
		"optimized_payload_sample": "TEXT", "cache_hit": "INTEGER", "upstream_input_tokens": "INTEGER",
		"upstream_output_tokens": "INTEGER", "upstream_cache_read_tokens": "INTEGER",
		"upstream_cache_creation_tokens": "INTEGER",
	}
	got := map[string]string{}
	rows, err := sqlDB.Query("PRAGMA table_info(request_logs)")
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		var cid, notnull, pk int
		var name, typ string
		var dflt any
		if err := rows.Scan(&cid, &name, &typ, &notnull, &dflt, &pk); err != nil {
			t.Fatal(err)
		}
		got[name] = typ
	}
	for k, v := range want {
		if got[k] != v {
			t.Errorf("column %s: type %q, want %q", k, got[k], v)
		}
	}
	if len(got) != len(want) {
		t.Errorf("request_logs has %d columns, want %d", len(got), len(want))
	}
}

func TestSchemaRoundTripAndReopen(t *testing.T) {
	dir := setupTestDB(t)
	id, err := LogRequest("claude-haiku-4-5-20251001", "anthropic", 1000, 400, 50, 120, "aggressive,prose", "hi", "ho", false)
	if err != nil || id == 0 {
		t.Fatalf("LogRequest id=%d err=%v", id, err)
	}
	// Dashboard noise must not be stored, and LogRequest reports that as 0.
	if noID, _ := LogRequest("gpt-4o", "openai", 100, 95, 0, 10, "passthrough", "", "", false); noID != 0 {
		t.Fatalf("noise row was stored with id %d", noID)
	}
	usage := pyjson.NewObject()
	usage.Set("input_tokens", int64(1200))
	usage.Set("output_tokens", int64(33))
	usage.Set("cache_read_input_tokens", int64(7))
	if err := UpdateUpstreamUsage(id, usage); err != nil {
		t.Fatal(err)
	}
	if err := StoreCachedResponse("h1", 200, `{"a":1}`, []byte("body")); err != nil {
		t.Fatal(err)
	}

	// Reopening the same directory must keep every row.
	if err := InitDB(dir); err != nil {
		t.Fatal(err)
	}
	logs, err := GetLogs(20, 0, true, "all")
	if err != nil {
		t.Fatal(err)
	}
	total, _ := logs.Get("total")
	if total != int64(1) {
		t.Fatalf("total after reopen = %v, want 1", total)
	}
	var upIn int64
	if err := sqlDB.QueryRow("SELECT upstream_input_tokens FROM request_logs WHERE id = ?", id).Scan(&upIn); err != nil || upIn != 1200 {
		t.Fatalf("upstream_input_tokens = %d, err %v", upIn, err)
	}
	cached, err := GetCachedResponse("h1")
	if err != nil || cached == nil || cached.StatusCode != 200 || string(cached.Body) != "body" {
		t.Fatalf("cached response = %+v err %v", cached, err)
	}
	if miss, _ := GetCachedResponse("nope"); miss != nil {
		t.Fatalf("expected cache miss, got %+v", miss)
	}
}

func TestDataDirWithSpecialCharacters(t *testing.T) {
	base := t.TempDir()
	dir := filepath.Join(base, "we?ird #dir %20")
	if err := InitDB(dir); err != nil {
		t.Fatalf("InitDB with special characters: %v", err)
	}
	t.Cleanup(func() {
		if sqlDB != nil {
			sqlDB.Close()
			sqlDB = nil
		}
	})
	if _, err := os.Stat(filepath.Join(dir, "proxy_data.db")); err != nil {
		t.Fatalf("database file not created at the literal path: %v", err)
	}
}

func TestClearRequestDataKeepsSettings(t *testing.T) {
	setupTestDB(t)
	if _, err := LogRequest("m", "p", 100, 10, 0, 5, "pruning", "a", "b", false); err != nil {
		t.Fatal(err)
	}
	o, err := ClearRequestData()
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := o.Get("cleared_logs"); n != int64(1) {
		t.Fatalf("cleared_logs = %v", n)
	}
	s, _ := GetSettings()
	if s.Len() != 5 {
		t.Fatalf("settings were touched by clear: %d keys", s.Len())
	}
}

// ---- settings --------------------------------------------------------------------

func TestSettingsForcingOnAggressive(t *testing.T) {
	setupTestDB(t)
	if err := SetSetting("history_pruning_enabled", 0); err != nil {
		t.Fatal(err)
	}
	if err := SetSetting("log_truncation_enabled", 0); err != nil {
		t.Fatal(err)
	}
	rec := do(t, http.MethodPost, "/api/settings/aggressive_enabled", `{"value": true}`,
		map[string]string{"Content-Type": "application/json"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	want := `{"key":"aggressive_enabled","value":1,"settings":{`
	if !strings.HasPrefix(rec.Body.String(), want) {
		t.Fatalf("body prefix %q", rec.Body.String())
	}
	s, _ := GetSettings()
	for _, k := range []string{"aggressive_enabled", "history_pruning_enabled", "log_truncation_enabled"} {
		if v, _ := s.Get(k); v != int64(1) {
			t.Errorf("%s = %v, want 1 after aggressive is turned on", k, v)
		}
	}
}

func TestGetSettingsHealsStaleState(t *testing.T) {
	setupTestDB(t)
	_ = SetSetting("aggressive_enabled", 1)
	_ = SetSetting("history_pruning_enabled", 0)
	s, err := GetSettings()
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := s.Get("history_pruning_enabled"); v != int64(1) {
		t.Fatalf("history_pruning_enabled not healed: %v", v)
	}
	// The key order must be unchanged after healing.
	if got := strings.Join(s.Keys(), ","); got != "compression_enabled,anti_yap_enabled,history_pruning_enabled,log_truncation_enabled,aggressive_enabled" {
		t.Fatalf("key order %s", got)
	}
}

func TestSettingsBodyTruthiness(t *testing.T) {
	setupTestDB(t)
	hdr := map[string]string{"Content-Type": "application/json"}
	cases := map[string]int64{`{"value": "0"}`: 1, `{"value": 0}`: 0, `{"value": ""}`: 0, `{"value": null}`: 0, `{"value": []}`: 0}
	for body, want := range cases {
		rec := do(t, http.MethodPost, "/api/settings/compression_enabled", body, hdr)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s: status %d", body, rec.Code)
		}
		o, err := pyjson.Loads(rec.Body.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		if v, _ := o.(*pyjson.Object).Get("value"); v != want {
			t.Errorf("%s: value %v, want %d", body, v, want)
		}
	}
}

// ---- routes: status codes and bodies ---------------------------------------------

func TestRouteStatusCodes(t *testing.T) {
	setupTestDB(t)
	repo := t.TempDir()
	old := repoDir
	repoDir = repo
	t.Cleanup(func() { repoDir = old })
	hdr := map[string]string{"Content-Type": "application/json"}

	cases := []struct {
		name, method, target, body string
		headers                    map[string]string
		want                       int
	}{
		{"server-info", "GET", "/api/server-info", "", nil, 200},
		{"settings", "GET", "/api/settings", "", nil, 200},
		{"settings missing value", "POST", "/api/settings/k", `{}`, hdr, 400},
		{"settings non-object list", "POST", "/api/settings/k", `[1]`, hdr, 400},
		{"settings list containing value", "POST", "/api/settings/k", `["value"]`, hdr, 500},
		{"settings number body", "POST", "/api/settings/k", `5`, hdr, 500},
		{"settings invalid json", "POST", "/api/settings/k", `not json`, hdr, 500},
		{"settings empty body", "POST", "/api/settings/k", ``, hdr, 500},
		{"stats", "GET", "/api/stats?range=1h", "", nil, 200},
		{"stats bad bool", "GET", "/api/stats?include_noise=maybe", "", nil, 422},
		{"logs", "GET", "/api/logs?limit=5&range=all", "", nil, 200},
		{"logs bad int", "GET", "/api/logs?limit=abc", "", nil, 422},
		{"prices", "GET", "/api/prices", "", nil, 200},
		{"clear", "POST", "/api/data/clear", "", nil, 200},
		{"root json", "GET", "/", "", nil, 200},
		{"root redirect for browsers", "GET", "/", "", map[string]string{"Accept": "text/html,*/*"}, 307},
		{"dashboard redirect", "GET", "/dashboard", "", nil, 307},
		{"wire script missing", "GET", "/install/wire-agents.ps1", "", nil, 404},
		{"unknown path forwarded", "GET", "/v1/messages", "", nil, 200},
		{"unsupported method", "TRACE", "/api/stats", "", nil, 405},
		{"post under dashboard without dist is forwarded", "POST", "/dashboard/x", "", nil, 200},
	}
	// Forwarded requests go to a local stand-in upstream, so the test stays offline.
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "forwarded")
	}))
	t.Cleanup(up.Close)
	t.Setenv("ANTHROPIC_BASE_URL", up.URL)
	t.Setenv("OPENAI_BASE_URL", up.URL)
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			rec := do(t, c.method, c.target, c.body, c.headers)
			if rec.Code != c.want {
				t.Fatalf("%s %s: status %d, want %d (body %q)", c.method, c.target, rec.Code, c.want, rec.Body.String())
			}
		})
	}
}

func TestWireScriptServedWhenPresent(t *testing.T) {
	setupTestDB(t)
	repo := t.TempDir()
	if err := os.MkdirAll(filepath.Join(repo, "install"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "install", "wire-agents.ps1"), []byte("# script\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := repoDir
	repoDir = repo
	t.Cleanup(func() { repoDir = old })
	rec := do(t, http.MethodGet, "/install/wire-agents.ps1", "", nil)
	if rec.Code != 200 || rec.Body.String() != "# script\n" {
		t.Fatalf("status %d body %q", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/plain; charset=utf-8" {
		t.Fatalf("content-type %q", ct)
	}
	if cd := rec.Header().Get("Content-Disposition"); cd != `attachment; filename="wire-agents.ps1"` {
		t.Fatalf("content-disposition %q", cd)
	}
}

func TestDashboardStatic(t *testing.T) {
	setupTestDB(t)
	dist := t.TempDir()
	if err := os.WriteFile(filepath.Join(dist, "index.html"), []byte("<html>x</html>"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dist, "app.js"), []byte("console.log(1)"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := frontendDistDir
	frontendDistDir = dist
	t.Cleanup(func() { frontendDistDir = old })

	rec := do(t, http.MethodGet, "/dashboard/", "", nil)
	if rec.Code != 200 || rec.Header().Get("Cache-Control") != "no-cache, no-store, must-revalidate" {
		t.Fatalf("index: status %d cache %q", rec.Code, rec.Header().Get("Cache-Control"))
	}
	rec = do(t, http.MethodGet, "/dashboard/app.js", "", nil)
	if rec.Code != 200 || rec.Header().Get("Cache-Control") != "" {
		t.Fatalf("js should not be marked no-cache: status %d cache %q", rec.Code, rec.Header().Get("Cache-Control"))
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/javascript") {
		t.Fatalf("js content-type %q", ct)
	}
	rec = do(t, http.MethodGet, "/dashboard/missing.css", "", nil)
	if rec.Code != 404 {
		t.Fatalf("missing file status %d", rec.Code)
	}
	// Traversal must not escape the dist directory.
	rec = do(t, http.MethodGet, "/dashboard/../../etc/passwd", "", nil)
	if rec.Code != 404 {
		t.Fatalf("traversal status %d", rec.Code)
	}
}

func TestDashboardMountSkippedWithoutDist(t *testing.T) {
	setupTestDB(t)
	old := frontendDistDir
	frontendDistDir = ""
	t.Cleanup(func() { frontendDistDir = old })
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "forwarded")
	}))
	t.Cleanup(up.Close)
	t.Setenv("ANTHROPIC_BASE_URL", up.URL)
	rec := do(t, http.MethodGet, "/dashboard/", "", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d, want the request forwarded (200) when the mount is skipped", rec.Code)
	}
}

// ---- prices -----------------------------------------------------------------------

func TestPricesRefreshFailureIs502(t *testing.T) {
	setupTestDB(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()
	old := litellmPricesURL
	litellmPricesURL = srv.URL
	t.Cleanup(func() { litellmPricesURL = old })

	rec := do(t, http.MethodPost, "/api/prices/refresh", "", nil)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "HTTP Error 500: Internal Server Error") {
		t.Fatalf("detail %q", rec.Body.String())
	}
}

func TestPricesRefreshStoresLiteLLMRows(t *testing.T) {
	setupTestDB(t)
	payload := `{
		"sample_spec": {"litellm_provider": "anthropic", "input_cost_per_token": 1, "output_cost_per_token": 1},
		"claude-test-x": {"litellm_provider": "anthropic", "input_cost_per_token": 0.000002, "output_cost_per_token": 0.00001},
		"random-model": {"litellm_provider": "unknown", "input_cost_per_token": 0.1, "output_cost_per_token": 0.1},
		"gpt-free": {"litellm_provider": "openai", "input_cost_per_token": 0, "output_cost_per_token": 0},
		"gpt-str": {"litellm_provider": "openai", "input_cost_per_token": "0.000001", "output_cost_per_token": 0}
	}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, payload)
	}))
	defer srv.Close()
	old := litellmPricesURL
	litellmPricesURL = srv.URL
	t.Cleanup(func() { litellmPricesURL = old })

	res, err := refreshPricesFromLiteLLM()
	if err != nil {
		t.Fatal(err)
	}
	if n, _ := res.Get("models"); n != int64(2) {
		t.Fatalf("models = %v, want 2 (sample_ and unknown-provider entries skipped)", n)
	}
	rows, err := allModelPrices(true)
	if err != nil {
		t.Fatal(err)
	}
	found := map[string]float64{}
	for _, r := range rows {
		found[r.Model] = r.InputUSD
	}
	near := func(a, b float64) bool { return a-b < 1e-9 && b-a < 1e-9 }
	if !near(found["claude-test-x"], 2.0) || !near(found["gpt-str"], 1.0) {
		t.Fatalf("stored rates %v", found)
	}
	if _, ok := found["random-model"]; ok {
		t.Fatal("unknown-provider model should not be stored")
	}
}

func TestMaybeAutoRefreshRecordsError(t *testing.T) {
	setupTestDB(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusNotFound)
	}))
	defer srv.Close()
	old := litellmPricesURL
	litellmPricesURL = srv.URL
	t.Cleanup(func() { litellmPricesURL = old })

	res := maybeAutoRefresh()
	if res == nil {
		t.Fatal("expected an error result when the table is empty")
	}
	if _, ok := res.Get("error"); !ok {
		t.Fatalf("no error key: %v", res.Keys())
	}
	last, _ := GetMeta("prices_last_error")
	if last == nil || *last != "HTTPError: HTTP Error 404: Not Found" {
		t.Fatalf("prices_last_error = %v", last)
	}
}

func TestMaybeAutoRefreshSkipsFreshTable(t *testing.T) {
	setupTestDB(t)
	if err := SetMeta("prices_updated_at", time.Now().UTC().Format(pricesTimeLayout)); err != nil {
		t.Fatal(err)
	}
	// The URL is unreachable on purpose. A fresh table must not trigger a fetch.
	old := litellmPricesURL
	litellmPricesURL = "http://127.0.0.1:1/unreachable"
	t.Cleanup(func() { litellmPricesURL = old })
	if res := maybeAutoRefresh(); res != nil {
		t.Fatalf("expected no refresh, got %v", res.Keys())
	}
}

func TestPricingPureHelpers(t *testing.T) {
	keys := normalizeKeys("claude-haiku-4-5-20251001")
	want := []string{"claude-haiku-4-5-20251001", "anthropic/claude-haiku-4-5-20251001", "claude-haiku-4-5", "anthropic/claude-haiku-4-5"}
	if strings.Join(keys, "|") != strings.Join(want, "|") {
		t.Fatalf("normalizeKeys = %v", keys)
	}
	if got := normalizeKeys("gpt-4o"); strings.Join(got, "|") != "gpt-4o|anthropic/gpt-4o" {
		t.Fatalf("short suffix should not be dropped: %v", got)
	}
	rows := []priceRow{{Model: "claude-haiku-4-5", InputUSD: 1, OutputUSD: 5}}
	if in, out := lookupRatesFrom(rows, "claude-haiku-4-5-20251001"); in != 1 || out != 5 {
		t.Fatalf("dated name should resolve to the base row: %v %v", in, out)
	}
	if in, out := lookupRatesFrom(nil, "mystery"); in != defaultInputUSD || out != defaultOutputUSD {
		t.Fatalf("default rates not used: %v %v", in, out)
	}
	match := matchPriceRow([]priceRow{{Model: "openai/gpt-4o-mini"}}, []string{"gpt-4o-mini"})
	if match == nil || match.Model != "openai/gpt-4o-mini" {
		t.Fatalf("suffix match failed: %v", match)
	}
}

func TestPyRoundMatchesPython(t *testing.T) {
	// Python: round(0.125, 2) == 0.12 (ties to even), round(2.675, 2) == 2.67.
	cases := []struct {
		x    float64
		n    int
		want float64
	}{{0.125, 2, 0.12}, {2.675, 2, 2.67}, {2.5, 0, 2}, {1.25, 1, 1.2}, {3.14159, 3, 3.142}}
	for _, c := range cases {
		if got := pyRound(c.x, c.n); got != c.want {
			t.Errorf("pyRound(%v, %d) = %v, want %v", c.x, c.n, got, c.want)
		}
	}
}

func TestParseLogTimestamp(t *testing.T) {
	ok := map[string]string{
		"2026-10-29 12:00:00":       "2026-10-29T12:00:00Z",
		"2026-10-29T12:00:00Z":      "2026-10-29T12:00:00Z",
		"2026-10-29T12:00:00+02:00": "2026-10-29T10:00:00Z",
		"2026-10-29 12:00:00.123":   "2026-10-29T12:00:00Z",
	}
	for in, want := range ok {
		got, valid := parseLogTimestamp(in)
		if !valid || got.UTC().Format("2006-01-02T15:04:05Z") != want {
			t.Errorf("parse %q = %v %v, want %s", in, got, valid, want)
		}
	}
	for _, bad := range []string{"", "2026-13-01 00:00:00", "not a time", "2026-02-30 00:00:00"} {
		if _, valid := parseLogTimestamp(bad); valid {
			t.Errorf("%q should not parse", bad)
		}
	}
}

func TestIsDashboardNoise(t *testing.T) {
	cases := []struct {
		cat    string
		saved  int64
		sample string
		want   bool
	}{
		{"passthrough", 0, "", true},
		{"aggressive", 600, "", false},
		{"aggressive", 10, "", true},
		{"unchanged", 0, "hello", false},
		{"", 0, "", true},
		{"passthrough", 0, `{"logs":[]}`, true},
		{"passthrough", 0, "agent.v1.thing with a long body that goes past twenty four chars", false},
		{"passthrough", 100, "", false},
	}
	for _, c := range cases {
		if got := isDashboardNoise(c.cat, c.saved, c.sample); got != c.want {
			t.Errorf("isDashboardNoise(%q, %d, %q) = %v, want %v", c.cat, c.saved, c.sample, got, c.want)
		}
	}
}

func TestNonFiniteFloatIs500(t *testing.T) {
	o := pyjson.NewObject()
	o.Set("x", math.NaN())
	rec := httptest.NewRecorder()
	writeJSON(rec, http.StatusOK, o)
	if rec.Code != http.StatusInternalServerError || rec.Body.String() != "Internal Server Error" {
		t.Fatalf("NaN gave %d %q, want a plain 500 like allow_nan=False", rec.Code, rec.Body.String())
	}
}

// ---- config and .env ----------------------------------------------------------------

func TestDotenvLineParsing(t *testing.T) {
	cases := map[string]string{
		"KEY=value":           "value",
		"export KEY=value":    "value",
		`KEY="a b\n\"c"`:      "a b\n\"c",
		"KEY='lit $x'":        "lit $x",
		"KEY=plain # comment": "plain",
		"KEY=":                "",
		"KEY = spaced":        "spaced",
	}
	for line, want := range cases {
		k, v, ok := parseDotenvLine(line)
		if !ok || k != "KEY" {
			t.Errorf("%q: key %q ok %v", line, k, ok)
		}
		if v != want {
			t.Errorf("%q: value %q, want %q", line, v, want)
		}
	}
	for _, skip := range []string{"", "# comment", "no equals sign", "1BAD=x"} {
		if _, _, ok := parseDotenvLine(skip); ok {
			t.Errorf("%q should be skipped", skip)
		}
	}
}

func TestConfigDefaultsAndPrecedence(t *testing.T) {
	for _, k := range []string{"GATEKEEP_HOST", "HOST", "GATEKEEP_PORT", "PORT", "GATEKEEP_AGENT_PORT", "GATEKEEP_API_KEY"} {
		t.Setenv(k, "")
		_ = os.Unsetenv(k)
	}
	// The dotenv loader reads .env from the cwd and parents. This test runs
	// in server/, and the repo has no .env there, so the defaults apply.
	if err := loadConfig(); err != nil {
		t.Fatal(err)
	}
	if cfgHost != "0.0.0.0" || cfgPort != 9477 || cfgAgentPort != 9478 {
		t.Fatalf("defaults wrong: %s %d %d", cfgHost, cfgPort, cfgAgentPort)
	}
	t.Setenv("HOST", "10.0.0.5")
	t.Setenv("GATEKEEP_HOST", "")
	if err := loadConfig(); err != nil {
		t.Fatal(err)
	}
	if cfgHost != "" {
		t.Fatalf("an empty GATEKEEP_HOST must win over HOST, got %q", cfgHost)
	}
}

func TestPublicURLs(t *testing.T) {
	t.Setenv("GATEKEEP_PUBLIC_URL", "https://gk.example.test/")
	t.Setenv("GATEKEEP_BASE_URL", "")
	t.Setenv("GATEKEEP_AGENT_URL", "")
	cfgAgentPort = 9478
	if got := agentPublicURL(); got != "https://gk.example.test:9478" {
		t.Fatalf("agentPublicURL = %s", got)
	}
	if scheme, host := pyURLSplit("HTTP://User@Host.Example:9477/x"); scheme != "http" || host != "host.example" {
		t.Fatalf("pyURLSplit = %q %q", scheme, host)
	}
}
