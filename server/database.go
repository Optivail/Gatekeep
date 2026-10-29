package main

// SQLite layer, a port of backend/database.py. The schema, column names,
// types, and SQL text are kept identical so that Python and Go can read and
// write the same proxy_data.db. The journal mode is left at SQLite's default
// (DELETE), not WAL, because WAL is persistent and would change the file
// layout that Python expects.

import (
	"database/sql"
	"errors"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	_ "modernc.org/sqlite"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

const (
	payloadSampleChars = 4000
	maxCachedResponses = 500
	dashboardRecent    = 40
	minDashboardSaved  = 50
)

var rangePresets = map[string]time.Duration{
	"1h":  time.Hour,
	"24h": 24 * time.Hour,
	"7d":  7 * 24 * time.Hour,
	"30d": 30 * 24 * time.Hour,
}

// Dashboard noise markers, copied from database.py. Order matters only for
// readability; each check is a plain substring or prefix test.
var noiseSampleMarkers = []string{
	"v1/traces",
	"analyticsservice/track",
	"analyticsservice/submit",
	"dashboardservice/",
	"aiservice/availablemodels",
	"aiservice/getusablemodels",
	"aiservice/getdefaultmodelforcli",
	"aiservice/getclidownloadurl",
	`{"logs":`,
	`{"events":`,
	"service.name",
	"cursor-agent-",
}

var meaningfulCategories = []string{
	"wire_proto_crush",
	"wire_proto_crush_aggressive",
	"cache_hit",
	"intent_cache",
	"anti_yap",
	"pruning",
	"ccr",
	"aggressive",
	"prose",
	"log_truncation",
	"unchanged",
}

var defaultSettings = []struct {
	key   string
	value int64
}{
	{"compression_enabled", 1},
	{"anti_yap_enabled", 1},
	{"history_pruning_enabled", 1},
	{"log_truncation_enabled", 1},
	{"aggressive_enabled", 1},
}

// requestLogColumns is the SELECT order used by the Python code. It fixes the
// key order of every log row in the API output.
var requestLogColumns = []string{
	"id", "timestamp", "model", "provider", "original_input_tokens",
	"optimized_input_tokens", "input_tokens_saved", "estimated_output_tokens_saved",
	"latency_ms", "savings_category", "original_payload_sample",
	"optimized_payload_sample", "cache_hit", "upstream_input_tokens",
	"upstream_output_tokens", "upstream_cache_read_tokens",
	"upstream_cache_creation_tokens",
}

// dbPath and sqlDB are set by InitDB. The Python module does the same at
// import time from GATEKEEP_DATA_DIR.
var (
	sqlDB *sql.DB
)

// InitDB opens (or creates) dataDir/proxy_data.db and creates the schema if
// it is missing. Existing tables and rows are left alone. Like init_db()
// in Python, it is safe to call on every start.
func InitDB(dataDir string) error {
	if err := os.MkdirAll(dataDir, 0o777); err != nil {
		return err
	}
	path := filepath.Join(dataDir, "proxy_data.db")
	if sqlDB != nil {
		sqlDB.Close()
		sqlDB = nil
	}
	db, err := sql.Open("sqlite", sqliteDSN(path))
	if err != nil {
		return err
	}
	// One connection inside this process: several pooled connections writing at
	// once make SQLite return SQLITE_BUSY immediately when a transaction
	// upgrades its lock (seen as "database is locked" 500s). Cross-process
	// access from the Python reference still uses the busy timeout.
	db.SetMaxOpenConns(1)
	// Python's sqlite3.connect uses a 5 second busy timeout.
	stmts := []string{
		"CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value INTEGER NOT NULL)",
		"CREATE TABLE IF NOT EXISTS request_logs (" +
			"id INTEGER PRIMARY KEY AUTOINCREMENT, " +
			"timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, " +
			"model TEXT, " +
			"provider TEXT, " +
			"original_input_tokens INTEGER, " +
			"optimized_input_tokens INTEGER, " +
			"input_tokens_saved INTEGER, " +
			"estimated_output_tokens_saved INTEGER, " +
			"latency_ms INTEGER, " +
			"savings_category TEXT, " +
			"original_payload_sample TEXT, " +
			"optimized_payload_sample TEXT, " +
			"cache_hit INTEGER DEFAULT 0, " +
			"upstream_input_tokens INTEGER, " +
			"upstream_output_tokens INTEGER, " +
			"upstream_cache_read_tokens INTEGER, " +
			"upstream_cache_creation_tokens INTEGER)",
		"CREATE INDEX IF NOT EXISTS idx_request_logs_timestamp ON request_logs(timestamp)",
		"CREATE TABLE IF NOT EXISTS response_cache (" +
			"request_hash TEXT PRIMARY KEY, " +
			"status_code INTEGER, " +
			"headers TEXT, " +
			"body BLOB, " +
			"created_at DATETIME DEFAULT CURRENT_TIMESTAMP)",
		"CREATE TABLE IF NOT EXISTS model_prices (" +
			"model TEXT PRIMARY KEY, " +
			"input_per_mtok REAL NOT NULL, " +
			"output_per_mtok REAL NOT NULL, " +
			"updated_at TEXT NOT NULL, " +
			"source TEXT)",
		"CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			db.Close()
			return fmt.Errorf("schema: %w", err)
		}
	}
	for _, d := range defaultSettings {
		if _, err := db.Exec("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", d.key, d.value); err != nil {
			db.Close()
			return err
		}
	}
	sqlDB = db
	return nil
}

// sqliteDSN builds a modernc DSN for path. The path is escaped so that '?',
// '#' and '%' in a directory name are not read as URI syntax.
func sqliteDSN(path string) string {
	esc := strings.NewReplacer("%", "%25", "?", "%3F", "#", "%23").Replace(path)
	return "file:" + esc + "?_pragma=busy_timeout(5000)"
}

func dbHandle() (*sql.DB, error) {
	if sqlDB == nil {
		return nil, errors.New("database not initialised: call InitDB first")
	}
	return sqlDB, nil
}

// ---- meta and model prices ---------------------------------------------------

// GetMeta returns the meta value, or nil when the key is absent. This mirrors
// Python returning None.
func GetMeta(key string) (*string, error) {
	db, err := dbHandle()
	if err != nil {
		return nil, err
	}
	var v string
	err = db.QueryRow("SELECT value FROM meta WHERE key = ?", key).Scan(&v)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &v, nil
}

// SetMeta upserts a meta value.
func SetMeta(key, value string) error {
	db, err := dbHandle()
	if err != nil {
		return err
	}
	_, err = db.Exec(
		"INSERT INTO meta (key, value) VALUES (?, ?) "+
			"ON CONFLICT(key) DO UPDATE SET value = excluded.value",
		key, value,
	)
	return err
}

// UpsertModelPrice stores one price row. Rates are USD per million tokens.
func UpsertModelPrice(model string, inputPerMTok, outputPerMTok float64, updatedAt, source string) error {
	db, err := dbHandle()
	if err != nil {
		return err
	}
	_, err = db.Exec(
		"INSERT INTO model_prices (model, input_per_mtok, output_per_mtok, updated_at, source) "+
			"VALUES (?, ?, ?, ?, ?) "+
			"ON CONFLICT(model) DO UPDATE SET "+
			"input_per_mtok = excluded.input_per_mtok, "+
			"output_per_mtok = excluded.output_per_mtok, "+
			"updated_at = excluded.updated_at, "+
			"source = excluded.source",
		model, inputPerMTok, outputPerMTok, updatedAt, source,
	)
	return err
}

// CountModelPrices returns the number of stored price rows.
func CountModelPrices() (int64, error) {
	db, err := dbHandle()
	if err != nil {
		return 0, err
	}
	var n int64
	err = db.QueryRow("SELECT COUNT(*) AS n FROM model_prices").Scan(&n)
	return n, err
}

// priceRow is one model_prices row. The SQL text for the full and the
// three-column reads is the same as in Python, so the row order matches too.
type priceRow struct {
	Model      string
	InputUSD   float64
	OutputUSD  float64
	UpdatedAt  string
	Source     string
	hasDetails bool
}

// allModelPrices reads the table in SQLite's scan order. Lookups depend on
// that order (the substring match returns the first hit), so no ORDER BY is
// added.
func allModelPrices(full bool) ([]priceRow, error) {
	db, err := dbHandle()
	if err != nil {
		return nil, err
	}
	q := "SELECT model, input_per_mtok, output_per_mtok FROM model_prices"
	if full {
		q = "SELECT model, input_per_mtok, output_per_mtok, updated_at, source FROM model_prices"
	}
	rows, err := db.Query(q)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []priceRow
	for rows.Next() {
		var p priceRow
		if full {
			var src sql.NullString
			if err := rows.Scan(&p.Model, &p.InputUSD, &p.OutputUSD, &p.UpdatedAt, &src); err != nil {
				return nil, err
			}
			p.Source = src.String
			p.hasDetails = true
		} else if err := rows.Scan(&p.Model, &p.InputUSD, &p.OutputUSD); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

// ---- helpers shared by the stats and logs queries ------------------------------

// logRow is a request_logs row keyed by column name, as Python's dict(row).
type logRow map[string]any

// asInt is Python's `value or 0` followed by integer arithmetic. Non-numeric
// values count as 0. Python would raise for those, but the proxy only writes
// integers, so the difference is not reachable in practice.
func asInt(v any) int64 {
	switch t := v.(type) {
	case int64:
		return t
	case float64:
		return int64(t)
	case bool:
		if t {
			return 1
		}
	}
	return 0
}

// strOr returns v when it is a non-empty string, else def. This is Python's
// `value or default` for string fields.
func strOr(v any, def string) string {
	if s, ok := v.(string); ok && s != "" {
		return s
	}
	return def
}

// pyTruthy follows Python's bool() for the value types pyjson produces.
func pyTruthy(v any) bool {
	switch t := v.(type) {
	case nil:
		return false
	case bool:
		return t
	case int64:
		return t != 0
	case int:
		return t != 0
	case *big.Int:
		return t.Sign() != 0
	case float64:
		return t != 0 // NaN is truthy in Python too, and NaN != 0 here.
	case string:
		return t != ""
	case []any:
		return len(t) > 0
	case *pyjson.Object:
		return t.Len() > 0
	}
	return true
}

var isoTimestampRE = regexp.MustCompile(
	`^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2})(?::(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?)?)?(Z|[+-]\d{2}(?::?\d{2})?)?$`,
)

// parseLogTimestamp is database.py's _parse_log_timestamp. Naive values are
// taken as UTC, since SQLite's CURRENT_TIMESTAMP is UTC. ok is false when the
// value does not parse.
func parseLogTimestamp(ts string) (time.Time, bool) {
	if ts == "" {
		return time.Time{}, false
	}
	s := strings.ReplaceAll(strings.TrimSpace(ts), " ", "T")
	if strings.HasSuffix(s, "Z") {
		s = s[:len(s)-1] + "+00:00"
	}
	m := isoTimestampRE.FindStringSubmatch(s)
	if m == nil {
		return time.Time{}, false
	}
	num := func(i int) int {
		if m[i] == "" {
			return 0
		}
		n, _ := strconv.Atoi(m[i])
		return n
	}
	year, month, day := num(1), num(2), num(3)
	hour, minute, sec := num(4), num(5), num(6)
	nanos := 0
	if m[7] != "" {
		frac := m[7] + strings.Repeat("0", 6-len(m[7]))
		n, _ := strconv.Atoi(frac)
		nanos = n * 1000
	}
	if hour > 23 || minute > 59 || sec > 59 {
		return time.Time{}, false
	}
	t := time.Date(year, time.Month(month), day, hour, minute, sec, nanos, time.UTC)
	if t.Year() != year || int(t.Month()) != month || t.Day() != day {
		return time.Time{}, false
	}
	offset := 0
	if zone := m[8]; zone != "" && zone != "Z" {
		sign := 1
		if zone[0] == '-' {
			sign = -1
		}
		digits := strings.ReplaceAll(zone[1:], ":", "")
		hh, _ := strconv.Atoi(digits[:2])
		mm := 0
		if len(digits) >= 4 {
			mm, _ = strconv.Atoi(digits[2:4])
		}
		offset = sign * (hh*3600 + mm*60)
	}
	return t.Add(-time.Duration(offset) * time.Second), true
}

// timestampToAPI formats a stored timestamp as UTC ISO-8601 with a Z suffix.
// Values that do not parse are returned unchanged, as in Python.
func timestampToAPI(ts string) string {
	t, ok := parseLogTimestamp(ts)
	if !ok {
		return ts
	}
	return t.UTC().Format("2006-01-02T15:04:05Z")
}

// rowForAPI copies a row and converts its timestamp. The input is not
// modified.
func rowForAPI(row logRow) logRow {
	out := make(logRow, len(row))
	for k, v := range row {
		out[k] = v
	}
	if ts, ok := out["timestamp"].(string); ok && ts != "" {
		out["timestamp"] = timestampToAPI(ts)
	}
	return out
}

// rowObject turns a row into an ordered JSON object in column order.
func rowObject(row logRow) *pyjson.Object {
	o := pyjson.NewObject()
	for _, c := range requestLogColumns {
		o.Set(c, row[c])
	}
	return o
}

func normalizeRange(key string) string {
	k := strings.ToLower(strings.TrimSpace(key))
	if k == "" {
		return "24h"
	}
	if _, ok := rangePresets[k]; ok || k == "all" {
		return k
	}
	return "24h"
}

// isDashboardNoise is database.py's is_dashboard_noise. It hides OTLP,
// analytics, and dashboard probes from the savings views.
func isDashboardNoise(category string, saved int64, sample string) bool {
	cat := strings.ToLower(strOr(category, "passthrough"))
	smp := strings.ToLower(strings.TrimSpace(sample))

	if strings.Contains(smp, "serverconfigservice") || strings.Contains(cat, "server_config") {
		return true
	}
	if saved >= minDashboardSaved {
		return false
	}
	if saved > 0 {
		return true
	}
	for _, tag := range meaningfulCategories {
		if strings.Contains(cat, tag) && tag != "unchanged" {
			return false
		}
	}
	if strings.HasPrefix(smp, "agent.v1.") {
		return false
	}
	for _, m := range noiseSampleMarkers {
		if strings.Contains(smp, m) {
			return true
		}
	}
	if strings.HasPrefix(smp, "post /aiserver.v1.") {
		return true
	}
	if (cat == "passthrough" || cat == "server_config_noop") && utf8.RuneCountInString(smp) <= 24 {
		return true
	}
	return cat == "passthrough" && smp == ""
}

func logIsVisible(row logRow, includeNoise bool) bool {
	if includeNoise {
		return true
	}
	return !isDashboardNoise(
		strOr(row["savings_category"], ""),
		asInt(row["input_tokens_saved"]),
		strOr(row["original_payload_sample"], ""),
	)
}

// visibleRows returns request_logs newest first, after the noise filter.
func visibleRows(includeNoise bool) ([]logRow, error) {
	db, err := dbHandle()
	if err != nil {
		return nil, err
	}
	// CAST keeps the timestamp as TEXT. Without it the driver converts
	// DATETIME columns to time.Time, which would break string parsing.
	q := "SELECT id, CAST(timestamp AS TEXT) AS timestamp, model, provider, original_input_tokens, " +
		"optimized_input_tokens, input_tokens_saved, estimated_output_tokens_saved, latency_ms, " +
		"savings_category, original_payload_sample, optimized_payload_sample, cache_hit, " +
		"upstream_input_tokens, upstream_output_tokens, upstream_cache_read_tokens, " +
		"upstream_cache_creation_tokens FROM request_logs ORDER BY id DESC"
	rows, err := db.Query(q)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []logRow
	for rows.Next() {
		vals := make([]any, len(requestLogColumns))
		ptrs := make([]any, len(vals))
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return nil, err
		}
		row := make(logRow, len(vals))
		for i, c := range requestLogColumns {
			row[c] = vals[i]
		}
		if logIsVisible(row, includeNoise) {
			out = append(out, row)
		}
	}
	return out, rows.Err()
}

// filterRowsByWindow keeps rows with since <= ts < until. Rows whose timestamp
// does not parse are dropped whenever a bound is set. With no bounds, every
// row is kept, as in Python.
func filterRowsByWindow(rows []logRow, since, until *time.Time) []logRow {
	if since == nil && until == nil {
		return rows
	}
	var out []logRow
	for _, row := range rows {
		ts, ok := parseLogTimestamp(strOr(row["timestamp"], ""))
		if !ok {
			continue
		}
		if since != nil && ts.Before(*since) {
			continue
		}
		if until != nil && !ts.Before(*until) {
			continue
		}
		out = append(out, row)
	}
	return out
}

// pyRound is Python's round(x, n) for floats. FormatFloat rounds the exact
// binary value with ties to even, which is the same rule CPython uses, so the
// results match bit for bit.
func pyRound(x float64, n int) float64 {
	f, err := strconv.ParseFloat(strconv.FormatFloat(x, 'f', n, 64), 64)
	if err != nil {
		return x
	}
	return f
}

// aggregate holds the totals for one window. Its fields follow the key order
// of database.py's _aggregate_rows.
type aggregate struct {
	totalRequests   int64
	originalInput   int64
	optimizedInput  int64
	inputSaved      int64
	outputSaved     int64
	tokensSaved     int64
	avgSavePct      float64
	stretch         float64
	estimatedUSD    float64
	avgLatencyMS    float64
	cacheHits       int64
	upCacheRead     int64
	upCacheCreation int64
	upInput         int64
	upOutput        int64
	techniqueHits   *pyjson.Object
}

// aggregateRows totals rows. The price table is loaded once by the caller
// rather than once per row, which gives the same result with fewer queries.
func aggregateRows(rows []logRow, prices []priceRow) aggregate {
	var a aggregate
	a.totalRequests = int64(len(rows))
	var latencySum int64
	for _, r := range rows {
		a.originalInput += asInt(r["original_input_tokens"])
		a.optimizedInput += asInt(r["optimized_input_tokens"])
		a.inputSaved += asInt(r["input_tokens_saved"])
		a.outputSaved += asInt(r["estimated_output_tokens_saved"])
		a.cacheHits += asInt(r["cache_hit"])
		a.upCacheRead += asInt(r["upstream_cache_read_tokens"])
		a.upCacheCreation += asInt(r["upstream_cache_creation_tokens"])
		a.upInput += asInt(r["upstream_input_tokens"])
		a.upOutput += asInt(r["upstream_output_tokens"])
		latencySum += asInt(r["latency_ms"])
	}
	if len(rows) > 0 {
		a.avgLatencyMS = float64(latencySum) / float64(len(rows))
	}
	a.tokensSaved = a.inputSaved + a.outputSaved

	// Technique tags keep the order in which they first appear, which is the
	// order of the rows (newest first). A plain map would lose that order.
	type tagAcc struct{ count, saved int64 }
	tags := []string{}
	acc := map[string]*tagAcc{}
	for _, r := range rows {
		raw := strOr(r["savings_category"], "unchanged")
		saved := asInt(r["input_tokens_saved"])
		for _, part := range strings.Split(raw, ",") {
			tag := strings.TrimSpace(part)
			if tag == "" {
				tag = "unchanged"
			}
			b, ok := acc[tag]
			if !ok {
				b = &tagAcc{}
				acc[tag] = b
				tags = append(tags, tag)
			}
			b.count++
			b.saved += saved
		}
	}
	a.techniqueHits = pyjson.NewObject()
	for _, tag := range tags {
		o := pyjson.NewObject()
		o.Set("count", acc[tag].count)
		o.Set("saved", acc[tag].saved)
		a.techniqueHits.Set(tag, o)
	}

	if a.originalInput != 0 {
		a.avgSavePct = float64(a.inputSaved) / float64(a.originalInput) * 100
	}
	if a.optimizedInput != 0 {
		a.stretch = float64(a.originalInput) / float64(a.optimizedInput)
	} else {
		a.stretch = 1.0
	}

	for _, r := range rows {
		a.estimatedUSD += estimateUSDSavedFrom(prices,
			asInt(r["input_tokens_saved"]),
			asInt(r["estimated_output_tokens_saved"]),
			strOr(r["model"], ""),
		)
	}

	a.avgSavePct = pyRound(a.avgSavePct, 1)
	a.stretch = pyRound(a.stretch, 2)
	a.estimatedUSD = pyRound(a.estimatedUSD, 6)
	a.avgLatencyMS = pyRound(a.avgLatencyMS, 1)
	return a
}

// toObject lists the fields in Python's key order.
func (a aggregate) toObject() *pyjson.Object {
	o := pyjson.NewObject()
	o.Set("total_requests", a.totalRequests)
	o.Set("total_original_input_tokens", a.originalInput)
	o.Set("total_optimized_input_tokens", a.optimizedInput)
	o.Set("total_input_tokens_saved", a.inputSaved)
	o.Set("total_output_tokens_saved", a.outputSaved)
	o.Set("total_tokens_saved", a.tokensSaved)
	o.Set("avg_save_pct", a.avgSavePct)
	o.Set("stretch_multiplier", a.stretch)
	o.Set("estimated_usd_saved", a.estimatedUSD)
	o.Set("avg_latency_ms", a.avgLatencyMS)
	o.Set("total_cache_hits", a.cacheHits)
	o.Set("total_upstream_cache_read_tokens", a.upCacheRead)
	o.Set("total_upstream_cache_creation_tokens", a.upCacheCreation)
	o.Set("total_upstream_input_tokens", a.upInput)
	o.Set("total_upstream_output_tokens", a.upOutput)
	o.Set("technique_hits", a.techniqueHits)
	return o
}

// pctChange is _pct_change. A nil result is JSON null.
func pctChange(current, previous float64) any {
	if previous == 0 {
		if current == 0 {
			return nil
		}
		return 100.0
	}
	return pyRound((current-previous)/previous*100, 1)
}

// granularityForRange is _granularity_for_range.
func granularityForRange(rk string, span *time.Duration) string {
	switch rk {
	case "1h":
		return "5m"
	case "24h":
		return "hour"
	case "7d", "30d":
		return "day"
	}
	if span == nil {
		return "day"
	}
	switch {
	case *span <= 2*time.Hour:
		return "5m"
	case *span <= 48*time.Hour:
		return "hour"
	case *span <= 90*24*time.Hour:
		return "day"
	}
	return "week"
}

// bucketStart truncates a UTC time to the start of its bucket. Weeks start on
// Monday.
func bucketStart(ts time.Time, granularity string) time.Time {
	ts = ts.UTC()
	y, mo, d := ts.Date()
	h, mi := ts.Hour(), ts.Minute()
	switch granularity {
	case "5m":
		return time.Date(y, mo, d, h, (mi/5)*5, 0, 0, time.UTC)
	case "hour":
		return time.Date(y, mo, d, h, 0, 0, 0, time.UTC)
	case "day":
		return time.Date(y, mo, d, 0, 0, 0, 0, time.UTC)
	}
	start := time.Date(y, mo, d, 0, 0, 0, 0, time.UTC)
	monday := (int(start.Weekday()) + 6) % 7
	return start.AddDate(0, 0, -monday)
}

func bucketStep(granularity string) time.Duration {
	switch granularity {
	case "5m":
		return 5 * time.Minute
	case "hour":
		return time.Hour
	case "day":
		return 24 * time.Hour
	}
	return 7 * 24 * time.Hour
}

// buildTimeseries is _build_timeseries. Every bucket between since and until
// is present, including empty ones.
func buildTimeseries(rows []logRow, since, until time.Time, granularity string) []any {
	since = bucketStart(since, granularity)
	until = bucketStart(until, granularity)
	type bucket struct {
		requests, orig, opt, saved, outSaved int64
	}
	buckets := map[int64]*bucket{}
	step := bucketStep(granularity)
	for cur := since; !cur.After(until); cur = cur.Add(step) {
		buckets[cur.Unix()] = &bucket{}
	}
	for _, r := range rows {
		ts, ok := parseLogTimestamp(strOr(r["timestamp"], ""))
		if !ok {
			continue
		}
		key := bucketStart(ts, granularity).Unix()
		b, ok := buckets[key]
		if !ok {
			b = &bucket{}
			buckets[key] = b
		}
		b.requests++
		b.orig += asInt(r["original_input_tokens"])
		b.opt += asInt(r["optimized_input_tokens"])
		b.saved += asInt(r["input_tokens_saved"])
		b.outSaved += asInt(r["estimated_output_tokens_saved"])
	}
	keys := make([]int64, 0, len(buckets))
	for k := range buckets {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	out := make([]any, 0, len(keys))
	for _, k := range keys {
		b := buckets[k]
		o := pyjson.NewObject()
		o.Set("t", time.Unix(k, 0).UTC().Format("2006-01-02T15:04:05Z"))
		o.Set("requests", b.requests)
		o.Set("original_input_tokens", b.orig)
		o.Set("optimized_input_tokens", b.opt)
		o.Set("input_tokens_saved", b.saved)
		o.Set("output_tokens_saved", b.outSaved)
		out = append(out, o)
	}
	return out
}

// ---- settings ---------------------------------------------------------------

// GetSettings returns the settings object in table order. It also repairs
// stale UI state: when aggressive mode is on, the history and log passes are
// forced on, as main.py does.
func GetSettings() (*pyjson.Object, error) {
	db, err := dbHandle()
	if err != nil {
		return nil, err
	}
	rows, err := db.Query("SELECT key, value FROM settings")
	if err != nil {
		return nil, err
	}
	settings := pyjson.NewObject()
	for rows.Next() {
		var k string
		var v any
		if err := rows.Scan(&k, &v); err != nil {
			rows.Close()
			return nil, err
		}
		settings.Set(k, v)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, err
	}

	if pyTruthy(mustGet(settings, "aggressive_enabled")) {
		for _, key := range []string{"history_pruning_enabled", "log_truncation_enabled"} {
			if !pyTruthy(mustGet(settings, key)) {
				if _, err := db.Exec(
					"INSERT INTO settings (key, value) VALUES (?, 1) ON CONFLICT(key) DO UPDATE SET value = 1",
					key,
				); err != nil {
					return nil, err
				}
				settings.Set(key, int64(1))
			}
		}
	}
	return settings, nil
}

func mustGet(o *pyjson.Object, k string) any {
	v, _ := o.Get(k)
	return v
}

// SetSetting upserts one setting.
func SetSetting(key string, value int64) error {
	db, err := dbHandle()
	if err != nil {
		return err
	}
	_, err = db.Exec(
		"INSERT INTO settings (key, value) VALUES (?, ?) "+
			"ON CONFLICT(key) DO UPDATE SET value = excluded.value",
		key, value,
	)
	return err
}

// ---- request logs and response cache -----------------------------------------

// LogRequest records one proxied request. Dashboard noise is skipped, and in
// that case it returns 0. Otherwise it returns the new row id. model takes the
// raw value the proxy read from the payload: nil is stored as NULL, as Python
// does, and a value SQLite cannot bind is an error.
func LogRequest(model any, provider string, originalInput, optimizedInput, estimatedOutputSaved, latencyMS int64,
	savingsCategory, originalSample, optimizedSample string, cacheHit bool) (int64, error) {
	saved := originalInput - optimizedInput
	if isDashboardNoise(savingsCategory, saved, originalSample) {
		return 0, nil
	}
	bound, err := bindValue(model)
	if err != nil {
		return 0, err
	}
	db, err := dbHandle()
	if err != nil {
		return 0, err
	}
	cache := int64(0)
	if cacheHit {
		cache = 1
	}
	res, err := db.Exec(
		"INSERT INTO request_logs ("+
			"model, provider, original_input_tokens, optimized_input_tokens, input_tokens_saved, "+
			"estimated_output_tokens_saved, latency_ms, savings_category, "+
			"original_payload_sample, optimized_payload_sample, cache_hit"+
			") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
		bound, provider, originalInput, optimizedInput, originalInput-optimizedInput,
		estimatedOutputSaved, latencyMS, savingsCategory, originalSample, optimizedSample, cache,
	)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

// bindValue converts a pyjson value to something database/sql accepts. Like
// Python's sqlite3, it refuses types it cannot bind.
func bindValue(v any) (any, error) {
	switch t := v.(type) {
	case nil, string, int64, float64:
		return t, nil
	case bool:
		if t {
			return int64(1), nil
		}
		return int64(0), nil
	case *big.Int:
		if t.IsInt64() {
			return t.Int64(), nil
		}
		return nil, errors.New("int too big to convert to SQLite INTEGER")
	}
	return nil, fmt.Errorf("type %T is not supported by SQLite binding", v)
}

// UpdateUpstreamUsage stores the provider's own usage figures against a logged
// request. Anthropic names (input_tokens) and OpenAI names (prompt_tokens) are
// both accepted. A zero id or empty usage is a no-op.
func UpdateUpstreamUsage(logID int64, usage *pyjson.Object) error {
	if usage == nil || usage.Len() == 0 || logID == 0 {
		return nil
	}
	db, err := dbHandle()
	if err != nil {
		return err
	}
	pick := func(primary, fallback string) any {
		v, _ := usage.Get(primary)
		if v == nil {
			v, _ = usage.Get(fallback)
		}
		return v
	}
	get := func(k string) any {
		v, _ := usage.Get(k)
		return v
	}
	vals := []any{
		pick("input_tokens", "prompt_tokens"),
		pick("output_tokens", "completion_tokens"),
		get("cache_read_input_tokens"),
		get("cache_creation_input_tokens"),
	}
	for i, v := range vals {
		b, err := bindValue(v)
		if err != nil {
			return err
		}
		vals[i] = b
	}
	_, err = db.Exec(
		"UPDATE request_logs SET "+
			"upstream_input_tokens = ?, upstream_output_tokens = ?, "+
			"upstream_cache_read_tokens = ?, upstream_cache_creation_tokens = ? "+
			"WHERE id = ?",
		append(vals, logID)...,
	)
	return err
}

// CachedResponse is one response_cache row.
type CachedResponse struct {
	StatusCode int64
	Headers    string
	Body       []byte
}

// GetCachedResponse returns the cached response for hash, or nil on a miss.
func GetCachedResponse(hash string) (*CachedResponse, error) {
	db, err := dbHandle()
	if err != nil {
		return nil, err
	}
	var status sql.NullInt64
	var headers sql.NullString
	var body []byte
	err = db.QueryRow(
		"SELECT status_code, headers, body FROM response_cache WHERE request_hash = ?", hash,
	).Scan(&status, &headers, &body)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &CachedResponse{StatusCode: status.Int64, Headers: headers.String, Body: body}, nil
}

// StoreCachedResponse upserts a cached response. When the cache is full it
// first evicts the oldest entries (FIFO by created_at), keeping at most
// maxCachedResponses rows.
func StoreCachedResponse(hash string, status int64, headers string, body []byte) error {
	db, err := dbHandle()
	if err != nil {
		return err
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var count int64
	if err := tx.QueryRow("SELECT COUNT(*) AS n FROM response_cache").Scan(&count); err != nil {
		return err
	}
	if count >= maxCachedResponses {
		if _, err := tx.Exec(
			"DELETE FROM response_cache WHERE request_hash IN ("+
				"SELECT request_hash FROM response_cache ORDER BY created_at ASC LIMIT ?)",
			count-maxCachedResponses+1,
		); err != nil {
			return err
		}
	}
	if _, err := tx.Exec(
		"INSERT OR REPLACE INTO response_cache (request_hash, status_code, headers, body) VALUES (?, ?, ?, ?)",
		hash, status, headers, body,
	); err != nil {
		return err
	}
	return tx.Commit()
}

// ClearRequestData wipes request logs and the response cache. Settings and
// prices are kept. It returns {"cleared_logs": n, "cleared_cache": m}.
func ClearRequestData() (*pyjson.Object, error) {
	db, err := dbHandle()
	if err != nil {
		return nil, err
	}
	tx, err := db.Begin()
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	var logs, cache int64
	if err := tx.QueryRow("SELECT COUNT(*) AS n FROM request_logs").Scan(&logs); err != nil {
		return nil, err
	}
	if err := tx.QueryRow("SELECT COUNT(*) AS n FROM response_cache").Scan(&cache); err != nil {
		return nil, err
	}
	if _, err := tx.Exec("DELETE FROM request_logs"); err != nil {
		return nil, err
	}
	if _, err := tx.Exec("DELETE FROM response_cache"); err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	o := pyjson.NewObject()
	o.Set("cleared_logs", logs)
	o.Set("cleared_cache", cache)
	return o, nil
}

// ---- dashboard queries --------------------------------------------------------

// GetStats is database.get_stats with recent_limit fixed at 40, as main.py
// calls it. The result is a JSON-ready object in Python's key order.
func GetStats(includeNoise bool, rangeKey string) (*pyjson.Object, error) {
	rk := normalizeRange(rangeKey)
	visible, err := visibleRows(includeNoise)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()

	var since *time.Time
	var until time.Time
	var span *time.Duration
	if rk == "all" {
		until = now
		var earliest *time.Time
		for _, r := range visible {
			if ts, ok := parseLogTimestamp(strOr(r["timestamp"], "")); ok {
				if earliest == nil || ts.Before(*earliest) {
					t := ts
					earliest = &t
				}
			}
		}
		if earliest != nil {
			since = earliest
			d := until.Sub(*earliest)
			span = &d
		}
	} else {
		delta := rangePresets[rk]
		s := now.Add(-delta)
		since = &s
		until = now
		d := delta
		span = &d
	}

	current := filterRowsByWindow(visible, since, nil)
	prices, err := allModelPrices(false)
	if err != nil {
		return nil, err
	}
	agg := aggregateRows(current, prices)

	recent := make([]any, 0)
	for i, r := range current {
		if i >= dashboardRecent {
			break
		}
		recent = append(recent, rowObject(rowForAPI(r)))
	}

	granularity := granularityForRange(rk, span)
	timeseries := []any{}
	if since != nil && len(current) > 0 {
		timeseries = buildTimeseries(current, *since, until, granularity)
	}

	var compare any
	if since != nil && rk != "all" {
		prevSince := since.Add(-until.Sub(*since))
		prevRows := filterRowsByWindow(visible, &prevSince, since)
		prev := aggregateRows(prevRows, prices)
		c := pyjson.NewObject()
		c.Set("requests_pct", pctChange(float64(agg.totalRequests), float64(prev.totalRequests)))
		c.Set("tokens_saved_pct", pctChange(float64(agg.tokensSaved), float64(prev.tokensSaved)))
		c.Set("original_input_pct", pctChange(float64(agg.originalInput), float64(prev.originalInput)))
		c.Set("save_rate_pct", pctChange(agg.avgSavePct, prev.avgSavePct))
		c.Set("usd_saved_pct", pctChange(agg.estimatedUSD, prev.estimatedUSD))
		compare = c
	}

	updatedAt, err := GetMeta("prices_updated_at")
	if err != nil {
		return nil, err
	}
	source, err := GetMeta("prices_source")
	if err != nil {
		return nil, err
	}
	count, err := CountModelPrices()
	if err != nil {
		return nil, err
	}
	pricing := pyjson.NewObject()
	pricing.Set("updated_at", metaValue(updatedAt))
	pricing.Set("source", metaValue(source))
	pricing.Set("count", count)

	settings, err := GetSettings()
	if err != nil {
		return nil, err
	}

	out := agg.toObject()
	out.Set("range", rk)
	out.Set("granularity", granularity)
	out.Set("timeseries", timeseries)
	out.Set("compare_previous", compare)
	out.Set("pricing", pricing)
	out.Set("settings", settings)
	out.Set("recent_logs", recent)
	out.Set("include_noise", includeNoise)
	return out, nil
}

func metaValue(p *string) any {
	if p == nil {
		return nil
	}
	return *p
}

// GetLogs is database.get_logs. limit is clamped to 1..200 and offset to
// at least 0.
func GetLogs(limit, offset int64, includeNoise bool, rangeKey string) (*pyjson.Object, error) {
	if limit < 1 {
		limit = 1
	}
	if limit > 200 {
		limit = 200
	}
	if offset < 0 {
		offset = 0
	}
	rk := normalizeRange(rangeKey)
	visible, err := visibleRows(includeNoise)
	if err != nil {
		return nil, err
	}
	var since *time.Time
	if rk != "all" {
		s := time.Now().UTC().Add(-rangePresets[rk])
		since = &s
	}
	filtered := filterRowsByWindow(visible, since, nil)
	total := int64(len(filtered))
	start := offset
	if start > total {
		start = total
	}
	end := start + limit
	if end > total {
		end = total
	}
	logs := make([]any, 0, end-start)
	for _, r := range filtered[start:end] {
		logs = append(logs, rowObject(rowForAPI(r)))
	}
	o := pyjson.NewObject()
	o.Set("logs", logs)
	o.Set("total", total)
	o.Set("limit", limit)
	o.Set("offset", offset)
	o.Set("has_more", offset+int64(len(logs)) < total)
	o.Set("include_noise", includeNoise)
	o.Set("range", rk)
	return o, nil
}
