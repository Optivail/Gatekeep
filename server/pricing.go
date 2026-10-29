package main

// Model pricing, a port of backend/engines/pricing.py. It keeps a local
// model_prices table seeded with common rates and refreshed from the public
// LiteLLM price table. Costs are estimated in USD per million tokens.

import (
	"errors"
	"fmt"
	"io"
	"math/big"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// litellmPricesURL and pricesHTTPTimeout are variables so tests can point the
// refresh at a local server.
var (
	litellmPricesURL  = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
	pricesHTTPTimeout = 30 * time.Second
)

const (
	staleAfterHours  = 24
	defaultInputUSD  = 3.0
	defaultOutputUSD = 15.0
	pricesTimeLayout = "2006-01-02T15:04:05Z"
)

// keepLiteLLMProviders are provider names kept from the LiteLLM table. Others
// are kept only when their name starts with one of litellmNamePrefixes.
var keepLiteLLMProviders = []string{
	"anthropic", "openai", "azure", "openrouter", "google", "deepseek", "meta",
	"mistral", "mistralai", "qwen", "xai",
}

var litellmNamePrefixes = []string{
	"claude", "gpt", "gemini", "deepseek", "llama", "mistral", "qwen", "grok",
}

// popularModels is the curated list shown on the dashboard, in display order.
// Each entry maps a display name to LiteLLM lookup keys tried in order.
var popularModels = []struct {
	display  string
	patterns []string
}{
	{"openai/gpt-4o", []string{"gpt-4o", "openai/gpt-4o", "openai/gpt-4o-2024-05-13"}},
	{"openai/gpt-4o-mini", []string{"gpt-4o-mini", "openai/gpt-4o-mini"}},
	{"openai/gpt-4.1", []string{"gpt-4.1", "openai/gpt-4.1"}},
	{"openai/gpt-4.1-mini", []string{"gpt-4.1-mini", "openai/gpt-4.1-mini"}},
	{"openai/o3-mini", []string{"o3-mini", "openai/o3-mini"}},
	{"anthropic/claude-sonnet-4", []string{"claude-sonnet-4-20250514", "claude-sonnet-4-5", "claude-sonnet-4"}},
	{"anthropic/claude-3.5-sonnet", []string{"claude-3-5-sonnet-20241022", "claude-3-5-sonnet-latest"}},
	{"anthropic/claude-haiku-4.5", []string{"claude-haiku-4-5-20251001", "claude-haiku-4-5"}},
	{"anthropic/claude-3.5-haiku", []string{"claude-3-5-haiku-20241022", "claude-3-5-haiku-latest"}},
	{"anthropic/claude-opus-4", []string{"claude-opus-4-20250514", "claude-opus-4-1", "claude-opus-4"}},
	{"google/gemini-2.0-flash", []string{"gemini/gemini-2.0-flash", "gemini-2.0-flash", "gemini-2.0-flash"}},
	{"google/gemini-1.5-pro", []string{"gemini/gemini-1.5-pro", "gemini-1.5-pro"}},
	{"deepseek/deepseek-chat", []string{"deepseek/deepseek-chat", "deepseek-chat"}},
	{"deepseek/deepseek-r1", []string{"deepseek/deepseek-r1", "deepseek-reasoner"}},
	{"meta-llama/llama-3.1-70b-instruct", []string{"meta-llama/llama-3.1-70b-instruct", "llama-3.1-70b-instruct"}},
	{"mistral/mistral-large", []string{"mistral/mistral-large-latest", "mistral-large-latest"}},
	{"qwen/qwen-2.5-coder-32b", []string{"qwen/qwen-2.5-coder-32b-instruct"}},
	{"x-ai/grok-2", []string{"xai/grok-2", "grok-2"}},
}

// seedPrices is in the same order as the Python dict, so that the rows get the
// same insertion order (and therefore the same lookup order).
var seedPrices = []struct {
	model   string
	in, out float64
}{
	{"gpt-4o", 2.5, 10.0},
	{"gpt-4o-mini", 0.15, 0.6},
	{"gpt-4.1", 2.0, 8.0},
	{"gpt-4.1-mini", 0.4, 1.6},
	{"o3-mini", 1.1, 4.4},
	{"claude-haiku-4-5-20251001", 1.0, 5.0},
	{"claude-sonnet-4-20250514", 3.0, 15.0},
	{"claude-3-5-sonnet-20241022", 3.0, 15.0},
	{"claude-3-5-haiku-20241022", 0.8, 4.0},
	{"claude-opus-4-20250514", 15.0, 75.0},
	{"gemini/gemini-2.0-flash", 0.1, 0.4},
	{"gemini/gemini-1.5-pro", 1.25, 5.0},
	{"deepseek/deepseek-chat", 0.14, 0.28},
	{"deepseek/deepseek-r1", 0.55, 2.19},
	{"meta-llama/llama-3.1-70b-instruct", 0.52, 0.75},
	{"mistral/mistral-large-latest", 2.0, 6.0},
	{"qwen/qwen-2.5-coder-32b-instruct", 0.2, 0.8},
}

func seedRate(model string) (float64, float64, bool) {
	for _, s := range seedPrices {
		if s.model == model {
			return s.in, s.out, true
		}
	}
	return 0, 0, false
}

// pricesNow is the UTC timestamp format used in model_prices.updated_at.
func pricesNow() string {
	return time.Now().UTC().Format(pricesTimeLayout)
}

// pricingError carries the Python exception class name. maybeAutoRefresh
// stores "Type: message" in prices_last_error, so the name must match.
type pricingError struct {
	typ, msg string
}

func (e *pricingError) Error() string { return e.msg }

func errTypeName(err error) string {
	var pe *pricingError
	if errors.As(err, &pe) {
		return pe.typ
	}
	return "Exception"
}

// ensureSeedPrices upserts the seed rates. As in Python, it runs on every
// dashboard price read, so seed rows are reset to seed values each time.
func ensureSeedPrices() error {
	ts := pricesNow()
	for _, s := range seedPrices {
		if err := UpsertModelPrice(s.model, s.in, s.out, ts, "seed"); err != nil {
			return err
		}
	}
	return nil
}

// normalizeKeys returns the lookup keys for model, in the order Python tries
// them: the name, the anthropic/ form, and the same two without a dated suffix.
func normalizeKeys(model string) []string {
	if model == "" {
		return nil
	}
	m := strings.TrimSpace(model)
	keys := []string{m}
	if strings.HasPrefix(m, "anthropic/") {
		keys = append(keys, strings.TrimPrefix(m, "anthropic/"))
	} else {
		keys = append(keys, "anthropic/"+m)
	}
	if i := strings.LastIndex(m, "-"); i >= 0 {
		base, suffix := m[:i], m[i+1:]
		if isASCIIDigits(suffix) && len(suffix) >= 8 {
			keys = append(keys, base, "anthropic/"+base)
		}
	}
	return keys
}

func isASCIIDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

// lookupRatesFrom resolves rates against an already loaded price table: an
// exact key first, then a case-insensitive substring match in table order,
// then the default rates.
func lookupRatesFrom(prices []priceRow, model string) (float64, float64) {
	for _, key := range normalizeKeys(model) {
		for _, p := range prices {
			if p.Model == key {
				return p.InputUSD, p.OutputUSD
			}
		}
	}
	lower := strings.ToLower(model)
	for _, p := range prices {
		name := strings.ToLower(p.Model)
		if strings.Contains(lower, name) || strings.Contains(name, lower) {
			return p.InputUSD, p.OutputUSD
		}
	}
	return defaultInputUSD, defaultOutputUSD
}

// estimateUSDSavedFrom is the cost of the saved tokens for one row, given a
// loaded price table.
func estimateUSDSavedFrom(prices []priceRow, inputSaved, outputSaved int64, model string) float64 {
	in, out := lookupRatesFrom(prices, model)
	return float64(inputSaved)/1_000_000*in + float64(outputSaved)/1_000_000*out
}

// matchPriceRow finds the first row that matches any pattern. Exact keys are
// tried first. Then rows are scanned in table order, and a row matches when
// its lowercased name equals the pattern, ends with "/"+pattern, or contains
// the pattern.
func matchPriceRow(rows []priceRow, patterns []string) *priceRow {
	byName := make(map[string]int, len(rows))
	for i, r := range rows {
		byName[r.Model] = i
	}
	for _, pat := range patterns {
		if i, ok := byName[pat]; ok {
			return &rows[i]
		}
	}
	lowered := make([]string, len(patterns))
	for i, p := range patterns {
		lowered[i] = strings.ToLower(p)
	}
	for i, r := range rows {
		nl := strings.ToLower(r.Model)
		for _, pat := range lowered {
			if nl == pat || strings.HasSuffix(nl, "/"+pat) || strings.Contains(nl, pat) {
				return &rows[i]
			}
		}
	}
	return nil
}

// popularPrice is one row of the dashboard list.
type popularPrice struct {
	row     priceRow
	display string
}

// listPopularModelPrices returns the curated models. Rows that are missing from
// the table fall back to the seed rate for the first matching seed key.
func listPopularModelPrices(limit int) ([]popularPrice, error) {
	if err := ensureSeedPrices(); err != nil {
		return nil, err
	}
	rows, err := allModelPrices(true)
	if err != nil {
		return nil, err
	}
	updatedFallback := pricesNow()
	if v, err := GetMeta("prices_updated_at"); err != nil {
		return nil, err
	} else if v != nil && *v != "" {
		updatedFallback = *v
	}

	var out []popularPrice
	seen := map[string]bool{}
	for _, pm := range popularModels {
		if len(out) >= limit {
			break
		}
		row := matchPriceRow(rows, pm.patterns)
		var fallback *priceRow
		if row == nil {
			for _, pat := range pm.patterns {
				if in, o, ok := seedRate(pat); ok {
					// The Python code reuses the name `out` here, which breaks
					// the list. The fix is to use a separate variable.
					fallback = &priceRow{Model: pat, InputUSD: in, OutputUSD: o, UpdatedAt: updatedFallback, Source: "seed", hasDetails: true}
					break
				}
			}
			row = fallback
		}
		if row == nil || seen[row.Model] {
			continue
		}
		seen[row.Model] = true
		out = append(out, popularPrice{row: *row, display: pm.display})
	}
	return out, nil
}

// popularToObject renders one popular entry with the Python key order.
func popularToObject(p popularPrice) *pyjson.Object {
	o := pyjson.NewObject()
	o.Set("model", p.row.Model)
	o.Set("input_per_mtok", p.row.InputUSD)
	o.Set("output_per_mtok", p.row.OutputUSD)
	o.Set("updated_at", p.row.UpdatedAt)
	o.Set("source", p.row.Source)
	o.Set("display", p.display)
	return o
}

// fetchLiteLLM downloads the price JSON. A non-2xx status is an error,
// matching urllib's HTTPError.
func fetchLiteLLM() ([]byte, error) {
	req, err := http.NewRequest(http.MethodGet, litellmPricesURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "ai-gateway-proxy/1.0")
	client := &http.Client{Timeout: pricesHTTPTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, &pricingError{typ: "URLError", msg: fmt.Sprintf("<urlopen error %v>", err)}
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return nil, &pricingError{typ: "HTTPError", msg: fmt.Sprintf("HTTP Error %d: %s", resp.StatusCode, http.StatusText(resp.StatusCode))}
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, &pricingError{typ: "URLError", msg: fmt.Sprintf("<urlopen error %v>", err)}
	}
	return raw, nil
}

// applyLiteLLMPayload parses the LiteLLM JSON and stores every usable entry.
// It returns the number stored. The parsing is pure apart from the upserts,
// so it is covered by unit tests with a fixed payload.
func applyLiteLLMPayload(raw []byte, ts string) (int, error) {
	parsed, err := pyjson.Loads(raw)
	if err != nil {
		return 0, &pricingError{typ: "JSONDecodeError", msg: err.Error()}
	}
	data, ok := parsed.(*pyjson.Object)
	if !ok {
		return 0, &pricingError{typ: "ValueError", msg: "unexpected prices payload"}
	}

	upserted := 0
	for _, name := range data.Keys() {
		infoAny, _ := data.Get(name)
		info, ok := infoAny.(*pyjson.Object)
		if strings.HasPrefix(name, "sample_") || !ok {
			continue
		}
		provider, err := pyLower(info, "litellm_provider")
		if err != nil {
			return upserted, err
		}
		if !pricingContains(keepLiteLLMProviders, provider) {
			n := strings.ToLower(name)
			if !pricingHasPrefix(n, litellmNamePrefixes) {
				continue
			}
		}
		inTok, ok1 := pyFloat(info, "input_cost_per_token")
		outTok, ok2 := pyFloat(info, "output_cost_per_token")
		if !ok1 || !ok2 {
			continue
		}
		if inTok <= 0 && outTok <= 0 {
			continue
		}
		if err := UpsertModelPrice(name, inTok*1_000_000, outTok*1_000_000, ts, "litellm"); err != nil {
			return upserted, err
		}
		upserted++
	}
	return upserted, nil
}

// pyLower reads info[key] as Python's `(info.get(key) or "").lower()`.
// A non-string truthy value raises AttributeError in Python, and it does here.
func pyLower(info *pyjson.Object, key string) (string, error) {
	v, _ := info.Get(key)
	if !pyTruthy(v) {
		return "", nil
	}
	s, ok := v.(string)
	if !ok {
		return "", &pricingError{typ: "AttributeError", msg: fmt.Sprintf("'%T' object has no attribute 'lower'", v)}
	}
	return strings.ToLower(s), nil
}

// pyFloat reads info[key] as Python's float(info.get(key) or 0). ok is false
// where Python would raise TypeError or ValueError and skip the entry.
func pyFloat(info *pyjson.Object, key string) (float64, bool) {
	v, _ := info.Get(key)
	if !pyTruthy(v) {
		return 0, true
	}
	switch t := v.(type) {
	case bool:
		return 1, true
	case int64:
		return float64(t), true
	case float64:
		return t, true
	case *big.Int:
		f, _ := new(big.Float).SetInt(t).Float64()
		return f, true
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(t), 64)
		if err != nil {
			return 0, false
		}
		return f, true
	}
	return 0, false
}

func pricingContains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

func pricingHasPrefix(s string, prefixes []string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

// refreshPricesFromLiteLLM downloads and stores the LiteLLM table. It returns
// {"updated_at", "models", "source"}. The metadata update happens only after
// a successful parse, and prices_last_error is left alone, as in Python.
func refreshPricesFromLiteLLM() (*pyjson.Object, error) {
	raw, err := fetchLiteLLM()
	if err != nil {
		return nil, err
	}
	ts := pricesNow()
	upserted, err := applyLiteLLMPayload(raw, ts)
	if err != nil {
		return nil, err
	}
	if err := SetMeta("prices_updated_at", ts); err != nil {
		return nil, err
	}
	if err := SetMeta("prices_source", "litellm"); err != nil {
		return nil, err
	}
	if err := SetMeta("prices_count", strconv.Itoa(upserted)); err != nil {
		return nil, err
	}
	o := pyjson.NewObject()
	o.Set("updated_at", ts)
	o.Set("models", int64(upserted))
	o.Set("source", "litellm")
	return o, nil
}

// maybeAutoRefresh refreshes when the table is missing or older than
// staleAfterHours. It returns nil when no refresh was needed. On failure it
// records the error in meta and returns an object with an "error" key. It
// never returns an error itself, so startup can continue on the seed rates.
func maybeAutoRefresh() *pyjson.Object {
	if err := ensureSeedPrices(); err != nil {
		return errorResult(err)
	}
	updated, err := GetMeta("prices_updated_at")
	if err != nil {
		return errorResult(err)
	}
	if updated != nil && *updated != "" {
		if then, err := time.Parse(pricesTimeLayout, *updated); err == nil {
			age := time.Since(then).Hours()
			if age < staleAfterHours {
				return nil
			}
		}
	}
	res, err := refreshPricesFromLiteLLM()
	if err == nil {
		return res
	}
	_ = SetMeta("prices_last_error", fmt.Sprintf("%s: %s", errTypeName(err), err.Error()))
	return errorResult(err)
}

func errorResult(err error) *pyjson.Object {
	o := pyjson.NewObject()
	o.Set("error", err.Error())
	return o
}

// pricingStatus is the GET /api/prices payload.
func pricingStatus() (*pyjson.Object, error) {
	popular, err := listPopularModelPrices(24)
	if err != nil {
		return nil, err
	}
	updated, err := GetMeta("prices_updated_at")
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
	lastErr, err := GetMeta("prices_last_error")
	if err != nil {
		return nil, err
	}

	// Python: `source or ("seed" if prices else None)`. An empty string is
	// falsy and falls through.
	var srcVal any
	if source != nil && *source != "" {
		srcVal = *source
	} else if len(popular) > 0 {
		srcVal = "seed"
	}

	models := make([]any, 0, len(popular))
	for _, p := range popular {
		models = append(models, popularToObject(p))
	}

	o := pyjson.NewObject()
	o.Set("updated_at", metaValue(updated))
	o.Set("source", srcVal)
	o.Set("count", count)
	o.Set("last_error", metaValue(lastErr))
	o.Set("stale_after_hours", int64(staleAfterHours))
	o.Set("models", models)
	return o, nil
}
