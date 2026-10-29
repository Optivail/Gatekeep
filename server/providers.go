package main

// Port of backend/core/providers.py: which upstream a request goes to, the
// base URL of each provider, and the payload-shape checks the proxy uses to
// decide how to treat a JSON body.
//
// Base URLs are read from the environment at each lookup. Python reads them
// once at import, after .env is loaded; the values match unless the process
// environment changes while it runs.

import (
	"fmt"
	"math/big"
	"strconv"
	"strings"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// headerMap holds request headers as Python's _header_map does: lower-cased
// names, and the last value wins when a name repeats.
type headerMap map[string]string

func (h headerMap) has(name string) bool {
	_, ok := h[name]
	return ok
}

var mistralModelPrefixes = []string{
	"mistral", "codestral", "devstral", "ministral", "pixtral",
	"magistral", "open-mistral", "open-mixtral",
}

// pathProviders maps path prefixes to a provider. The first match wins, and
// only after the header hints in resolveProvider have been checked.
var pathProviders = []struct{ prefix, provider string }{
	{"aiserver.", "cursor"},
	{"agent.v1.", "cursor"},
	{"api/v1/freebuff", "freebuff"},
	{"backend-api/", "chatgpt"},
	{"api/codex", "chatgpt"},
	{"codex/", "chatgpt"},
	{"codex-backend/", "chatgpt"},
	{"ps/", "chatgpt"},
	{"plugins/", "chatgpt"},
	{"v3beta", "devin"},
	{"v3/", "devin"},
	{"v1/messages", "anthropic"},
	{"v1/complete", "anthropic"},
	{"v1/responses", "openai"},
	{"v1/chat/completions", "openai"},
	{"v1/completions", "openai"},
	{"v1/embeddings", "openai"},
	{"v1/models", "openai"},
	{"v1beta/", "google"},
	{"v1/projects/", "google"},
}

var overrideProviders = map[string]bool{
	"anthropic": true, "openai": true, "chatgpt": true, "google": true,
	"mistral": true, "cursor": true, "freebuff": true, "devin": true,
}

// isMistralModel mirrors _is_mistral_model. An empty model is not Mistral.
func isMistralModel(model string) bool {
	if model == "" {
		return false
	}
	m := strings.ToLower(strings.TrimSpace(model))
	for _, p := range mistralModelPrefixes {
		if strings.HasPrefix(m, p) || strings.Contains(m, "/"+p) {
			return true
		}
	}
	return false
}

// resolveProvider mirrors resolve_provider. model is "" when Python passes
// None. The order of the checks matters: the first match wins.
func resolveProvider(h headerMap, path, model string) string {
	pathL := strings.ToLower(strings.TrimLeft(path, "/"))
	ctype := h["content-type"]

	// An explicit override from the client (for example a Vibe custom provider).
	if o := strings.ToLower(strings.TrimSpace(h["x-gatekeep-provider"])); overrideProviders[o] {
		return o
	}

	// Cursor agent CLI (ConnectRPC). Routed here so the caller can refuse it.
	if strings.HasPrefix(pathL, "aiserver.") || strings.HasPrefix(pathL, "agent.v1.") {
		return "cursor"
	}
	if strings.Contains(ctype, "connect+") || strings.HasPrefix(ctype, "application/connect") {
		return "cursor"
	}
	if strings.Contains(ctype, "application/grpc") {
		return "cursor"
	}

	if strings.HasPrefix(pathL, "api/v1/freebuff") {
		return "freebuff"
	}

	// Codex ChatGPT subscription backend.
	for _, p := range []string{"backend-api/", "api/codex", "codex/", "codex-backend/", "ps/", "oauth/codex", "plugins/"} {
		if strings.HasPrefix(pathL, p) {
			return "chatgpt"
		}
	}

	if strings.HasPrefix(pathL, "v3/") || strings.HasPrefix(pathL, "v3beta") {
		return "devin"
	}
	if h["x-devin-org"] != "" || h["x-cognition-org"] != "" {
		return "devin"
	}

	// Google: the key header wins over the path.
	if h.has("x-goog-api-key") || h.has("x-goog-user-project") {
		return "google"
	}
	if strings.HasPrefix(pathL, "v1beta/") || strings.Contains(pathL, "generativelanguage") {
		return "google"
	}

	// Anthropic signals come before Authorization, because Claude Code can send both.
	if h.has("anthropic-version") || h.has("anthropic-beta") {
		return "anthropic"
	}
	if strings.HasPrefix(pathL, "v1/messages") || strings.HasPrefix(pathL, "v1/complete") {
		return "anthropic"
	}

	if isMistralModel(model) {
		return "mistral"
	}

	if strings.HasPrefix(pathL, "v1/responses") || strings.HasPrefix(pathL, "v1/chat/") {
		return "openai"
	}
	if strings.HasPrefix(pathL, "v1/embeddings") || pathL == "v1/models" || strings.HasPrefix(pathL, "v1/models/") {
		// Python returns openai for a Bearer key and for any Authorization header,
		// so the presence of the header is the whole test.
		if h.has("authorization") {
			return "openai"
		}
	}

	if h.has("x-api-key") {
		if strings.HasPrefix(h["x-api-key"], "sk-ant") {
			return "anthropic"
		}
		if strings.HasPrefix(pathL, "v1beta") {
			return "google"
		}
		return "anthropic"
	}

	if h.has("authorization") {
		return "openai"
	}

	for _, pp := range pathProviders {
		if strings.HasPrefix(pathL, pp.prefix) {
			return pp.provider
		}
	}
	return "anthropic"
}

// providerBase returns the environment override for a base URL, or the default.
// Like Python's os.getenv(name, default), a set but empty variable wins.
func providerBase(envName, def string) string {
	return strings.TrimRight(envOr(envName, def), "/")
}

// baseURLFor mirrors base_url_for for a provider without a path. An agent.v1
// path of the cursor provider has its own rule: see baseURLForPath.
func baseURLFor(provider string) string {
	switch provider {
	case "cursor":
		return providerBase("CURSOR_BASE_URL", "https://api2.cursor.sh")
	case "openai":
		return providerBase("OPENAI_BASE_URL", "https://api.openai.com")
	case "chatgpt":
		return providerBase("CHATGPT_BASE_URL", "https://chatgpt.com")
	case "google":
		return providerBase("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com")
	case "mistral":
		return providerBase("MISTRAL_BASE_URL", "https://api.mistral.ai")
	case "freebuff":
		return providerBase("FREEBUFF_BASE_URL", "https://www.codebuff.com")
	case "devin":
		return providerBase("DEVIN_BASE_URL", "https://api.devin.ai")
	}
	return providerBase("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
}

// shouldOptimizePayload mirrors should_optimize_payload: only conversation-shaped
// JSON goes through the crush. Analytics, plugins, token counting and the
// proprietary paths are passed through.
func shouldOptimizePayload(p *pyjson.Object, path string) bool {
	if p == nil {
		return false
	}
	pathL := strings.ToLower(strings.TrimLeft(path, "/"))
	if strings.Contains(pathL, "count_tokens") {
		return false
	}
	if strings.HasPrefix(pathL, "aiserver.") || strings.HasPrefix(pathL, "api/v1/freebuff") {
		return false
	}
	if strings.HasPrefix(pathL, "v3/") || strings.HasPrefix(pathL, "v3beta") {
		return false
	}
	if strings.HasPrefix(pathL, "codex/analytics") || strings.Contains(pathL, "/analytics") {
		return false
	}
	if strings.HasPrefix(pathL, "ps/plugins") || strings.HasSuffix(pathL, "/mcp") {
		return false
	}
	for _, k := range []string{"messages", "prompt", "input", "instructions"} {
		if _, ok := p.Get(k); ok {
			return true
		}
	}
	return false
}

// pyStrOf mirrors str(v or ""): the text of a value, or "" when it is falsy.
func pyStrOf(v any) string {
	if !pyTruthy(v) {
		return ""
	}
	switch t := v.(type) {
	case string:
		return t
	case bool:
		if t {
			return "True"
		}
		return "False"
	case int64:
		return strconv.FormatInt(t, 10)
	case *big.Int:
		return t.String()
	case float64:
		return pyjson.FloatRepr(t)
	}
	// Containers: Python's repr differs from JSON, but the proxy only tests
	// the text for substrings such as "anthropic", which survive either way.
	b, err := pyjson.Dumps(v, pyjson.Default)
	if err != nil {
		return ""
	}
	return string(b)
}

// pyIterable returns the items a `for x in v` loop would see, for the
// `payload.get(...) or []` idiom. Non-iterable truthy values raise TypeError in
// Python, which the caller turns into a 500.
func pyIterable(v any) ([]any, error) {
	if !pyTruthy(v) {
		return nil, nil
	}
	switch t := v.(type) {
	case []any:
		return t, nil
	case string, *pyjson.Object:
		// Characters and dict keys are strings, never dicts, so the callers'
		// checks match nothing in them.
		return nil, nil
	}
	return nil, fmt.Errorf("TypeError: '%T' object is not iterable", v)
}

func objValue(o *pyjson.Object, key string) any {
	v, _ := o.Get(key)
	return v
}

// isAnthropicMessagesPayload mirrors is_anthropic_messages_payload. It is true
// when the model name mentions anthropic, when a message carries an Anthropic
// block type, or when the payload has a string or list system prompt.
func isAnthropicMessagesPayload(p *pyjson.Object) (bool, error) {
	if p == nil {
		return false, nil
	}
	if strings.Contains(strings.ToLower(pyStrOf(objValue(p, "model"))), "anthropic") {
		return true, nil
	}
	msgs, err := pyIterable(objValue(p, "messages"))
	if err != nil {
		return false, err
	}
	for _, m := range msgs {
		mo, ok := m.(*pyjson.Object)
		if !ok {
			continue
		}
		blocks, ok := objValue(mo, "content").([]any)
		if !ok {
			continue
		}
		for _, b := range blocks {
			bo, ok := b.(*pyjson.Object)
			if !ok {
				continue
			}
			if s, ok := objValue(bo, "type").(string); ok {
				switch s {
				case "tool_use", "tool_result", "thinking":
					return true, nil
				}
			}
		}
	}
	if _, ok := p.Get("system"); ok {
		switch objValue(p, "system").(type) {
		case string, []any:
			return true, nil
		}
	}
	return false, nil
}

// isOpenAIChatPayload mirrors is_openai_chat_payload: OpenAI or Mistral
// chat.completions JSON, which is not Anthropic Messages.
func isOpenAIChatPayload(p *pyjson.Object) (bool, error) {
	if p == nil {
		return false, nil
	}
	anth, err := isAnthropicMessagesPayload(p)
	if err != nil || anth {
		return false, err
	}
	msgs, ok := objValue(p, "messages").([]any)
	if !ok || len(msgs) == 0 {
		return false, nil
	}
	if tools, ok := objValue(p, "tools").([]any); ok {
		for _, t := range tools {
			to, ok := t.(*pyjson.Object)
			if !ok {
				continue
			}
			if s, _ := objValue(to, "type").(string); s == "function" {
				return true, nil
			}
			if _, ok := objValue(to, "function").(*pyjson.Object); ok {
				return true, nil
			}
		}
	}
	for _, m := range msgs {
		mo, ok := m.(*pyjson.Object)
		if !ok {
			continue
		}
		if s, _ := objValue(mo, "role").(string); s == "system" || s == "user" || s == "assistant" || s == "tool" {
			return true, nil
		}
	}
	return false, nil
}

// isOpenAIResponsesPayload mirrors is_openai_responses_payload: the Responses
// API shape, with an input that is a list or a string.
func isOpenAIResponsesPayload(p *pyjson.Object) (bool, error) {
	if p == nil {
		return false, nil
	}
	anth, err := isAnthropicMessagesPayload(p)
	if err != nil || anth {
		return false, err
	}
	chat, err := isOpenAIChatPayload(p)
	if err != nil || chat {
		return false, err
	}
	switch objValue(p, "input").(type) {
	case []any, string:
		return true, nil
	}
	return false, nil
}
