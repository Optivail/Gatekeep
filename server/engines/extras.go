// Mirrors backend/engines/extras.py: extra savings techniques that need no
// model routing (cache fixes, extractive digest, deferred tool loading, media
// stripping, and the intent-cache key).
package engines

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"

	"github.com/Optivail/Gatekeep/server/pyjson"
	"github.com/Optivail/Gatekeep/server/pyre"
)

const (
	// deferKeepFull is how many tools keep their full schema; the rest become
	// defer_loading stubs.
	deferKeepFull          = 5
	digestMinChars         = 400
	preserveRecentMessages = 4
)

var (
	volatileSystem = []*pyre.Regexp{
		pyre.MustCompile(`^.*\bcc_version\b.*$`, pyre.IgnoreCase|pyre.Multiline),
		pyre.MustCompile(`^.*\bclaude.?code.?version\b.*$`, pyre.IgnoreCase|pyre.Multiline),
		pyre.MustCompile(`\bcurrent (date|time)\s*[:=]\s*[^\n]+`, pyre.IgnoreCase),
		pyre.MustCompile(`\btoday'?s date\s*[:=]\s*[^\n]+`, pyre.IgnoreCase),
		pyre.MustCompile(`\b20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b`),
	}
	reTripleNewline = pyre.MustCompile(`\n{3,}`)
	reDigestPath    = pyre.MustCompile(`[\\/][\w.-]+\.\w+`)
	reNumRun        = pyre.MustCompile(`\d+`)
	reTimestamp     = pyre.MustCompile(`\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?`)
	reWhitespaceRun = pyre.MustCompile(`\s+`)
)

// scrubVolatile mirrors the scrub() closure in strip_volatile_system. Falsy
// values come back unchanged; a truthy non-str raises TypeError in re.sub.
func scrubVolatile(v any) (any, error) {
	if !pyTruthy(v) {
		return v, nil
	}
	s, ok := v.(string)
	if !ok {
		return nil, pyError("TypeError", "expected string or bytes-like object")
	}
	text := s
	for _, pat := range volatileSystem {
		text = pat.Sub("", text)
	}
	text = pyStrip(reTripleNewline.Sub("\n\n", text))
	if text == "" {
		// Never emit empty text blocks: Anthropic rejects them.
		return s, nil
	}
	return text, nil
}

// StripVolatileSystem mirrors strip_volatile_system: it removes fingerprint
// and timestamp lines that would otherwise bust the Anthropic prefix cache.
func StripVolatileSystem(system any) (any, error) {
	switch sys := system.(type) {
	case string:
		return scrubVolatile(sys)
	case []any:
		var out []any
		for _, block := range sys {
			bo, ok := block.(*pyjson.Object)
			if !ok || !pyEqual(objGet(bo, "type"), "text") {
				out = append(out, block)
				continue
			}
			// block.get("text", "") defaults to "" only when the key is absent.
			rawText := objGetOr(bo, "text", "")
			scrubbed, err := scrubVolatile(rawText)
			if err != nil {
				return nil, err
			}
			text, isStr := scrubbed.(string)
			if !isStr {
				// .strip() on a non-str raises AttributeError.
				return nil, pyError("AttributeError", "object has no attribute 'strip'")
			}
			if pyStrip(text) == "" {
				continue
			}
			nb := copyObject(bo)
			nb.Set("text", text)
			out = append(out, nb)
		}
		if len(out) == 0 {
			return system, nil
		}
		return out, nil
	}
	return system, nil
}

// objGetOr mirrors dict.get(key, default): the default applies only when the
// key is absent, not when its value is null.
func objGetOr(o *pyjson.Object, key string, def any) any {
	if v, ok := o.Get(key); ok {
		return v
	}
	return def
}

// nameKey mirrors the sort key in stabilize_tools: (name or "") for dicts and
// "" for everything else.
func nameKey(t any) any {
	to, ok := t.(*pyjson.Object)
	if !ok {
		return ""
	}
	name := objGet(to, "name")
	if !pyTruthy(name) {
		return ""
	}
	return name
}

// StabilizeTools mirrors stabilize_tools: deterministic order by name, with
// the CCR retrieve tool kept last. Incomparable names raise TypeError.
func StabilizeTools(tools any) (any, error) {
	list, ok := tools.([]any)
	if !ok || len(list) < 2 {
		return tools, nil
	}
	var retrieve, rest []any
	for _, t := range list {
		if to, ok := t.(*pyjson.Object); ok && pyEqual(objGet(to, "name"), RetrieveToolName) {
			retrieve = append(retrieve, t)
		} else {
			rest = append(rest, t)
		}
	}
	if err := pySortByKey(rest, nameKey); err != nil {
		return nil, err
	}
	out := make([]any, 0, len(rest)+len(retrieve))
	out = append(out, rest...)
	return append(out, retrieve...), nil
}

// ApplyDeferLoading mirrors apply_defer_loading: tools past the first few
// (and MCP tools) become defer_loading stubs. changed reports any edit.
func ApplyDeferLoading(tools any) (any, bool, error) {
	list, ok := tools.([]any)
	if !ok || len(list) == 0 {
		return tools, false, nil
	}
	changed := false
	var out []any
	fullKept := 0
	for _, tool := range list {
		to, ok := tool.(*pyjson.Object)
		if !ok {
			out = append(out, tool)
			continue
		}
		nameVal := objGet(to, "name")
		if !pyTruthy(nameVal) {
			nameVal = ""
		}
		t := copyObject(to)
		if pyEqual(nameVal, RetrieveToolName) {
			out = append(out, t)
			continue
		}
		name, isStr := nameVal.(string)
		if !isStr {
			// name.startswith() on a non-str raises AttributeError.
			return nil, false, pyError("AttributeError", "object has no attribute 'startswith'")
		}
		isMCP := strings.HasPrefix(name, "mcp__") || strings.Contains(pyPrefix(name, 6), "__")
		if fullKept < deferKeepFull && !isMCP {
			fullKept++
			out = append(out, t)
			continue
		}
		// Stub: name, a short description, an empty schema, and defer_loading.
		if !pyTruthy(objGet(t, "defer_loading")) {
			t.Set("defer_loading", true)
			changed = true
		}
		if desc, ok := objGet(t, "description").(string); ok && pyLen(desc) > 32 {
			t.Set("description", pyPrefix(desc, 29)+"…")
			changed = true
		}
		if schema, ok := objGet(t, "input_schema").(*pyjson.Object); ok &&
			(pyTruthy(objGet(schema, "properties")) || pyTruthy(objGet(schema, "required"))) {
			t.Set("input_schema", objectOf("type", "object", "properties", pyjson.NewObject()))
			changed = true
		}
		out = append(out, t)
	}
	return out, changed, nil
}

// digestText mirrors _digest_text. Falsy values become "" (and then fail the
// caller's != check); a truthy non-str raises AttributeError in .strip().
func digestText(v any) (string, error) {
	text, err := strOrEmpty(v)
	if err != nil {
		return "", err
	}
	text = pyStrip(text)
	if pyLen(text) < digestMinChars {
		return text, nil
	}
	// Keep signal lines (errors, paths, headings), else head and tail.
	var keep []string
	for _, line := range strings.Split(text, "\n") {
		if pyStrip(line) == "" {
			continue
		}
		low := pyLower(line)
		if containsAnyStr(low, "error", "traceback", "failed", "exception", "todo", "fix") {
			keep = append(keep, pyPrefix(line, 200))
		} else if reDigestPath.MatchString(line) {
			keep = append(keep, pyPrefix(line, 200))
		}
		if len(keep) >= 8 {
			break
		}
	}
	if len(keep) > 0 {
		return "[DIGEST " + itoa(pyLen(text)) + " chars]\n" + strings.Join(keep, "\n"), nil
	}
	return "[DIGEST " + itoa(pyLen(text)) + " chars]\n" +
		pyPrefix(text, 220) + "\n…\n" + pySuffix(text, 180), nil
}

func containsAnyStr(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// ApplyExtractiveDigest mirrors apply_extractive_digest: it shrinks old
// text-only turns in place. Messages with tool_use or tool_result blocks are
// skipped so tool pairing stays intact.
func ApplyExtractiveDigest(payload *pyjson.Object) (bool, error) {
	messages, ok := objGet(payload, "messages").([]any)
	if !ok || len(messages) <= preserveRecentMessages+1 {
		return false, nil
	}
	changed := false
	cutoff := len(messages) - preserveRecentMessages
	for _, m := range messages[1:cutoff] {
		msg, ok := m.(*pyjson.Object)
		if !ok {
			continue
		}
		content := objGet(msg, "content")
		if s, isStr := content.(string); isStr {
			dig, err := digestText(s)
			if err != nil {
				return changed, err
			}
			if dig != s {
				msg.Set("content", dig)
				changed = true
			}
			continue
		}
		list, isList := content.([]any)
		if !isList {
			continue
		}
		// Python builds a set of block types, which raises for unhashable types.
		hasToolBlock := false
		for _, b := range list {
			bo, ok := b.(*pyjson.Object)
			if !ok {
				continue
			}
			bt := objGet(bo, "type")
			if _, hashable := pyKey(bt); !hashable {
				return changed, pyError("TypeError", "unhashable type")
			}
			if pyEqual(bt, "tool_use") || pyEqual(bt, "tool_result") {
				hasToolBlock = true
			}
		}
		if hasToolBlock {
			continue
		}
		newBlocks := make([]any, 0, len(list))
		for _, b := range list {
			bo, ok := b.(*pyjson.Object)
			if !ok || !pyEqual(objGet(bo, "type"), "text") {
				newBlocks = append(newBlocks, b)
				continue
			}
			nb := copyObject(bo)
			dig, err := digestText(objGetOr(nb, "text", ""))
			if err != nil {
				return changed, err
			}
			// dig != block.get("text"): a missing key reads as None, so it differs.
			if cur, isStr := objGet(nb, "text").(string); !isStr || cur != dig {
				nb.Set("text", dig)
				changed = true
			}
			newBlocks = append(newBlocks, nb)
		}
		msg.Set("content", newBlocks)
	}
	return changed, nil
}

// StripMediaBlocks mirrors strip_media_blocks: image, document and file blocks
// in history become a short text note (they are large and rarely needed).
func StripMediaBlocks(payload *pyjson.Object) bool {
	messages, ok := objGet(payload, "messages").([]any)
	if !ok {
		return false
	}
	changed := false
	for _, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok {
			continue
		}
		content, ok := objGet(msg, "content").([]any)
		if !ok {
			continue
		}
		newBlocks := make([]any, 0, len(content))
		for _, b := range content {
			bo, ok := b.(*pyjson.Object)
			if ok {
				bt := objGet(bo, "type")
				if pyEqual(bt, "image") || pyEqual(bt, "document") || pyEqual(bt, "file") {
					newBlocks = append(newBlocks, objectOf("type", "text", "text", "["+pyStr(bt)+" omitted by proxy]"))
					changed = true
					continue
				}
			}
			newBlocks = append(newBlocks, b)
		}
		// Python assigns inside the loop once any change has happened, so the
		// flag is global across messages. The output is the same either way.
		if changed {
			msg.Set("content", newBlocks)
		}
	}
	return changed
}

// SanitizeThinkingBlocks mirrors sanitize_thinking_blocks: empty or omitted
// thinking blocks desync the Claude Code prompt cache, so they are dropped.
func SanitizeThinkingBlocks(payload *pyjson.Object) bool {
	messages, ok := objGet(payload, "messages").([]any)
	if !ok {
		return false
	}
	changed := false
	for _, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok || !pyEqual(objGet(msg, "role"), "assistant") {
			continue
		}
		content, ok := objGet(msg, "content").([]any)
		if !ok {
			continue
		}
		newBlocks := make([]any, 0, len(content))
		for _, b := range content {
			bo, ok := b.(*pyjson.Object)
			if !ok {
				newBlocks = append(newBlocks, b)
				continue
			}
			bt := objGet(bo, "type")
			if pyEqual(bt, "thinking") {
				th, isStr := objGet(bo, "thinking").(string)
				if !isStr || pyStrip(th) == "" {
					changed = true
					continue
				}
			}
			if pyEqual(bt, "redacted_thinking") && !pyTruthy(objGet(bo, "data")) {
				changed = true
				continue
			}
			newBlocks = append(newBlocks, b)
		}
		if len(newBlocks) != len(content) {
			msg.Set("content", newBlocks)
			changed = true
		}
	}
	return changed
}

// FoldRepeatedLogLines mirrors fold_repeated_log_lines: runs of four or more
// near-identical log lines (digits and timestamps vary) collapse to one line.
func FoldRepeatedLogLines(text string) string {
	if text == "" || strings.Count(text, "\n") < 8 {
		return text
	}
	templ := func(line string) string {
		return reNumRun.Sub("{}", reTimestamp.Sub("{}", line))
	}
	lines := strings.Split(text, "\n")
	var out []string
	for i := 0; i < len(lines); {
		line := lines[i]
		if pyStrip(line) == "" {
			out = append(out, line)
			i++
			continue
		}
		t := templ(line)
		j := i + 1
		for j < len(lines) && pyStrip(lines[j]) != "" && templ(lines[j]) == t {
			j++
		}
		if run := j - i; run >= 4 {
			out = append(out, t+"  [×"+itoa(run)+" folded]")
		} else {
			out = append(out, lines[i:j]...)
		}
		i = j
	}
	return strings.Join(out, "\n")
}

// ApplyCacheAndStructureFixes mirrors apply_cache_and_structure_fixes. It runs
// the cache-fix, digest, defer and media passes and returns the category tags.
func ApplyCacheAndStructureFixes(payload *pyjson.Object) ([]string, error) {
	var cats []string
	if SanitizeThinkingBlocks(payload) {
		cats = append(cats, "thinking_sanitize")
	}

	if _, has := payload.Get("system"); has {
		before := ContentText(objGet(payload, "system"))
		newSystem, err := StripVolatileSystem(objGet(payload, "system"))
		if err != nil {
			return nil, err
		}
		payload.Set("system", newSystem)
		if ContentText(newSystem) != before {
			cats = append(cats, "cache_fix")
		}
	}

	digested, err := ApplyExtractiveDigest(payload)
	if err != nil {
		return nil, err
	}
	if digested {
		cats = append(cats, "digest")
	}

	if StripMediaBlocks(payload) {
		cats = append(cats, "media_strip")
	}

	if tools, ok := objGet(payload, "tools").([]any); ok && len(tools) > 0 {
		beforeNames := toolNames(tools)
		stabilized, err := StabilizeTools(tools)
		if err != nil {
			return nil, err
		}
		payload.Set("tools", stabilized)
		afterNames := toolNames(stabilized.([]any))
		deferred, deferredChanged, err := ApplyDeferLoading(objGet(payload, "tools"))
		if err != nil {
			return nil, err
		}
		payload.Set("tools", deferred)
		if !namesEqual(beforeNames, afterNames) && !containsStr(cats, "cache_fix") {
			cats = append(cats, "cache_fix")
		}
		if deferredChanged {
			cats = append(cats, "defer_loading")
		}
	}
	return cats, nil
}

// toolNames mirrors [t.get("name") if isinstance(t, dict) else None for t in tools].
func toolNames(tools []any) []any {
	out := make([]any, len(tools))
	for i, t := range tools {
		if to, ok := t.(*pyjson.Object); ok {
			out[i] = objGet(to, "name")
		}
	}
	return out
}

func namesEqual(a, b []any) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !pyEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}

func containsStr(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

// NormalizeIntentText mirrors normalize_intent_text: trim, lowercase, and
// collapse whitespace runs to a single space.
func NormalizeIntentText(text string) string {
	return reWhitespaceRun.Sub(" ", pyLower(pyStrip(text)))
}

// ComputeIntentHash mirrors compute_intent_hash. It returns the cache key for
// short plain-text user turns (retries and identical asks). ok is false
// where Python returns None.
func ComputeIntentHash(payload *pyjson.Object, provider string) (string, bool, error) {
	messages, ok := objGet(payload, "messages").([]any)
	if !ok || len(messages) == 0 {
		return "", false, nil
	}
	last, ok := messages[len(messages)-1].(*pyjson.Object)
	if !ok || !pyEqual(objGet(last, "role"), "user") {
		return "", false, nil
	}
	var text string
	switch content := objGet(last, "content").(type) {
	case []any:
		for _, b := range content {
			if bo, ok := b.(*pyjson.Object); ok && pyEqual(objGet(bo, "type"), "tool_result") {
				return "", false, nil
			}
		}
		text = ContentText(content)
	case string:
		text = content
	default:
		return "", false, nil
	}
	text = NormalizeIntentText(text)
	if text == "" || pyLen(text) > 800 {
		return "", false, nil
	}
	first, isDict := messages[0].(*pyjson.Object)
	if !isDict {
		first = pyjson.NewObject()
	}
	firstText := NormalizeIntentText(ContentText(objGet(first, "content")))
	seed := objectOf(
		"provider", provider,
		"model", objGet(payload, "model"),
		"system", objGet(payload, "system"),
		"first", firstText,
		"n", int64(len(messages)),
		"intent", text,
	)
	b, err := canonicalJSON(seed)
	if err != nil {
		return "", false, err
	}
	sum := sha256.Sum256(b)
	return "intent:" + hex.EncodeToString(sum[:]), true, nil
}
