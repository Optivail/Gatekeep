// Mirrors backend/engines/aggressive.py: the aggressive tier for Anthropic
// Messages, OpenAI-compatible chat and Codex Responses payloads. It stores
// large tool outputs in CCR, elides old ones, dedups identical bodies, trims
// system and developer prompts, minifies tool schemas, and records each change
// as a category tag.
//
// The Python passes mutate the payload in place. The port does the same on
// *pyjson.Object, and it keeps the cache-safety rules from the source: a
// transform is a pure function of its content, and the only position-based
// rewrite is the explicit age elision.
package engines

import (
	"bytes"
	"strings"

	"github.com/Optivail/Gatekeep/server/pyjson"
	"github.com/Optivail/Gatekeep/server/pyre"
)

const (
	ccrMinChars     = 800
	ccrPreviewChars = 180
	// toolResultMax caps every tool_result in the CCR path.
	toolResultMax = 220
	staleStub     = "[STALE]"
	// openAIKeepTools: newest N OpenAI tool messages stay intact after crush.
	openAIKeepTools = 4
	openAIToolMax   = 160
)

var (
	reJSONArrayHint     = pyre.MustCompile(`^\s*\[`)
	reGrepLine          = pyre.MustCompile(`^[^:\n]+:\d+`)
	reSystemBoilerplate = pyre.MustCompile(`(?is)\n{0,2}(output format guidelines)[^\n]{0,200}(\n[^\n]{0,200}){0,4}`)
	// reHardNoiseBlocks removes Codex plugin and permission catalogs. It is
	// stronger than the Claude-safe trim, so it is used only for the OpenAI and
	// Codex paths.
	reHardNoiseBlocks = pyre.MustCompile(
		`(?is)` +
			`<recommended_plugins\b[^>]*>.*?</recommended_plugins>` +
			`|<apps_instructions\b[^>]*>.*?</apps_instructions>` +
			`|<skills_instructions\b[^>]*>.*?</skills_instructions>` +
			`|<permissions[_ ]instructions\b[^>]*>.*?</permissions[_ ]instructions>`)
	reHardMDSections = pyre.MustCompile(
		`(?ims)^#{1,3}\s+(Instruction hierarchy|Critical instructions|` +
			`Output format|Communication|Tone|Safety|Refusal|Formatting)\b.*?` +
			`(?=^#{1,3}\s+|\Z)`)
)

// trimSystemBlocks mirrors the list branch of the two system trimmers. A text
// block whose trimmed text is empty is kept when its original text is not
// blank, and dropped otherwise. A non-str original raises, because Python calls
// .strip() on it.
func trimSystemBlocks(system any, blocks []any, trim func(any) (any, error)) (any, error) {
	out := make([]any, 0, len(blocks))
	for _, b := range blocks {
		block, isObj := b.(*pyjson.Object)
		if !isObj || !typeIs(objGet(block, "type"), "text", "input_text") {
			out = append(out, b)
			continue
		}
		original := objGetOr(block, "text", "")
		trimmed, err := trim(original)
		if err != nil {
			return nil, err
		}
		ts, isStr := trimmed.(string)
		if !isStr || pyStrip(ts) == "" {
			os, ok := original.(string)
			if !ok {
				return nil, pyError("AttributeError", "text block is not a str")
			}
			if pyStrip(os) != "" {
				nb := copyObject(block)
				nb.Set("text", original)
				out = append(out, nb)
			}
			continue
		}
		nb := copyObject(block)
		nb.Set("text", ts)
		out = append(out, nb)
	}
	if len(out) > 0 {
		return out, nil
	}
	return system, nil
}

// AggressiveTrimSystemHard mirrors aggressive_trim_system_hard: a quota-stretch
// trim for Mistral and Vibe system prompts (not the Codex OS prompt). It drops
// the Codex noise blocks and some markdown sections, then keeps a head and a
// tail of very long prompts.
func AggressiveTrimSystemHard(system any) (any, error) {
	switch s := system.(type) {
	case string:
		text := OptimizeText(s)
		text = reSystemBoilerplate.Sub("\n", text)
		text = reHardNoiseBlocks.Sub("\n[omitted]\n", text)
		text = reHardMDSections.Sub("\n", text)
		text = pyStrip(reBlankRun.Sub("\n\n", text))
		n := pyLen(text)
		switch {
		case n > 4500:
			text = pyPrefix(text, 2000) + "\n…[system trimmed]…\n" + pySuffix(text, 1500)
		case n > 2800:
			text = pyPrefix(text, 1400) + "\n…[system trimmed]…\n" + pySuffix(text, 900)
		}
		if text != "" {
			return text, nil
		}
		return system, nil
	case []any:
		return trimSystemBlocks(system, s, AggressiveTrimSystemHard)
	}
	return system, nil
}

// AggressiveTrimSystem mirrors aggressive_trim_system: a light cleanup that
// keeps the prompt intact. Only prompts over 20000 characters are cut.
func AggressiveTrimSystem(system any) (any, error) {
	switch s := system.(type) {
	case string:
		text := OptimizeText(s)
		text = reSystemBoilerplate.Sub("\n", text)
		text = pyStrip(reBlankRun.Sub("\n\n", text))
		if pyLen(text) > 20000 {
			text = pyPrefix(text, 8000) + "\n…[system trimmed]…\n" + pySuffix(text, 6000)
		}
		if text != "" {
			return text, nil
		}
		return system, nil
	case []any:
		return trimSystemBlocks(system, s, AggressiveTrimSystem)
	}
	return system, nil
}

// AggressiveTrimCodexDeveloper mirrors aggressive_trim_codex_developer: strip
// the Codex noise blocks and keep the agent's operating instructions.
func AggressiveTrimCodexDeveloper(text string) string {
	if pyLen(text) < 400 {
		return text
	}
	out := OptimizeText(text)
	out = reHardNoiseBlocks.Sub("\n[omitted]\n", out)
	out = pyStrip(reBlankRun.Sub("\n\n", out))
	if out != "" {
		return out
	}
	return text
}

// crushLogLines mirrors _crush_log_lines. Noise lines go, short logs are kept
// whole, and long logs keep the head, the tail and every error line with one
// line of context. Each dropped run becomes a counted marker.
func crushLogLines(text string) string {
	if IsErrorPayload(text, "", false) {
		return PreserveStacktraceBlocks(text, StacktraceCeiling)
	}
	var lines []string
	for _, l := range strings.Split(text, "\n") {
		if !reNoiseLine.MatchString(l) {
			lines = append(lines, l)
		}
	}
	if len(lines) <= 24 {
		return strings.Join(lines, "\n")
	}
	n := len(lines)
	keep := make([]bool, n)
	for i := 0; i < 4; i++ {
		keep[i] = true
	}
	for i := n - 6; i < n; i++ {
		keep[i] = true
	}
	for i, line := range lines {
		if reErrorKeywords.MatchString(line) {
			keep[i] = true
			for j := max(0, i-1); j < min(n, i+2); j++ {
				keep[j] = true
			}
		}
	}
	var out []string
	for i := 0; i < n; {
		if keep[i] {
			out = append(out, lines[i])
			i++
			continue
		}
		start := i
		for i < n && !keep[i] {
			i++
		}
		out = append(out, "[…"+itoa(i-start)+" lines crushed…]")
	}
	return strings.Join(out, "\n")
}

// crushJSONArray mirrors _crush_json_array. A JSON array of more than 12
// elements becomes a summary object with the count, the first and last three
// items, and four items sampled from the middle.
func crushJSONArray(text string) (string, error) {
	data, ok, err := jsonLoadsValue(text)
	if err != nil {
		return "", err
	}
	if !ok {
		return text, nil
	}
	list, isList := data.([]any)
	if !isList || len(list) <= 12 {
		return text, nil
	}
	const head, tail, sample = 3, 3, 4
	middle := list[head : len(list)-tail]
	step := 1
	if len(middle) > 0 {
		step = max(1, len(middle)/sample)
	}
	sampled := []any{}
	for k := 0; k < sample; k++ {
		idx := k * step
		if idx >= len(middle) {
			break
		}
		sampled = append(sampled, middle[idx])
	}
	view := objectOf(
		"_n", int64(len(list)),
		"_head", list[:head],
		"_sample", sampled,
		"_tail", list[len(list)-tail:],
	)
	b, err := dumpsCompact(view)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// crushGrepish mirrors _crush_grepish. A grep-style output keeps only its
// match lines and error lines. It is used when they are at least eight lines
// and under 80 percent of the total.
func crushGrepish(text string) string {
	lines := strings.Split(text, "\n")
	var hits []string
	for _, l := range lines {
		if reGrepLine.MatchString(l) || reErrorKeywords.MatchString(l) {
			hits = append(hits, l)
		}
	}
	if len(hits) < 8 || float64(len(hits)) >= float64(len(lines))*0.8 {
		return text
	}
	miss := itoa(len(lines) - len(hits))
	if len(hits) <= 40 {
		return strings.Join(hits, "\n") + "\n[" + miss + " non-match lines omitted]"
	}
	return strings.Join(hits[:20], "\n") + "\n[…" + itoa(len(hits)-40) + " matches omitted…]\n" +
		strings.Join(hits[len(hits)-20:], "\n") + "\n[" + miss + " non-match lines omitted]"
}

// CrushText mirrors crush_text. It runs the content-specific crushers in order
// (folded repeats, JSON array, grep output, then log lines) and returns the
// first change. Error payloads are preserved whole.
func CrushText(text string) (string, error) {
	if text == "" || pyLen(text) < 400 {
		return text, nil
	}
	if IsErrorPayload(text, "", false) {
		return PreserveDiagnosticText(text, StacktraceCeiling), nil
	}
	// Fold repeated log lines before the coarser crushers (lossless template fold).
	if folded := FoldRepeatedLogLines(text); folded != text {
		text = folded
	}
	stripped := pyStrip(text)
	if reJSONArrayHint.MatchString(stripped) && strings.HasSuffix(stripped, "]") {
		crushed, err := crushJSONArray(stripped)
		if err != nil {
			return "", err
		}
		if crushed != stripped {
			return crushed, nil
		}
	}
	grepLines := 0
	for _, l := range strings.Split(text, "\n") {
		if reGrepLine.MatchString(l) {
			grepLines++
		}
	}
	if grepLines >= 8 {
		return crushGrepish(text), nil
	}
	if len(strings.Split(text, "\n")) > 30 {
		return crushLogLines(text), nil
	}
	return text, nil
}

// ccrWrap mirrors _ccr_wrap for a str. The text is replaced by a CCR marker
// when crushing still leaves at least 800 characters. The original is stored
// in the CCR store, keyed by its hash, so a later ccr_retrieve call can fetch it.
func ccrWrap(text string) (string, error) {
	if text == "" || pyLen(text) < 400 {
		return text, nil
	}
	if IsErrorPayload(text, "", false) {
		return PreserveDiagnosticText(text, StacktraceCeiling), nil
	}
	crushed, err := CrushText(text)
	if err != nil {
		return "", err
	}
	if pyLen(text) < ccrMinChars || pyLen(crushed) < ccrMinChars {
		return truncHeadTail(crushed, toolResultMax), nil
	}
	h := Store(text)
	preview := pyPrefix(crushed, ccrPreviewChars)
	return truncHeadTail(Marker(h, pyLen(text), preview), toolResultMax), nil
}

// ccrWrapValue applies ccrWrap to a dynamic block text. Falsy values pass
// through. A str is wrapped. A list or dict shorter than 400 passes through,
// because Python only calls len() on it. Longer values and numbers raise.
func ccrWrapValue(v any) (any, error) {
	if !pyTruthy(v) {
		return v, nil
	}
	switch t := v.(type) {
	case string:
		s, err := ccrWrap(t)
		if err != nil {
			return nil, err
		}
		return s, nil
	case []any:
		if len(t) < 400 {
			return v, nil
		}
	case *pyjson.Object:
		if t.Len() < 400 {
			return v, nil
		}
	}
	return nil, pyError("TypeError", "ccr wrap on a non-str block text")
}

// preserveDiagnosticValue applies preserve_diagnostic_text to a dynamic value.
// A truthy non-str raises, because the traceback regex search needs a str.
func preserveDiagnosticValue(v any) (any, error) {
	if !pyTruthy(v) {
		return v, nil
	}
	s, ok := v.(string)
	if !ok {
		return nil, pyError("TypeError", "expected string or bytes-like object")
	}
	return PreserveDiagnosticText(s, StacktraceCeiling), nil
}

// mapResultContent mirrors _map_tool_result_content. A str is mapped; in a list,
// only the text blocks are mapped and the rest are kept. Other content is
// returned unchanged.
func mapResultContent(content any, mapper func(any) (any, error)) (any, error) {
	switch c := content.(type) {
	case string:
		return mapper(c)
	case []any:
		out := make([]any, 0, len(c))
		for _, b := range c {
			bo, isObj := b.(*pyjson.Object)
			if isObj && pyEqual(objGet(bo, "type"), "text") {
				nb := copyObject(bo)
				v, err := mapper(objGetOr(bo, "text", ""))
				if err != nil {
					return nil, err
				}
				nb.Set("text", v)
				out = append(out, nb)
				continue
			}
			out = append(out, b)
		}
		return out, nil
	}
	return content, nil
}

// ApplyPureToolResultCCR mirrors apply_pure_tool_result_ccr: tool_results are
// crushed by a content-pure function, so the cached prefix stays stable. The
// live zone and results with loop relax are left intact. Error payloads are
// preserved whole. Results that are already CCR markers, stale stubs or
// unchanged-file markers are skipped.
func ApplyPureToolResultCCR(payload *pyjson.Object, relax bool) (bool, error) {
	if relax {
		return false, nil
	}
	messages, ok := objGet(payload, "messages").([]any)
	if !ok {
		return false, nil
	}
	live := liveZoneResultLocs(messages)
	changed := false
	for mi, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok || !pyEqual(objGet(msg, "role"), "user") {
			continue
		}
		content, ok := objGet(msg, "content").([]any)
		if !ok {
			continue
		}
		for bi, b := range content {
			block, ok := b.(*pyjson.Object)
			if !ok || !pyEqual(objGet(block, "type"), "tool_result") {
				continue
			}
			if live[resultLoc{mi, bi}] {
				continue
			}
			if IsCCRFulfilled(block) {
				continue
			}
			before := ContentText(objGet(block, "content"))
			if pyLen(before) < 400 {
				continue
			}
			if IsErrorPayload(before, "", pyTruthy(objGet(block, "is_error"))) {
				nv, err := mapResultContent(objGet(block, "content"), preserveDiagnosticValue)
				if err != nil {
					return changed, err
				}
				block.Set("content", nv)
				if ContentText(objGet(block, "content")) != before {
					changed = true
				}
				continue
			}
			if strings.HasPrefix(before, "[CCR hash=") || strings.HasPrefix(before, "[STALE:") {
				continue
			}
			if strings.HasPrefix(before, "[FILE UNCHANGED:") {
				continue
			}
			nv, err := mapResultContent(objGet(block, "content"), ccrWrapValue)
			if err != nil {
				return changed, err
			}
			block.Set("content", nv)
			if ContentText(objGet(block, "content")) != before {
				changed = true
			}
		}
	}
	return changed, nil
}

// staleOccurrence is one tool_result that answers a read of a file path.
type staleOccurrence struct {
	mi, bi int
	path   string
}

// ApplyStaleToolCompaction mirrors apply_stale_tool_compaction. When a file is
// read more than once, every read except the newest becomes STALE_STUB. The
// rewrite happens once, and after that the stub is stable.
func ApplyStaleToolCompaction(payload *pyjson.Object) (bool, error) {
	messages, ok := objGet(payload, "messages").([]any)
	if !ok {
		return false, nil
	}

	pending := map[string]string{}
	var occurrences []staleOccurrence
	for mi, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok {
			continue
		}
		content, ok := objGet(msg, "content").([]any)
		if !ok {
			continue
		}
		switch {
		case pyEqual(objGet(msg, "role"), "assistant"):
			for _, b := range content {
				block, ok := b.(*pyjson.Object)
				if !ok || !pyEqual(objGet(block, "type"), "tool_use") {
					continue
				}
				path, found, err := extractFilePath(block)
				if err != nil {
					return false, err
				}
				if found && path != "" && pyTruthy(objGet(block, "id")) {
					key, hashable := pyKey(objGet(block, "id"))
					if !hashable {
						return false, pyError("TypeError", "unhashable type")
					}
					pending[key] = path
				}
			}
		case pyEqual(objGet(msg, "role"), "user"):
			for bi, b := range content {
				block, ok := b.(*pyjson.Object)
				if !ok || !pyEqual(objGet(block, "type"), "tool_result") {
					continue
				}
				key, hashable := pyKey(objGet(block, "tool_use_id"))
				if !hashable {
					return false, pyError("TypeError", "unhashable type")
				}
				if path, found := pending[key]; found && path != "" {
					occurrences = append(occurrences, staleOccurrence{mi, bi, path})
				}
			}
		}
	}

	latest := map[string][2]int{}
	for _, o := range occurrences {
		latest[o.path] = [2]int{o.mi, o.bi}
	}
	changed := false
	for _, o := range occurrences {
		if latest[o.path] == [2]int{o.mi, o.bi} {
			continue
		}
		msg := messages[o.mi].(*pyjson.Object)
		content := append([]any(nil), contentList(msg)...)
		block := copyObject(content[o.bi].(*pyjson.Object))
		if ContentText(objGet(block, "content")) == staleStub {
			continue
		}
		block.Set("content", staleStub)
		content[o.bi] = block
		msg.Set("content", content)
		changed = true
	}
	return changed, nil
}

// stripSchemaNode mirrors the strip_schema helper shared by the minifiers. It
// drops description, title, examples, default, $schema and $defs, and recurses
// into properties and items. Other keys are kept as they are.
func stripSchemaNode(node any) any {
	o, ok := node.(*pyjson.Object)
	if !ok {
		return node
	}
	out := pyjson.NewObject()
	for _, k := range o.Keys() {
		switch k {
		case "description", "title", "examples", "default", "$schema", "$defs":
			continue
		}
		v := objGet(o, k)
		if po, isObj := v.(*pyjson.Object); isObj && k == "properties" {
			np := pyjson.NewObject()
			for _, pk := range po.Keys() {
				np.Set(pk, stripSchemaNode(objGet(po, pk)))
			}
			out.Set(k, np)
			continue
		}
		if io, isObj := v.(*pyjson.Object); isObj && k == "items" {
			out.Set(k, stripSchemaNode(io))
			continue
		}
		out.Set(k, v)
	}
	return out
}

// junkToolKeys are top-level tool keys that carry no structure for the model.
var junkToolKeys = []string{"title", "examples", "$schema", "$defs", "default"}

// HardMinifyTools mirrors hard_minify_tools: a deterministic shrink of the
// Anthropic tool definitions. Identical tools in give identical tools out. The
// retrieve tool is left alone. Non-list input is returned as is.
func HardMinifyTools(tools any) any {
	list, ok := tools.([]any)
	if !ok {
		return tools
	}
	out := make([]any, 0, len(list))
	for _, t := range list {
		tool, ok := t.(*pyjson.Object)
		if !ok {
			out = append(out, t)
			continue
		}
		tc := copyObject(tool)
		if pyEqual(objGet(tc, "name"), RetrieveToolName) {
			out = append(out, tc)
			continue
		}
		if d, isStr := objGet(tc, "description").(string); isStr {
			tc.Set("description", pyPrefix(firstSentence(d), 48))
		}
		if schema, isObj := objGet(tc, "input_schema").(*pyjson.Object); isObj {
			tc.Set("input_schema", stripSchemaNode(schema))
		}
		for _, junk := range junkToolKeys {
			tc.Delete(junk)
		}
		out = append(out, tc)
	}
	return out
}

// codexMinifyOne mirrors minify_one in hard_minify_codex_tools. It keeps enough
// of the exec tool's documentation for tool use and trims the rest. Nested
// namespace tools are minified recursively.
func codexMinifyOne(tool any) any {
	t0, ok := tool.(*pyjson.Object)
	if !ok {
		return tool
	}
	t := copyObject(t0)
	name := objGet(t, "name")
	d, isStr := objGet(t, "description").(string)
	switch {
	case pyEqual(name, "exec") && isStr:
		if pyLen(d) > 3500 {
			t.Set("description", pyPrefix(d, 2800)+"\n…[trimmed]…")
		}
	case isStr:
		t.Set("description", pyPrefix(firstSentence(d), 120))
	}
	if fn, isObj := objGet(t, "function").(*pyjson.Object); isObj {
		fc := copyObject(fn)
		if fd, isStr := objGet(fc, "description").(string); isStr {
			fc.Set("description", pyPrefix(firstSentence(fd), 120))
		}
		if p, isObj := objGet(fc, "parameters").(*pyjson.Object); isObj {
			fc.Set("parameters", stripSchemaNode(p))
		}
		t.Set("function", fc)
	}
	if p, isObj := objGet(t, "parameters").(*pyjson.Object); isObj {
		t.Set("parameters", stripSchemaNode(p))
	}
	if nested, isList := objGet(t, "tools").([]any); isList {
		mapped := make([]any, len(nested))
		for i, x := range nested {
			mapped[i] = codexMinifyOne(x)
		}
		t.Set("tools", mapped)
	}
	return t
}

// HardMinifyCodexTools mirrors hard_minify_codex_tools: a mild shrink of Codex
// tools. It never guts exec, because a tool-use loop needs its documentation.
func HardMinifyCodexTools(tools any) any {
	list, ok := tools.([]any)
	if !ok {
		return tools
	}
	out := make([]any, 0, len(list))
	for _, t := range list {
		out = append(out, codexMinifyOne(t))
	}
	return out
}

// openAIMinifyOne mirrors minify_one in hard_minify_openai_tools. It handles a
// function wrapper and a top-level tool, and recurses into namespace trees.
// The retrieve tool is returned with its schema unchanged.
func openAIMinifyOne(tool any) any {
	t0, ok := tool.(*pyjson.Object)
	if !ok {
		return tool
	}
	t := copyObject(t0)
	if fn, isObj := objGet(t, "function").(*pyjson.Object); isObj {
		fc := copyObject(fn)
		if pyEqual(objGet(fc, "name"), RetrieveToolName) {
			return t
		}
		if fd, isStr := objGet(fc, "description").(string); isStr {
			fc.Set("description", pyPrefix(firstSentence(fd), 48))
		}
		if p, isObj := objGet(fc, "parameters").(*pyjson.Object); isObj {
			fc.Set("parameters", stripSchemaNode(p))
		}
		t.Set("function", fc)
	}
	if pyEqual(objGet(t, "name"), RetrieveToolName) {
		return t
	}
	if d, isStr := objGet(t, "description").(string); isStr {
		t.Set("description", pyPrefix(firstSentence(d), 48))
	}
	if p, isObj := objGet(t, "parameters").(*pyjson.Object); isObj {
		t.Set("parameters", stripSchemaNode(p))
	}
	if s, isObj := objGet(t, "input_schema").(*pyjson.Object); isObj {
		t.Set("input_schema", stripSchemaNode(s))
	}
	if nested, isList := objGet(t, "tools").([]any); isList {
		mapped := make([]any, len(nested))
		for i, x := range nested {
			mapped[i] = openAIMinifyOne(x)
		}
		t.Set("tools", mapped)
	}
	for _, junk := range junkToolKeys {
		t.Delete(junk)
	}
	return t
}

// HardMinifyOpenAITools mirrors hard_minify_openai_tools: it shrinks OpenAI,
// Mistral and Codex tools, including nested namespace and custom trees.
func HardMinifyOpenAITools(tools any) any {
	list, ok := tools.([]any)
	if !ok {
		return tools
	}
	out := make([]any, 0, len(list))
	for _, t := range list {
		out = append(out, openAIMinifyOne(t))
	}
	return out
}

// lastNInts returns the set of the last n indices, as xs[-n:] does.
func lastNInts(xs []int, n int) map[int]bool {
	out := map[int]bool{}
	start := len(xs) - n
	if start < 0 {
		start = 0
	}
	for _, x := range xs[start:] {
		out[x] = true
	}
	return out
}

// dedupCats mirrors list(dict.fromkeys(cats)): first occurrences, in order.
func dedupCats(cats []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(cats))
	for _, c := range cats {
		if !seen[c] {
			seen[c] = true
			out = append(out, c)
		}
	}
	return out
}

// ApplyAggressiveOpenAI mirrors apply_aggressive_openai: the aggressive crush
// for OpenAI-compatible chat payloads (Mistral, Vibe, Codex API key). Tool
// messages are deduped by digest. The newest four keep their content, after
// the crush, and older ones are elided or kept only for their errors. Long
// user and assistant messages are crushed, and the tool schemas are minified.
// The session key is not needed here, because nothing in this path is stateful
// beyond CCR.
func ApplyAggressiveOpenAI(payload *pyjson.Object) ([]string, error) {
	var cats []string
	messages, ok := objGet(payload, "messages").([]any)
	if !ok {
		return cats, nil
	}

	var toolIdxs []int
	for i, m := range messages {
		if msg, ok := m.(*pyjson.Object); ok && pyEqual(objGet(msg, "role"), "tool") {
			toolIdxs = append(toolIdxs, i)
		}
	}
	live := lastNInts(toolIdxs, openAIKeepTools)
	seen := map[string]bool{}

	for i, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok {
			continue
		}
		role := objGet(msg, "role")
		if pyEqual(role, "system") {
			before, isStr := objGet(msg, "content").(string)
			if isStr && pyLen(before) > 200 {
				trimmed, err := AggressiveTrimSystemHard(before)
				if err != nil {
					return cats, err
				}
				if ts, ok := trimmed.(string); ok && ts != before {
					msg.Set("content", ts)
					cats = append(cats, "system_trim")
				}
			}
			continue
		}
		if !pyEqual(role, "tool") {
			content, isStr := objGet(msg, "content").(string)
			if !isStr {
				continue
			}
			// User pastes and long assistant turns are the history bloat.
			if pyEqual(role, "user") && pyLen(content) > 1200 {
				pre, err := PreprocessToolText(content)
				if err != nil {
					return cats, err
				}
				crushed, err := CrushText(pre)
				if err != nil {
					return cats, err
				}
				if pyLen(crushed) > 900 {
					crushed = truncHeadTail(crushed, 800)
				}
				if crushed != content {
					msg.Set("content", crushed)
					cats = append(cats, "user_crush")
				}
			} else if pyEqual(role, "assistant") && pyLen(content) > 1500 {
				crushed, err := CrushText(content)
				if err != nil {
					return cats, err
				}
				if pyLen(crushed) > 1000 {
					crushed = truncHeadTail(crushed, 900)
				}
				if crushed != content {
					msg.Set("content", crushed)
					cats = append(cats, "assistant_crush")
				}
			}
			continue
		}

		content, isStr := objGet(msg, "content").(string)
		if !isStr || pyLen(content) < 60 {
			continue
		}
		if content == microStub || content == dupStub || strings.HasPrefix(content, "[CCR") {
			continue
		}

		digest := sha256Hex(encodeUTF8Replace(content))[:16]
		if seen[digest] {
			msg.Set("content", dupStub)
			cats = append(cats, "tool_dedup")
			continue
		}
		seen[digest] = true

		if !live[i] {
			if IsErrorPayload(content, "", false) {
				crushed := PreserveDiagnosticText(content, StacktraceCeiling)
				if crushed != content {
					msg.Set("content", crushed)
					cats = append(cats, "error_preserve")
				}
				continue
			}
			msg.Set("content", microStub)
			cats = append(cats, "age_elide")
			continue
		}

		pre, err := PreprocessToolText(content)
		if err != nil {
			return cats, err
		}
		var crushed string
		if IsErrorPayload(pre, "", false) {
			crushed = PreserveDiagnosticText(pre, StacktraceCeiling)
		} else {
			crushed, err = CrushText(pre)
			if err != nil {
				return cats, err
			}
		}
		switch {
		case pyLen(content) >= ccrMinChars && pyLen(crushed) >= 400:
			h := Store(content)
			crushed = truncHeadTail(Marker(h, pyLen(content), pyPrefix(crushed, ccrPreviewChars)), openAIToolMax)
			cats = append(cats, "ccr")
		case pyLen(crushed) > openAIToolMax:
			crushed = truncHeadTail(crushed, openAIToolMax)
			cats = append(cats, "tool_crush")
		case crushed != content:
			cats = append(cats, "tool_crush")
		}
		if crushed != content {
			msg.Set("content", crushed)
		}
	}

	if tools, ok := objGet(payload, "tools").([]any); ok && len(tools) > 0 {
		payload.Set("tools", HardMinifyOpenAITools(tools))
		// The CCR retrieve tool is not injected here: it adds tokens, and the
		// OpenAI-compat clients rarely call it.
		cats = append(cats, "hard_tool_minify")
	}
	return dedupCats(cats), nil
}

// crushResponsesOutputText mirrors _crush_responses_output_text: the same
// crush as the chat path, for Responses output text. Large outputs go to CCR.
func crushResponsesOutputText(text string) (string, error) {
	if pyLen(text) < 80 {
		return text, nil
	}
	pre, err := PreprocessToolText(text)
	if err != nil {
		return "", err
	}
	if IsErrorPayload(pre, "", false) {
		return PreserveDiagnosticText(pre, StacktraceCeiling), nil
	}
	crushed, err := CrushText(pre)
	if err != nil {
		return "", err
	}
	if pyLen(text) >= ccrMinChars && pyLen(crushed) >= 400 {
		h := Store(text)
		return truncHeadTail(Marker(h, pyLen(text), pyPrefix(crushed, ccrPreviewChars)), openAIToolMax), nil
	}
	if pyLen(crushed) > openAIToolMax {
		return truncHeadTail(crushed, openAIToolMax), nil
	}
	return crushed, nil
}

// ApplyAggressiveResponses mirrors apply_aggressive_responses: the aggressive
// crush for the OpenAI and Codex Responses API (input items). It covers the
// instructions, the input string or item list, tool outputs (age elision,
// dedup and crush), user blobs, Codex additional_tools, developer prompts and
// the top-level tools.
func ApplyAggressiveResponses(payload *pyjson.Object) ([]string, error) {
	var cats []string
	if instr, isStr := objGet(payload, "instructions").(string); isStr && pyLen(instr) > 400 {
		trimmed, err := AggressiveTrimSystemHard(instr)
		if err != nil {
			return cats, err
		}
		if ts, ok := trimmed.(string); ok && ts != instr {
			payload.Set("instructions", ts)
			cats = append(cats, "system_trim")
		}
	}

	rawInput := objGet(payload, "input")
	if s, isStr := rawInput.(string); isStr && pyLen(s) > 1200 {
		crushed, err := crushResponsesOutputText(s)
		if err != nil {
			return cats, err
		}
		if pyLen(crushed) > 900 {
			crushed = truncHeadTail(crushed, 800)
		}
		if crushed != s {
			payload.Set("input", crushed)
			cats = append(cats, "user_crush")
		}
		return dedupCats(cats), nil
	}

	list, isList := rawInput.([]any)
	if !isList {
		if tools, ok := objGet(payload, "tools").([]any); ok && len(tools) > 0 {
			payload.Set("tools", HardMinifyOpenAITools(tools))
			cats = append(cats, "hard_tool_minify")
		}
		return dedupCats(cats), nil
	}

	// Tool outputs are collected first, so age elision knows the newest four.
	var toolIdxs []int
	for i, item := range list {
		obj, ok := item.(*pyjson.Object)
		if !ok {
			continue
		}
		if typeIs(objGet(obj, "type"), "function_call_output", "tool_result", "computer_call_output") {
			toolIdxs = append(toolIdxs, i)
		} else if pyEqual(objGet(obj, "role"), "tool") {
			toolIdxs = append(toolIdxs, i)
		}
	}
	live := lastNInts(toolIdxs, openAIKeepTools)
	seen := map[string]bool{}

	for i, item := range list {
		obj, ok := item.(*pyjson.Object)
		if !ok {
			continue
		}
		t := objGet(obj, "type")

		// function_call_output.output, computer_call_output.output, tool_result.
		if typeIs(t, "function_call_output", "computer_call_output", "tool_result") {
			out := objGet(obj, "output")
			if _, isStr := out.(string); !isStr {
				// Some clients send the output as a list of content parts.
				if parts, isParts := out.([]any); isParts {
					for _, p := range parts {
						part, ok := p.(*pyjson.Object)
						if !ok {
							continue
						}
						before, isStr := objGet(part, "text").(string)
						if !isStr || pyLen(before) < 60 {
							continue
						}
						digest := sha256Hex(encodeUTF8Replace(before))[:16]
						if seen[digest] || !live[i] {
							if seen[digest] {
								part.Set("text", dupStub)
								cats = append(cats, "tool_dedup")
							} else {
								part.Set("text", microStub)
								cats = append(cats, "age_elide")
							}
							continue
						}
						seen[digest] = true
						crushed, err := crushResponsesOutputText(before)
						if err != nil {
							return cats, err
						}
						if crushed != before {
							part.Set("text", crushed)
							cats = append(cats, "tool_crush")
						}
					}
				}
				continue
			}
			outStr := out.(string)
			if pyLen(outStr) < 60 {
				continue
			}
			digest := sha256Hex(encodeUTF8Replace(outStr))[:16]
			if seen[digest] {
				obj.Set("output", dupStub)
				cats = append(cats, "tool_dedup")
				continue
			}
			seen[digest] = true
			if !live[i] {
				if IsErrorPayload(outStr, "", false) {
					crushed := PreserveDiagnosticText(outStr, StacktraceCeiling)
					if crushed != outStr {
						obj.Set("output", crushed)
						cats = append(cats, "error_preserve")
					}
					continue
				}
				obj.Set("output", microStub)
				cats = append(cats, "age_elide")
				continue
			}
			crushed, err := crushResponsesOutputText(outStr)
			if err != nil {
				return cats, err
			}
			if crushed != outStr {
				obj.Set("output", crushed)
				cats = append(cats, "tool_crush")
			}
			continue
		}

		if pyEqual(objGet(obj, "role"), "tool") {
			if content, isStr := objGet(obj, "content").(string); isStr {
				if pyLen(content) < 60 {
					continue
				}
				digest := sha256Hex(encodeUTF8Replace(content))[:16]
				if seen[digest] {
					obj.Set("content", dupStub)
					cats = append(cats, "tool_dedup")
					continue
				}
				seen[digest] = true
				if !live[i] {
					if IsErrorPayload(content, "", false) {
						crushed := PreserveDiagnosticText(content, StacktraceCeiling)
						if crushed != content {
							obj.Set("content", crushed)
							cats = append(cats, "error_preserve")
						}
						continue
					}
					obj.Set("content", microStub)
					cats = append(cats, "age_elide")
					continue
				}
				crushed, err := crushResponsesOutputText(content)
				if err != nil {
					return cats, err
				}
				if crushed != content {
					obj.Set("content", crushed)
					cats = append(cats, "tool_crush")
				}
				continue
			}
		}

		// User and environment blobs: plugins, cwd dumps and pastes.
		if pyEqual(objGet(obj, "role"), "user") {
			content := objGet(obj, "content")
			if cs, isStr := content.(string); isStr && pyLen(cs) > 1200 {
				crushed, err := crushResponsesOutputText(cs)
				if err != nil {
					return cats, err
				}
				if pyLen(crushed) > 900 {
					crushed = truncHeadTail(crushed, 800)
				}
				if crushed != cs {
					obj.Set("content", crushed)
					cats = append(cats, "user_crush")
				}
			} else if parts, isParts := content.([]any); isParts {
				for _, p := range parts {
					part, ok := p.(*pyjson.Object)
					if !ok || !typeIs(objGet(part, "type"), "input_text", "text") {
						continue
					}
					before, isStr := objGet(part, "text").(string)
					if !isStr || pyLen(before) <= 900 {
						continue
					}
					// Recurring Codex plugin catalogs are dropped entirely.
					if strings.Contains(before, "<recommended_plugins") || strings.Contains(before, "<apps_instructions") {
						part.Set("text", "[omitted]")
						cats = append(cats, "user_crush")
						continue
					}
					// Real user tasks stay intact; only huge dumps are cut.
					if pyLen(before) > 6000 {
						crushed := truncHeadTail(before, 4000)
						if crushed != before {
							part.Set("text", crushed)
							cats = append(cats, "user_crush")
						}
					}
				}
			}
		}
	}

	// Codex keeps tools and developer prompts inside input[], not at the top level.
	for _, item := range list {
		obj, ok := item.(*pyjson.Object)
		if !ok {
			continue
		}
		if pyEqual(objGet(obj, "type"), "additional_tools") {
			if tools, isList := objGet(obj, "tools").([]any); isList {
				before, err := dumpsCompact(tools)
				if err != nil {
					return cats, err
				}
				minified := HardMinifyCodexTools(tools)
				obj.Set("tools", minified)
				after, err := dumpsCompact(minified)
				if err != nil {
					return cats, err
				}
				if !bytes.Equal(after, before) {
					cats = append(cats, "hard_tool_minify")
				}
			}
		}
		if pyEqual(objGet(obj, "role"), "developer") {
			switch c := objGet(obj, "content").(type) {
			case string:
				if pyLen(c) > 400 {
					if trimmed := AggressiveTrimCodexDeveloper(c); trimmed != c {
						obj.Set("content", trimmed)
						cats = append(cats, "system_trim")
					}
				}
			case []any:
				for _, p := range c {
					part, ok := p.(*pyjson.Object)
					if !ok || !typeIs(objGet(part, "type"), "input_text", "text") {
						continue
					}
					before, isStr := objGet(part, "text").(string)
					if !isStr || pyLen(before) <= 400 {
						continue
					}
					// The main Codex OS prompt is trimmed only of skills and apps noise.
					if trimmed := AggressiveTrimCodexDeveloper(before); trimmed != before {
						part.Set("text", trimmed)
						cats = append(cats, "system_trim")
					}
				}
			}
		}
	}

	if tools, ok := objGet(payload, "tools").([]any); ok && len(tools) > 0 {
		payload.Set("tools", HardMinifyOpenAITools(tools))
		cats = append(cats, "hard_tool_minify")
	}
	return dedupCats(cats), nil
}

// ApplyAggressive mirrors apply_aggressive: the Anthropic Messages pipeline.
// The order matters. CCR fulfilment and loop observation run first, then the
// maxsave preprocess, stale and pure CCR, the late age elision and schema
// passes, the system trim, and finally the cache and structure fixes.
// sessionKey "" stands for Python's None (no session).
func ApplyAggressive(payload *pyjson.Object, sessionKey string) ([]string, error) {
	var cats []string
	fulfilled, err := FulfillRetrieveToolResults(objGet(payload, "messages"))
	if err != nil {
		return cats, err
	}
	if fulfilled {
		cats = append(cats, "ccr_fulfill")
	}

	observed, err := Observe(payload, sessionKey)
	if err != nil {
		return cats, err
	}
	cats = append(cats, observed...)
	relax := IsRelaxed(sessionKey) || IsDecompress(sessionKey)

	if IsDecompress(sessionKey) {
		// Loop decompress: lossless passes only, with no CCR and no elision.
		lossless, err := ApplyMaxsave(payload, true)
		if err != nil {
			return cats, err
		}
		cats = append(cats, lossless...)
		cats = append(cats, "loop_decompress")
		ClearCCRFulfilledFlags(objGet(payload, "messages"))
		return cats, nil
	}

	// Preprocess before CCR, so the crushed form is what gets hashed and stored.
	pre, err := ApplyMaxsave(payload, false)
	if err != nil {
		return cats, err
	}
	cats = append(cats, pre...)

	stale, err := ApplyStaleToolCompaction(payload)
	if err != nil {
		return cats, err
	}
	if stale {
		cats = append(cats, "stale_compaction")
	}

	pureCCR, err := ApplyPureToolResultCCR(payload, relax)
	if err != nil {
		return cats, err
	}
	if pureCCR {
		cats = append(cats, "ccr")
	}

	// Age elision runs after CCR, so spill and retrieve still work for the
	// crushed bodies.
	late, err := ApplyMaxsaveLate(payload, relax)
	if err != nil {
		return cats, err
	}
	cats = append(cats, late...)

	if tools, ok := objGet(payload, "tools").([]any); ok {
		minified := HardMinifyTools(tools)
		payload.Set("tools", InjectRetrieveTool(minified))
		cats = append(cats, "hard_tool_minify")
	}

	if hasKey(payload, "system") {
		before := ContentText(objGet(payload, "system"))
		sys, err := AggressiveTrimSystem(objGet(payload, "system"))
		if err != nil {
			return cats, err
		}
		payload.Set("system", sys)
		if ContentText(sys) != before {
			cats = append(cats, "system_trim")
		}
	}

	fixes, err := ApplyCacheAndStructureFixes(payload)
	if err != nil {
		return cats, err
	}
	cats = append(cats, fixes...)

	// Re-run the unused-tool pass after defer, so mid-session schemas stay tiny.
	unused, err := DropUnusedToolSchemas(payload)
	if err != nil {
		return cats, err
	}
	if unused && !containsStr(cats, "unused_tools") {
		cats = append(cats, "unused_tools")
	}
	if err := DropEmptyTextBlocks(payload); err != nil {
		return cats, err
	}
	ClearCCRFulfilledFlags(objGet(payload, "messages"))
	return cats, nil
}
