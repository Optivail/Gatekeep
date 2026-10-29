// Mirrors backend/engines/optimizer.py: token counting, caveman/prose
// compression, history pruning, cache alignment, file delta tracking, tool
// minification, and the request-level cache key helpers.
//
// Python values reach this module as dict/list/str/int/float/bool/None. Each
// function here takes the same value shapes and reproduces the Python type
// rules (truthiness, isinstance, .get on non-dicts, str-only operations), so
// an input the proxy can send raises or passes through exactly as Python does.
package engines

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"math/big"
	"strconv"
	"strings"
	"sync"

	"github.com/Optivail/Gatekeep/server/pyjson"
	"github.com/Optivail/Gatekeep/server/pyre"
	"github.com/Optivail/Gatekeep/server/pytok"
)

// ErrUnsupported marks an input this port cannot reproduce byte for byte. The
// caller must send that request to the Python backend. Guessing would change
// the bytes sent upstream, which the compression rules forbid.
var ErrUnsupported = errors.New("engines: unsupported input, route to the Python backend")

// isPyKind reports whether err carries the Python exception kind (for example
// "ValueError"), as produced by pyError.
func isPyKind(err error, kind string) bool {
	return strings.HasPrefix(err.Error(), kind+":")
}

const antiYap = "[SYSTEM DIRECTIVE: Output code and tool calls ONLY. " +
	"Omit all preambles, greetings, and conversational wrap-ups.]"

// Phase 2.5: history pruning (sliding window over long multi-turn conversations).
const (
	historyWindowThreshold   = 6
	preserveTailMessages     = 4
	largeCodeBlockMinLines   = 5
	codeOmitPlaceholder      = "[Previous context/code omitted for brevity]"
	logTruncateMinLines      = 20
	logHeadLines             = 5
	logTailLines             = 10
	maxCacheBreakpoints      = 4
	cacheSystemCharThreshold = 3500
	cacheHistoryDepth        = 3
	newestToolResultMaxChars = 800
	maxTrackedSessions       = 200
	antiYapMinChars          = 500
	explorationRunMin        = 3
	explorationMinResultLns  = 40
	maxTokensClampTarget     = 2048
	simpleQueryCharThreshold = 100
)

var (
	reCodeFenceSplit = pyre.MustCompile("(```.*?```)", pyre.DotAll)
	fillerPatterns   = []*pyre.Regexp{
		pyre.MustCompile(`(?im)^[ \t]*(certainly|sure(?: thing)?|of course|great question|absolutely)!\s*`),
		pyre.MustCompile(`(?i)i hope (this|that) helps!?\.?`),
		pyre.MustCompile(`(?i)please let me know if you (have any questions|need (any )?(further )?(help|assistance)|need anything else)\.?`),
		pyre.MustCompile(`(?i)i'?d be happy to help( you)?( with (that|this))?!?\.?`),
		pyre.MustCompile(`(?i)feel free to (ask|reach out) if you (have|need) [^.\n]*\.?`),
		pyre.MustCompile(`(?i)thank you for (your patience|asking)!?\.?`),
	}
	reInlineCodeSplit = pyre.MustCompile("(`[^`\n]+`)")
	reDeterminers     = pyre.MustCompile(`(?i)\b(a|an|the)\b ?`)
	reFillerPreamble  = pyre.MustCompile(`(?im)^[ \t]*(so|well|basically|essentially|actually|just|simply)\b[,:]?\s*`)
	rePoliteAux       = pyre.MustCompile(`(?i)\b(could you please|would you please|can you please|could you|would you|can you)\b\s*`)
	reSpaceTabs       = pyre.MustCompile(`[ \t]+`)
	reNoiseLine       = pyre.MustCompile(`^[\s\-\\/|.⠀-⣿]*$`)
	reErrorKeywords   = pyre.MustCompile(`(?i)\b(error|traceback|exception|failed)\b`)
	rePythonFence     = pyre.MustCompile("^```python\\s*\\n(.*)\\n```$", pyre.DotAll, pyre.IgnoreCase)
	reSentenceBreak   = pyre.MustCompile(`[.\n]`)
	reSSEData         = pyre.MustCompile(`^data:\s*(.+)$`)
)

// EstimatedAntiYapOutputTokensSaved mirrors ESTIMATED_ANTI_YAP_OUTPUT_TOKENS_SAVED.
// It is a flat guess for the dashboard, not a measured value: the proxy never
// sees the model's reply.
const EstimatedAntiYapOutputTokensSaved = 60

// OptimizeOptions carries the optimize_payload switches. Python defaults are
// all true; DefaultOptimizeOptions returns them.
type OptimizeOptions struct {
	Compression    bool
	AntiYap        bool
	HistoryPruning bool
	LogTruncation  bool
}

// DefaultOptimizeOptions mirrors optimize_payload's default arguments.
func DefaultOptimizeOptions() OptimizeOptions {
	return OptimizeOptions{Compression: true, AntiYap: true, HistoryPruning: true, LogTruncation: true}
}

// CountTokens mirrors count_tokens: the cl100k_base token count, or an error
// where tiktoken raises (e.g. a text containing a special token).
func CountTokens(text string) (int, error) {
	if text == "" {
		return 0, nil
	}
	return pytok.CountTokens(text)
}

// canonicalJSON mirrors json.dumps(v, ensure_ascii=False, separators=(",",":")).
// The Python .encode("utf-8") raises on lone surrogates; pyjson reports the
// same case as an error.
func canonicalJSON(v any) ([]byte, error) {
	return pyjson.Dumps(v, pyjson.Compact)
}

// CanonicalJSONDumps mirrors canonical_json_dumps.
func CanonicalJSONDumps(data any) ([]byte, error) {
	return canonicalJSON(data)
}

// sha256Hex returns the lowercase hex SHA-256 digest of b.
func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// itoa formats a Python int for an f-string.
func itoa(n int) string { return strconv.Itoa(n) }

// splitCaptured mirrors re.split on a pattern whose only group is the whole
// match (no empty matches are possible for the patterns that use it). The
// result alternates text, match, text, ..., text, as Python's does.
func splitCaptured(re *pyre.Regexp, s string) []string {
	locs := re.FindAllIndex(s, -1)
	out := make([]string, 0, 2*len(locs)+1)
	prev := 0
	for _, loc := range locs {
		out = append(out, s[prev:loc[0]], s[loc[0]:loc[1]])
		prev = loc[1]
	}
	return append(out, s[prev:])
}

// cavemanWords strips filler, polite prefixes and articles from prose. The
// \b boundaries keep identifiers and paths intact.
func cavemanWords(text string) string {
	if isASCII(text) {
		return cavemanWordsASCII(text) // exact; see caveman_ascii.go
	}
	text = reFillerPreamble.Sub("", text)
	text = rePoliteAux.Sub("", text)
	text = reDeterminers.Sub("", text)
	return text
}

// cleanProse compresses text outside code fences.
func cleanProse(text string) string {
	text = stripFillers(text)
	segments := splitCaptured(reInlineCodeSplit, text)
	var b strings.Builder
	for i, seg := range segments {
		if i%2 == 1 {
			b.WriteString(seg) // inline `code` is never touched
		} else {
			b.WriteString(cavemanWords(seg))
		}
	}
	text = reSpaceTabs.Sub(" ", b.String())
	return reTripleNewline.Sub("\n\n", text)
}

// cleanCodeBlock strips trailing whitespace and collapses blank-line runs
// inside a fenced block (the fence lines are included in block).
func cleanCodeBlock(block string) string {
	lines := strings.Split(block, "\n")
	cleaned := make([]string, 0, len(lines))
	blankRun := 0
	for _, line := range lines {
		line = pyRStrip(line)
		if line == "" {
			blankRun++
			if blankRun > 1 {
				continue
			}
		} else {
			blankRun = 0
		}
		cleaned = append(cleaned, line)
	}
	return strings.Join(cleaned, "\n")
}

// OptimizeText mirrors optimize_text for a str.
func OptimizeText(text string) string {
	if text == "" {
		return text
	}
	segments := splitCaptured(reCodeFenceSplit, text)
	var b strings.Builder
	for i, seg := range segments {
		if i%2 == 1 {
			b.WriteString(cleanCodeBlock(seg))
		} else {
			b.WriteString(cleanProse(seg))
		}
	}
	return pyStrip(b.String())
}

// optimizeTextValue applies optimize_text to a dynamically typed block text.
// Falsy values pass through unchanged; a truthy non-str raises TypeError in
// the regex split.
func optimizeTextValue(v any) (any, error) {
	if !pyTruthy(v) {
		return v, nil
	}
	s, ok := v.(string)
	if !ok {
		return nil, pyError("TypeError", "expected string or bytes-like object")
	}
	return OptimizeText(s), nil
}

// InjectAntiYap mirrors inject_anti_yap. The list form returns a new list;
// the caller's list is not modified. Other shapes pass through untouched.
func InjectAntiYap(system any) any {
	switch s := system.(type) {
	case nil:
		return antiYap
	case string:
		return s + "\n\n" + antiYap
	case []any:
		out := make([]any, 0, len(s)+1)
		out = append(out, s...)
		return append(out, objectOf("type", "text", "text", antiYap))
	}
	return system
}

// ContentText mirrors _content_text: the plain text of a message content
// (str, or a list of text and tool_result blocks), joined with newlines.
func ContentText(content any) string {
	switch c := content.(type) {
	case string:
		return c
	case []any:
		var texts []string
		for _, block := range c {
			switch b := block.(type) {
			case string:
				texts = append(texts, b)
			case *pyjson.Object:
				btype := objGet(b, "type")
				text, textIsStr := objGet(b, "text").(string)
				isTextType := pyEqual(btype, "text") || pyEqual(btype, "input_text") || pyEqual(btype, "output_text")
				notCall := !pyEqual(btype, "tool_use") && !pyEqual(btype, "function_call")
				if isTextType || (textIsStr && notCall) {
					if textIsStr {
						texts = append(texts, text)
					}
				} else if pyEqual(btype, "tool_result") && hasKey(b, "content") {
					texts = append(texts, ContentText(objGet(b, "content")))
				}
			}
		}
		return strings.Join(texts, "\n")
	}
	return ""
}

// hasKey mirrors `key in dict`.
func hasKey(o *pyjson.Object, key string) bool {
	_, ok := o.Get(key)
	return ok
}

// optimizeContent mirrors _optimize_content: it cleans text blocks and
// returns a new block list for list content.
func optimizeContent(content any) (any, error) {
	switch c := content.(type) {
	case string:
		return OptimizeText(c), nil
	case []any:
		out := make([]any, 0, len(c))
		for _, block := range c {
			bo, ok := block.(*pyjson.Object)
			if ok && isTextBlockType(objGet(bo, "type")) {
				nb := copyObject(bo)
				t, err := optimizeTextValue(objGetOr(nb, "text", ""))
				if err != nil {
					return nil, err
				}
				nb.Set("text", t)
				out = append(out, nb)
				continue
			}
			out = append(out, block)
		}
		return out, nil
	}
	return content, nil
}

func isTextBlockType(v any) bool {
	return pyEqual(v, "text") || pyEqual(v, "input_text") || pyEqual(v, "output_text")
}

// walkSchema collects the description strings of a JSON-schema fragment.
func walkSchema(node any, parts *[]string) {
	o, ok := node.(*pyjson.Object)
	if !ok {
		return
	}
	if d, ok := objGet(o, "description").(string); ok {
		*parts = append(*parts, d)
	}
	if props, ok := objGet(o, "properties").(*pyjson.Object); ok {
		for _, k := range props.Keys() {
			walkSchema(objGet(props, k), parts)
		}
	}
	walkSchema(objGet(o, "items"), parts)
}

// toolsText flattens tool names, descriptions and schema descriptions for
// token accounting. Empty parts are kept (they join as blank lines).
func toolsText(tools any) string {
	list, ok := tools.([]any)
	if !ok {
		return ""
	}
	var parts []string
	for _, t := range list {
		to, ok := t.(*pyjson.Object)
		if !ok {
			continue
		}
		if n, ok := objGet(to, "name").(string); ok {
			parts = append(parts, n)
		}
		if d, ok := objGet(to, "description").(string); ok {
			parts = append(parts, d)
		}
		walkSchema(objGet(to, "input_schema"), &parts)
		if fn, ok := objGet(to, "function").(*pyjson.Object); ok {
			if n, ok := objGet(fn, "name").(string); ok {
				parts = append(parts, n)
			}
			if d, ok := objGet(fn, "description").(string); ok {
				parts = append(parts, d)
			}
			walkSchema(objGet(fn, "parameters"), &parts)
		}
	}
	return strings.Join(parts, "\n")
}

// toolsTextNested flattens Codex namespace, custom and function tool trees.
func toolsTextNested(tools any) string {
	list, ok := tools.([]any)
	if !ok {
		return ""
	}
	var parts []string
	for _, t := range list {
		to, ok := t.(*pyjson.Object)
		if !ok {
			continue
		}
		parts = append(parts, toolsText([]any{to}))
		if nested, ok := objGet(to, "tools").([]any); ok {
			parts = append(parts, toolsTextNested(nested))
		}
		if p, ok := objGet(to, "parameters").(*pyjson.Object); ok {
			walkSchema(p, &parts)
		}
	}
	return joinNonEmpty(parts, "\n")
}

// joinNonEmpty mirrors "\n".join(p for p in parts if p).
func joinNonEmpty(parts []string, sep string) string {
	var kept []string
	for _, p := range parts {
		if p != "" {
			kept = append(kept, p)
		}
	}
	return strings.Join(kept, sep)
}

// ExtractText mirrors extract_text: every piece of request text that the
// token estimate counts (system, messages, prompt, Responses-style input,
// instructions, and tool definitions).
func ExtractText(payload *pyjson.Object) (string, error) {
	var parts []string
	if hasKey(payload, "system") {
		parts = append(parts, ContentText(objGet(payload, "system")))
	}
	msgs, err := pyIterOr(objGet(payload, "messages"))
	if err != nil {
		return "", err
	}
	for _, m := range msgs {
		if mo, ok := m.(*pyjson.Object); ok {
			parts = append(parts, ContentText(objGet(mo, "content")))
		}
	}
	if s, ok := objGet(payload, "prompt").(string); ok {
		parts = append(parts, s)
	}
	switch inp := objGet(payload, "input").(type) {
	case string:
		parts = append(parts, inp)
	case []any:
		for _, item := range inp {
			switch it := item.(type) {
			case string:
				parts = append(parts, it)
			case *pyjson.Object:
				parts = append(parts, ContentText(objGet(it, "content")))
				if s, ok := objGet(it, "text").(string); ok {
					parts = append(parts, s)
				}
				if s, ok := objGet(it, "output").(string); ok {
					parts = append(parts, s)
				} else if list, ok := objGet(it, "output").([]any); ok {
					for _, part := range list {
						if po, ok := part.(*pyjson.Object); ok {
							if s, ok := objGet(po, "text").(string); ok {
								parts = append(parts, s)
							}
						}
					}
				}
				if s, ok := objGet(it, "instructions").(string); ok {
					parts = append(parts, s)
				}
				// Codex ChatGPT: tools live under input[].type=additional_tools.
				if pyEqual(objGet(it, "type"), "additional_tools") {
					if tools, ok := objGet(it, "tools").([]any); ok {
						if tt := toolsTextNested(tools); tt != "" {
							parts = append(parts, tt)
						}
					}
				}
			}
		}
	}
	if s, ok := objGet(payload, "instructions").(string); ok {
		parts = append(parts, s)
	}
	if tt := toolsText(objGet(payload, "tools")); tt != "" {
		parts = append(parts, tt)
	}
	return joinNonEmpty(parts, "\n"), nil
}

// MeasureWireTokens mirrors measure_wire_tokens: a BPE estimate of the request
// (or a byte/4 estimate for opaque bodies) plus a sample string for the
// dashboard. Pass payload as the decoded JSON value; non-dicts are ignored,
// as isinstance does.
func MeasureWireTokens(body []byte, payload any) (int, string, error) {
	const sampleCap = 4000
	sample := ""
	text := ""
	if po, ok := payload.(*pyjson.Object); ok {
		t, err := ExtractText(po)
		if err != nil {
			return 0, "", err
		}
		text = t
		if text != "" {
			sample = pyPrefix(text, sampleCap)
		}
	}
	if text == "" && len(body) > 0 {
		text = pyDecodeIgnore(body)
		if sample == "" {
			sample = pyPrefix(text, sampleCap)
		}
	}
	if text != "" {
		n, err := CountTokens(text)
		return n, sample, err
	}
	if len(body) > 0 {
		approx := len(body) / 4
		if approx < 1 {
			approx = 1
		}
		return approx, "[binary " + itoa(len(body)) + "B ~" + itoa(approx) + " tok]", nil
	}
	return 0, "", nil
}

// truncateVerboseLogs mirrors _truncate_verbose_logs: error payloads are kept
// whole, spinner and blank lines are dropped, and long logs keep their head,
// tail and error lines.
func truncateVerboseLogs(text string) (string, error) {
	if IsErrorPayload(text, "", false) {
		return PreserveDiagnosticText(text, StacktraceCeiling), nil
	}
	var lines []string
	for _, l := range strings.Split(text, "\n") {
		if !reNoiseLine.MatchString(l) {
			lines = append(lines, l)
		}
	}
	if len(lines) <= logTruncateMinLines {
		return strings.Join(lines, "\n"), nil
	}

	keep := make([]bool, len(lines))
	for i := 0; i < logHeadLines; i++ {
		keep[i] = true
	}
	for i := len(lines) - logTailLines; i < len(lines); i++ {
		keep[i] = true
	}
	for i, line := range lines {
		if reErrorKeywords.MatchString(line) {
			keep[i] = true
		}
	}

	var out []string
	for i := 0; i < len(lines); {
		if keep[i] {
			out = append(out, lines[i])
			i++
			continue
		}
		start := i
		for i < len(lines) && !keep[i] {
			i++
		}
		out = append(out, "[... Truncated "+itoa(i-start)+" lines of verbose logs ...]")
	}
	return strings.Join(out, "\n"), nil
}

// pythonFenceStub mirrors the first branch of _prune_middle_content for a
// fenced block. A block that matches ```python ... ``` is distilled to its
// public signatures (distillPythonSignatures, pyast.go). The stub replaces the
// block only when it is non-blank and differs from the original code. ok is
// false when Python would fall through to the truncation branches. err is
// ErrUnsupported when the distiller declines the input.
func pythonFenceStub(seg string) (stub string, ok bool, err error) {
	loc := rePythonFence.FindSubmatchIndex(seg)
	if loc == nil {
		return "", false, nil
	}
	code := seg[loc[2]:loc[3]]
	distilled, err := distillPythonSignatures(code)
	if err != nil {
		return "", false, err
	}
	if distilled == "" || pyStrip(distilled) == "" || distilled == code {
		return "", false, nil
	}
	return distilled, true, nil
}

// pruneMiddleContent mirrors _prune_middle_content. Falsy values pass through
// unchanged; a truthy non-str raises TypeError in the regex split.
func pruneMiddleContent(text any, logTruncation bool) (any, error) {
	if !pyTruthy(text) {
		return text, nil
	}
	s, ok := text.(string)
	if !ok {
		return nil, pyError("TypeError", "expected string or bytes-like object")
	}
	segments := splitCaptured(reCodeFenceSplit, s)
	var out strings.Builder
	for i, seg := range segments {
		lines := strings.Split(seg, "\n")
		if i%2 == 1 {
			// Fenced code block, fence lines included.
			stub, stubbed, err := pythonFenceStub(seg)
			if err != nil {
				return nil, err
			}
			if stubbed {
				// Python reference code: the signature stub always wins.
				out.WriteString("```python\n" + stub + "\n```")
				continue
			}
			if logTruncation && len(lines) > logTruncateMinLines {
				t, err := truncateVerboseLogs(seg)
				if err != nil {
					return nil, err
				}
				out.WriteString(t)
			} else if len(lines)-2 >= largeCodeBlockMinLines {
				out.WriteString(codeOmitPlaceholder)
			} else {
				out.WriteString(seg)
			}
		} else {
			// Prose, which may itself be an unfenced log or stacktrace dump.
			if logTruncation && len(lines) > logTruncateMinLines {
				t, err := truncateVerboseLogs(seg)
				if err != nil {
					return nil, err
				}
				out.WriteString(t)
			} else {
				out.WriteString(seg)
			}
		}
	}
	return out.String(), nil
}

// pruneContentValue mirrors _prune_content_value.
func pruneContentValue(value any, logTruncation bool) (any, error) {
	switch v := value.(type) {
	case string:
		return pruneMiddleContent(v, logTruncation)
	case []any:
		out := make([]any, 0, len(v))
		for _, b := range v {
			nb, err := pruneContentBlock(b, logTruncation)
			if err != nil {
				return nil, err
			}
			out = append(out, nb)
		}
		return out, nil
	}
	return value, nil
}

// pruneContentBlock mirrors _prune_content_block. tool_use blocks and anything
// else pass through untouched.
func pruneContentBlock(block any, logTruncation bool) (any, error) {
	bo, ok := block.(*pyjson.Object)
	if !ok {
		return block, nil
	}
	if pyEqual(objGet(bo, "type"), "text") {
		nb := copyObject(bo)
		t, err := pruneMiddleContent(objGetOr(nb, "text", ""), logTruncation)
		if err != nil {
			return nil, err
		}
		nb.Set("text", t)
		return nb, nil
	}
	if pyEqual(objGet(bo, "type"), "tool_result") && hasKey(bo, "content") {
		nb := copyObject(bo)
		c, err := pruneContentValue(objGet(nb, "content"), logTruncation)
		if err != nil {
			return nil, err
		}
		nb.Set("content", c)
		return nb, nil
	}
	return block, nil
}

var exploratoryToolNames = map[string]bool{
	"glob": true, "grep": true, "ls": true, "list_dir": true, "list_files": true,
	"find": true, "search": true, "search_files": true,
}

// exploratoryToolName mirrors _exploratory_tool_name: the lowercase name of a
// single-block assistant tool_use for an exploratory tool, else "".
func exploratoryToolName(msg any) (string, error) {
	mo, ok := msg.(*pyjson.Object)
	if !ok || !pyEqual(objGet(mo, "role"), "assistant") {
		return "", nil
	}
	content, ok := objGet(mo, "content").([]any)
	if !ok || len(content) != 1 {
		return "", nil
	}
	bo, ok := content[0].(*pyjson.Object)
	if !ok || !pyEqual(objGet(bo, "type"), "tool_use") {
		return "", nil
	}
	name, err := lowerOrEmpty(objGet(bo, "name"))
	if err != nil {
		return "", err
	}
	if exploratoryToolNames[name] {
		return name, nil
	}
	return "", nil
}

// squashExploratoryChains mirrors _squash_exploratory_chains: runs of at least
// three exploratory tool_use/tool_result pairs have long results replaced by a
// one-line marker. Only tool_result content changes, so every id still pairs.
func squashExploratoryChains(messages []any) ([]any, error) {
	out := make([]any, len(messages))
	copy(out, messages)
	n := len(out)
	i := 0
	for i < n-1 {
		type run struct {
			resIdx int
			name   string
		}
		var runs []run
		j := i
		for j < n-1 {
			name, err := exploratoryToolName(out[j])
			if err != nil {
				return nil, err
			}
			nxt, isDict := out[j+1].(*pyjson.Object)
			if name == "" || !isDict || !pyEqual(objGet(nxt, "role"), "user") {
				break
			}
			nc, isList := objGet(nxt, "content").([]any)
			if !isList || len(nc) == 0 {
				break
			}
			first, isDict := nc[0].(*pyjson.Object)
			if !isDict || !pyEqual(objGet(first, "type"), "tool_result") {
				break
			}
			runs = append(runs, run{resIdx: j + 1, name: name})
			j += 2
		}
		if len(runs) >= explorationRunMin {
			for _, r := range runs {
				resultMsg := copyObject(out[r.resIdx].(*pyjson.Object))
				blocks := append([]any(nil), objGet(resultMsg, "content").([]any)...)
				block := copyObject(blocks[0].(*pyjson.Object))
				if strings.Count(ContentText(objGet(block, "content")), "\n")+1 < explorationMinResultLns {
					continue
				}
				marker := "[Exploration: " + r.name + " — output condensed]"
				if _, isList := objGet(block, "content").([]any); isList {
					block.Set("content", []any{objectOf("type", "text", "text", marker)})
				} else {
					block.Set("content", marker)
				}
				blocks[0] = block
				resultMsg.Set("content", blocks)
				out[r.resIdx] = resultMsg
			}
			i = j
		} else {
			i++
		}
	}
	return out, nil
}

// PruneHistory mirrors prune_history: keep the first message and the last
// four verbatim, and shrink large code and log content in between. Messages
// are never dropped or reordered, so tool_use/tool_result pairing survives.
func PruneHistory(messages any, logTruncation, turnSquashing bool) (any, error) {
	list, ok := messages.([]any)
	if !ok || len(list) <= historyWindowThreshold {
		return messages, nil
	}
	first := list[0]
	tail := list[len(list)-preserveTailMessages:]
	middle := list[1 : len(list)-preserveTailMessages]

	if turnSquashing {
		var err error
		middle, err = squashExploratoryChains(middle)
		if err != nil {
			return nil, err
		}
	}

	pruned := make([]any, 0, len(middle))
	for _, msg := range middle {
		mo, ok := msg.(*pyjson.Object)
		if !ok || !hasKey(mo, "content") {
			pruned = append(pruned, msg)
			continue
		}
		nm := copyObject(mo)
		c, err := pruneContentValue(objGet(nm, "content"), logTruncation)
		if err != nil {
			return nil, err
		}
		nm.Set("content", c)
		pruned = append(pruned, nm)
	}

	out := make([]any, 0, len(pruned)+len(tail)+1)
	out = append(out, first)
	out = append(out, pruned...)
	return append(out, tail...), nil
}

// countCacheBreakpoints mirrors _count_cache_breakpoints: cache_control markers
// already present anywhere in the payload.
func countCacheBreakpoints(payload *pyjson.Object) (int, error) {
	count := 0
	if system, ok := objGet(payload, "system").([]any); ok {
		for _, b := range system {
			if bo, ok := b.(*pyjson.Object); ok && hasKey(bo, "cache_control") {
				count++
			}
		}
	}
	if tools, ok := objGet(payload, "tools").([]any); ok {
		for _, t := range tools {
			if to, ok := t.(*pyjson.Object); ok && hasKey(to, "cache_control") {
				count++
			}
		}
	}
	msgs, err := pyIterOr(objGet(payload, "messages"))
	if err != nil {
		return 0, err
	}
	for _, msg := range msgs {
		mo, ok := msg.(*pyjson.Object)
		if !ok {
			continue
		}
		if content, ok := objGet(mo, "content").([]any); ok {
			for _, b := range content {
				if bo, ok := b.(*pyjson.Object); ok && hasKey(bo, "cache_control") {
					count++
				}
			}
		}
	}
	return count, nil
}

// ephemeralCache is a fresh {"type": "ephemeral"} marker value.
func ephemeralCache() *pyjson.Object {
	return objectOf("type", "ephemeral")
}

// markLastDict sets cache_control on the last element when it is a dict.
func markLastDict(list []any) bool {
	if len(list) == 0 {
		return false
	}
	last, ok := list[len(list)-1].(*pyjson.Object)
	if !ok {
		return false
	}
	last.Set("cache_control", ephemeralCache())
	return true
}

// ApplyAnthropicCacheAlignment mirrors apply_anthropic_cache_alignment. It
// marks up to four cacheable breakpoints (large system prompt, tool
// definitions, second-to-last message) for Anthropic only, and does nothing
// when the caller already set any cache_control.
func ApplyAnthropicCacheAlignment(payload *pyjson.Object, provider string) (*pyjson.Object, error) {
	if provider != "anthropic" {
		return payload, nil
	}
	n, err := countCacheBreakpoints(payload)
	if err != nil {
		return nil, err
	}
	if n > 0 {
		return payload, nil
	}

	breakpoints := 0
	system := objGet(payload, "system")
	var systemText string
	if s, ok := system.(string); ok {
		systemText = s
	} else {
		systemText = ContentText(system)
	}
	if breakpoints < maxCacheBreakpoints && pyLen(systemText) > cacheSystemCharThreshold {
		switch sys := system.(type) {
		case string:
			payload.Set("system", []any{objectOf("type", "text", "text", sys, "cache_control", ephemeralCache())})
			breakpoints++
		case []any:
			if markLastDict(sys) {
				breakpoints++
			}
		}
	}

	if tools, ok := objGet(payload, "tools").([]any); ok && breakpoints < maxCacheBreakpoints {
		if markLastDict(tools) {
			breakpoints++
		}
	}

	if msgs, ok := objGet(payload, "messages").([]any); ok && breakpoints < maxCacheBreakpoints && len(msgs) >= cacheHistoryDepth {
		if target, ok := msgs[len(msgs)-2].(*pyjson.Object); ok {
			switch c := objGet(target, "content").(type) {
			case string:
				target.Set("content", []any{objectOf("type", "text", "text", c, "cache_control", ephemeralCache())})
			case []any:
				markLastDict(c)
			}
		}
	}
	return payload, nil
}

// truncateHeadTailChars mirrors _truncate_head_tail_chars. Only str input is
// meaningful; list and dict input follow Python's len() and slicing rules, so
// a long list is rendered with its repr and a dict raises.
func truncateHeadTailChars(text any, maxChars int) (any, error) {
	switch t := text.(type) {
	case string:
		if pyLen(t) <= maxChars*2 {
			return t, nil
		}
		omitted := pyLen(t) - maxChars*2
		return pyPrefix(t, maxChars) + "\n... [" + itoa(omitted) + " chars truncated] ...\n" +
			pySuffix(t, maxChars), nil
	case []any:
		if len(t) <= maxChars*2 {
			return t, nil
		}
		omitted := len(t) - maxChars*2
		head := pyRepr(t[:maxChars])
		tail := pyRepr(t[len(t)-maxChars:])
		return head + "\n... [" + itoa(omitted) + " chars truncated] ...\n" + tail, nil
	case *pyjson.Object:
		if t.Len() <= maxChars*2 {
			return t, nil
		}
		return nil, pyError("TypeError", "unhashable type: 'slice'")
	}
	return nil, pyError("TypeError", "object has no len()")
}

// ApplyNewestToolResultCompaction mirrors apply_newest_tool_result_compaction:
// it caps tool_result text in the last user message only (the mutable tail),
// leaving older results alone so the cached prefix stays stable.
func ApplyNewestToolResultCompaction(payload *pyjson.Object) (*pyjson.Object, error) {
	msgs, ok := objGet(payload, "messages").([]any)
	if !ok || len(msgs) == 0 {
		return payload, nil
	}
	last, ok := msgs[len(msgs)-1].(*pyjson.Object)
	if !ok || !pyEqual(objGet(last, "role"), "user") {
		return payload, nil
	}
	content, ok := objGet(last, "content").([]any)
	if !ok {
		return payload, nil
	}
	for _, block := range content {
		bo, ok := block.(*pyjson.Object)
		if !ok || !pyEqual(objGet(bo, "type"), "tool_result") {
			continue
		}
		switch inner := objGet(bo, "content").(type) {
		case string:
			t, err := truncateHeadTailChars(inner, newestToolResultMaxChars)
			if err != nil {
				return nil, err
			}
			bo.Set("content", t)
		case []any:
			for _, b := range inner {
				bb, ok := b.(*pyjson.Object)
				if !ok || !pyEqual(objGet(bb, "type"), "text") {
					continue
				}
				t, err := truncateHeadTailChars(objGetOr(bb, "text", ""), newestToolResultMaxChars)
				if err != nil {
					return nil, err
				}
				bb.Set("text", t)
			}
		}
	}
	return payload, nil
}

var (
	fileReadToolNames = map[string]bool{
		"read": true, "read_file": true, "view": true, "view_file": true,
		"cat": true, "get_file_contents": true,
	}
	filePathInputKeys = []string{"file_path", "path", "target_file", "filepath"}
)

// extractFilePath mirrors _extract_file_path: the path a read-style tool_use
// names. ok is false where Python returns None. A non-dict input raises
// AttributeError once a read tool is seen.
func extractFilePath(block *pyjson.Object) (string, bool, error) {
	name, err := lowerOrEmpty(objGet(block, "name"))
	if err != nil {
		return "", false, err
	}
	if !fileReadToolNames[name] {
		return "", false, nil
	}
	input := objGet(block, "input")
	if !pyTruthy(input) {
		input = pyjson.NewObject()
	}
	io, ok := input.(*pyjson.Object)
	if !ok {
		return "", false, pyError("AttributeError", "object has no attribute 'get'")
	}
	for _, key := range filePathInputKeys {
		if s, ok := objGet(io, key).(string); ok {
			return s, true, nil
		}
	}
	return "", false, nil
}

// fileEntry is one remembered file version: the SHA-256 of the UTF-8 bytes
// and the text itself (for the diff).
type fileEntry struct {
	hash    string
	content string
}

// fileTracker mirrors SessionFileStateTracker. A repeated read of an unchanged
// file collapses to a marker, and a changed file to a unified diff. Each call
// re-walks the whole message list, so the output is a pure function of the
// payload; the per-session map only isolates concurrent conversations. Python
// dict order decides which session is evicted, so the order is kept.
type fileTracker struct {
	mu       sync.Mutex
	sessions map[string]map[string]fileEntry
	order    []string
}

func newFileTracker() *fileTracker {
	return &fileTracker{sessions: map[string]map[string]fileEntry{}}
}

// fileTrackerGlobal is the process-wide tracker (_file_tracker in Python).
var fileTrackerGlobal = newFileTracker()

// replace mirrors _replace: it records the new version and returns what the
// client should see for this read.
func (t *fileTracker) replace(files map[string]fileEntry, path, content string) string {
	newHash := sha256Hex(pyEncodeIgnore(content))
	prev, had := files[path]
	files[path] = fileEntry{hash: newHash, content: content}
	if !had {
		return content
	}
	if prev.hash == newHash {
		return "[FILE UNCHANGED: " + path + "]"
	}
	diff := strings.Join(unifiedDiffLines(pySplitLines(prev.content), pySplitLines(content), path, path), "\n")
	if diff == "" {
		return "[FILE UNCHANGED: " + path + "]"
	}
	return diff
}

// process mirrors SessionFileStateTracker.process: it rewrites tool_result
// content in place for re-reads of the same file in this messages list.
func (t *fileTracker) process(sessionKey string, messages []any) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, exists := t.sessions[sessionKey]; !exists {
		if len(t.sessions) >= maxTrackedSessions && len(t.order) > 0 {
			victim := t.order[0]
			t.order = t.order[1:]
			delete(t.sessions, victim)
		}
		t.order = append(t.order, sessionKey)
	}
	files := map[string]fileEntry{}
	t.sessions[sessionKey] = files

	// pending maps a tool_use id to its file path. Keys follow Python dict
	// identity, so an unhashable id raises TypeError.
	pending := map[string]string{}
	for _, msg := range messages {
		mo, ok := msg.(*pyjson.Object)
		if !ok {
			continue
		}
		content, ok := objGet(mo, "content").([]any)
		if !ok {
			continue
		}
		if pyEqual(objGet(mo, "role"), "assistant") {
			for _, block := range content {
				bo, ok := block.(*pyjson.Object)
				if !ok || !pyEqual(objGet(bo, "type"), "tool_use") {
					continue
				}
				path, hasPath, err := extractFilePath(bo)
				if err != nil {
					return err
				}
				if !hasPath || path == "" {
					continue
				}
				key, hashable := pyKey(objGet(bo, "id"))
				if !hashable {
					return pyError("TypeError", "unhashable type")
				}
				pending[key] = path
			}
		} else if pyEqual(objGet(mo, "role"), "user") {
			for _, block := range content {
				bo, ok := block.(*pyjson.Object)
				if !ok || !pyEqual(objGet(bo, "type"), "tool_result") {
					continue
				}
				lookup, hashable := pyKey(objGet(bo, "tool_use_id"))
				if !hashable {
					return pyError("TypeError", "unhashable type")
				}
				path := pending[lookup]
				if path == "" {
					continue
				}
				text := ContentText(objGet(bo, "content"))
				if text == "" {
					continue
				}
				replacement := t.replace(files, path, text)
				switch objGet(bo, "content").(type) {
				case string:
					bo.Set("content", replacement)
				case []any:
					bo.Set("content", []any{objectOf("type", "text", "text", replacement)})
				}
			}
		}
	}
	return nil
}

// DeriveSessionKey mirrors derive_session_key: a 16-hex-digit fingerprint of
// the system prompt and the first message, which stay constant across turns.
func DeriveSessionKey(payload *pyjson.Object) string {
	var first any
	if list, ok := objGet(payload, "messages").([]any); ok && len(list) > 0 {
		first = list[0]
	}
	// json.dumps(..., sort_keys=True, default=str) with the default ASCII escaping.
	seedBytes, _ := pyjson.Dumps([]any{objGet(payload, "system"), first},
		pyjson.Options{ItemSep: ", ", KeySep: ": ", EnsureASCII: true, SortKeys: true})
	seed := string(seedBytes)
	if len(seed) > 2000 {
		seed = seed[:2000]
	}
	return sha256Hex([]byte(seed))[:16]
}

// ApplyFileDeltaTracking mirrors apply_file_delta_tracking. It rewrites the
// payload in place and returns it.
func ApplyFileDeltaTracking(payload *pyjson.Object, sessionKey string) (*pyjson.Object, error) {
	if msgs, ok := objGet(payload, "messages").([]any); ok {
		if err := fileTrackerGlobal.process(sessionKey, msgs); err != nil {
			return nil, err
		}
	}
	return payload, nil
}

// ClampMaxTokens mirrors clamp_max_tokens: short, plain-text, non-code
// exchanges get a 2048-token cap. The payload is modified in place.
func ClampMaxTokens(payload *pyjson.Object) *pyjson.Object {
	mt := objGet(payload, "max_tokens")
	if !pyIsInt(mt) {
		return payload
	}
	n, _ := pyIntOf(mt)
	if n.Cmp(big.NewInt(maxTokensClampTarget)) <= 0 {
		return payload
	}
	msgs, ok := objGet(payload, "messages").([]any)
	if !ok || len(msgs) == 0 {
		return payload
	}
	last, ok := msgs[len(msgs)-1].(*pyjson.Object)
	if !ok || !pyEqual(objGet(last, "role"), "user") {
		return payload
	}
	text := pyStrip(ContentText(objGet(last, "content")))
	if text == "" || pyLen(text) >= simpleQueryCharThreshold {
		return payload
	}
	for _, marker := range []string{"```", "def ", "class ", "http://", "https://"} {
		if strings.Contains(text, marker) {
			return payload
		}
	}
	payload.Set("max_tokens", int64(maxTokensClampTarget))
	return payload
}

// firstSentence mirrors _first_sentence: cut at the first '.' or newline and
// keep the terminator. Abbreviations are not handled, by design.
func firstSentence(text string) string {
	if text == "" {
		return text
	}
	var head string
	if loc := reSentenceBreak.FindIndex(text); loc != nil {
		head = pyStrip(text[:loc[0]])
	} else {
		head = pyStrip(text)
	}
	if head == "" {
		return pyStrip(text)
	}
	if strings.HasSuffix(head, ".") || strings.HasSuffix(head, "!") || strings.HasSuffix(head, "?") {
		return head
	}
	return head + "."
}

// minifySchemaNode mirrors _minify_schema_node: it shortens description
// strings and leaves structure, types and key order alone.
func minifySchemaNode(node any) any {
	o, ok := node.(*pyjson.Object)
	if !ok {
		return node
	}
	nn := copyObject(o)
	if d, ok := objGet(nn, "description").(string); ok {
		nn.Set("description", firstSentence(d))
	}
	if props, ok := objGet(nn, "properties").(*pyjson.Object); ok {
		np := pyjson.NewObject()
		for _, k := range props.Keys() {
			np.Set(k, minifySchemaNode(objGet(props, k)))
		}
		nn.Set("properties", np)
	}
	if items, ok := objGet(nn, "items").(*pyjson.Object); ok {
		nn.Set("items", minifySchemaNode(items))
	}
	return nn
}

// MinifyToolsDeterministic mirrors minify_tools_deterministic. It returns a
// new tool list; the input is not modified.
func MinifyToolsDeterministic(tools any) any {
	list, ok := tools.([]any)
	if !ok {
		return tools
	}
	out := make([]any, 0, len(list))
	for _, tool := range list {
		to, ok := tool.(*pyjson.Object)
		if !ok {
			out = append(out, tool)
			continue
		}
		nt := copyObject(to)
		if d, ok := objGet(nt, "description").(string); ok {
			nt.Set("description", firstSentence(d))
		}
		if schema, ok := objGet(nt, "input_schema").(*pyjson.Object); ok {
			nt.Set("input_schema", minifySchemaNode(schema))
		}
		// OpenAI and Mistral function tools.
		if fo, ok := objGet(nt, "function").(*pyjson.Object); ok {
			nf := copyObject(fo)
			if d, ok := objGet(nf, "description").(string); ok {
				nf.Set("description", firstSentence(d))
			}
			if params, ok := objGet(nf, "parameters").(*pyjson.Object); ok {
				nf.Set("parameters", minifySchemaNode(params))
			}
			nt.Set("function", nf)
		}
		out = append(out, nt)
	}
	return out
}

// OptimizePayload mirrors optimize_payload. It rewrites the payload in place
// and returns it with the category tags (for the dashboard badge).
func OptimizePayload(payload *pyjson.Object, opts OptimizeOptions) (*pyjson.Object, []string, error) {
	var cats []string

	if opts.Compression {
		if tools, ok := objGet(payload, "tools").([]any); ok && len(tools) > 0 {
			payload.Set("tools", MinifyToolsDeterministic(tools))
			cats = append(cats, "tool_minification")
		}
	}

	// Short prompts never get the anti-yap directive: its own cost can exceed
	// anything a one-line query has to save. Python short-circuits, so the
	// text is only extracted when the switch is on.
	antiYapActive := false
	if opts.AntiYap {
		text, err := ExtractText(payload)
		if err != nil {
			return nil, nil, err
		}
		antiYapActive = pyLen(text) > antiYapMinChars
	}

	if antiYapActive && hasKey(payload, "system") {
		payload.Set("system", InjectAntiYap(objGet(payload, "system")))
		cats = append(cats, "anti_yap")
	}

	if opts.HistoryPruning {
		if list, ok := objGet(payload, "messages").([]any); ok && len(list) > historyWindowThreshold {
			pruned, err := PruneHistory(list, opts.LogTruncation, true)
			if err != nil {
				return nil, nil, err
			}
			payload.Set("messages", pruned)
			cats = append(cats, "pruning")
			if opts.LogTruncation {
				cats = append(cats, "log_truncation")
			}
			for _, m := range pruned.([]any) {
				if mo, ok := m.(*pyjson.Object); ok && strings.Contains(ContentText(objGet(mo, "content")), "[Exploration:") {
					cats = append(cats, "turn_squashing")
					break
				}
			}
		}
	}

	if opts.Compression {
		payload = ClampMaxTokens(payload)
		cats = append(cats, "prose")
	}

	msgs, err := pyIterOr(objGet(payload, "messages"))
	if err != nil {
		return nil, nil, err
	}
	for _, m := range msgs {
		mo, ok := m.(*pyjson.Object)
		if !ok {
			continue
		}
		if pyEqual(objGet(mo, "role"), "system") {
			if antiYapActive {
				mo.Set("content", InjectAntiYap(objGet(mo, "content")))
			}
		} else if opts.Compression {
			c, err := optimizeContent(objGet(mo, "content"))
			if err != nil {
				return nil, nil, err
			}
			mo.Set("content", c)
		}
	}

	if opts.Compression {
		if p, ok := objGet(payload, "prompt").(string); ok {
			payload.Set("prompt", OptimizeText(p))
		}
	}
	return payload, cats, nil
}

// CredentialFingerprint mirrors credential_fingerprint: a short digest of the
// caller's upstream credential headers, so one caller never receives a cached
// response that was produced for another. get returns the value of a
// lowercase header name ("" when absent). Pass header values as Python sees
// them (the proxy decodes header bytes as latin-1).
func CredentialFingerprint(get func(name string) string) string {
	parts := make([]string, 0, 4)
	for _, h := range []string{"authorization", "x-api-key", "x-goog-api-key", "api-key"} {
		parts = append(parts, h+"="+pyStrip(get(h)))
	}
	return sha256Hex([]byte(strings.Join(parts, "\n")))[:32]
}

// ComputeRequestHash mirrors compute_request_hash: a cache key over the whole
// original request plus the caller's credential fingerprint.
func ComputeRequestHash(payload *pyjson.Object, provider, credential string) (string, error) {
	kp := objectOf("provider", provider, "credential", credential, "payload", payload)
	b, err := canonicalJSON(kp)
	if err != nil {
		return "", err
	}
	return sha256Hex(b), nil
}

var reSSEDataLine = reSSEData

// normalizeInt returns int64 when the value fits, else *big.Int.
func normalizeInt(n *big.Int) any {
	if n.IsInt64() {
		return n.Int64()
	}
	return n
}

// ExtractUpstreamUsage mirrors extract_upstream_usage: the provider's own
// usage figures from an SSE stream or a plain JSON body. Usage events are
// merged in order, so Anthropic's message_start cache counts and the final
// message_delta output count both survive.
func ExtractUpstreamUsage(rawBody []byte) (*pyjson.Object, error) {
	text := pyDecodeIgnore(rawBody)
	usage := pyjson.NewObject()
	sawSSE := false

	ingest := func(u any) error {
		uo, ok := u.(*pyjson.Object)
		if !ok {
			return nil
		}
		flat := pyjson.NewObject()
		for _, k := range uo.Keys() {
			v := objGet(uo, k)
			if v == nil {
				continue
			}
			switch v.(type) {
			case *pyjson.Object, []any:
				continue
			}
			flat.Set(k, v)
		}
		// Newer Anthropic: cache_creation: {ephemeral_5m_input_tokens, ...}
		if creation, ok := objGet(uo, "cache_creation").(*pyjson.Object); ok {
			total := big.NewInt(0)
			for _, k := range creation.Keys() {
				v := objGet(creation, k)
				if !pyIsNumber(v) {
					continue
				}
				n, err := pyIntConv(v)
				if err != nil {
					return err
				}
				total.Add(total, n)
			}
			if total.Sign() != 0 && !hasKey(flat, "cache_creation_input_tokens") {
				flat.Set("cache_creation_input_tokens", normalizeInt(total))
			}
		}
		for _, k := range flat.Keys() {
			usage.Set(k, objGet(flat, k))
		}
		return nil
	}

	for _, line := range pySplitLines(text) {
		stripped := pyStrip(line)
		loc := reSSEDataLine.FindSubmatchIndex(stripped)
		if loc == nil {
			continue
		}
		payloadText := stripped[loc[2]:loc[3]]
		if payloadText == "" || payloadText == "[DONE]" {
			continue
		}
		obj, err := loadsStr(payloadText)
		if err != nil {
			continue // json.loads ValueError: skip the line
		}
		sawSSE = true
		if o, ok := obj.(*pyjson.Object); ok {
			if err := ingest(objGet(o, "usage")); err != nil {
				return nil, err
			}
			if msg, ok := objGet(o, "message").(*pyjson.Object); ok {
				if err := ingest(objGet(msg, "usage")); err != nil {
					return nil, err
				}
			}
		}
	}
	if !sawSSE {
		obj, err := loadsStr(text)
		if err != nil {
			return usage, nil
		}
		if o, ok := obj.(*pyjson.Object); ok {
			if err := ingest(objGet(o, "usage")); err != nil && !isPyKind(err, "ValueError") {
				return nil, err
			}
		}
	}
	return usage, nil
}

// loadsStr mirrors json.loads(str). A leading U+FEFF is an error for str input
// (Python rejects the BOM), while pyjson.Loads strips a BOM from bytes, so the
// check is done here.
func loadsStr(s string) (any, error) {
	if strings.HasPrefix(s, "\uFEFF") {
		return nil, pyError("ValueError", "Unexpected UTF-8 BOM")
	}
	return pyjson.Loads([]byte(s))
}
