// Mirrors backend/engines/maxsave.py: lossless-ish tool-output reducers (ANSI
// and progress stripping, JSON and CLI-table compaction, duplicate and stale
// stubs, age elision of old tool results, unused tool schemas, git-workflow
// boilerplate trimming) and the empty-block scrub that runs last.
//
// Values are the JSON shapes Python sees: *pyjson.Object for dict, []any for
// list, string, int64/*big.Int, float64, bool and nil. Where Python calls a
// str method on a value that may not be a str, the port returns an error
// (Python raises). Falsy values pass through, as the Python guards do.
package engines

import (
	"errors"
	"strings"

	"github.com/Optivail/Gatekeep/server/pyjson"
	"github.com/Optivail/Gatekeep/server/pyre"
)

const (
	// keepRecentToolResults: newest tool_result blocks kept full; older ones
	// become microStub when age elision runs.
	keepRecentToolResults = 16
	// liveZoneSkipCCR: newest tool_result blocks skip CCR entirely.
	liveZoneSkipCCR = 12
	microStub       = "·"
	dupStub         = "[dup]"
	bashGitStub     = "Git: commit/push/PR only if user asks. No force, no config edits, " +
		"no hook skips, no amend unless asked. Prefer named git add. HEREDOC messages."
)

var (
	bashNames   = map[string]bool{"Bash": true, "bash": true, "Shell": true, "shell": true, "BashTool": true}
	fileRWNames = map[string]bool{
		"read": true, "read_file": true, "Read": true, "view": true, "view_file": true,
		"Write": true, "write": true, "Edit": true, "edit": true, "cat": true, "get_file_contents": true,
	}

	reANSI         = pyre.MustCompile(`\x1b\[[0-9;]*[A-Za-z]|\x1b\].*?\x07|\r`)
	reProgress     = pyre.MustCompile(`(?im)^.*(\d+%|eta\s|\belapsed\b|[⠀-⣿]{3,}|[=-]{3,}>|downloading|extracting|resolving|fetching packages).*$`)
	reBlankRun     = pyre.MustCompile(`\n{3,}`)
	reTrailWS      = pyre.MustCompile(`[ \t]+\n`)
	reBase64Blob   = pyre.MustCompile(`(?:data:[^;]+;base64,)?[A-Za-z0-9+/]{400,}={0,2}`)
	reCLIPadded    = pyre.MustCompile(`\S  +\S  +\S`)
	reMultiSpace   = pyre.MustCompile(`  +`)
	reSysReminder  = pyre.MustCompile(`(?is)<system-reminder>.*?</system-reminder>`)
	reBashGitWall  = pyre.MustCompile(`(?is)#\s*Committing changes with git\b.*?(?=#\s*Creating pull requests|\Z)`)
	reBashPRWall   = pyre.MustCompile(`(?is)#\s*Creating pull requests\b.*?(?=#\s*Other common operations|\Z)`)
	reBashOtherGH  = pyre.MustCompile(`(?is)#\s*Other common operations\b.*`)
	reGitSafetyInl = pyre.MustCompile(`(?is)Git Safety Protocol:.*?(?=\n\n|\n# |\Z)`)
	reNewlines2    = pyre.MustCompile(`\n{2,}`)

	// gitInstructionPatterns are the Claude Code git-workflow walls removed
	// from system text (strip_git_instruction_bloat).
	gitInstructionPatterns = []*pyre.Regexp{
		pyre.MustCompile(`\n{0,2}#?\s*Committing changes with git\b.*?(?=\n# |\n## |\n[A-Z][a-z].{20,}|\Z)`, pyre.IgnoreCase|pyre.DotAll),
		pyre.MustCompile(`\n{0,2}#?\s*Creating pull requests\b.*?(?=\n# |\n## |\Z)`, pyre.IgnoreCase|pyre.DotAll),
		pyre.MustCompile(`\n{0,2}IMPORTANT:.*?(?:git commit|HEREDOC).*?(?=\n\n|\Z)`, pyre.IgnoreCase|pyre.DotAll),
	}
)

// resultLoc is a (message index, block index) pair for a tool_result block.
type resultLoc struct{ mi, bi int }

// hasLoneSurrogate reports whether s holds an encoded surrogate code point.
// Python str can hold one (JSON "\ud800" produces it); Go strings carry it as
// WTF-8, and UTF-8 encoding of such a string is what Python rejects.
func hasLoneSurrogate(s string) bool {
	for i := 0; i < len(s); i++ {
		if isSurrogateAt(s, i) {
			return true
		}
	}
	return false
}

// encodeUTF8Strict mirrors str.encode("utf-8"). A lone surrogate makes Python
// raise, and this port declines the input rather than guess.
func encodeUTF8Strict(s string) ([]byte, error) {
	if hasLoneSurrogate(s) {
		return nil, ErrUnsupported
	}
	return []byte(s), nil
}

// encodeUTF8Replace mirrors str.encode("utf-8", errors="replace"): each lone
// surrogate becomes one '?'. The engines hash this form for dedup keys.
func encodeUTF8Replace(s string) []byte {
	out := make([]byte, 0, len(s))
	for i := 0; i < len(s); {
		if isSurrogateAt(s, i) {
			out = append(out, '?')
			i += 3
			continue
		}
		out = append(out, s[i])
		i++
	}
	return out
}

// dumpsCompact mirrors json.dumps(v, ensure_ascii=False, separators=(",",":")).
// CPython returns a str even for lone surrogates; pyjson reports them as
// ErrLoneSurrogate, so this port declines.
func dumpsCompact(v any) ([]byte, error) {
	b, err := pyjson.Dumps(v, pyjson.Compact)
	if errors.Is(err, pyjson.ErrLoneSurrogate) {
		return nil, ErrUnsupported
	}
	return b, err
}

// jsonLoadsValue mirrors json.loads(str). ok is false where Python raises
// ValueError (the engines catch it). Nesting past CPython's recursion limit
// raises RecursionError instead, which those catches do not cover, so the
// port declines there.
func jsonLoadsValue(s string) (any, bool, error) {
	v, err := loadsStr(s)
	if err == nil {
		return v, true, nil
	}
	if strings.Contains(err.Error(), "maximum recursion depth") {
		return nil, false, ErrUnsupported
	}
	return nil, false, nil
}

// truncHeadTail mirrors _truncate_head_tail_chars for a str.
func truncHeadTail(s string, maxChars int) string {
	v, _ := truncateHeadTailChars(s, maxChars) // a str never raises here
	return v.(string)
}

// pyInStrSet mirrors `v in frozenset_of_str`. A list or dict is unhashable,
// so Python raises TypeError; any other non-str value is simply not a member.
func pyInStrSet(v any, set map[string]bool) (bool, error) {
	switch t := v.(type) {
	case string:
		return set[t], nil
	case []any, *pyjson.Object:
		return false, pyError("TypeError", "unhashable type")
	}
	return false, nil
}

// typeIs reports whether block["type"] equals one of names (dict `in` tuple).
func typeIs(v any, names ...string) bool {
	s, ok := v.(string)
	if !ok {
		return false
	}
	for _, n := range names {
		if s == n {
			return true
		}
	}
	return false
}

// StripANSI mirrors strip_ansi.
func StripANSI(text string) string {
	if text == "" || (!strings.Contains(text, "\x1b") && !strings.Contains(text, "\r")) {
		return text
	}
	return reANSI.Sub("", text)
}

// SqueezeWS mirrors squeeze_ws.
func SqueezeWS(text string) string {
	if text == "" {
		return text
	}
	text = reTrailWS.Sub("\n", text)
	text = reBlankRun.Sub("\n\n", text)
	if strings.HasSuffix(text, "\n") || strings.HasPrefix(text, "\n") {
		return pyStrip(text)
	}
	return text
}

// StripBase64Blobs mirrors strip_base64_blobs.
func StripBase64Blobs(text string) string {
	if text == "" || pyLen(text) < 400 {
		return text
	}
	return reBase64Blob.Sub("[b64]", text)
}

// DropProgressLines mirrors drop_progress_lines. Nothing is dropped when that
// would remove more than two lines, so a mostly-progress log stays intact.
func DropProgressLines(text string) string {
	if text == "" || strings.Count(text, "\n") < 6 {
		return text
	}
	lines := strings.Split(text, "\n")
	kept := make([]string, 0, len(lines))
	for _, l := range lines {
		if !reProgress.MatchString(l) {
			kept = append(kept, l)
		}
	}
	if len(kept) >= len(lines)-2 {
		return text
	}
	return strings.Join(kept, "\n")
}

// CompactJSONText mirrors compact_json_text: JSON is minified, and a uniform
// array of objects becomes a "[n=..]" header plus pipe-separated rows when that
// is shorter.
func CompactJSONText(text string) (string, error) {
	s := pyStrip(text)
	if pyLen(s) < 80 || (s[0] != '{' && s[0] != '[') {
		return text, nil
	}
	data, ok, err := jsonLoadsValue(s)
	if err != nil {
		return "", err
	}
	if !ok {
		return text, nil
	}

	if list, isList := data.([]any); isList && len(list) > 0 && allObjects(list) {
		first := list[0].(*pyjson.Object)
		keys := first.Keys()
		if len(keys) > 0 && allSameKeys(list, keys) {
			rows := []string{strings.Join(keys, "|")}
			for _, item := range list {
				x := item.(*pyjson.Object)
				cells := make([]string, 0, len(keys))
				for _, k := range keys {
					c, err := compactCell(objGet(x, k))
					if err != nil {
						return "", err
					}
					cells = append(cells, c)
				}
				rows = append(rows, strings.Join(cells, "|"))
			}
			toon := "[n=" + itoa(len(list)) + "]\n" + strings.Join(rows, "\n")
			compact, err := dumpsCompact(data)
			if err != nil {
				return "", err
			}
			if pyLen(toon) < pyLen(string(compact)) {
				return toon, nil
			}
			return string(compact), nil
		}
	}

	compact, err := dumpsCompact(data)
	if err != nil {
		return "", err
	}
	if pyLen(string(compact)) < pyLen(s) {
		return string(compact), nil
	}
	return text, nil
}

// compactCell mirrors the cell() helper in compact_json_text: None is empty,
// containers are compact JSON, and anything else is str() with "|" escaped and
// newlines folded to spaces.
func compactCell(v any) (string, error) {
	switch v.(type) {
	case nil:
		return "", nil
	case []any, *pyjson.Object:
		b, err := dumpsCompact(v)
		if err != nil {
			return "", err
		}
		return string(b), nil
	}
	s := pyStr(v)
	s = strings.ReplaceAll(s, "|", `\|`)
	return strings.ReplaceAll(s, "\n", " "), nil
}

// allObjects reports whether every list item is a dict.
func allObjects(list []any) bool {
	for _, x := range list {
		if _, ok := x.(*pyjson.Object); !ok {
			return false
		}
	}
	return true
}

// allSameKeys reports whether every dict has exactly keys, in the same order
// (list(x.keys()) == keys).
func allSameKeys(list []any, keys []string) bool {
	for _, x := range list {
		k := x.(*pyjson.Object).Keys()
		if len(k) != len(keys) {
			return false
		}
		for i := range k {
			if k[i] != keys[i] {
				return false
			}
		}
	}
	return true
}

// CrushCLITables mirrors crush_cli_tables: wide space-padded tables (docker
// ps, kubectl, ls -l) collapse to tab-separated cells.
func CrushCLITables(text string) string {
	if text == "" || strings.Count(text, "\n") < 4 {
		return text
	}
	lines := strings.Split(text, "\n")
	padded := 0
	for _, l := range lines {
		if reCLIPadded.MatchString(l) {
			padded++
		}
	}
	threshold := len(lines) / 3
	if threshold < 4 {
		threshold = 4
	}
	if padded < threshold {
		return text
	}
	out := make([]string, len(lines))
	for i, l := range lines {
		stripped := pyStrip(l)
		if stripped != "" {
			out[i] = reMultiSpace.Sub("\t", stripped)
		} else {
			out[i] = l
		}
	}
	return strings.Join(out, "\n")
}

// PreprocessToolText mirrors preprocess_tool_text for a str. Each step is
// lossless for the model's purposes; the result falls back to the input when
// the steps empty it.
func PreprocessToolText(text string) (string, error) {
	if text == "" {
		return text, nil
	}
	before := text
	text = StripANSI(text)
	text = StripBase64Blobs(text)
	text = DropProgressLines(text)
	text = CrushCLITables(text)
	text, err := CompactJSONText(text)
	if err != nil {
		return "", err
	}
	text = SqueezeWS(text)
	if text != "" {
		return text, nil
	}
	return before, nil
}

// preprocessToolValue applies preprocess_tool_text to a dynamic value. Falsy
// values pass through. A truthy non-str raises: strip_ansi and
// compact_json_text call str methods on it.
func preprocessToolValue(v any) (any, error) {
	if !pyTruthy(v) {
		return v, nil
	}
	s, ok := v.(string)
	if !ok {
		return nil, pyError("AttributeError", "preprocess_tool_text expects str")
	}
	return PreprocessToolText(s)
}

// mapText mirrors _map_text: str content goes to mapper; list content maps the
// text blocks and replaces image, document and file blocks with a micro stub.
// Other content is returned as is.
func mapText(content any, mapper func(any) (any, error)) (any, error) {
	switch c := content.(type) {
	case string:
		return mapper(c)
	case []any:
		out := make([]any, 0, len(c))
		for _, b := range c {
			bo, isObj := b.(*pyjson.Object)
			switch {
			case isObj && pyEqual(objGet(bo, "type"), "text"):
				nb := copyObject(bo)
				v, err := mapper(objGetOr(bo, "text", ""))
				if err != nil {
					return nil, err
				}
				nb.Set("text", v)
				out = append(out, nb)
			case isObj && typeIs(objGet(bo, "type"), "image", "document", "file"):
				out = append(out, objectOf("type", "text", "text", microStub))
			default:
				out = append(out, b)
			}
		}
		return out, nil
	}
	return content, nil
}

// ApplyToolResultPreprocess mirrors apply_tool_result_preprocess: every user
// tool_result of at least 40 characters is run through preprocess_tool_text.
func ApplyToolResultPreprocess(payload *pyjson.Object) (bool, error) {
	messages, ok := objGet(payload, "messages").([]any)
	if !ok {
		return false, nil
	}
	changed := false
	for _, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok || !pyEqual(objGet(msg, "role"), "user") {
			continue
		}
		content, ok := objGet(msg, "content").([]any)
		if !ok {
			continue
		}
		for _, b := range content {
			block, ok := b.(*pyjson.Object)
			if !ok || !pyEqual(objGet(block, "type"), "tool_result") {
				continue
			}
			if IsCCRFulfilled(block) {
				continue
			}
			before := ContentText(objGet(block, "content"))
			if pyLen(before) < 40 || before == microStub || before == dupStub || strings.HasPrefix(before, "[") {
				continue
			}
			nv, err := mapText(objGet(block, "content"), preprocessToolValue)
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

// ApplyDuplicateToolResultStubs mirrors apply_duplicate_tool_result_stubs: an
// identical tool_result body after its first occurrence becomes dupStub. The
// key is the full SHA-256 of the body, encoded with errors="replace".
func ApplyDuplicateToolResultStubs(payload *pyjson.Object) bool {
	messages, ok := objGet(payload, "messages").([]any)
	if !ok {
		return false
	}
	seen := map[string]bool{}
	changed := false
	for _, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok || !pyEqual(objGet(msg, "role"), "user") {
			continue
		}
		content, ok := objGet(msg, "content").([]any)
		if !ok {
			continue
		}
		for i, b := range content {
			block, ok := b.(*pyjson.Object)
			if !ok || !pyEqual(objGet(block, "type"), "tool_result") {
				continue
			}
			text := ContentText(objGet(block, "content"))
			if pyLen(text) < 80 || text == microStub || text == dupStub {
				continue
			}
			if strings.HasPrefix(text, "[STALE") || strings.HasPrefix(text, "[CCR") || strings.HasPrefix(text, "[FILE") {
				continue
			}
			h := sha256Hex(encodeUTF8Replace(text))
			if seen[h] {
				nb := copyObject(block)
				nb.Set("content", dupStub)
				content[i] = nb
				changed = true
			} else {
				seen[h] = true
			}
		}
		if changed {
			msg.Set("content", content)
		}
	}
	return changed
}

// iterToolResultLocs mirrors _iter_tool_result_locs: every tool_result block in
// a user message, in order.
func iterToolResultLocs(messages []any) []resultLoc {
	var locs []resultLoc
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
			if block, ok := b.(*pyjson.Object); ok && pyEqual(objGet(block, "type"), "tool_result") {
				locs = append(locs, resultLoc{mi, bi})
			}
		}
	}
	return locs
}

// liveZoneResultLocs mirrors live_zone_result_locs: the newest
// liveZoneSkipCCR tool_result blocks.
func liveZoneResultLocs(messages []any) map[resultLoc]bool {
	locs := iterToolResultLocs(messages)
	out := map[resultLoc]bool{}
	for _, l := range lastN(locs, liveZoneSkipCCR) {
		out[l] = true
	}
	return out
}

// lastN mirrors xs[-n:].
func lastN(xs []resultLoc, n int) []resultLoc {
	if len(xs) <= n {
		return xs
	}
	return xs[len(xs)-n:]
}

// toolUseNames mirrors _tool_use_name_by_id: the name of each assistant
// tool_use, keyed by its id. A list or dict id is unhashable, so Python raises.
func toolUseNames(messages []any) (map[string]any, error) {
	names := map[string]any{}
	for _, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok || !pyEqual(objGet(msg, "role"), "assistant") {
			continue
		}
		content, ok := objGet(msg, "content").([]any)
		if !ok {
			continue
		}
		for _, b := range content {
			block, ok := b.(*pyjson.Object)
			if !ok || !pyEqual(objGet(block, "type"), "tool_use") || !pyTruthy(objGet(block, "id")) {
				continue
			}
			key, hashable := pyKey(objGet(block, "id"))
			if !hashable {
				return nil, pyError("TypeError", "unhashable type")
			}
			name := objGet(block, "name")
			if !pyTruthy(name) {
				name = ""
			}
			names[key] = name
		}
	}
	return names, nil
}

// unresolvedErrorLocs mirrors _unresolved_error_locs: a tool_result that
// reports an error and has no later successful result stays protected.
func unresolvedErrorLocs(messages []any) map[resultLoc]bool {
	locs := iterToolResultLocs(messages)
	protected := map[resultLoc]bool{}
	for idx, loc := range locs {
		block := blockAt(messages, loc)
		text := ContentText(objGet(block, "content"))
		if !IsErrorPayload(text, "", pyTruthy(objGet(block, "is_error"))) {
			continue
		}
		laterSuccess := false
		for _, loc2 := range locs[idx+1:] {
			block2 := blockAt(messages, loc2)
			text2 := ContentText(objGet(block2, "content"))
			if !IsErrorPayload(text2, "", pyTruthy(objGet(block2, "is_error"))) {
				laterSuccess = true
				break
			}
		}
		if !laterSuccess {
			protected[loc] = true
		}
	}
	return protected
}

// blockAt returns the tool_result block at loc. iterToolResultLocs has already
// checked that the message and the block are dicts.
func blockAt(messages []any, loc resultLoc) *pyjson.Object {
	return contentList(messages[loc.mi].(*pyjson.Object))[loc.bi].(*pyjson.Object)
}

// contentList returns msg["content"] for a message that iterToolResultLocs
// has already checked to hold a list.
func contentList(msg *pyjson.Object) []any {
	c, _ := objGet(msg, "content").([]any)
	return c
}

// ApplyAgeElide mirrors apply_age_elide: tool_results older than the newest
// keepRecentToolResults become microStub, and the matching tool_use inputs
// become {}. Errors with no later success stay visible. One-way: once stubbed,
// a result is never restored.
func ApplyAgeElide(payload *pyjson.Object, relax bool) (bool, error) {
	if relax {
		return false, nil
	}
	messages, ok := objGet(payload, "messages").([]any)
	if !ok {
		return false, nil
	}
	locs := iterToolResultLocs(messages)
	if len(locs) <= keepRecentToolResults {
		return false, nil
	}

	keep := map[resultLoc]bool{}
	for _, l := range lastN(locs, keepRecentToolResults) {
		keep[l] = true
	}
	names, err := toolUseNames(messages)
	if err != nil {
		return false, err
	}
	sticky := unresolvedErrorLocs(messages)
	// Python's live set is lastN(locs, max(16, 12)), which is the keep set
	// itself, so the live-only checks below never fire. They are kept to
	// match the source line for line.
	live := keep
	elideIDs := map[string]bool{}
	changed := false
	for _, loc := range locs {
		if keep[loc] || sticky[loc] {
			continue
		}
		msg := messages[loc.mi].(*pyjson.Object)
		content := contentList(msg)
		block := content[loc.bi].(*pyjson.Object)
		if IsCCRFulfilled(block) {
			continue
		}
		tid := objGet(block, "tool_use_id")
		lookup := tid
		if !pyTruthy(lookup) {
			lookup = ""
		}
		key, hashable := pyKey(lookup)
		if !hashable {
			return changed, pyError("TypeError", "unhashable type")
		}
		tname := names[key]
		if live[loc] {
			if isBash, err := pyInStrSet(tname, bashNames); err != nil {
				return changed, err
			} else if isBash {
				continue
			}
			if isFile, err := pyInStrSet(tname, fileRWNames); err != nil {
				return changed, err
			} else if isFile {
				continue
			}
		}
		if pyTruthy(tid) {
			tk, _ := pyKey(tid)
			elideIDs[tk] = true
		}
		if ContentText(objGet(block, "content")) == microStub {
			continue
		}
		newContent := append([]any(nil), content...)
		nb := copyObject(block)
		nb.Set("content", microStub)
		newContent[loc.bi] = nb
		msg.Set("content", newContent)
		changed = true
	}

	if len(elideIDs) == 0 {
		return changed, nil
	}

	for _, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok || !pyEqual(objGet(msg, "role"), "assistant") {
			continue
		}
		content, ok := objGet(msg, "content").([]any)
		if !ok {
			continue
		}
		newContent := make([]any, 0, len(content))
		local := false
		for _, b := range content {
			block, isObj := b.(*pyjson.Object)
			if isObj && pyEqual(objGet(block, "type"), "tool_use") {
				idKey, hashable := pyKey(objGet(block, "id"))
				if !hashable {
					return changed, pyError("TypeError", "unhashable type")
				}
				if elideIDs[idKey] {
					inp := objGet(block, "input")
					if !isEmptyDict(inp) {
						nb := copyObject(block)
						nb.Set("input", pyjson.NewObject())
						b = nb
						local = true
						changed = true
					}
				}
			}
			newContent = append(newContent, b)
		}
		if local {
			msg.Set("content", newContent)
		}
	}
	return changed, nil
}

// isEmptyDict reports whether v == {} in Python (only an empty dict is equal).
func isEmptyDict(v any) bool {
	o, ok := v.(*pyjson.Object)
	return ok && o.Len() == 0
}

// DropUnusedToolSchemas mirrors drop_unused_tool_schemas: tools never called in
// the history lose their schema and get a short description and
// defer_loading. The retrieve tool is always kept.
func DropUnusedToolSchemas(payload *pyjson.Object) (bool, error) {
	tools, okTools := objGet(payload, "tools").([]any)
	messages, okMsgs := objGet(payload, "messages").([]any)
	if !okTools || !okMsgs {
		return false, nil
	}
	used := map[string]bool{}
	for _, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok || !pyEqual(objGet(msg, "role"), "assistant") {
			continue
		}
		content, ok := objGet(msg, "content").([]any)
		if !ok {
			continue
		}
		for _, b := range content {
			block, ok := b.(*pyjson.Object)
			if !ok || !pyEqual(objGet(block, "type"), "tool_use") || !pyTruthy(objGet(block, "name")) {
				continue
			}
			key, hashable := pyKey(objGet(block, "name"))
			if !hashable {
				return false, pyError("TypeError", "unhashable type")
			}
			used[key] = true
		}
	}
	if len(used) == 0 {
		// First turns: defer_loading already handles overflow.
		return false, nil
	}

	changed := false
	out := make([]any, 0, len(tools))
	for _, t := range tools {
		tool, ok := t.(*pyjson.Object)
		if !ok {
			out = append(out, t)
			continue
		}
		nameVal := objGet(tool, "name")
		if !pyTruthy(nameVal) {
			nameVal = ""
		}
		nameKey, hashable := pyKey(nameVal)
		if !hashable {
			return false, pyError("TypeError", "unhashable type")
		}
		if used[nameKey] {
			out = append(out, tool)
			continue
		}
		// `name.endswith` and `name.lower()` run only when the name is unused,
		// so a non-str name fails here, as in Python.
		name, isStr := nameVal.(string)
		if !isStr {
			return false, pyError("AttributeError", "name is not a str")
		}
		if strings.HasSuffix(name, "ccr_retrieve") || strings.Contains(pyLower(name), "retrieve") {
			out = append(out, tool)
			continue
		}
		t2 := copyObject(tool)
		if d, isStr := objGet(t2, "description").(string); isStr && pyLen(d) > 24 {
			t2.Set("description", pyPrefix(d, 21)+"…")
			changed = true
		}
		if schema, ok := objGet(t2, "input_schema").(*pyjson.Object); ok && pyTruthy(objGet(schema, "properties")) {
			t2.Set("input_schema", objectOf("type", "object", "properties", pyjson.NewObject()))
			changed = true
		}
		t2.Set("defer_loading", true)
		out = append(out, t2)
	}
	payload.Set("tools", out)
	return changed, nil
}

// stripGitScrub mirrors the scrub() closure in strip_git_instruction_bloat for
// a str. The original is returned unchanged when scrubbing would empty it.
func stripGitScrub(text string) (string, bool) {
	if text == "" || pyLen(text) < 400 {
		return text, false
	}
	original := text
	for _, pat := range gitInstructionPatterns {
		text = pat.Sub("\n", text)
	}
	text = pyStrip(reBlankRun.Sub("\n\n", text))
	if text == "" {
		return original, false
	}
	return text, text != original
}

// StripGitInstructionBloat mirrors strip_git_instruction_bloat. It returns the
// new system value and whether anything changed. In list form, a text block
// whose text is not a str raises, because the Python scrub calls str methods
// on it.
func StripGitInstructionBloat(system any) (any, bool, error) {
	switch s := system.(type) {
	case string:
		t, ch := stripGitScrub(s)
		return t, ch, nil
	case []any:
		changed := false
		out := make([]any, 0, len(s))
		for _, b := range s {
			block, isObj := b.(*pyjson.Object)
			if !isObj || !pyEqual(objGet(block, "type"), "text") {
				out = append(out, b)
				continue
			}
			tv := objGetOr(block, "text", "")
			text, isStr := tv.(string)
			if !isStr {
				return nil, false, pyError("AttributeError", "text block is not a str")
			}
			newT, ch := stripGitScrub(text)
			if ch {
				nb := copyObject(block)
				nb.Set("text", newT)
				block = nb
				changed = true
			}
			if pyStrip(newT) != "" {
				out = append(out, block)
			}
		}
		if len(out) > 0 {
			return out, changed, nil
		}
		return system, changed, nil
	}
	return system, false, nil
}

// StripSystemReminders mirrors strip_system_reminders for a str: the
// <system-reminder> blocks are removed and the whitespace squeezed.
func StripSystemReminders(text string) string {
	if text == "" || !strings.Contains(pyLower(text), "<system-reminder>") {
		return text
	}
	out := reSysReminder.Sub("", text)
	if out != text {
		return SqueezeWS(out)
	}
	return text
}

// stripSystemRemindersValue applies StripSystemReminders to a dynamic value.
// Falsy values pass through; a truthy non-str raises at .lower().
func stripSystemRemindersValue(v any) (any, error) {
	if !pyTruthy(v) {
		return v, nil
	}
	s, ok := v.(string)
	if !ok {
		return nil, pyError("AttributeError", "lower")
	}
	return StripSystemReminders(s), nil
}

// ShrinkBashGitBloat mirrors shrink_bash_git_bloat: the Bash-tool git and PR
// instructions collapse to a one-line stub.
func ShrinkBashGitBloat(text string) string {
	if text == "" || !strings.Contains(text, "Committing changes with git") {
		return text
	}
	text = reBashGitWall.Sub("\n"+bashGitStub+"\n", text)
	text = reBashPRWall.Sub("\n", text)
	text = reBashOtherGH.Sub("\n", text)
	text = reGitSafetyInl.Sub("", text)
	return SqueezeWS(text)
}

// ApplyMessageReminderStrip mirrors apply_message_reminder_strip: system
// reminders are removed from message text and from tool_result text.
func ApplyMessageReminderStrip(payload *pyjson.Object) (bool, error) {
	messages, ok := objGet(payload, "messages").([]any)
	if !ok {
		return false, nil
	}
	changed := false
	for _, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok {
			continue
		}
		content := objGet(msg, "content")
		if s, isStr := content.(string); isStr {
			nv, err := stripSystemRemindersValue(s)
			if err != nil {
				return changed, err
			}
			if nv.(string) != s {
				msg.Set("content", nv)
				changed = true
			}
			continue
		}
		list, isList := content.([]any)
		if !isList {
			continue
		}
		for i, b := range list {
			block, isObj := b.(*pyjson.Object)
			if !isObj {
				continue
			}
			switch {
			case pyEqual(objGet(block, "type"), "text") && isStringValue(objGet(block, "text")):
				text := objGet(block, "text").(string)
				nv := StripSystemReminders(text)
				if nv != text {
					nb := copyObject(block)
					nb.Set("text", nv)
					list[i] = nb
					changed = true
				}
			case pyEqual(objGet(block, "type"), "tool_result"):
				before := ContentText(objGet(block, "content"))
				mapped, err := mapText(objGet(block, "content"), stripSystemRemindersValue)
				if err != nil {
					return changed, err
				}
				if ContentText(mapped) != before {
					nb := copyObject(block)
					nb.Set("content", mapped)
					list[i] = nb
					changed = true
				}
			}
		}
		if changed {
			msg.Set("content", list)
		}
	}
	return changed, nil
}

// isStringValue reports whether v is a str.
func isStringValue(v any) bool {
	_, ok := v.(string)
	return ok
}

// ShrinkToolDescriptionWalls mirrors shrink_tool_description_walls. Long tool
// descriptions lose their git walls, and a long tail after the first paragraph
// is dropped. The tools slice is updated in place, as the Python list is.
func ShrinkToolDescriptionWalls(tools []any) bool {
	changed := false
	for i, t := range tools {
		tool, ok := t.(*pyjson.Object)
		if !ok {
			continue
		}
		desc, isStr := objGet(tool, "description").(string)
		if !isStr || pyLen(desc) < 200 {
			continue
		}
		newDesc := ShrinkBashGitBloat(desc)
		if pyLen(newDesc) > 400 {
			parts := reNewlines2.Split(newDesc, 1)
			if len(parts) == 2 && pyLen(parts[1]) > 300 {
				newDesc = pyStrip(parts[0])
			}
		}
		if newDesc != desc {
			nt := copyObject(tool)
			nt.Set("description", newDesc)
			tools[i] = nt
			changed = true
		}
	}
	return changed
}

// ApplyMaxsave mirrors apply_maxsave, the early pass that runs before CCR.
// losslessOnly skips the passes that change wording (reminder, git and
// duplicate stubs) and keeps only the CLI and JSON compaction.
func ApplyMaxsave(payload *pyjson.Object, losslessOnly bool) ([]string, error) {
	var cats []string
	if !losslessOnly {
		ch, err := ApplyMessageReminderStrip(payload)
		if err != nil {
			return cats, err
		}
		if ch {
			cats = append(cats, "reminder_strip")
		}
		if tools, ok := objGet(payload, "tools").([]any); ok && ShrinkToolDescriptionWalls(tools) {
			cats = append(cats, "git_trim")
		}
	}
	ch, err := ApplyToolResultPreprocess(payload)
	if err != nil {
		return cats, err
	}
	if ch {
		cats = append(cats, "cli_toon")
	}
	if !losslessOnly {
		if ApplyDuplicateToolResultStubs(payload) {
			cats = append(cats, "dup_stub")
		}
		if hasKey(payload, "system") {
			newSys, ch, err := StripGitInstructionBloat(objGet(payload, "system"))
			if err != nil {
				return cats, err
			}
			if ch {
				payload.Set("system", newSys)
				if !containsStr(cats, "git_trim") {
					cats = append(cats, "git_trim")
				}
			}
		}
	}
	return cats, nil
}

// cleanSystemForDrop mirrors clean_system in drop_empty_text_blocks: an empty
// string system becomes " ", and empty text blocks are dropped from a list.
func cleanSystemForDrop(system any) any {
	switch s := system.(type) {
	case string:
		if pyStrip(s) == "" {
			return " "
		}
		return s
	case []any:
		out := make([]any, 0, len(s))
		for _, b := range s {
			block, isObj := b.(*pyjson.Object)
			if isObj && pyEqual(objGet(block, "type"), "text") {
				if pyStrip(pyStr(textOrEmpty(objGet(block, "text")))) == "" {
					continue
				}
			}
			out = append(out, b)
		}
		if len(out) > 0 {
			return out
		}
		return system
	}
	return system
}

// textOrEmpty mirrors `x or ""`.
func textOrEmpty(v any) any {
	if !pyTruthy(v) {
		return ""
	}
	return v
}

// DropEmptyTextBlocks mirrors drop_empty_text_blocks. Anthropic rejects empty
// text blocks, so they are removed after the aggressive transforms. A user
// message left with no blocks gets a single ".". The Python code calls .strip()
// on a non-str text inside a tool_result list, and that raises here too.
func DropEmptyTextBlocks(payload *pyjson.Object) error {
	if hasKey(payload, "system") {
		payload.Set("system", cleanSystemForDrop(objGet(payload, "system")))
	}

	messages, ok := objGet(payload, "messages").([]any)
	if !ok {
		return nil
	}
	for _, m := range messages {
		msg, ok := m.(*pyjson.Object)
		if !ok {
			continue
		}
		content := objGet(msg, "content")
		if s, isStr := content.(string); isStr {
			if pyStrip(s) == "" {
				msg.Set("content", ".")
			}
			continue
		}
		list, isList := content.([]any)
		if !isList {
			continue
		}
		newBlocks := make([]any, 0, len(list))
		for _, b := range list {
			block, isObj := b.(*pyjson.Object)
			switch {
			case isObj && pyEqual(objGet(block, "type"), "text"):
				if pyStrip(pyStr(textOrEmpty(objGet(block, "text")))) == "" {
					continue
				}
				newBlocks = append(newBlocks, block)
			case isObj && pyEqual(objGet(block, "type"), "tool_result"):
				c := objGet(block, "content")
				if s, isStr := c.(string); isStr && pyStrip(s) == "" {
					nb := copyObject(block)
					nb.Set("content", microStub)
					block = nb
				} else if cl, isList := c.([]any); isList {
					nl := make([]any, 0, len(cl))
					for _, cb := range cl {
						if co, isObj := cb.(*pyjson.Object); isObj && pyEqual(objGet(co, "type"), "text") {
							tv, err := textStripped(objGet(co, "text"))
							if err != nil {
								return err
							}
							if tv == "" {
								continue
							}
						}
						nl = append(nl, cb)
					}
					nb := copyObject(block)
					if len(nl) > 0 {
						nb.Set("content", nl)
					} else {
						nb.Set("content", microStub)
					}
					block = nb
				}
				newBlocks = append(newBlocks, block)
			default:
				newBlocks = append(newBlocks, block)
			}
		}
		if len(newBlocks) == 0 && pyEqual(objGet(msg, "role"), "user") {
			newBlocks = []any{objectOf("type", "text", "text", ".")}
		}
		msg.Set("content", newBlocks)
	}
	return nil
}

// textStripped mirrors `(v or "").strip()`. A truthy non-str raises.
func textStripped(v any) (string, error) {
	if !pyTruthy(v) {
		return "", nil
	}
	s, ok := v.(string)
	if !ok {
		return "", pyError("AttributeError", "object has no attribute 'strip'")
	}
	return pyStrip(s), nil
}

// ApplyMaxsaveLate mirrors apply_maxsave_late, the pass that runs after CCR:
// age elision, unused tool schemas and the empty-block scrub. The originals
// were already spilled to CCR by then.
func ApplyMaxsaveLate(payload *pyjson.Object, relax bool) ([]string, error) {
	var cats []string
	elided, err := ApplyAgeElide(payload, relax)
	if err != nil {
		return cats, err
	}
	if elided {
		cats = append(cats, "age_elide")
	}
	unused, err := DropUnusedToolSchemas(payload)
	if err != nil {
		return cats, err
	}
	if unused {
		cats = append(cats, "unused_tools")
	}
	if err := DropEmptyTextBlocks(payload); err != nil {
		return cats, err
	}
	return cats, nil
}
