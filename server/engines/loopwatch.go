// Mirrors backend/engines/loopwatch.py: a light loop watcher that detects
// repeated identical tool calls and relaxes compression for the session.
// Per-session state is process-wide and guarded by one mutex (Python used
// threading.Lock; Go handlers run concurrently).
package engines

import (
	"strings"
	"sync"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

const (
	loopRepeat       = 3
	decompressWindow = 4
	decompressRepeat = 2
	decompressTurns  = 2
	loopNudgeMarker  = "[proxy] Repeated identical tool calls detected; stop re-running the same command; finish the task with what you have."
)

var (
	loopMu      sync.Mutex
	loopRelaxed = map[string]bool{}
	loopDecomp  = map[string]int{}

	loopReadTools = map[string]bool{
		"read": true, "read_file": true, "Read": true, "view": true,
		"view_file": true, "cat": true, "get_file_contents": true,
	}
	loopSearchTools = map[string]bool{
		"grep": true, "Grep": true, "glob": true, "Glob": true, "ls": true,
		"list_dir": true, "list_files": true, "find": true, "search": true,
		"search_files": true,
	}
)

// ClearSession mirrors loopwatch.clear_session. An empty key is a no-op.
func ClearSession(sessionKey string) {
	if sessionKey == "" {
		return
	}
	loopMu.Lock()
	defer loopMu.Unlock()
	delete(loopRelaxed, sessionKey)
	delete(loopDecomp, sessionKey)
}

// IsRelaxed mirrors loopwatch.is_relaxed.
func IsRelaxed(sessionKey string) bool {
	if sessionKey == "" {
		return false
	}
	loopMu.Lock()
	defer loopMu.Unlock()
	return loopRelaxed[sessionKey]
}

// IsDecompress mirrors loopwatch.is_decompress.
func IsDecompress(sessionKey string) bool {
	if sessionKey == "" {
		return false
	}
	loopMu.Lock()
	defer loopMu.Unlock()
	return loopDecomp[sessionKey] > 0
}

// TickDecompress mirrors loopwatch.tick_decompress: one lossless turn used up.
func TickDecompress(sessionKey string) {
	if sessionKey == "" {
		return
	}
	loopMu.Lock()
	defer loopMu.Unlock()
	remaining := loopDecomp[sessionKey]
	if remaining <= 0 {
		return
	}
	remaining--
	if remaining <= 0 {
		delete(loopDecomp, sessionKey)
	} else {
		loopDecomp[sessionKey] = remaining
	}
}

// IsBypassCompression mirrors loopwatch.is_bypass_compression.
func IsBypassCompression(sessionKey string) bool {
	return IsRelaxed(sessionKey) || IsDecompress(sessionKey)
}

// loopFingerprint mirrors _fingerprint(name, inp): name plus the canonical
// JSON of the input. Python falls back to str(inp) when the canonical dump
// fails (a lone surrogate cannot be encoded as UTF-8). A nil input gives "".
func loopFingerprint(name string, inp any) string {
	body := ""
	if inp != nil {
		if b, err := canonicalJSON(inp); err == nil {
			body = string(b)
		} else {
			body = pyStr(inp)
		}
	}
	return name + "|" + body
}

// pyFingerprintName mirrors the f-string "{name}|" when the name is not a str.
// Only the trailing-repeat path uses the raw name, so Python str() applies.
func pyFingerprintName(v any) string {
	if !pyTruthy(v) {
		return ""
	}
	return pyStr(v)
}

// toolUse is one trailing tool_use as (lowered name, input, file path).
type toolUse struct {
	name    string
	inp     any
	path    string
	hasPath bool
}

// recentToolUses mirrors _recent_tool_uses: the trailing tool_use blocks,
// newest message first, reversed at the end so they read oldest-to-newest.
func recentToolUses(messages []any, limit int) ([]toolUse, error) {
	var uses []toolUse
	for mi := len(messages) - 1; mi >= 0; mi-- {
		msg, ok := messages[mi].(*pyjson.Object)
		if !ok || !pyEqual(objGet(msg, "role"), "assistant") {
			continue
		}
		content, ok := objGet(msg, "content").([]any)
		if !ok {
			continue
		}
		for _, b := range content {
			bo, ok := b.(*pyjson.Object)
			if !ok || !pyEqual(objGet(bo, "type"), "tool_use") {
				continue
			}
			name, err := lowerOrEmpty(objGet(bo, "name"))
			if err != nil {
				return nil, err
			}
			inp := objGet(bo, "input")
			if !pyTruthy(inp) {
				inp = pyjson.NewObject()
			}
			path, hasPath, err := extractFilePath(bo)
			if err != nil {
				return nil, err
			}
			uses = append(uses, toolUse{name: name, inp: inp, path: path, hasPath: hasPath})
		}
		if len(uses) >= limit {
			break
		}
	}
	// reverse: the Python list is built newest-message-first, then reversed.
	for i, j := 0, len(uses)-1; i < j; i, j = i+1, j-1 {
		uses[i], uses[j] = uses[j], uses[i]
	}
	return uses, nil
}

// trailingRepeatCount mirrors _trailing_repeat_count: how many consecutive
// tool_uses at the end share the last fingerprint.
func trailingRepeatCount(messages []any) (int, string, error) {
	var fps []string
	for _, msg := range messages {
		mo, ok := msg.(*pyjson.Object)
		if !ok || !pyEqual(objGet(mo, "role"), "assistant") {
			continue
		}
		content, ok := objGet(mo, "content").([]any)
		if !ok {
			continue
		}
		for _, b := range content {
			bo, ok := b.(*pyjson.Object)
			if !ok || !pyEqual(objGet(bo, "type"), "tool_use") {
				continue
			}
			fps = append(fps, loopFingerprint(pyFingerprintName(objGet(bo, "name")), objGet(bo, "input")))
		}
	}
	if len(fps) == 0 {
		return 0, "", nil
	}
	last := fps[len(fps)-1]
	n := 0
	for i := len(fps) - 1; i >= 0; i-- {
		if fps[i] != last {
			break
		}
		n++
	}
	return n, last, nil
}

// anyCountAtLeast reports whether some value occurs at least min times,
// mirroring any(c >= min for c in Counter(values).values()).
func anyCountAtLeast(values []string, min int) bool {
	counts := map[string]int{}
	for _, v := range values {
		counts[v]++
		if counts[v] >= min {
			return true
		}
	}
	return false
}

// detectDecompress mirrors _detect_decompress: the same file read, or the
// same search, at least twice within the last few tool uses.
func detectDecompress(messages []any) (bool, error) {
	recent, err := recentToolUses(messages, decompressWindow)
	if err != nil {
		return false, err
	}
	if len(recent) < decompressRepeat {
		return false, nil
	}
	var readPaths []string
	for _, u := range recent {
		if loopReadTools[u.name] && u.hasPath && u.path != "" {
			readPaths = append(readPaths, u.path)
		}
	}
	if len(readPaths) > 0 && anyCountAtLeast(readPaths, decompressRepeat) {
		return true, nil
	}
	var searchFPs []string
	for _, u := range recent {
		if loopSearchTools[u.name] {
			searchFPs = append(searchFPs, loopFingerprint(u.name, u.inp))
		}
	}
	if len(searchFPs) >= decompressRepeat && anyCountAtLeast(searchFPs, decompressRepeat) {
		return true, nil
	}
	return false, nil
}

// injectNudge mirrors _inject_nudge: append the nudge once to a trailing user
// message.
func injectNudge(payload *pyjson.Object) (bool, error) {
	messages, ok := objGet(payload, "messages").([]any)
	if !ok || len(messages) == 0 {
		return false, nil
	}
	last, ok := messages[len(messages)-1].(*pyjson.Object)
	if !ok || !pyEqual(objGet(last, "role"), "user") {
		return false, nil
	}
	switch content := objGet(last, "content").(type) {
	case string:
		if strings.Contains(content, loopNudgeMarker) {
			return false, nil
		}
		last.Set("content", content+"\n\n"+loopNudgeMarker)
		return true, nil
	case []any:
		for _, b := range content {
			bo, ok := b.(*pyjson.Object)
			if !ok || !pyEqual(objGet(bo, "type"), "text") {
				continue
			}
			found, err := pyContains(loopNudgeMarker, pyOrEmptyValue(objGet(bo, "text")))
			if err != nil {
				return false, err
			}
			if found {
				return false, nil
			}
		}
		out := make([]any, 0, len(content)+1)
		out = append(out, content...)
		out = append(out, objectOf("type", "text", "text", loopNudgeMarker))
		last.Set("content", out)
		return true, nil
	}
	return false, nil
}

// pyOrEmptyValue mirrors (v or ""): falsy values become the empty string.
func pyOrEmptyValue(v any) any {
	if pyTruthy(v) {
		return v
	}
	return ""
}

// pyContains mirrors `needle in haystack` for a str needle and a JSON value:
// substring for str, element equality for list, key membership for dict, and
// TypeError for anything else.
func pyContains(needle string, haystack any) (bool, error) {
	switch h := haystack.(type) {
	case string:
		return strings.Contains(h, needle), nil
	case []any:
		for _, x := range h {
			if pyEqual(x, needle) {
				return true, nil
			}
		}
		return false, nil
	case *pyjson.Object:
		_, ok := h.Get(needle)
		return ok, nil
	}
	return false, pyError("TypeError", "argument of type is not iterable")
}

// Observe mirrors loopwatch.observe. It updates the relax and decompress
// flags from the history and injects a soft nudge once. The returned slice is
// the category tags (Python returns a list).
func Observe(payload *pyjson.Object, sessionKey string) ([]string, error) {
	var cats []string
	messages, ok := objGet(payload, "messages").([]any)
	if !ok {
		return cats, nil
	}

	if IsDecompress(sessionKey) {
		return append(cats, "loop_decompress"), nil
	}
	if IsRelaxed(sessionKey) {
		return append(cats, "loop_relax"), nil
	}

	detected, err := detectDecompress(messages)
	if err != nil {
		return nil, err
	}
	if detected {
		if sessionKey != "" {
			loopMu.Lock()
			loopDecomp[sessionKey] = decompressTurns
			loopMu.Unlock()
		}
		return append(cats, "loop_decompress"), nil
	}

	n, _, err := trailingRepeatCount(messages)
	if err != nil {
		return nil, err
	}
	if n < loopRepeat || sessionKey == "" {
		return cats, nil
	}

	loopMu.Lock()
	loopRelaxed[sessionKey] = true
	loopMu.Unlock()
	cats = append(cats, "loop_relax")
	nudged, err := injectNudge(payload)
	if err != nil {
		return nil, err
	}
	if nudged {
		cats = append(cats, "loop_nudge")
	}
	return cats, nil
}
