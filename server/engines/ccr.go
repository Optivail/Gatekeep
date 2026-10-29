// Mirrors backend/engines/ccr.py: the Compress-Cache-Retrieve store. Originals
// stay in process RAM; the proxy rewrites failed or empty ccr_retrieve
// tool_results on the next request because the client cannot run the tool.
package engines

import (
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

const (
	ccrMaxEntries = 256
	ccrDefaultTTL = 6 * 3600 * time.Second

	// RetrieveToolName is the name of the injected retrieve tool.
	RetrieveToolName = "ccr_retrieve"
	// CCRFulfilledKey marks a tool_result whose content came from the store.
	CCRFulfilledKey = "_ccr_fulfilled"
)

// retrieveToolDescription and retrieveToolSchema are the Python dict literals
// RETRIEVE_TOOL is built from. RetrieveTool() copies them on every call,
// because Python's dict(RETRIEVE_TOOL) is a shallow copy and the schema must
// never be shared between requests.
const retrieveToolDescription = "Retrieve a full tool output previously compressed by the proxy. " +
	"Call with the hash from a [CCR hash=...] marker when the truncated view is insufficient."

// RetrieveTool returns a fresh copy of the RETRIEVE_TOOL definition.
func RetrieveTool() *pyjson.Object {
	hashProp := objectOf("type", "string", "description", "CCR hash from the compression marker.")
	schema := objectOf(
		"type", "object",
		"properties", objectOf("hash", hashProp),
		"required", []any{"hash"},
	)
	return objectOf(
		"name", RetrieveToolName,
		"description", retrieveToolDescription,
		"input_schema", schema,
	)
}

type ccrEntry struct {
	seq  uint64    // logical clock; eviction picks the smallest, like min(ts)
	ts   time.Time // wall clock, used only for the TTL check
	ord  uint64    // first-insertion position; breaks ties like dict order
	text string
}

// ccrStore mirrors CcrStore. Python's min(ts) eviction is replaced by a
// logical sequence number: both pick the least recently touched entry, and
// the sequence never ties.
type ccrStore struct {
	mu      sync.Mutex
	max     int
	ttl     time.Duration
	items   map[string]*ccrEntry
	clock   uint64
	nextOrd uint64
	now     func() time.Time
}

func newCCRStore(max int, ttl time.Duration) *ccrStore {
	return &ccrStore{max: max, ttl: ttl, items: map[string]*ccrEntry{}, now: time.Now}
}

// defaultStore is the process-wide store (_store in Python).
var defaultStore = newCCRStore(ccrMaxEntries, ccrDefaultTTL)

// keyOf mirrors sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16].
func keyOf(text string) string {
	sum := sha256.Sum256(pyEncodeIgnore(text))
	return hex.EncodeToString(sum[:])[:16]
}

func (s *ccrStore) evictLocked(now time.Time) {
	for k, e := range s.items {
		if now.Sub(e.ts) > s.ttl {
			delete(s.items, k)
		}
	}
}

func (s *ccrStore) put(text string) string {
	key := keyOf(text)
	now := s.now()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.evictLocked(now)
	if _, exists := s.items[key]; !exists && len(s.items) >= s.max {
		s.evictOldestLocked()
	}
	s.clock++
	if old, exists := s.items[key]; exists {
		old.seq, old.ts, old.text = s.clock, now, text
		return key
	}
	s.nextOrd++
	s.items[key] = &ccrEntry{seq: s.clock, ts: now, ord: s.nextOrd, text: text}
	return key
}

// evictOldestLocked removes the least recently touched entry. Equal sequence
// numbers cannot occur, so the ord tie-break is only a guard.
func (s *ccrStore) evictOldestLocked() {
	var oldestKey string
	var oldest *ccrEntry
	for k, e := range s.items {
		if oldest == nil || e.seq < oldest.seq || (e.seq == oldest.seq && e.ord < oldest.ord) {
			oldestKey, oldest = k, e
		}
	}
	if oldest != nil {
		delete(s.items, oldestKey)
	}
}

// get mirrors CcrStore.get. A hit refreshes the entry's recency.
func (s *ccrStore) get(key string) (string, bool) {
	if key == "" {
		return "", false
	}
	now := s.now()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.evictLocked(now)
	e, ok := s.items[key]
	if !ok {
		return "", false
	}
	s.clock++
	e.seq, e.ts = s.clock, now
	return e.text, true
}

// Store mirrors ccr.store: keeps text and returns its 16-hex-digit key.
func Store(text string) string {
	return defaultStore.put(text)
}

// Recall mirrors ccr.recall. ok is false where Python returns None.
func Recall(key string) (string, bool) {
	return defaultStore.get(key)
}

// Marker mirrors ccr.marker: the one-block summary the client sees in place
// of a compressed tool result.
func Marker(hashKey string, originalChars int, preview string) string {
	preview = pyStrip(replaceNewlines(preview))
	if pyLen(preview) > 240 {
		preview = pyPrefix(preview, 240) + "…"
	}
	return "[CCR hash=" + hashKey + " chars=" + itoa(originalChars) + "] " + preview + "\n" +
		"(Full output stored locally. Call tool " + RetrieveToolName +
		` with {"hash":"` + hashKey + `"} if you need more detail.)`
}

func replaceNewlines(s string) string {
	out := []byte(s)
	for i, c := range out {
		if c == '\n' {
			out[i] = ' '
		}
	}
	return string(out)
}

// IsCCRFulfilled mirrors is_ccr_fulfilled: truthy marker on a dict block.
func IsCCRFulfilled(block any) bool {
	o, ok := block.(*pyjson.Object)
	return ok && pyTruthy(objGet(o, CCRFulfilledKey))
}

// ClearCCRFulfilledFlags mirrors clear_ccr_fulfilled_flags. It mutates the
// messages in place and drops the marker from every dict block.
func ClearCCRFulfilledFlags(messages any) {
	list, ok := messages.([]any)
	if !ok {
		return
	}
	for _, msg := range list {
		mo, ok := msg.(*pyjson.Object)
		if !ok {
			continue
		}
		content, ok := objGet(mo, "content").([]any)
		if !ok {
			continue
		}
		for _, block := range content {
			if bo, ok := block.(*pyjson.Object); ok {
				bo.Delete(CCRFulfilledKey)
			}
		}
	}
}

// InjectRetrieveTool mirrors inject_retrieve_tool. A non-list becomes an
// empty list, and the same list is returned when the tool is already present.
func InjectRetrieveTool(tools any) []any {
	list, ok := tools.([]any)
	if !ok {
		list = []any{}
	}
	for _, t := range list {
		if to, ok := t.(*pyjson.Object); ok && pyEqual(objGet(to, "name"), RetrieveToolName) {
			return list
		}
	}
	out := make([]any, 0, len(list)+1)
	out = append(out, list...)
	return append(out, RetrieveTool())
}

// FulfillRetrieveToolResults mirrors fulfill_retrieve_tool_results. It
// replaces tool_results that answer a ccr_retrieve call with the stored
// original, or with a miss marker flagged as an error. The bool reports
// whether anything was rewritten.
func FulfillRetrieveToolResults(messages any) (bool, error) {
	list, ok := messages.([]any)
	if !ok {
		return false, nil
	}
	// Map tool_use id -> requested hash, from assistant ccr_retrieve calls.
	idToHash := map[string]string{}
	for _, msg := range list {
		mo, ok := msg.(*pyjson.Object)
		if !ok || !pyEqual(objGet(mo, "role"), "assistant") {
			continue
		}
		content, ok := objGet(mo, "content").([]any)
		if !ok {
			continue
		}
		for _, block := range content {
			bo, ok := block.(*pyjson.Object)
			if !ok || !pyEqual(objGet(bo, "type"), "tool_use") {
				continue
			}
			nameVal := objGet(bo, "name")
			if !pyTruthy(nameVal) {
				nameVal = ""
			}
			if !pyEqual(nameVal, RetrieveToolName) {
				continue
			}
			input := objGet(bo, "input")
			if !pyTruthy(input) {
				input = pyjson.NewObject()
			}
			io, isDict := input.(*pyjson.Object)
			if !isDict {
				continue
			}
			h, isStr := objGet(io, "hash").(string)
			if !isStr || !pyTruthy(objGet(bo, "id")) {
				continue
			}
			key, hashable := pyKey(objGet(bo, "id"))
			if !hashable {
				return false, pyError("TypeError", "unhashable type")
			}
			idToHash[key] = h
		}
	}
	if len(idToHash) == 0 {
		return false, nil
	}

	changed := false
	for _, msg := range list {
		mo, ok := msg.(*pyjson.Object)
		if !ok || !pyEqual(objGet(mo, "role"), "user") {
			continue
		}
		content, ok := objGet(mo, "content").([]any)
		if !ok {
			continue
		}
		for _, block := range content {
			bo, ok := block.(*pyjson.Object)
			if !ok || !pyEqual(objGet(bo, "type"), "tool_result") {
				continue
			}
			lookup, hashable := pyKey(objGet(bo, "tool_use_id"))
			if !hashable {
				return changed, pyError("TypeError", "unhashable type")
			}
			h, found := idToHash[lookup]
			if !found || h == "" {
				continue
			}
			text, hit := Recall(h)
			if !hit {
				bo.Set("content", "[CCR miss: hash="+h+" expired or unknown]")
				bo.Set("is_error", true)
			} else {
				bo.Set("content", text)
				bo.Set(CCRFulfilledKey, true)
				bo.Delete("is_error")
			}
			changed = true
		}
	}
	return changed, nil
}
