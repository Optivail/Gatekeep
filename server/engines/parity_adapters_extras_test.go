// Mirrors the extras.py, loopwatch.py and error_guard.py adapters in
// server/parity/engine_cases.py. Test-only. The entries are merged into
// parityAdapters at init time, next to the optimizer adapters.
package engines

import (
	"github.com/Optivail/Gatekeep/server/pyjson"
)

func init() {
	for name, fn := range parityExtrasAdapters {
		parityAdapters[name] = fn
	}
}

// argKey returns a session key argument; Python None and "" are both falsy and
// both mean "no session", so None maps to "".
func argKey(a *pyjson.Object, key string) string {
	s, _ := objGet(a, key).(string)
	return s
}

func stringsToAny(in []string) []any {
	out := make([]any, 0, len(in))
	for _, s := range in {
		out = append(out, s)
	}
	return out
}

var parityExtrasAdapters = map[string]func(*pyjson.Object) (any, error){
	"extras.strip_volatile_system": func(a *pyjson.Object) (any, error) {
		return StripVolatileSystem(argAny(a, "system"))
	},
	"extras.stabilize_tools": func(a *pyjson.Object) (any, error) {
		return StabilizeTools(argAny(a, "tools"))
	},
	"extras.apply_defer_loading": func(a *pyjson.Object) (any, error) {
		tools, changed, err := ApplyDeferLoading(argAny(a, "tools"))
		if err != nil {
			return nil, err
		}
		return []any{tools, changed}, nil
	},
	"extras.apply_extractive_digest": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		changed, err := ApplyExtractiveDigest(p)
		if err != nil {
			return nil, err
		}
		return []any{changed, p}, nil
	},
	"extras.strip_media_blocks": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		return []any{StripMediaBlocks(p), p}, nil
	},
	"extras.sanitize_thinking_blocks": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		return []any{SanitizeThinkingBlocks(p), p}, nil
	},
	"extras.fold_repeated_log_lines": func(a *pyjson.Object) (any, error) {
		text, err := argStr(a, "text")
		if err != nil {
			return nil, err
		}
		return FoldRepeatedLogLines(text), nil
	},
	"extras.apply_cache_and_structure_fixes": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		cats, err := ApplyCacheAndStructureFixes(p)
		if err != nil {
			return nil, err
		}
		return []any{stringsToAny(cats), p}, nil
	},
	"extras.normalize_intent_text": func(a *pyjson.Object) (any, error) {
		text := argKey(a, "text") // None and "" both normalise to ""
		return NormalizeIntentText(text), nil
	},
	"extras.compute_intent_hash": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		provider, err := argStr(a, "provider")
		if err != nil {
			return nil, err
		}
		h, ok, err := ComputeIntentHash(p, provider)
		if err != nil || !ok {
			return nil, err
		}
		return h, nil
	},
	"loopwatch.observe": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		cats, err := Observe(p, argKey(a, "session_key"))
		if err != nil {
			return nil, err
		}
		return []any{stringsToAny(cats), p}, nil
	},
	"loopwatch.clear_session": func(a *pyjson.Object) (any, error) {
		ClearSession(argKey(a, "session_key"))
		return nil, nil
	},
	"loopwatch.is_relaxed": func(a *pyjson.Object) (any, error) {
		return IsRelaxed(argKey(a, "session_key")), nil
	},
	"loopwatch.is_decompress": func(a *pyjson.Object) (any, error) {
		return IsDecompress(argKey(a, "session_key")), nil
	},
	"loopwatch.tick_decompress": func(a *pyjson.Object) (any, error) {
		TickDecompress(argKey(a, "session_key"))
		return nil, nil
	},
	"loopwatch.is_bypass_compression": func(a *pyjson.Object) (any, error) {
		return IsBypassCompression(argKey(a, "session_key")), nil
	},
	"error_guard.extract_stacktrace_blocks": func(a *pyjson.Object) (any, error) {
		text, err := argStr(a, "text")
		if err != nil {
			return nil, err
		}
		blocks := ExtractStacktraceBlocks(text)
		out := make([]any, 0, len(blocks))
		for _, b := range blocks {
			out = append(out, []any{int64(b[0]), int64(b[1])})
		}
		return out, nil
	},
	"error_guard.preserve_stacktrace_blocks": func(a *pyjson.Object) (any, error) {
		text, err := argStr(a, "text")
		if err != nil {
			return nil, err
		}
		ceiling, err := argInt(a, "ceiling")
		if err != nil {
			return nil, err
		}
		return PreserveStacktraceBlocks(text, ceiling), nil
	},
}
