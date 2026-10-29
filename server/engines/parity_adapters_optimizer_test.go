// Mirrors the optimizer.py and error_guard.py adapters in server/parity/engine_cases.py.
// Test-only. Each entry runs the Go function that ports the Python call of the
// same name and returns a value in the same shape (tuples become lists, bytes
// stay bytes). The canonical form is defined in parity_test.go.
package engines

import (
	"encoding/base64"
	"errors"
	"fmt"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// parityResets mirrors RESET in engine_cases.py. Each case starts from a
// fresh file-read tracker, as the Python side does.
var parityResets = []func(){
	func() { fileTrackerGlobal = newFileTracker() },
	// loopwatch keeps per-session flags in package maps (_RELAXED, _DECOMPRESS).
	func() {
		loopMu.Lock()
		loopRelaxed = map[string]bool{}
		loopDecomp = map[string]int{}
		loopMu.Unlock()
	},
}

// parityAdapters maps a Python engine function name to its Go adapter.
var parityAdapters = map[string]func(*pyjson.Object) (any, error){
	"optimizer.count_tokens": func(a *pyjson.Object) (any, error) {
		text, err := argStr(a, "text")
		if err != nil {
			return nil, err
		}
		n, err := CountTokens(text)
		return int64(n), err
	},
	"optimizer.optimize_text": func(a *pyjson.Object) (any, error) {
		return optimizeTextValue(argAny(a, "text"))
	},
	"optimizer.inject_anti_yap": func(a *pyjson.Object) (any, error) {
		return InjectAntiYap(argAny(a, "system")), nil
	},
	"optimizer._content_text": func(a *pyjson.Object) (any, error) {
		return ContentText(argAny(a, "content")), nil
	},
	"optimizer._optimize_content": func(a *pyjson.Object) (any, error) {
		return optimizeContent(argAny(a, "content"))
	},
	"optimizer._prune_middle_content": func(a *pyjson.Object) (any, error) {
		return pruneMiddleContent(argAny(a, "text"), argBool(a, "log_truncation_enabled"))
	},
	"optimizer._truncate_verbose_logs": func(a *pyjson.Object) (any, error) {
		text, err := argStr(a, "text")
		if err != nil {
			return nil, err
		}
		return truncateVerboseLogs(text)
	},
	"optimizer._first_sentence": func(a *pyjson.Object) (any, error) {
		text, err := argStr(a, "text")
		if err != nil {
			return nil, err
		}
		return firstSentence(text), nil
	},
	"optimizer._extract_file_path": func(a *pyjson.Object) (any, error) {
		block, err := argObj(a, "block")
		if err != nil {
			return nil, err
		}
		path, ok, err := extractFilePath(block)
		if err != nil || !ok {
			return nil, err
		}
		return path, nil
	},
	"optimizer._squash_exploratory_chains": func(a *pyjson.Object) (any, error) {
		msgs, err := argList(a, "messages")
		if err != nil {
			return nil, err
		}
		return squashExploratoryChains(msgs)
	},
	"optimizer.extract_text": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		return ExtractText(p)
	},
	"optimizer.prune_history": func(a *pyjson.Object) (any, error) {
		return PruneHistory(argAny(a, "messages"), argBool(a, "log_truncation_enabled"),
			argBool(a, "turn_squashing_enabled"))
	},
	"optimizer.optimize_payload": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		opts := OptimizeOptions{
			Compression:    argBool(a, "compression_enabled"),
			AntiYap:        argBool(a, "anti_yap_enabled"),
			HistoryPruning: argBool(a, "history_pruning_enabled"),
			LogTruncation:  argBool(a, "log_truncation_enabled"),
		}
		out, cats, err := OptimizePayload(p, opts)
		if err != nil {
			return nil, err
		}
		catList := make([]any, 0, len(cats))
		for _, c := range cats {
			catList = append(catList, c)
		}
		return []any{out, catList}, nil
	},
	"optimizer.apply_anthropic_cache_alignment": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		provider, err := argStr(a, "provider")
		if err != nil {
			return nil, err
		}
		return ApplyAnthropicCacheAlignment(p, provider)
	},
	"optimizer.apply_newest_tool_result_compaction": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		return ApplyNewestToolResultCompaction(p)
	},
	"optimizer.clamp_max_tokens": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		return ClampMaxTokens(p), nil
	},
	"optimizer.minify_tools_deterministic": func(a *pyjson.Object) (any, error) {
		return MinifyToolsDeterministic(argAny(a, "tools")), nil
	},
	"optimizer.derive_session_key": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		return DeriveSessionKey(p), nil
	},
	"optimizer.apply_file_delta_tracking": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		key, err := argStr(a, "session_key")
		if err != nil {
			return nil, err
		}
		return ApplyFileDeltaTracking(p, key)
	},
	"optimizer.credential_fingerprint": func(a *pyjson.Object) (any, error) {
		h, err := argObj(a, "headers")
		if err != nil {
			return nil, err
		}
		// Python: (headers.get(h) or "").strip(). A truthy non-str raises
		// AttributeError at .strip(), so the adapter reports that case.
		vals := map[string]string{}
		for _, name := range []string{"authorization", "x-api-key", "x-goog-api-key", "api-key"} {
			v := objGet(h, name)
			if !pyTruthy(v) {
				vals[name] = ""
				continue
			}
			s, ok := v.(string)
			if !ok {
				return nil, pyError("AttributeError", "object has no attribute 'strip'")
			}
			vals[name] = s
		}
		return CredentialFingerprint(func(name string) string { return vals[name] }), nil
	},
	"optimizer.compute_request_hash": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		provider, err := argStr(a, "provider")
		if err != nil {
			return nil, err
		}
		cred, err := argStr(a, "credential")
		if err != nil {
			return nil, err
		}
		return ComputeRequestHash(p, provider, cred)
	},
	"optimizer.canonical_json_dumps": func(a *pyjson.Object) (any, error) {
		return CanonicalJSONDumps(argAny(a, "data"))
	},
	"optimizer.measure_wire_tokens": func(a *pyjson.Object) (any, error) {
		body, err := argB64(a, "body_b64")
		if err != nil {
			return nil, err
		}
		n, sample, err := MeasureWireTokens(body, argAny(a, "payload"))
		if err != nil {
			return nil, err
		}
		return []any{int64(n), sample}, nil
	},
	"optimizer.extract_upstream_usage": func(a *pyjson.Object) (any, error) {
		body, err := argB64(a, "raw_body_b64")
		if err != nil {
			return nil, err
		}
		return ExtractUpstreamUsage(body)
	},
	"error_guard.is_error_payload": func(a *pyjson.Object) (any, error) {
		text, err := argStr(a, "text")
		if err != nil {
			return nil, err
		}
		// tool_name None and "" both behave as no tool name in Python.
		tool := ""
		if s, ok := argAny(a, "tool_name").(string); ok {
			tool = s
		}
		return IsErrorPayload(text, tool, argBool(a, "is_error_flag")), nil
	},
	"error_guard.preserve_diagnostic_text": func(a *pyjson.Object) (any, error) {
		text, err := argStr(a, "text")
		if err != nil {
			return nil, err
		}
		ceiling, err := argInt(a, "ceiling")
		if err != nil {
			return nil, err
		}
		return PreserveDiagnosticText(text, ceiling), nil
	},
}

// argAny returns the raw JSON value of key (nil when absent, like dict.get).
func argAny(a *pyjson.Object, key string) any { return objGet(a, key) }

func argStr(a *pyjson.Object, key string) (string, error) {
	s, ok := objGet(a, key).(string)
	if !ok {
		return "", fmt.Errorf("parity: arg %q is not a string", key)
	}
	return s, nil
}

func argObj(a *pyjson.Object, key string) (*pyjson.Object, error) {
	o, ok := objGet(a, key).(*pyjson.Object)
	if !ok {
		return nil, fmt.Errorf("parity: arg %q is not an object", key)
	}
	return o, nil
}

func argList(a *pyjson.Object, key string) ([]any, error) {
	l, ok := objGet(a, key).([]any)
	if !ok {
		return nil, fmt.Errorf("parity: arg %q is not a list", key)
	}
	return l, nil
}

func argBool(a *pyjson.Object, key string) bool {
	b, _ := objGet(a, key).(bool)
	return b
}

func argInt(a *pyjson.Object, key string) (int, error) {
	n, ok := objGet(a, key).(int64)
	if !ok {
		return 0, fmt.Errorf("parity: arg %q is not an int", key)
	}
	return int(n), nil
}

func argB64(a *pyjson.Object, key string) ([]byte, error) {
	s, err := argStr(a, key)
	if err != nil {
		return nil, err
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, errors.New("parity: bad base64 in " + key)
	}
	return b, nil
}
