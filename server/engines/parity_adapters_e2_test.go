// Mirrors the aggressive.py, maxsave.py and wirecrush.py adapters in
// server/parity/engine_cases.py. Test-only. Each entry runs the Go port with the
// same argument shapes that the Python adapter sees, and returns the same
// canonical JSON shape. Dynamic inputs follow the Python rule for each function:
// falsy values pass through, and a non-str truthy value raises where Python does.
package engines

import (
	"encoding/base64"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// e2Seen is the wirecrush seen set shared by the steps of one case. RESET clears it.
var e2Seen = map[string]bool{}

func init() {
	for name, fn := range parityE2Adapters {
		parityAdapters[name] = fn
	}
	parityResets = append(parityResets, func() {
		e2Seen = map[string]bool{}
		defaultStore = newCCRStore(ccrMaxEntries, ccrDefaultTTL)
		// The Python RESET also pins the debug-capture counters. Go writes
		// nothing unless a capture dir is set, and none is set here.
		SetWireCaptureDir("")
	})
}

// b64Canon is the canonical form of bytes that Python adapters return as
// "b64:" + base64.
func b64Canon(b []byte) string {
	return "b64:" + base64.StdEncoding.EncodeToString(b)
}

// seenArg returns the shared seen set when use_seen is set, else nil (None).
func seenArg(a *pyjson.Object) map[string]bool {
	if argBool(a, "use_seen") {
		return e2Seen
	}
	return nil
}

// toolsArg returns the tools list argument.
func toolsArg(a *pyjson.Object) []any {
	list, _ := argAny(a, "tools").([]any)
	return list
}

// strArg mirrors a Python function that calls str methods on its argument: a
// falsy value passes through, a str is processed, and anything else raises.
func strArg(a *pyjson.Object, key string, f func(string) (any, error)) (any, error) {
	v := argAny(a, key)
	if !pyTruthy(v) {
		return v, nil
	}
	s, ok := v.(string)
	if !ok {
		return nil, pyError("AttributeError", "expected str")
	}
	return f(s)
}

// strStrictArg mirrors a Python function that calls .split on its argument:
// any non-str raises, falsy values included.
func strStrictArg(a *pyjson.Object, key string, f func(string) (any, error)) (any, error) {
	s, ok := argAny(a, key).(string)
	if !ok {
		return nil, pyError("AttributeError", "expected str")
	}
	return f(s)
}

// strPassArg mirrors a Python function that returns any non-str unchanged.
func strPassArg(a *pyjson.Object, key string, f func(string) (any, error)) (any, error) {
	v := argAny(a, key)
	s, ok := v.(string)
	if !ok {
		return v, nil
	}
	return f(s)
}

// payloadPair runs a payload pass and returns [result, payload], the shape the
// Python adapters use.
func payloadPair(a *pyjson.Object, run func(p *pyjson.Object) (any, error)) (any, error) {
	p, err := argObj(a, "payload")
	if err != nil {
		return nil, err
	}
	res, err := run(p)
	if err != nil {
		return nil, err
	}
	return []any{res, p}, nil
}

// boolAny converts a bool result to its canonical JSON value.
func boolAny(b bool, err error) (any, error) {
	return b, err
}

var parityE2Adapters = map[string]func(*pyjson.Object) (any, error){
	// --- aggressive ----------------------------------------------------------
	"aggressive.apply_aggressive": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		cats, err := ApplyAggressive(p, argKey(a, "session_key"))
		if err != nil {
			return nil, err
		}
		return []any{stringsToAny(cats), p}, nil
	},
	"aggressive.apply_aggressive_openai": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		cats, err := ApplyAggressiveOpenAI(p)
		if err != nil {
			return nil, err
		}
		return []any{stringsToAny(cats), p}, nil
	},
	"aggressive.apply_aggressive_responses": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		cats, err := ApplyAggressiveResponses(p)
		if err != nil {
			return nil, err
		}
		return []any{stringsToAny(cats), p}, nil
	},
	"aggressive.aggressive_trim_system": func(a *pyjson.Object) (any, error) {
		return AggressiveTrimSystem(argAny(a, "system"))
	},
	"aggressive.aggressive_trim_system_hard": func(a *pyjson.Object) (any, error) {
		return AggressiveTrimSystemHard(argAny(a, "system"))
	},
	"aggressive.aggressive_trim_codex_developer": func(a *pyjson.Object) (any, error) {
		return strPassArg(a, "text", func(s string) (any, error) {
			return AggressiveTrimCodexDeveloper(s), nil
		})
	},
	"aggressive.crush_text": func(a *pyjson.Object) (any, error) {
		return strArg(a, "text", func(s string) (any, error) { return CrushText(s) })
	},
	"aggressive._ccr_wrap": func(a *pyjson.Object) (any, error) {
		return ccrWrapValue(argAny(a, "text"))
	},
	"aggressive._crush_log_lines": func(a *pyjson.Object) (any, error) {
		return strStrictArg(a, "text", func(s string) (any, error) { return crushLogLines(s), nil })
	},
	"aggressive._crush_json_array": func(a *pyjson.Object) (any, error) {
		return strPassArg(a, "text", func(s string) (any, error) { return crushJSONArray(s) })
	},
	"aggressive._crush_grepish": func(a *pyjson.Object) (any, error) {
		return strStrictArg(a, "text", func(s string) (any, error) { return crushGrepish(s), nil })
	},
	"aggressive._crush_responses_output_text": func(a *pyjson.Object) (any, error) {
		return strPassArg(a, "text", func(s string) (any, error) { return crushResponsesOutputText(s) })
	},
	"aggressive.apply_pure_tool_result_ccr": func(a *pyjson.Object) (any, error) {
		return payloadPair(a, func(p *pyjson.Object) (any, error) {
			return boolAny(ApplyPureToolResultCCR(p, argBool(a, "relax")))
		})
	},
	"aggressive.apply_stale_tool_compaction": func(a *pyjson.Object) (any, error) {
		return payloadPair(a, func(p *pyjson.Object) (any, error) {
			return boolAny(ApplyStaleToolCompaction(p))
		})
	},
	"aggressive.hard_minify_tools": func(a *pyjson.Object) (any, error) {
		return HardMinifyTools(argAny(a, "tools")), nil
	},
	"aggressive.hard_minify_codex_tools": func(a *pyjson.Object) (any, error) {
		return HardMinifyCodexTools(argAny(a, "tools")), nil
	},
	"aggressive.hard_minify_openai_tools": func(a *pyjson.Object) (any, error) {
		return HardMinifyOpenAITools(argAny(a, "tools")), nil
	},

	// --- maxsave -------------------------------------------------------------
	"maxsave.strip_ansi": func(a *pyjson.Object) (any, error) {
		return strArg(a, "text", func(s string) (any, error) { return StripANSI(s), nil })
	},
	"maxsave.squeeze_ws": func(a *pyjson.Object) (any, error) {
		return strArg(a, "text", func(s string) (any, error) { return SqueezeWS(s), nil })
	},
	"maxsave.strip_base64_blobs": func(a *pyjson.Object) (any, error) {
		return strArg(a, "text", func(s string) (any, error) { return StripBase64Blobs(s), nil })
	},
	"maxsave.drop_progress_lines": func(a *pyjson.Object) (any, error) {
		return strArg(a, "text", func(s string) (any, error) { return DropProgressLines(s), nil })
	},
	"maxsave.crush_cli_tables": func(a *pyjson.Object) (any, error) {
		return strArg(a, "text", func(s string) (any, error) { return CrushCLITables(s), nil })
	},
	"maxsave.compact_json_text": func(a *pyjson.Object) (any, error) {
		return strArg(a, "text", func(s string) (any, error) { return CompactJSONText(s) })
	},
	"maxsave.preprocess_tool_text": func(a *pyjson.Object) (any, error) {
		return preprocessToolValue(argAny(a, "text"))
	},
	"maxsave.strip_system_reminders": func(a *pyjson.Object) (any, error) {
		return stripSystemRemindersValue(argAny(a, "text"))
	},
	"maxsave.shrink_bash_git_bloat": func(a *pyjson.Object) (any, error) {
		return strArg(a, "text", func(s string) (any, error) { return ShrinkBashGitBloat(s), nil })
	},
	"maxsave.apply_tool_result_preprocess": func(a *pyjson.Object) (any, error) {
		return payloadPair(a, func(p *pyjson.Object) (any, error) { return boolAny(ApplyToolResultPreprocess(p)) })
	},
	"maxsave.apply_duplicate_tool_result_stubs": func(a *pyjson.Object) (any, error) {
		return payloadPair(a, func(p *pyjson.Object) (any, error) { return ApplyDuplicateToolResultStubs(p), nil })
	},
	"maxsave.apply_message_reminder_strip": func(a *pyjson.Object) (any, error) {
		return payloadPair(a, func(p *pyjson.Object) (any, error) { return boolAny(ApplyMessageReminderStrip(p)) })
	},
	"maxsave.apply_age_elide": func(a *pyjson.Object) (any, error) {
		return payloadPair(a, func(p *pyjson.Object) (any, error) {
			return boolAny(ApplyAgeElide(p, argBool(a, "relax")))
		})
	},
	"maxsave.drop_unused_tool_schemas": func(a *pyjson.Object) (any, error) {
		return payloadPair(a, func(p *pyjson.Object) (any, error) { return boolAny(DropUnusedToolSchemas(p)) })
	},
	"maxsave.apply_maxsave": func(a *pyjson.Object) (any, error) {
		return payloadPair(a, func(p *pyjson.Object) (any, error) {
			cats, err := ApplyMaxsave(p, argBool(a, "lossless_only"))
			return stringsToAny(cats), err
		})
	},
	"maxsave.apply_maxsave_late": func(a *pyjson.Object) (any, error) {
		return payloadPair(a, func(p *pyjson.Object) (any, error) {
			cats, err := ApplyMaxsaveLate(p, argBool(a, "relax"))
			return stringsToAny(cats), err
		})
	},
	"maxsave.drop_empty_text_blocks": func(a *pyjson.Object) (any, error) {
		p, err := argObj(a, "payload")
		if err != nil {
			return nil, err
		}
		if err := DropEmptyTextBlocks(p); err != nil {
			return nil, err
		}
		return p, nil
	},
	"maxsave.strip_git_instruction_bloat": func(a *pyjson.Object) (any, error) {
		newSys, ch, err := StripGitInstructionBloat(argAny(a, "system"))
		if err != nil {
			return nil, err
		}
		return []any{newSys, ch}, nil
	},
	"maxsave.shrink_tool_description_walls": func(a *pyjson.Object) (any, error) {
		t := toolsArg(a)
		return []any{ShrinkToolDescriptionWalls(t), t}, nil
	},

	// --- wirecrush -----------------------------------------------------------
	"wirecrush.crush_connect_body": func(a *pyjson.Object) (any, error) {
		body, err := argB64(a, "body_b64")
		if err != nil {
			return nil, err
		}
		out, saved, err := CrushConnectBody(body, argBool(a, "aggressive"), seenArg(a))
		if err != nil {
			return nil, err
		}
		return []any{b64Canon(out), int64(saved)}, nil
	},
	"wirecrush.crush_protobuf_message": func(a *pyjson.Object) (any, error) {
		body, err := argB64(a, "body_b64")
		if err != nil {
			return nil, err
		}
		out, saved, err := CrushProtobufMessage(body, argBool(a, "aggressive"), seenArg(a))
		if err != nil {
			return nil, err
		}
		return []any{b64Canon(out), int64(saved)}, nil
	},
	"wirecrush.crush_wire_body": func(a *pyjson.Object) (any, error) {
		body, err := argB64(a, "body_b64")
		if err != nil {
			return nil, err
		}
		ps, _ := argAny(a, "path").(string)
		ct, _ := argAny(a, "content_type").(string)
		out, cats, saved, err := CrushWireBody(body, ps, ct, argBool(a, "aggressive"))
		if err != nil {
			return nil, err
		}
		return []any{b64Canon(out), stringsToAny(cats), int64(saved)}, nil
	},
	"wirecrush.drain_crush_frames": func(a *pyjson.Object) (any, error) {
		body, err := argB64(a, "body_b64")
		if err != nil {
			return nil, err
		}
		rest, frames, err := DrainCrushFrames(body, argBool(a, "aggressive"), seenArg(a))
		if err != nil {
			return nil, err
		}
		canon := make([]any, 0, len(frames))
		for _, f := range frames {
			canon = append(canon, b64Canon(f))
		}
		return []any{b64Canon(rest), canon}, nil
	},
	"wirecrush.crush_json_body": func(a *pyjson.Object) (any, error) {
		body, err := argB64(a, "body_b64")
		if err != nil {
			return nil, err
		}
		out, ok, err := CrushJSONBody(body, argBool(a, "aggressive"))
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, nil // Python returns None
		}
		return out, nil
	},
	"wirecrush.crush_json_value": func(a *pyjson.Object) (any, error) {
		return CrushJSONValue(argAny(a, "node"), argBool(a, "aggressive"))
	},
}
