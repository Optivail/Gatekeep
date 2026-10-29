package main

// Port of the CPU-heavy steps of backend/core/proxy.py: _crush_json_conversation
// (the JSON conversation crush) and the wire crush used for non-conversation
// bodies. The engines do the rewriting. This file decides which engine runs
// and in what order, and keeps the token and sample bookkeeping.
//
// ErrUnsupported policy: when an engine declines an input, the caller forwards
// the ORIGINAL body unchanged, logs a warning, and records the request with
// the category "unoptimized". Python would rewrite the body in that case, so
// this is a known divergence (see forward.go).

import (
	"bytes"
	"errors"
	"fmt"

	"github.com/Optivail/Gatekeep/server/engines"
	"github.com/Optivail/Gatekeep/server/pyjson"
)

// optFlags are the settings read from the database for one request.
type optFlags struct {
	compression, antiYap, historyPruning, logTruncation, aggressive bool
}

// crushResult is what _crush_json_conversation returns.
type crushResult struct {
	body                            []byte
	payload                         *pyjson.Object
	cats                            []string
	originalTokens, optimizedTokens int
	originalSample, optimizedSample string
}

// stepErr names the engine step that failed. The name shows in the log line
// for an ErrUnsupported decline. errors.Is still matches the wrapped error.
func stepErr(step string, err error) error {
	return fmt.Errorf("%s: %w", step, err)
}

// crushJSONConversation mirrors _crush_json_conversation. payload is mutated in
// place, as the Python dict is. On an engine error the result still carries
// the original token figures, so an ErrUnsupported caller can log them.
func crushJSONConversation(payload *pyjson.Object, provider string, originalBody []byte, f optFlags) (crushResult, error) {
	var res crushResult

	origText, err := engines.ExtractText(payload)
	if err != nil {
		return res, stepErr("extract_text", err)
	}
	origTokens, err := engines.CountTokens(origText)
	if err != nil {
		return res, stepErr("count_tokens", err)
	}
	res.originalTokens = origTokens
	res.originalSample = truncRunes(origText, payloadSampleChars)

	bypass := false
	sessionKey := ""
	if f.compression {
		sessionKey = engines.DeriveSessionKey(payload)
		if _, err := engines.Observe(payload, sessionKey); err != nil {
			return res, stepErr("loopwatch.observe", err)
		}
		bypass = engines.IsBypassCompression(sessionKey)
		anth, err := isAnthropicMessagesPayload(payload)
		if err != nil {
			return res, stepErr("payload_shape", err)
		}
		if anth {
			payload, err = engines.ApplyFileDeltaTracking(payload, sessionKey)
			if err != nil {
				return res, stepErr("file_delta_tracking", err)
			}
			if !f.aggressive && !bypass {
				payload, err = engines.ApplyNewestToolResultCompaction(payload)
				if err != nil {
					return res, stepErr("newest_tool_result", err)
				}
			}
		}
	}

	anth, err := isAnthropicMessagesPayload(payload)
	if err != nil {
		return res, stepErr("payload_shape", err)
	}
	openaiCompat := !anth
	pruneHist := f.historyPruning && (!f.aggressive || openaiCompat) && !bypass
	pruneLogs := f.logTruncation && (!f.aggressive || openaiCompat) && !bypass

	payload, cats, err := engines.OptimizePayload(payload, engines.OptimizeOptions{
		Compression:    f.compression,
		AntiYap:        f.antiYap,
		HistoryPruning: pruneHist,
		LogTruncation:  pruneLogs,
	})
	if err != nil {
		return res, stepErr("optimize_payload", err)
	}

	more, err := applyAggressivePass(payload, f, bypass, sessionKey)
	if err != nil {
		return res, stepErr("aggressive", err)
	}
	cats = append(append([]string(nil), cats...), more...)

	optimizedText, err := engines.ExtractText(payload)
	if err != nil {
		return res, stepErr("extract_text", err)
	}
	optimizedTokens, err := engines.CountTokens(optimizedText)
	if err != nil {
		return res, stepErr("count_tokens", err)
	}
	res.optimizedSample = truncRunes(optimizedText, payloadSampleChars)

	payload, err = engines.ApplyAnthropicCacheAlignment(payload, provider)
	if err != nil {
		return res, stepErr("cache_alignment", err)
	}

	// The rewrite must not make the request bigger. When the count goes up,
	// send the original bytes.
	if origTokens-optimizedTokens < 0 {
		res.body = originalBody
		res.optimizedTokens = origTokens
		cats = nil
	} else {
		res.body, err = engines.CanonicalJSONDumps(payload)
		if err != nil {
			return res, stepErr("canonical_json", err)
		}
		res.optimizedTokens = optimizedTokens
	}
	if f.compression && sessionKey != "" && engines.IsDecompress(sessionKey) {
		engines.TickDecompress(sessionKey)
	}
	res.payload = payload
	res.cats = cats
	return res, nil
}

// applyAggressivePass runs the aggressive pass for the payload's shape. It
// mirrors the elif chain in _crush_json_conversation. Shape checks run lazily,
// in the same order as Python, so an error is raised only where Python raises.
func applyAggressivePass(payload *pyjson.Object, f optFlags, bypass bool, sessionKey string) ([]string, error) {
	if f.aggressive && f.compression {
		anth, err := isAnthropicMessagesPayload(payload)
		if err != nil {
			return nil, err
		}
		if anth {
			return engines.ApplyAggressive(payload, sessionKey)
		}
		chat, err := isOpenAIChatPayload(payload)
		if err != nil {
			return nil, err
		}
		if chat && !bypass {
			return engines.ApplyAggressiveOpenAI(payload)
		}
		resp, err := isOpenAIResponsesPayload(payload)
		if err != nil {
			return nil, err
		}
		if resp {
			return engines.ApplyAggressiveResponses(payload)
		}
	}
	if f.compression {
		anth, err := isAnthropicMessagesPayload(payload)
		if err != nil {
			return nil, err
		}
		if anth {
			return engines.ApplyCacheAndStructureFixes(payload)
		}
	}
	return nil, nil
}

// wireCrushResult is what the wire branch of forward_request needs back.
type wireCrushResult struct {
	body                            []byte
	cats                            []string
	originalTokens, optimizedTokens int
	originalSample, optimizedSample string
}

// crushWire mirrors the non-JSON crush in forward_request: the Connect, gRPC
// and protobuf bodies, and any other body that the wire crush can shrink. It
// returns the body to forward. A decline returns ErrUnsupported with the
// original body, which the caller forwards unchanged.
func crushWire(body []byte, path, contentType string, origTokens int, origSample string,
	compression, aggressive bool) (wireCrushResult, error) {
	out := wireCrushResult{body: body, originalTokens: origTokens, optimizedTokens: origTokens,
		originalSample: origSample, optimizedSample: origSample}
	if !compression || len(body) < 200 {
		return out, nil
	}
	origLen := len(body)
	newBody, cats, saved, err := engines.CrushWireBody(body, path, contentType, compression && aggressive)
	if err != nil {
		return out, stepErr("wire_crush", err)
	}
	if len(cats) == 0 || (saved <= 0 && len(newBody) >= origLen) {
		return out, nil
	}

	out.body = newBody
	out.cats = cats
	// Python checks the body before the crush, with bytes.lstrip().
	if t := bytes.TrimLeft(body, " \t\n\r\v\f"); len(t) > 0 && (t[0] == '{' || t[0] == '[') {
		parsed, perr := pyjson.Loads(newBody)
		if perr != nil {
			// Python falls back to a byte estimate when json.loads raises ValueError.
			out.optimizedTokens = max(1, len(newBody)/4)
			out.optimizedSample = fmt.Sprintf("[wire crush %d->%dB]", origLen, len(newBody))
			return out, nil
		}
		tokens, sample, err := engines.MeasureWireTokens(newBody, parsed)
		if err != nil {
			return out, stepErr("measure_wire_tokens", err)
		}
		out.optimizedTokens = tokens
		out.optimizedSample = sample
		return out, nil
	}
	if saved == 0 {
		saved = origLen - len(newBody)
	}
	out.originalTokens = max(origTokens, max(1, origLen/4))
	out.optimizedTokens = max(1, len(newBody)/4)
	out.optimizedSample = fmt.Sprintf("[wire crush %d->%dB saved=%d]", origLen, len(newBody), saved)
	return out, nil
}

// isUnsupported reports whether err is an engine decline (see the policy above).
func isUnsupported(err error) bool {
	return errors.Is(err, engines.ErrUnsupported)
}

// truncRunes returns the first n characters of s, cut on a character boundary.
// Python slices by code point, so a multi-byte character counts as one.
func truncRunes(s string, n int) string {
	count := 0
	for i := range s {
		if count == n {
			return s[:i]
		}
		count++
	}
	return s
}
