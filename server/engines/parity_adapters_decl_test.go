// Mirrors the body_codec and connect_frames adapters in server/parity/
// engine_cases.py (the decline-class section). Test-only.
//
// Connect frames that were re-compressed are compared at decoded-payload
// level: the view lists each frame's flags and its gunzipped payload, never
// the gzip bytes. gzip writes the current time into its header, and the
// deflate output of Go's compress/flate is not zlib's, so bytes can differ
// between two correct implementations.
package engines

import (
	"encoding/base64"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

func init() {
	parityAdapters["body_codec.decode_request_body"] = parityBodyCodecDecode
	parityAdapters["connect_frames.map_frame_payloads"] = parityConnectMap
}

func parityBodyCodecDecode(a *pyjson.Object) (any, error) {
	body, err := argB64(a, "body_b64")
	if err != nil {
		return nil, err
	}
	headers, err := argObj(a, "headers")
	if err != nil {
		return nil, err
	}
	out, ok, err := DecodeRequestBody(body, headers)
	if err != nil {
		return nil, err
	}
	return []any{base64.StdEncoding.EncodeToString(out), ok, headers}, nil
}

// parityTransforms holds the payload maps used by the corpus. Each matches the
// lambda of the same name in engine_cases.py.
var parityTransforms = map[string]func([]byte) []byte{
	"identity": func(b []byte) []byte { return b },
	"upper": func(b []byte) []byte {
		// bytes.upper() changes ASCII letters only.
		out := append([]byte(nil), b...)
		for i, c := range out {
			if c >= 'a' && c <= 'z' {
				out[i] = c - 32
			}
		}
		return out
	},
	"append_x": func(b []byte) []byte { return append(append([]byte(nil), b...), 'x') },
	"halve":    func(b []byte) []byte { return b[: len(b)/2 : len(b)/2] },
}

func parityConnectMap(a *pyjson.Object) (any, error) {
	body, err := argB64(a, "body_b64")
	if err != nil {
		return nil, err
	}
	name, err := argStr(a, "transform")
	if err != nil {
		return nil, err
	}
	fn, ok := parityTransforms[name]
	if !ok {
		return nil, pyError("KeyError", name)
	}
	out, changed, err := MapFramePayloads(body, fn, argBool(a, "skip_end_stream"))
	if err != nil {
		return nil, err
	}
	return []any{changed, parityFrameView(out)}, nil
}

// parityFrameView mirrors _frame_view in engine_cases.py.
func parityFrameView(body []byte) any {
	frames, ok := IterFrames(body)
	if !ok {
		return []any{"raw", base64.StdEncoding.EncodeToString(body)}
	}
	view := make([]any, 0, len(frames))
	for _, f := range frames {
		decoded := f.Payload
		if f.Flags&0x01 != 0 {
			if d, err := gzipDecompress(f.Payload); err == nil {
				decoded = d
			}
		}
		view = append(view, []any{int64(f.Flags), base64.StdEncoding.EncodeToString(decoded)})
	}
	return []any{"frames", view}
}
