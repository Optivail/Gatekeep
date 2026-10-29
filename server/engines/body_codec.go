// Mirrors backend/engines/body_codec.py: decode a gzip or zstd request body
// before the engines read it.
package engines

import (
	"bytes"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// maxDecodedBody caps the decoded size (Python's _MAX_OUT).
const maxDecodedBody = 256 * 1024 * 1024

var (
	zstdMagic = []byte{0x28, 0xb5, 0x2f, 0xfd}
	gzipMagic = []byte{0x1f, 0x8b}
)

// DecodeRequestBody mirrors decode_request_body. On success the
// Content-Encoding entries are removed from headers (every case variant), as
// the Python dict is mutated in place. Any decode failure returns the body
// unchanged and false; Python catches every exception there, so no error is
// returned. The error result only reports a header value that is not a str,
// which Python's .strip() would raise on.
func DecodeRequestBody(body []byte, headers *pyjson.Object) ([]byte, bool, error) {
	if len(body) == 0 {
		return body, false, nil
	}

	enc := ""
	for _, key := range headers.Keys() {
		if pyLower(key) == "content-encoding" {
			val, err := strOrEmpty(objGet(headers, key))
			if err != nil {
				return body, false, err
			}
			enc = pyLower(pyStrip(val))
			break
		}
	}

	var codecs []string
	if enc != "" {
		for _, c := range splitComma(enc) {
			if c = pyStrip(c); c != "" {
				codecs = append(codecs, c)
			}
		}
	}
	if len(codecs) == 0 {
		switch {
		case bytes.HasPrefix(body, zstdMagic):
			codecs = []string{"zstd"}
		case bytes.HasPrefix(body, gzipMagic):
			codecs = []string{"gzip"}
		default:
			return body, false, nil
		}
	}
	if len(codecs) != 1 {
		return body, false, nil
	}

	var out []byte
	switch codecs[0] {
	case "zstd", "zst":
		// Python decodes this with the zstandard package. No zstd decoder is
		// available to this module, so the caller must hand the request to
		// the Python backend. Returning the body unchanged here would forward
		// bytes the client did not mean to send.
		return body, false, ErrUnsupported
	case "gzip", "x-gzip":
		d, err := gzipDecompress(body)
		if err != nil {
			return body, false, nil
		}
		if len(d) > maxDecodedBody {
			return body, false, nil
		}
		out = d
	default:
		return body, false, nil
	}
	if len(out) > maxDecodedBody {
		return body, false, nil
	}

	for _, key := range headers.Keys() {
		if pyLower(key) == "content-encoding" {
			headers.Delete(key)
		}
	}
	return out, true, nil
}

// splitComma mirrors str.split(",").
func splitComma(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == ',' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return append(out, s[start:])
}
