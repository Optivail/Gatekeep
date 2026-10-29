// Mirrors backend/engines/body_codec.py: decode a gzip or zstd request body
// before the engines read it.
package engines

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"

	"github.com/klauspost/compress/zstd"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// maxDecodedBody caps the decoded size (Python's _MAX_OUT).
const maxDecodedBody = 256 * 1024 * 1024

// zstdWindowLimit is libzstd's default decoder window limit (windowLog 27,
// 128 MiB). python-zstandard keeps it, so a multi-segment frame whose window
// is larger fails there with "Frame requires too much memory".
const zstdWindowLimit = 1 << 27

// zstdMaxBlock is the largest block size in the zstd format (128 KiB).
const zstdMaxBlock = 128 * 1024

var (
	zstdMagic = []byte{0x28, 0xb5, 0x2f, 0xfd}
	gzipMagic = []byte{0x1f, 0x8b}

	errZstdFrame = errors.New("body_codec: malformed zstd frame")
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
		d, err := zstdDecodeFirstFrame(body)
		if err != nil {
			return body, false, nil
		}
		out = d
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

	for _, key := range headers.Keys() {
		if pyLower(key) == "content-encoding" {
			headers.Delete(key)
		}
	}
	return out, true, nil
}

// zstdDecodeFirstFrame mirrors python-zstandard's
// ZstdDecompressor().decompress(body, max_output_size=_MAX_OUT) for one
// request body. That call decodes only the first frame: bytes after it are
// ignored, and a leading skippable frame is an error. The output limit only
// applies when the frame header carries no content size. A declared content
// size is trusted as it is, so a larger output is accepted (Python allocates
// that size and does not check it against max_output_size).
//
// Frame extent is found by walking the header and block headers (RFC 8878).
// The extracted frame is then decoded by klauspost/compress/zstd, which
// rejects corrupt data, checksum mismatches and size mismatches as libzstd
// does. Python's exact error text is not reproduced; only the success or
// failure, and the decoded bytes, are compared.
func zstdDecodeFirstFrame(body []byte) ([]byte, error) {
	if !bytes.HasPrefix(body, zstdMagic) {
		return nil, errZstdFrame
	}
	hdrEnd, csize, known, err := zstdFrameHeader(body)
	if err != nil {
		return nil, err
	}
	// python-zstandard returns b"" for a declared content size of 0 without
	// reading the blocks, so block data and checksum are not validated.
	if known && csize == 0 {
		return []byte{}, nil
	}
	n, err := zstdFrameEnd(body, hdrEnd)
	if err != nil {
		return nil, err
	}
	dec, err := zstd.NewReader(nil,
		zstd.WithDecoderConcurrency(1),
		zstd.WithDecoderMaxWindow(1<<41),
		zstd.WithDecoderMaxMemory(1<<40),
	)
	if err != nil {
		return nil, err
	}
	defer dec.Close()
	if err := dec.Reset(bytes.NewReader(body[:n])); err != nil {
		return nil, err
	}
	limit := uint64(maxDecodedBody)
	if known {
		limit = math.MaxInt64 - 1
	}
	out, err := io.ReadAll(io.LimitReader(dec, int64(limit)+1))
	if err != nil {
		return nil, err
	}
	if uint64(len(out)) > limit {
		return nil, errZstdFrame
	}
	return out, nil
}

// zstdFrameHeader parses the frame header at the start of b (magic included).
// It returns the offset just past the header, the declared content size, and
// whether that size is declared. It rejects what libzstd rejects at header
// level: the reserved bit, a window above zstdWindowLimit for multi-segment
// frames, and a truncated header.
func zstdFrameHeader(b []byte) (pos int, csize uint64, known bool, err error) {
	if len(b) < 5 {
		return 0, 0, false, errZstdFrame
	}
	fhd := b[4]
	if fhd&0x08 != 0 { // reserved bit must be zero
		return 0, 0, false, errZstdFrame
	}
	single := fhd&0x20 != 0
	fcsFlag := fhd >> 6
	dictSize := [4]int{0, 1, 2, 4}[fhd&0x03]
	pos = 5

	if !single {
		if pos >= len(b) {
			return 0, 0, false, errZstdFrame
		}
		wd := b[pos]
		pos++
		base := uint64(1) << (10 + uint64(wd>>3))
		if base+base/8*uint64(wd&7) > zstdWindowLimit {
			return 0, 0, false, errZstdFrame
		}
	}

	pos += dictSize
	fcsSize := 0
	switch fcsFlag {
	case 0:
		if single {
			fcsSize = 1
		}
	case 1:
		fcsSize = 2
	case 2:
		fcsSize = 4
	case 3:
		fcsSize = 8
	}
	known = single || fcsFlag != 0
	if pos+fcsSize > len(b) {
		return 0, 0, false, errZstdFrame
	}
	switch fcsSize {
	case 1:
		csize = uint64(b[pos])
	case 2:
		csize = uint64(binary.LittleEndian.Uint16(b[pos:])) + 256
	case 4:
		csize = uint64(binary.LittleEndian.Uint32(b[pos:]))
	case 8:
		csize = binary.LittleEndian.Uint64(b[pos:])
	}
	pos += fcsSize
	return pos, csize, known, nil
}

// zstdFrameEnd walks the block headers from pos and returns the offset just
// past the frame, including the optional checksum. It rejects a block above
// 128 KiB, a reserved block type, and a frame cut short. The checksum flag is
// read from the header byte at b[4].
func zstdFrameEnd(b []byte, pos int) (int, error) {
	hasChecksum := b[4]&0x04 != 0
	for {
		if pos+3 > len(b) {
			return 0, errZstdFrame
		}
		h := uint32(b[pos]) | uint32(b[pos+1])<<8 | uint32(b[pos+2])<<16
		pos += 3
		last := h&1 != 0
		typ := (h >> 1) & 3
		size := int(h >> 3)
		if size > zstdMaxBlock {
			return 0, errZstdFrame
		}
		switch typ {
		case 0, 2: // raw, compressed: size bytes follow
			pos += size
		case 1: // RLE: one byte follows
			pos++
		default: // reserved
			return 0, errZstdFrame
		}
		if pos > len(b) {
			return 0, errZstdFrame
		}
		if last {
			break
		}
	}
	if hasChecksum {
		pos += 4
		if pos > len(b) {
			return 0, errZstdFrame
		}
	}
	return pos, nil
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
