// Mirrors backend/engines/connect_frames.py: Connect protocol framing,
// [flags:1][len:4 BE][payload], with gzip on flag bit 0 (0x01) and EndStream
// on flag bit 1 (0x02).
package engines

import (
	"bytes"
	"encoding/binary"
	"math"
)

// Frame is one Connect envelope: (flags, payload).
type Frame struct {
	Flags   int
	Payload []byte
}

// IterFrames mirrors iter_frames. ok is false where Python returns None
// (fewer than 5 bytes, a truncated frame, or trailing bytes).
func IterFrames(body []byte) (frames []Frame, ok bool) {
	if len(body) < 5 {
		return nil, false
	}
	i := 0
	for i+5 <= len(body) {
		flags := int(body[i])
		ln := int(binary.BigEndian.Uint32(body[i+1 : i+5]))
		i += 5
		if i+ln > len(body) {
			return nil, false
		}
		frames = append(frames, Frame{Flags: flags, Payload: body[i : i+ln]})
		i += ln
	}
	if i != len(body) {
		return nil, false
	}
	return frames, true
}

// EncodeFrames mirrors encode_frames. Flags outside a byte, or payloads too
// large for a 4-byte length, raise in Python (ValueError/OverflowError).
func EncodeFrames(frames []Frame) ([]byte, error) {
	var out bytes.Buffer
	for _, f := range frames {
		if f.Flags < 0 || f.Flags > 255 {
			return nil, pyError("ValueError", "byte must be in range(0, 256)")
		}
		if uint64(len(f.Payload)) > math.MaxUint32 {
			return nil, pyError("OverflowError", "int too big to convert")
		}
		out.WriteByte(byte(f.Flags))
		out.Write(binary.BigEndian.AppendUint32(nil, uint32(len(f.Payload))))
		out.Write(f.Payload)
	}
	return out.Bytes(), nil
}

// DrainCompleteFrames mirrors drain_complete_frames: it returns the unparsed
// remainder and every complete frame in buf.
func DrainCompleteFrames(buf []byte) (rest []byte, frames []Frame) {
	i := 0
	n := len(buf)
	for i+5 <= n {
		flags := int(buf[i])
		ln := int(binary.BigEndian.Uint32(buf[i+1 : i+5]))
		end := i + 5 + ln
		if end > n {
			break
		}
		frames = append(frames, Frame{Flags: flags, Payload: buf[i+5 : end]})
		i = end
	}
	return buf[i:], frames
}

// FrameToBytes mirrors frame_to_bytes. With decompress set, a gzip payload
// is inflated and its flag cleared. A bad gzip header or CRC keeps the frame
// as it was; truncation and corrupt deflate data are returned as errors, as
// Python lets them propagate.
func FrameToBytes(flags int, payload []byte, decompress bool) ([]byte, error) {
	if decompress && flags&0x01 != 0 {
		out, err := gzipDecompress(payload)
		switch {
		case err == nil:
			payload = out
			flags &^= 0x01
		case !isGzipOSError(err):
			return nil, err
		}
	}
	if flags < 0 || flags > 255 {
		return nil, pyError("ValueError", "bytes must be in range(0, 256)")
	}
	out := make([]byte, 0, 5+len(payload))
	out = append(out, byte(flags))
	out = binary.BigEndian.AppendUint32(out, uint32(len(payload)))
	return append(out, payload...), nil
}

// ConnectFrameBuffer mirrors ConnectFrameBuffer: it reassembles frames that
// arrive split across HTTP/2 DATA chunks. It is not safe for concurrent use,
// as in Python.
type ConnectFrameBuffer struct {
	buf        []byte
	decompress bool
}

// NewConnectFrameBuffer mirrors ConnectFrameBuffer(decompress=...).
func NewConnectFrameBuffer(decompress bool) *ConnectFrameBuffer {
	return &ConnectFrameBuffer{decompress: decompress}
}

// Feed mirrors ConnectFrameBuffer.feed. The buffer is advanced before frames
// are converted, so a conversion error leaves the same leftover state Python
// does.
func (b *ConnectFrameBuffer) Feed(chunk []byte) ([][]byte, error) {
	b.buf = append(append([]byte(nil), b.buf...), chunk...)
	var out [][]byte
	for {
		rest, frames := DrainCompleteFrames(b.buf)
		b.buf = rest
		if len(frames) == 0 {
			break
		}
		for _, f := range frames {
			fb, err := FrameToBytes(f.Flags, f.Payload, b.decompress)
			if err != nil {
				return nil, err
			}
			out = append(out, fb)
		}
	}
	return out, nil
}

// FlushComplete mirrors ConnectFrameBuffer.flush_complete.
func (b *ConnectFrameBuffer) FlushComplete() ([][]byte, error) {
	return b.Feed(nil)
}

// DebugConnectChunk mirrors debug_connect_chunk. The Python logging side
// effect is dropped; the return value (the EndStream bit) is what callers use.
func DebugConnectChunk(data []byte, label string) bool {
	if len(data) < 5 {
		return false
	}
	return data[0]&0x02 != 0
}

// DecompressConnectBody mirrors decompress_connect_body: it strips gzip from
// flagged frames for clients that expect plain envelopes. The body is returned
// untouched when nothing changed.
func DecompressConnectBody(body []byte) ([]byte, error) {
	frames, ok := IterFrames(body)
	if !ok {
		return body, nil
	}
	out := make([]Frame, 0, len(frames))
	changed := false
	for _, f := range frames {
		flags, payload := f.Flags, f.Payload
		if flags&0x01 != 0 {
			d, err := gzipDecompress(payload)
			switch {
			case err == nil:
				payload = d
				flags &^= 0x01
				changed = true
			case !isGzipOSError(err):
				return nil, err
			}
		}
		out = append(out, Frame{Flags: flags, Payload: payload})
	}
	if !changed {
		return body, nil
	}
	return EncodeFrames(out)
}

// MapFramePayloads mirrors map_frame_payloads. fn is applied to each frame
// payload (inflated first when flagged). ok reports whether the body changed.
//
// Caveat: a changed gzip-flagged frame is re-compressed with gzipCompress,
// which is not byte-identical to Python's zlib output. Its decompressed
// content matches; only the deflate bytes and the mtime differ.
func MapFramePayloads(body []byte, fn func([]byte) []byte, skipEndStream bool) ([]byte, bool, error) {
	frames, ok := IterFrames(body)
	if !ok {
		nw := fn(body)
		return nw, !bytes.Equal(nw, body), nil
	}
	out := make([]Frame, 0, len(frames))
	changed := false
	for _, f := range frames {
		flags, payload := f.Flags, f.Payload
		if skipEndStream && flags&0x02 != 0 {
			out = append(out, f)
			continue
		}
		raw := payload
		compressed := flags&0x01 != 0
		if compressed {
			d, err := gzipDecompress(payload)
			switch {
			case err == nil:
				raw = d
			case isGzipOSError(err):
				raw = payload
				compressed = false
			default:
				return nil, false, err
			}
		}
		mapped := fn(raw)
		if !bytes.Equal(mapped, raw) {
			changed = true
		}
		if compressed {
			out = append(out, Frame{Flags: flags | 0x01, Payload: gzipCompress(mapped)})
		} else {
			out = append(out, Frame{Flags: flags, Payload: mapped})
		}
	}
	if !changed {
		return body, false, nil
	}
	enc, err := EncodeFrames(out)
	if err != nil {
		return nil, false, err
	}
	return enc, true, nil
}
