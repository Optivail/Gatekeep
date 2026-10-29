package engines

import (
	"testing"
)

// MapFramePayloads re-compresses a changed gzip frame with Go's compress/flate,
// so its bytes differ from Python's gzip.compress output. These tests check the
// decoded payload and the frame flags, which are the parts that must match.
func TestMapFramePayloadsDecodedLevel(t *testing.T) {
	body := encodeFrameBytes(t, []Frame{
		{Flags: 0x01, Payload: gzipCompress([]byte("abc"))},
		{Flags: 0x00, Payload: []byte("xyz")},
		{Flags: 0x02, Payload: []byte("{}")},
	})
	out, changed, err := MapFramePayloads(body, parityTransforms["upper"], true)
	if err != nil || !changed {
		t.Fatalf("MapFramePayloads = changed %v, err %v", changed, err)
	}
	frames, ok := IterFrames(out)
	if !ok || len(frames) != 3 {
		t.Fatalf("output frames = %d, ok %v; want 3", len(frames), ok)
	}
	if frames[0].Flags != 0x01 {
		t.Fatalf("frame 0 flags = %#x, want 0x01", frames[0].Flags)
	}
	if d, err := gzipDecompress(frames[0].Payload); err != nil || string(d) != "ABC" {
		t.Fatalf("frame 0 decoded = %q, %v; want ABC", d, err)
	}
	if frames[1].Flags != 0x00 || string(frames[1].Payload) != "XYZ" {
		t.Fatalf("frame 1 = %#x %q; want 0x00 XYZ", frames[1].Flags, frames[1].Payload)
	}
	// EndStream is skipped: its payload is untouched.
	if frames[2].Flags != 0x02 || string(frames[2].Payload) != "{}" {
		t.Fatalf("frame 2 = %#x %q; want 0x02 {}", frames[2].Flags, frames[2].Payload)
	}
}

func encodeFrameBytes(t *testing.T, frames []Frame) []byte {
	t.Helper()
	out, err := EncodeFrames(frames)
	if err != nil {
		t.Fatalf("EncodeFrames: %v", err)
	}
	return out
}
