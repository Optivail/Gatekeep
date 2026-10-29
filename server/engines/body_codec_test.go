package engines

import (
	"encoding/hex"
	"testing"

	"github.com/Optivail/Gatekeep/server/pyjson"
)

// Frames below are python-zstandard 0.25 output or hand-built frames. The
// expected results are CPython's decompress(body, max_output_size=256 MiB):
// only the first frame is decoded, a declared content size of 0 returns b""
// without reading the blocks, and any other error leaves the body unchanged.
func TestZstdFirstFrameMatchesPythonZstandard(t *testing.T) {
	cases := []struct {
		name, hexBody string
		ok            bool
		want          string
	}{
		{"two frames: only the first", "28b52ffd2005290000666972737428b52ffd20063100007365636f6e64", true, "first"},
		{"trailing bytes ignored", "28b52ffd200529000066697273746a756e6b21", true, "first"},
		{"content size 0 skips blocks", "28b52ffd20002900006162636465", true, ""},
		{"RLE block", "28b52ffd20042300007a", true, "zzzz"},
		{"leading skippable frame", "502a4d180300000061626328b52ffd20052900006669727374", false, ""},
		{"reserved block type", "28b52ffd00581f0000616263", false, ""},
		{"window above 128 MiB", "28b52ffd00a0190000616263", false, ""},
		{"bad checksum", "28b52ffd24073900007061796c6f61645c2c2200", false, ""},
	}
	for _, c := range cases {
		body, _ := hex.DecodeString(c.hexBody)
		headers := pyjson.NewObject()
		headers.Set("content-encoding", "zstd")
		out, ok, err := DecodeRequestBody(body, headers)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if ok != c.ok {
			t.Errorf("%s: decoded = %v, want %v", c.name, ok, c.ok)
			continue
		}
		if ok && string(out) != c.want {
			t.Errorf("%s: out = %q, want %q", c.name, out, c.want)
		}
		if !ok && string(out) != string(body) {
			t.Errorf("%s: a failed decode must return the body unchanged", c.name)
		}
	}
}
