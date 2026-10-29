package engines

import (
	"bytes"
	"testing"
)

// pbVarint encodes a varint field (wire type 0).
func pbVarint(fn uint64, v uint64) []byte {
	out, ok := appendField(nil, fn, 0, writeVarint(v))
	if !ok {
		panic("bad field")
	}
	return out
}

// pbBytes encodes a length-delimited field (wire type 2).
func pbBytes(fn uint64, val []byte) []byte {
	out, ok := appendField(nil, fn, 2, val)
	if !ok {
		panic("bad field")
	}
	return out
}

// pbStr encodes a string field.
func pbStr(fn uint64, s string) []byte { return pbBytes(fn, []byte(s)) }

func cat(parts ...[]byte) []byte {
	var out []byte
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// connectFrame wraps a payload in a Connect envelope with the given flags.
func connectFrame(flags byte, payload []byte) []byte {
	out := []byte{flags, 0, 0, 0, 0}
	n := len(payload)
	out[1], out[2], out[3], out[4] = byte(n>>24), byte(n>>16), byte(n>>8), byte(n)
	return append(out, payload...)
}

func TestIsServerConfigPath(t *testing.T) {
	cases := map[string]bool{
		"aiserver.v1.ServerConfigService/GetServerConfig": true,
		"/aiserver.v1.GetServerConfigRequest":             true,
		"aiserver.v1.ServerConfigService/Other":           true,
		"AISERVER.V1.GETSERVERCONFIG":                     true,
		"aiserver.v1.AiService/StreamChat":                false,
		"agent.v1.AgentService/Run":                       false,
		"":                                                false,
	}
	for path, want := range cases {
		if got := IsServerConfigPath(path); got != want {
			t.Errorf("IsServerConfigPath(%q) = %v, want %v", path, got, want)
		}
	}
}

func TestIsAgentRunURL(t *testing.T) {
	cases := []struct {
		url  string
		want bool
	}{
		{"https://agentn.global.api5.cursor.sh", true},
		{"https://agent.cursor.sh/x", true},
		{"https://agent-gcpp.cursor.sh", true},
		{"https://agent-anything.cursor.sh", true},
		{"https://AGENT.CURSOR.SH", true},
		{"https://user@agent.cursor.sh:443/path", true},
		{"https://agent.cursor.sh\t", true},
		{"https://agen\tt.cursor.sh", true},
		{"https://agent-x%.cursor.sh", true},
		{"https://agent%.cursor.sh", false},
		{"https://agentx.cursor.sh", false},
		{"https://api2.cursor.sh", false},
		{"https://api.cursor.sh", false},
		{"https://repo42.cursor.sh", false},
		{"https://staging.cursor.sh", false},
		{"http://agent.cursor.sh", false},
		{"https://agent.cursor.sh.evil.example", false},
		{"https://agent.example.com", false},
		{"https://[::1]/", false},
		{"https://[agent.cursor.sh", false},
		{"https://agent.cursor.sh]", false},
		{"https://", false},
		{"", false},
		{"  https://agent.cursor.sh  ", true},
	}
	for _, tc := range cases {
		if got := isAgentRunURL(tc.url); got != tc.want {
			t.Errorf("isAgentRunURL(%q) = %v, want %v", tc.url, got, tc.want)
		}
	}
}

func TestRewriteProtobufAgentURLs(t *testing.T) {
	const gk = "http://gk.test:9478/"
	agent := "https://agentn.global.api5.cursor.sh"
	cases := []struct {
		name string
		in   []byte
		want []byte
	}{
		{
			name: "agent url in a string field",
			in:   pbStr(1, agent),
			want: pbStr(1, "http://gk.test:9478"),
		},
		{
			name: "http2 config is forced to the forced-bidi-disabled value",
			in:   pbVarint(7, 0),
			want: pbVarint(7, 3),
		},
		{
			name: "http2 config already forced is unchanged",
			in:   pbVarint(7, 3),
			want: pbVarint(7, 3),
		},
		{
			name: "a varint in another field is not touched",
			in:   pbVarint(2, 0),
			want: pbVarint(2, 0),
		},
		{
			name: "nested message with an agent url",
			in:   pbBytes(2, pbStr(1, agent)),
			want: pbBytes(2, pbStr(1, "http://gk.test:9478")),
		},
		{
			name: "non-agent host is left alone",
			in:   pbStr(1, "https://api2.cursor.sh/x"),
			want: pbStr(1, "https://api2.cursor.sh/x"),
		},
		{
			name: "non-utf8 bytes that are not a message are left alone",
			in:   pbBytes(1, []byte{0xff, 0xfe, 0xfd}),
			want: pbBytes(1, []byte{0xff, 0xfe, 0xfd}),
		},
		{
			name: "garbage that does not parse is returned as is",
			in:   []byte{0x08},
			want: []byte{0x08},
		},
		{
			name: "empty input",
			in:   nil,
			want: nil,
		},
		{
			name: "two agent urls and http2 config together",
			in: cat(pbStr(1, agent), pbStr(3, "https://agent-gcpp.cursor.sh"),
				pbVarint(7, 1)),
			want: cat(pbStr(1, "http://gk.test:9478"), pbStr(3, "http://gk.test:9478"),
				pbVarint(7, 3)),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ForgetAgentUpstream()
			got := RewriteProtobufAgentURLs(tc.in, gk)
			if !bytes.Equal(got, tc.want) {
				t.Fatalf("got %x, want %x", got, tc.want)
			}
		})
	}
}

func TestRewriteProtobufAgentURLsCachesUpstream(t *testing.T) {
	ForgetAgentUpstream()
	t.Cleanup(ForgetAgentUpstream)
	RewriteProtobufAgentURLs(pbStr(1, "https://agent.cursor.sh/run"), "http://gk")
	if got := GetAgentUpstream(); got != "https://agent.cursor.sh/run" {
		t.Fatalf("cached upstream %q", got)
	}
	// A non-https URL is never cached.
	ForgetAgentUpstream()
	RememberAgentUpstream("http://agent.cursor.sh")
	if got := GetAgentUpstream(); got != "" {
		t.Fatalf("http upstream was cached: %q", got)
	}
	RememberAgentUpstream("  https://agentn.global.api5.cursor.sh/  ")
	if got := GetAgentUpstream(); got != "https://agentn.global.api5.cursor.sh" {
		t.Fatalf("trimmed upstream %q", got)
	}
}

func TestRewriteConnectResponse(t *testing.T) {
	ForgetAgentUpstream()
	t.Cleanup(ForgetAgentUpstream)
	const gk = "http://gk:9478"
	agentMsg := pbStr(1, "https://agentn.global.api5.cursor.sh")
	plainMsg := pbStr(1, "hello")

	// Frame with an agent url, a frame that is unchanged, and an EndStream frame.
	body := cat(
		connectFrame(0, agentMsg),
		connectFrame(0, plainMsg),
		connectFrame(2, []byte(`{"error":{"code":"x"}}`)),
	)
	got, err := RewriteConnectResponse(body, gk)
	if err != nil {
		t.Fatalf("RewriteConnectResponse: %v", err)
	}
	want := cat(
		connectFrame(0, pbStr(1, gk)),
		connectFrame(0, plainMsg),
		connectFrame(2, []byte(`{"error":{"code":"x"}}`)),
	)
	if !bytes.Equal(got, want) {
		t.Fatalf("framed body\n got %x\nwant %x", got, want)
	}

	// No change: the body comes back as it was.
	unchanged := cat(connectFrame(0, plainMsg))
	got, err = RewriteConnectResponse(unchanged, gk)
	if err != nil || !bytes.Equal(got, unchanged) {
		t.Fatalf("unchanged body: %x, %v", got, err)
	}

	// A body that is not framed is rewritten as one message.
	raw := pbStr(1, "https://agent.cursor.sh")
	got, err = RewriteConnectResponse(raw, gk)
	if err != nil || !bytes.Equal(got, pbStr(1, gk)) {
		t.Fatalf("raw body: %x, %v", got, err)
	}

	// Empty body.
	got, err = RewriteConnectResponse(nil, gk)
	if err != nil || len(got) != 0 {
		t.Fatalf("empty body: %x, %v", got, err)
	}
}

func TestMaybeRewriteServerConfig(t *testing.T) {
	const gk = "http://gk.test:9478"
	agentMsg := pbStr(1, "https://agentn.global.api5.cursor.sh")
	body := connectFrame(0, agentMsg)
	rewritten := connectFrame(0, pbStr(1, gk))

	t.Run("not a config path", func(t *testing.T) {
		ForgetAgentUpstream()
		got, rw, err := MaybeRewriteServerConfig("aiserver.v1.AiService/StreamChat", body, gk)
		if err != nil || !bytes.Equal(got, body) || rw != nil {
			t.Fatalf("got %x %v %v", got, rw, err)
		}
		if GetAgentUpstream() != "" {
			t.Fatal("a non-config path cached an upstream")
		}
	})

	t.Run("config path, first and second call", func(t *testing.T) {
		ForgetAgentUpstream()
		t.Cleanup(ForgetAgentUpstream)
		path := "aiserver.v1.ServerConfigService/GetServerConfig"
		got, rw, err := MaybeRewriteServerConfig(path, body, gk)
		if err != nil || !bytes.Equal(got, rewritten) {
			t.Fatalf("first call body %x, %v", got, err)
		}
		wantRW := []string{"agent_urls->" + gk, "cached_upstream=https://agentn.global.api5.cursor.sh"}
		if !equalStrings(rw, wantRW) {
			t.Fatalf("first call rewrites %q, want %q", rw, wantRW)
		}
		// Second call: the body has no agent url left to change, but the
		// upstream is already cached, so it is reported as unchanged.
		got, rw, err = MaybeRewriteServerConfig(path, rewritten, gk)
		if err != nil || !bytes.Equal(got, rewritten) {
			t.Fatalf("second call body %x, %v", got, err)
		}
		wantRW = []string{"agent_urls_unchanged->" + gk}
		if !equalStrings(rw, wantRW) {
			t.Fatalf("second call rewrites %q, want %q", rw, wantRW)
		}
	})

	t.Run("empty body and empty origin are no-ops", func(t *testing.T) {
		ForgetAgentUpstream()
		path := "aiserver.v1.ServerConfigService/GetServerConfig"
		if got, rw, _ := MaybeRewriteServerConfig(path, nil, gk); len(got) != 0 || rw != nil {
			t.Fatalf("empty body: %x %v", got, rw)
		}
		if got, rw, _ := MaybeRewriteServerConfig(path, body, ""); !bytes.Equal(got, body) || rw != nil {
			t.Fatalf("empty origin: %x %v", got, rw)
		}
	})
}

func TestTakeTailReturnsPartialFrame(t *testing.T) {
	b := NewConnectFrameBuffer(false)
	frame := connectFrame(0, []byte("abcdef"))
	frames, err := b.Feed(frame[:4])
	if err != nil || len(frames) != 0 {
		t.Fatalf("partial feed: %v %d", err, len(frames))
	}
	if tail := b.TakeTail(); !bytes.Equal(tail, frame[:4]) {
		t.Fatalf("tail %x, want %x", tail, frame[:4])
	}
	if tail := b.TakeTail(); len(tail) != 0 {
		t.Fatalf("second take returned %x", tail)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
