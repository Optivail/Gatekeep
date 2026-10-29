package engines

// Port of backend/engines/cursor_rewrite.py: the GetServerConfig response of
// Cursor names the agent endpoint. These functions rewrite that endpoint to
// Gatekeep's own origin, so the client's agent traffic comes back through the
// agent listener. The Gatekeep origin is an argument here; Python reads it from
// config.agent_public_url() inside the module.

import (
	"bytes"
	"strings"
	"sync"
	"unicode/utf8"
)

const (
	// http2ConfigField is aiserver.v1.GetServerConfigResponse.http2_config.
	http2ConfigField = 7
	// forceBidiDisabled is HTTP2_CONFIG_FORCE_BIDI_DISABLED = 3. It forces the
	// client onto the HTTP/1.1 agent path.
	forceBidiDisabled = 3
)

// cursorBlockedHosts are the Cursor API hosts that never carry agent traffic.
var cursorBlockedHosts = map[string]bool{
	"api2.cursor.sh":        true,
	"api3.cursor.sh":        true,
	"api.cursor.sh":         true,
	"repo42.cursor.sh":      true,
	"staging.cursor.sh":     true,
	"dev-staging.cursor.sh": true,
}

// agentUpstream is _agent_upstream: the last agent endpoint seen in a
// GetServerConfig response. It stays set for the life of the process.
var agentUpstream struct {
	mu  sync.Mutex
	url string
}

// GetAgentUpstream returns the cached agent endpoint, or "" when none has been
// seen (Python's None).
func GetAgentUpstream() string {
	agentUpstream.mu.Lock()
	defer agentUpstream.mu.Unlock()
	return agentUpstream.url
}

// RememberAgentUpstream caches url as the agent endpoint when it is an https
// URL (_remember_agent_upstream). It is exported for the proxy and for tests.
func RememberAgentUpstream(url string) {
	u := strings.TrimRight(pyStrip(url), "/")
	if !strings.HasPrefix(u, "https://") {
		return
	}
	agentUpstream.mu.Lock()
	agentUpstream.url = u
	agentUpstream.mu.Unlock()
}

// ForgetAgentUpstream clears the cached agent endpoint. Tests use it to start
// from the same state as a fresh process.
func ForgetAgentUpstream() {
	agentUpstream.mu.Lock()
	agentUpstream.url = ""
	agentUpstream.mu.Unlock()
}

// IsServerConfigPath reports whether a request path is a GetServerConfig call
// (is_server_config_path).
func IsServerConfigPath(path string) bool {
	p := pyLower(strings.TrimLeft(path, "/"))
	return strings.Contains(p, "getserverconfig") || strings.Contains(p, "serverconfigservice")
}

// pyHostname returns the hostname of an https URL the way urllib.parse does:
// tab, CR and LF are removed first, the netloc ends at the first "/", "?" or
// "#", and the hostname is lower-cased with any "%zone" kept as written. ok is
// false where urlsplit raises ValueError (an unbalanced IPv6 bracket).
func pyHostname(raw string) (host string, ok bool) {
	s := strings.NewReplacer("\t", "", "\r", "", "\n", "").Replace(raw)
	rest := strings.TrimPrefix(s, "https://")
	netloc := rest
	if i := strings.IndexAny(rest, "/?#"); i >= 0 {
		netloc = rest[:i]
	}
	hasOpen := strings.Contains(netloc, "[")
	hasClose := strings.Contains(netloc, "]")
	if hasOpen != hasClose {
		return "", false
	}
	hostinfo := netloc
	if i := strings.LastIndex(netloc, "@"); i >= 0 {
		hostinfo = netloc[i+1:]
	}
	var hostname string
	if i := strings.Index(hostinfo, "["); i >= 0 {
		bracketed := hostinfo[i+1:]
		if j := strings.Index(bracketed, "]"); j >= 0 {
			hostname = bracketed[:j]
		} else {
			hostname = bracketed
		}
	} else {
		hostname = hostinfo
		if j := strings.Index(hostname, ":"); j >= 0 {
			hostname = hostname[:j]
		}
	}
	if hostname == "" {
		return "", true
	}
	if i := strings.Index(hostname, "%"); i >= 0 {
		return pyLower(hostname[:i]) + hostname[i:], true
	}
	return pyLower(hostname), true
}

// isAgentRunURL reports whether a string is a Cursor agent endpoint
// (_is_agent_run_url).
func isAgentRunURL(raw string) bool {
	s := pyStrip(raw)
	if !strings.HasPrefix(s, "https://") {
		return false
	}
	host, ok := pyHostname(s)
	if !ok {
		return false
	}
	if !strings.HasSuffix(host, ".cursor.sh") {
		return false
	}
	if cursorBlockedHosts[host] {
		return false
	}
	for _, p := range []string{"agent.", "agentn.", "agent-gcpp", "agent-"} {
		if strings.HasPrefix(host, p) {
			return true
		}
	}
	return false
}

// rewriteAgentURL returns the Gatekeep origin for an agent endpoint, after
// caching the endpoint. Any other string is returned unchanged.
func rewriteAgentURL(s, gatekeep string) string {
	if !isAgentRunURL(s) {
		return s
	}
	RememberAgentUpstream(s)
	return strings.TrimRight(gatekeep, "/")
}

// walkStrings is _walk_strings. It returns the message with its agent URLs and
// http2_config rewritten, and whether anything changed. A message that does not
// parse is returned unchanged.
func walkStrings(data []byte, gatekeep string) ([]byte, bool) {
	fields, ok := parseFields(data)
	if !ok {
		return data, false
	}
	var out []byte
	changed := false
	for _, f := range fields {
		val := f.val
		switch {
		case f.wt == 0 && f.fn == http2ConfigField:
			if !bytes.Equal(val, []byte{forceBidiDisabled}) {
				val = []byte{forceBidiDisabled}
				changed = true
			}
		case f.wt == 2:
			if s, valid := decodeStrictUTF8(val); valid && isAgentRunURL(s) {
				if nw := rewriteAgentURL(s, gatekeep); nw != s {
					val = []byte(nw)
					changed = true
				}
			} else if looksLikeMessage(val) {
				if nested, nch := walkStrings(val, gatekeep); nch {
					val = nested
					changed = true
				}
			}
		}
		enc, encOK := appendField(out, f.fn, f.wt, val)
		if !encOK {
			// The key does not fit in 64 bits and cannot be re-encoded exactly.
			// The message is left as it was.
			return data, false
		}
		out = enc
	}
	if !changed {
		return data, false
	}
	return out, true
}

// decodeStrictUTF8 mirrors bytes.decode("utf-8") in Python: it fails on invalid
// sequences and on encoded surrogates.
func decodeStrictUTF8(b []byte) (string, bool) {
	if !utf8.Valid(b) {
		return "", false
	}
	return string(b), true
}

// RewriteProtobufAgentURLs rewrites the agent endpoints inside one protobuf
// message (rewrite_protobuf_agent_urls).
func RewriteProtobufAgentURLs(data []byte, gatekeep string) []byte {
	out, _ := walkStrings(data, gatekeep)
	return out
}

// RewriteConnectResponse rewrites every Connect frame of a response body
// (rewrite_connect_response). A body that is not framed is rewritten as one
// message. The error case is a gzip-flagged frame that does not decompress.
func RewriteConnectResponse(body []byte, gatekeep string) ([]byte, error) {
	if len(body) == 0 {
		return body, nil
	}
	mapped, changed, err := MapFramePayloads(body, func(raw []byte) []byte {
		return RewriteProtobufAgentURLs(raw, gatekeep)
	}, true)
	if err != nil {
		return nil, err
	}
	if changed {
		return mapped, nil
	}
	return RewriteProtobufAgentURLs(body, gatekeep), nil
}

// MaybeRewriteServerConfig rewrites a GetServerConfig response body and reports
// what it did (maybe_rewrite_server_config). The rewrite labels match Python's.
func MaybeRewriteServerConfig(path string, body []byte, gatekeep string) ([]byte, []string, error) {
	if !IsServerConfigPath(path) {
		return body, nil, nil
	}
	if len(body) == 0 || gatekeep == "" {
		return body, nil, nil
	}
	before := GetAgentUpstream()
	newBody, err := RewriteConnectResponse(body, gatekeep)
	if err != nil {
		return nil, nil, err
	}
	var rewrites []string
	cur := GetAgentUpstream()
	if !bytes.Equal(newBody, body) {
		rewrites = append(rewrites, "agent_urls->"+gatekeep)
	} else if cur != "" {
		rewrites = append(rewrites, "agent_urls_unchanged->"+gatekeep)
	}
	if cur != "" && cur != before {
		rewrites = append(rewrites, "cached_upstream="+cur)
	}
	return newBody, rewrites, nil
}

// TakeTail returns the bytes of an incomplete frame that the buffer still
// holds, and empties the buffer. Python reads req_buf._buf for this; the relay
// sends the tail as it is when the stream ends.
func (b *ConnectFrameBuffer) TakeTail() []byte {
	t := b.buf
	b.buf = nil
	return t
}
