// Mirrors the cursor_rewrite adapters in server/parity/engine_cases.py. Test-only.
// The entries are merged into parityAdapters at init time, and the cursor
// rewrite state is cleared before every case, as the Python RESET does.
package engines

import (
	"github.com/Optivail/Gatekeep/server/pyjson"
)

func init() {
	for name, fn := range parityCursorAdapters {
		parityAdapters[name] = fn
	}
	parityResets = append(parityResets, ForgetAgentUpstream)
}

// anyOrNil returns nil for "", which Python's None is in the corpus.
func anyOrNil(s string) any {
	if s == "" {
		return nil
	}
	return s
}

var parityCursorAdapters = map[string]func(*pyjson.Object) (any, error){
	"cursor_rewrite.rewrite_protobuf_agent_urls": func(a *pyjson.Object) (any, error) {
		body, err := argB64(a, "body_b64")
		if err != nil {
			return nil, err
		}
		gk, err := argStr(a, "gatekeep")
		if err != nil {
			return nil, err
		}
		return RewriteProtobufAgentURLs(body, gk), nil
	},
	"cursor_rewrite.rewrite_connect_response": func(a *pyjson.Object) (any, error) {
		body, err := argB64(a, "body_b64")
		if err != nil {
			return nil, err
		}
		gk, err := argStr(a, "gatekeep")
		if err != nil {
			return nil, err
		}
		out, err := RewriteConnectResponse(body, gk)
		if err != nil {
			return nil, err
		}
		// Compared without gzip on the frames, as engine_cases.py does.
		return DecompressConnectBody(out)
	},
	"cursor_rewrite.maybe_rewrite_server_config": func(a *pyjson.Object) (any, error) {
		path, err := argStr(a, "path")
		if err != nil {
			return nil, err
		}
		body, err := argB64(a, "body_b64")
		if err != nil {
			return nil, err
		}
		gk, err := argStr(a, "gatekeep")
		if err != nil {
			return nil, err
		}
		newBody, rw, err := MaybeRewriteServerConfig(path, body, gk)
		if err != nil {
			return nil, err
		}
		plain, err := DecompressConnectBody(newBody)
		if err != nil {
			return nil, err
		}
		return []any{b64Canon(plain), stringsToAny(rw)}, nil
	},
	"cursor_rewrite.is_server_config_path": func(a *pyjson.Object) (any, error) {
		path, err := argStr(a, "path")
		if err != nil {
			return nil, err
		}
		return IsServerConfigPath(path), nil
	},
	"cursor_rewrite.is_agent_run_url": func(a *pyjson.Object) (any, error) {
		u, err := argStr(a, "url")
		if err != nil {
			return nil, err
		}
		return isAgentRunURL(u), nil
	},
	"cursor_rewrite.remember_agent_upstream": func(a *pyjson.Object) (any, error) {
		u, err := argStr(a, "url")
		if err != nil {
			return nil, err
		}
		RememberAgentUpstream(u)
		return anyOrNil(GetAgentUpstream()), nil
	},
	"cursor_rewrite.get_agent_upstream": func(*pyjson.Object) (any, error) {
		return anyOrNil(GetAgentUpstream()), nil
	},
}
