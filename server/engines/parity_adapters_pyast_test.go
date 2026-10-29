package engines

import "github.com/Optivail/Gatekeep/server/pyjson"

func init() {
	parityAdapters["pyast.distill_python_signatures"] = func(a *pyjson.Object) (any, error) {
		code, err := argStr(a, "code")
		if err != nil {
			return nil, err
		}
		out, err := distillPythonSignatures(code)
		if err != nil {
			return nil, err
		}
		return out, nil
	}
	parityAdapters["pyast.literal_name"] = func(a *pyjson.Object) (any, error) {
		name, err := argStr(a, "name")
		if err != nil {
			return nil, err
		}
		r, ok, err := pyNameLookup(name)
		if err != nil {
			return nil, err
		}
		if !ok {
			return nil, pyError("SyntaxError", "unknown Unicode character name")
		}
		return int64(r), nil
	}
}
