#!/usr/bin/env python3
"""Parity corpus for the Go engines port (server/engines).

The Python engines are the oracle. This script generates realistic and
adversarial inputs, runs every engine function on them, and writes one JSON
case per line:

    {"id": n,
     "steps": [{"fn": "optimizer.optimize_text", "args": {...}}, ...],
     "want": ["<canonical>", ...],
     "err": [true|false, ...]}

Steps in one case share global state (the file-read tracker), so a case
resets RESET before it starts. The Go test (server/engines/parity_test.go)
reads the file, runs the same steps through its own adapters, and compares.

Canonical form of a result:
  str    -> itself. A str that holds a surrogate code point becomes
            "b64:" + base64(UTF-8 with surrogatepass), because a JSON line
            cannot keep two adjacent surrogates apart reliably.
  bytes  -> "b64:" + base64.
  other  -> json.dumps(value, ensure_ascii=False, separators=(",", ":"),
            allow_nan=True). If that text holds a surrogate, the
            ensure_ascii=True form is used instead. Go mirrors both choices.
  raised -> want "" and err true.

Args never carry adjacent surrogates (the generator avoids them), so a JSON
round trip is exact. Bytes travel as base64 under the keys body_b64 and
raw_body_b64.

Run with the Gatekeep venv (needs tiktoken):

    python -I server/parity/engine_cases.py --out <file> --count 5000 --seed 1
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from engines import error_guard, extras, loopwatch, optimizer  # noqa: E402

_SURROGATE = re.compile("[\ud800-\udfff]")


# --- canonical form ---------------------------------------------------------


def canonical(value: Any) -> str:
    if isinstance(value, bytes):
        return "b64:" + base64.b64encode(value).decode("ascii")
    if isinstance(value, str):
        if _SURROGATE.search(value):
            return "b64:" + base64.b64encode(value.encode("utf-8", "surrogatepass")).decode("ascii")
        return value
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=True)
    if _SURROGATE.search(text):
        text = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=True)
    return text


# --- adapters: name -> callable(args dict) -> Python value ------------------


def _b64(args: dict, key: str) -> bytes:
    return base64.b64decode(args[key])


def _count_tokens(a):
    return optimizer.count_tokens(a["text"])


def _optimize_text(a):
    return optimizer.optimize_text(a["text"])


def _inject_anti_yap(a):
    return optimizer.inject_anti_yap(a["system"])


def _content_text(a):
    return optimizer._content_text(a["content"])


def _optimize_content(a):
    return optimizer._optimize_content(a["content"])


def _prune_middle_content(a):
    return optimizer._prune_middle_content(a["text"], a["log_truncation_enabled"])


def _truncate_verbose_logs(a):
    return optimizer._truncate_verbose_logs(a["text"])


def _is_error_payload(a):
    return error_guard.is_error_payload(
        a["text"], tool_name=a["tool_name"], is_error_flag=a["is_error_flag"]
    )


def _preserve_diagnostic_text(a):
    return error_guard.preserve_diagnostic_text(a["text"], ceiling=a["ceiling"])


def _extract_text(a):
    return optimizer.extract_text(a["payload"])


def _prune_history(a):
    return optimizer.prune_history(
        a["messages"],
        log_truncation_enabled=a["log_truncation_enabled"],
        turn_squashing_enabled=a["turn_squashing_enabled"],
    )


def _squash_exploratory_chains(a):
    return optimizer._squash_exploratory_chains(a["messages"])


def _optimize_payload(a):
    payload, categories = optimizer.optimize_payload(
        a["payload"],
        compression_enabled=a["compression_enabled"],
        anti_yap_enabled=a["anti_yap_enabled"],
        history_pruning_enabled=a["history_pruning_enabled"],
        log_truncation_enabled=a["log_truncation_enabled"],
    )
    return [payload, categories]


def _cache_alignment(a):
    return optimizer.apply_anthropic_cache_alignment(a["payload"], a["provider"])


def _newest_tool_result(a):
    return optimizer.apply_newest_tool_result_compaction(a["payload"])


def _clamp_max_tokens(a):
    return optimizer.clamp_max_tokens(a["payload"])


def _minify_tools(a):
    return optimizer.minify_tools_deterministic(a["tools"])


def _first_sentence(a):
    return optimizer._first_sentence(a["text"])


def _extract_file_path(a):
    return optimizer._extract_file_path(a["block"])


def _derive_session_key(a):
    return optimizer.derive_session_key(a["payload"])


def _file_delta(a):
    return optimizer.apply_file_delta_tracking(a["payload"], a["session_key"])


def _credential_fingerprint(a):
    return optimizer.credential_fingerprint(a["headers"])


def _request_hash(a):
    return optimizer.compute_request_hash(a["payload"], a["provider"], a["credential"])


def _canonical_json(a):
    return optimizer.canonical_json_dumps(a["data"])


def _measure_wire(a):
    payload = a.get("payload")
    return list(optimizer.measure_wire_tokens(_b64(a, "body_b64"), payload))


def _upstream_usage(a):
    return optimizer.extract_upstream_usage(_b64(a, "raw_body_b64"))


def _strip_volatile_system(a):
    return extras.strip_volatile_system(a["system"])


def _stabilize_tools(a):
    return extras.stabilize_tools(a["tools"])


def _apply_defer_loading(a):
    tools, changed = extras.apply_defer_loading(a["tools"])
    return [tools, changed]


def _extractive_digest(a):
    payload = a["payload"]
    changed = extras.apply_extractive_digest(payload)
    return [changed, payload]


def _strip_media(a):
    payload = a["payload"]
    changed = extras.strip_media_blocks(payload)
    return [changed, payload]


def _sanitize_thinking(a):
    payload = a["payload"]
    changed = extras.sanitize_thinking_blocks(payload)
    return [changed, payload]


def _fold_repeated_log_lines(a):
    return extras.fold_repeated_log_lines(a["text"])


def _cache_and_structure(a):
    payload = a["payload"]
    cats = extras.apply_cache_and_structure_fixes(payload)
    return [cats, payload]


def _normalize_intent(a):
    return extras.normalize_intent_text(a["text"])


def _intent_hash(a):
    return extras.compute_intent_hash(a["payload"], a["provider"])


def _observe(a):
    payload = a["payload"]
    cats = loopwatch.observe(payload, a["session_key"])
    return [cats, payload]


def _loop_clear_session(a):
    return loopwatch.clear_session(a["session_key"])


def _loop_is_relaxed(a):
    return loopwatch.is_relaxed(a["session_key"])


def _loop_is_decompress(a):
    return loopwatch.is_decompress(a["session_key"])


def _loop_tick_decompress(a):
    return loopwatch.tick_decompress(a["session_key"])


def _loop_is_bypass(a):
    return loopwatch.is_bypass_compression(a["session_key"])


def _extract_stacktrace_blocks(a):
    return error_guard.extract_stacktrace_blocks(a["text"])


def _preserve_stacktrace_blocks(a):
    return error_guard.preserve_stacktrace_blocks(a["text"], ceiling=a["ceiling"])


ADAPTERS: dict[str, Callable[[dict], Any]] = {
    "extras.strip_volatile_system": _strip_volatile_system,
    "extras.stabilize_tools": _stabilize_tools,
    "extras.apply_defer_loading": _apply_defer_loading,
    "extras.apply_extractive_digest": _extractive_digest,
    "extras.strip_media_blocks": _strip_media,
    "extras.sanitize_thinking_blocks": _sanitize_thinking,
    "extras.fold_repeated_log_lines": _fold_repeated_log_lines,
    "extras.apply_cache_and_structure_fixes": _cache_and_structure,
    "extras.normalize_intent_text": _normalize_intent,
    "extras.compute_intent_hash": _intent_hash,
    "loopwatch.observe": _observe,
    "loopwatch.clear_session": _loop_clear_session,
    "loopwatch.is_relaxed": _loop_is_relaxed,
    "loopwatch.is_decompress": _loop_is_decompress,
    "loopwatch.tick_decompress": _loop_tick_decompress,
    "loopwatch.is_bypass_compression": _loop_is_bypass,
    "error_guard.extract_stacktrace_blocks": _extract_stacktrace_blocks,
    "error_guard.preserve_stacktrace_blocks": _preserve_stacktrace_blocks,
    "optimizer.count_tokens": _count_tokens,
    "optimizer.optimize_text": _optimize_text,
    "optimizer.inject_anti_yap": _inject_anti_yap,
    "optimizer._content_text": _content_text,
    "optimizer._optimize_content": _optimize_content,
    "optimizer._prune_middle_content": _prune_middle_content,
    "optimizer._truncate_verbose_logs": _truncate_verbose_logs,
    "optimizer._first_sentence": _first_sentence,
    "optimizer._extract_file_path": _extract_file_path,
    "optimizer._squash_exploratory_chains": _squash_exploratory_chains,
    "optimizer.extract_text": _extract_text,
    "optimizer.prune_history": _prune_history,
    "optimizer.optimize_payload": _optimize_payload,
    "optimizer.apply_anthropic_cache_alignment": _cache_alignment,
    "optimizer.apply_newest_tool_result_compaction": _newest_tool_result,
    "optimizer.clamp_max_tokens": _clamp_max_tokens,
    "optimizer.minify_tools_deterministic": _minify_tools,
    "optimizer.derive_session_key": _derive_session_key,
    "optimizer.apply_file_delta_tracking": _file_delta,
    "optimizer.credential_fingerprint": _credential_fingerprint,
    "optimizer.compute_request_hash": _request_hash,
    "optimizer.canonical_json_dumps": _canonical_json,
    "optimizer.measure_wire_tokens": _measure_wire,
    "optimizer.extract_upstream_usage": _upstream_usage,
    "error_guard.is_error_payload": _is_error_payload,
    "error_guard.preserve_diagnostic_text": _preserve_diagnostic_text,
}


def _reset_file_tracker() -> None:
    optimizer._file_tracker = optimizer.SessionFileStateTracker()


def _reset_loopwatch() -> None:
    loopwatch._RELAXED.clear()
    loopwatch._DECOMPRESS.clear()


RESET: list[Callable[[], None]] = [_reset_file_tracker, _reset_loopwatch]


# --- generators -------------------------------------------------------------

# Words chosen to hit every regex in optimizer.py: filler phrases, polite
# auxiliaries, determiners, preambles, inline code, \b boundaries around
# identifiers and paths, error keywords, and non-ASCII text.
PROSE = [
    "So", "well", "basically", "Sure!", "Certainly!", "Of course!", "Absolutely!",
    "Great question!", "I hope this helps!", "I hope that helps.",
    "Please let me know if you have any questions.",
    "Please let me know if you need further help.",
    "Please let me know if you need anything else.",
    "I'd be happy to help you with that!", "I'd be happy to help", "I d be happy to help with this.",
    "Feel free to ask if you have more questions.", "Feel free to reach out if you need help.",
    "Thank you for your patience!", "Thank you for asking.", "Could you please", "would you",
    "can you please", "could you", "can you", "the", "The", "A", "an", "An", "THE", "the function",
    "Actually,", "just", "simply", "not", "never", "if", "else", "true", "false", "error",
    "failed", "Traceback", "exception", "snake_case_name", "camelCaseName", "a_b.c", "x.the.y",
    "/usr/local/bin/tool", "src/app/main.py", "café", "naïve", "日本語",
    "emoji \U0001F600", "combining é", "zero​width", "nbsp x", "line sep",
    "x\u0085y", "1,000", "https://example.com/a?b=1", "`inline code`", "`a`", "tabs\tand  spaces",
    "　ideo", "a.", "b!", "c?", "d:", "(the)", "'a'", "\"the\"", "--", "|", "ERROR:", "",
    "sure thing!", "of course!", "Absolutely!!", "I'd be happy to help you with this!?",
]

SEPARATORS = [
    " ", "  ", "\t", "\n", "\n\n", "\n\n\n", "\r\n", " ", "\x0b", "\x0c", "\x1c",
    "\x85", "　", ". ", ", ", " \n ", "\n   \n", "\n\n\n\n", "\n \t\n",
]

CODE_LINES = [
    "def compute(x: int, y: str = 'a') -> dict[str, int]:",
    "    return {'a': x}",
    "class Widget(Base):",
    "    def render(self, *args, **kwargs):",
    "        '''Docstring.'''",
    "        pass   ",
    "_private_helper(a):",
    "async def fetch(url: str) -> bytes:",
    "    x = [i for i in range(10)]",
    "import os",
    "from typing import Optional",
    "if __name__ == '__main__':",
    "    main()",
    "",
    "",
    "   ",
    "\t\tindented = True\t",
    "const x = `tmpl`;",
    "fn main() {}",
    "SELECT * FROM t WHERE a = 1;",
    '{"k": [1, 2, 3]}',
    "    if x is None: return",
    "x = 1  # the comment",
]

LOG_LINES = [
    "⠋ Loading modules", "⠙", "-----", "|", "/", "\\", "", "   ",
    "INFO: started worker 3", "DEBUG: cache miss key=abc",
    "ERROR: connection refused", "Traceback (most recent call last):",
    '  File "/srv/app.py", line 42, in handler', "    raise ValueError('bad')",
    "ValueError: bad value", "FAILED tests/test_x.py::test_y",
    "stderr: segfault at 0x0 while loading", "exit code 1",
    "command failed with exit code 2", "normal progress 45%", "warning: deprecated",
    "npm ERR! code ELIFECYCLE", "fatal error: out of memory", "error ts2322: type mismatch",
    "Exception in thread main", "⠀⠁⣿", "--- separator ---", "data 123 ok",
    "PASSED", "ok", "  ", "\t", "exit code 0", "Error: x failed",
]

EXPLORE_TOOLS = ["glob", "Grep", "ls", "LS", "list_dir", "find", "Search", "search_files", "list_files", "Glob"]
READ_TOOLS = ["Read", "read_file", "view", "view_file", "cat", "get_file_contents", "Bash", "write"]
PATHS = ["/srv/app/main.py", "src/util.py", "README.md", "a/b/c.txt", "", "/x/the.py"]

# Surrogates are added alone, never next to another surrogate (see module doc).
LONE_SURROGATES = ["\ud800", "\udfff", "\udc00", "\udbff"]


def _rand_text(r: random.Random, n_pieces: int | None = None) -> str:
    kind = r.random()
    if kind < 0.45:
        out = []
        for _ in range(n_pieces or r.randint(1, 40)):
            out.append(r.choice(PROSE))
            out.append(r.choice(SEPARATORS))
        text = "".join(out)
    elif kind < 0.62:
        text = _code_block(r, _pick_lang(r)) + r.choice(SEPARATORS) + r.choice(PROSE)
    elif kind < 0.80:
        text = "\n".join(r.choice(LOG_LINES) for _ in range(r.randint(3, 90)))
    elif kind < 0.86:
        text = json.dumps(_rand_json(r, 0), indent=r.choice([None, 2]))
    elif kind < 0.90:
        text = "x" * r.randint(1000, 6000) + r.choice([" ", "\n", ""]) + "y" * r.randint(0, 50)
    elif kind < 0.93:
        text = r.choice(["", " ", "\n\n\n", "\t \n", "   ", "\n"])
    elif kind < 0.96:
        text = "Traceback (most recent call last):\n" + "\n".join(
            r.choice(LOG_LINES) for _ in range(r.randint(0, 25))
        )
    else:
        text = r.choice(PROSE) + r.choice(SEPARATORS) + _code_block(r, "")
    if r.random() < 0.04:
        text += r.choice(LONE_SURROGATES) + " " + r.choice(PROSE)
    if r.random() < 0.01:
        text += "<|endoftext|>"
    return text


def _pick_lang(r: random.Random) -> str:
    return r.choice(["python", "python", "Python", "js", "", "py", "PYTHON", "json", "text", "python3"])


def _code_block(r: random.Random, lang: str) -> str:
    lines = [r.choice(CODE_LINES) for _ in range(r.randint(0, 28))]
    body = "\n".join(lines)
    if r.random() < 0.05:
        return "```" + lang + "\n" + body  # unclosed fence
    if r.random() < 0.03:
        return "``` " + lang + "\n" + body + "\n```"
    return "```" + lang + "\n" + body + "\n```"


def _rand_json(r: random.Random, depth: int) -> Any:
    if depth > 2 or r.random() < 0.4:
        return r.choice([1, -3, 2.5, "text", None, True, 10**20, "unicode é"])
    if r.random() < 0.5:
        return [_rand_json(r, depth + 1) for _ in range(r.randint(0, 4))]
    return {f"k{i}": _rand_json(r, depth + 1) for i in range(r.randint(0, 4))}


def _tool_block(r: random.Random, tid: str) -> dict:
    name = r.choice(READ_TOOLS + EXPLORE_TOOLS)
    shape = r.random()
    if shape < 0.5:
        inp: Any = {r.choice(["file_path", "path", "target_file", "filepath", "command"]): r.choice(PATHS)}
    elif shape < 0.8:
        inp = {"pattern": r.choice(PROSE)}
    elif shape < 0.9:
        inp = None
    else:
        inp = r.choice(["str-input", 5, []])
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _result_block(r: random.Random, tid: str) -> dict:
    shape = r.random()
    if shape < 0.6:
        content: Any = _rand_text(r)
    elif shape < 0.85:
        content = [{"type": "text", "text": _rand_text(r)} for _ in range(r.randint(0, 3))]
    elif shape < 0.92:
        content = None
    else:
        content = r.choice([{"type": "text", "text": "dict"}, 7, ""])
    return {"type": "tool_result", "tool_use_id": tid, "content": content}


def _block(r: random.Random, depth: int) -> Any:
    k = r.random()
    if k < 0.28:
        return {"type": "text", "text": _rand_text(r)}
    if k < 0.34:
        return {"type": r.choice(["input_text", "output_text"]), "text": _rand_text(r)}
    if k < 0.50:
        return _tool_block(r, f"tu_{r.randint(0, 9)}")
    if k < 0.64:
        return _result_block(r, f"tu_{r.randint(0, 9)}")
    if k < 0.68:
        return {"type": "image", "source": {"type": "base64", "data": "AAAA"}}
    if k < 0.71:
        return {"type": "thinking", "thinking": _rand_text(r)}
    if k < 0.75:
        return _rand_text(r)
    if k < 0.77:
        return r.choice([7, None, True, 2.5])
    if k < 0.80:
        return {"text": _rand_text(r)}
    if k < 0.83:
        return {"type": "text"}
    if k < 0.85:
        return {"type": "text", "text": r.choice([5, None, ["x"], 0])}
    if k < 0.87:
        return {"type": "function_call", "text": "fc", "name": r.choice(["read", "glob", 5])}
    if k < 0.89:
        return {"type": "tool_use", "name": r.choice([None, 5, "glob"]), "id": "x"}
    block = {"type": "text", "text": _rand_text(r), "cache_control": {"type": "ephemeral"}}
    return block


def _content(r: random.Random) -> Any:
    k = r.random()
    if k < 0.55:
        return _rand_text(r)
    if k < 0.90:
        return [_block(r, 0) for _ in range(r.randint(0, 4))]
    return r.choice([None, 3, "", [], True, {"type": "text", "text": "dict content"}])


def _messages(r: random.Random, n: int) -> list:
    out: list = []
    while len(out) < n:
        if r.random() < 0.2:
            run = r.randint(1, 5)
            for _ in range(run):
                if len(out) >= n + 2:
                    break
                tid = f"ex_{len(out)}"
                tool = r.choice(EXPLORE_TOOLS)
                if r.random() < 0.05:
                    tool = r.choice(["", None, 5])
                use_block: Any = {"type": "tool_use", "id": tid, "name": tool, "input": {"pattern": "x"}}
                out.append({"role": "assistant", "content": [use_block]})
                lines = "\n".join(r.choice(LOG_LINES + ["src/a.py:1: x"]) for _ in range(r.randint(0, 70)))
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": tid, "content": lines}
                ]})
            continue
        role = r.choice(["user", "assistant", "user", "user", "system"])
        msg: dict = {"role": role, "content": _content(r)}
        if r.random() < 0.02:
            msg = {"content": _content(r)}
        out.append(msg)
    return out[:n] if r.random() < 0.7 else out


def _tools(r: random.Random) -> list:
    tools = []
    for i in range(r.randint(0, 7)):
        desc = " ".join(r.choice(PROSE) for _ in range(r.randint(1, 30))) + r.choice(["", ".", "!"])
        schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": _rand_text(r)},
                "items": {"type": "array", "items": {"type": "string", "description": "one. two"}},
                "n": {"type": "integer", "description": r.choice(["Count. More", "", "Plain"])},
            },
            "required": ["path"],
        }
        if r.random() < 0.15:
            tool: dict = {
                "type": "function",
                "function": {"name": f"fn{i}", "description": desc, "parameters": schema},
            }
        else:
            tool = {"name": f"tool_{i}", "description": desc, "input_schema": schema}
        if r.random() < 0.05:
            tool = r.choice([7, "str-tool", None])
        tools.append(tool)
    if tools and r.random() < 0.05 and isinstance(tools[-1], dict):
        tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools


def _payload(r: random.Random) -> dict:
    n = r.choice([0, 1, 2, 3, 5, 6, 7, 8, 9, 12, 15])
    p: dict = {"model": "claude-test", "max_tokens": r.choice([1, 100, 2048, 2049, 4096, 8192, None, "8", True])}
    if r.random() < 0.85:
        p["messages"] = _messages(r, n)
    else:
        p["messages"] = r.choice([None, "not a list", 5, {}, []])
    if r.random() < 0.7:
        if r.random() < 0.2:
            p["system"] = " ".join(r.choice(PROSE) for _ in range(r.randint(400, 900)))
        else:
            p["system"] = _content(r) if r.random() < 0.5 else _rand_text(r)
    if r.random() < 0.6:
        p["tools"] = _tools(r)
    if r.random() < 0.08:
        p["prompt"] = _rand_text(r)
    if r.random() < 0.08:
        items: list = []
        for _ in range(r.randint(0, 5)):
            k = r.random()
            if k < 0.25:
                items.append(_rand_text(r))
            elif k < 0.5:
                items.append({"type": "message", "content": _content(r)})
            elif k < 0.7:
                items.append({"type": "additional_tools", "tools": [
                    {"type": "namespace", "tools": _tools(r)},
                    {"type": "function", "name": "f", "description": "Desc. More", "parameters": {
                        "type": "object", "properties": {"q": {"description": "Q. R"}}}},
                ]})
            else:
                items.append({"output": r.choice([_rand_text(r), [{"text": _rand_text(r)}, "x"]]),
                              "instructions": _rand_text(r), "text": _rand_text(r)})
        p["input"] = r.choice([_rand_text(r), items, items])
    if r.random() < 0.05:
        p["instructions"] = _rand_text(r)
    if r.random() < 0.3:
        p["stream"] = r.random() < 0.5
    return p


def _file_sequence(r: random.Random, session: str, steps: int) -> list:
    """A list of apply_file_delta_tracking steps that share global state."""
    path = r.choice(PATHS[:-1])
    versions = [
        "line one\nline two\nline three\n",
        "line one\nline TWO\nline three\n",
        "line one\nline two\nline three",
        "line one\nline two\nline three\n\n\n",
        _rand_text(r) + "\n",
    ]
    out = []
    for _ in range(steps):
        msgs = []
        for _ in range(r.randint(1, 4)):
            tid = f"r{r.randint(0, 5)}"
            msgs.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": tid, "name": r.choice(READ_TOOLS),
                 "input": {"file_path": path if r.random() < 0.9 else r.choice(PATHS)}}]})
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tid,
                 "content": r.choice(versions) if r.random() < 0.8 else [
                     {"type": "text", "text": r.choice(versions)}]}]})
        msgs.insert(0, {"role": "user", "content": "start"})
        pay = {"model": "m", "messages": msgs}
        out.append({"fn": "optimizer.apply_file_delta_tracking",
                    "args": {"payload": pay, "session_key": session}})
    return out


def _headers(r: random.Random) -> dict:
    h: dict = {}
    for name in ["authorization", "x-api-key", "x-goog-api-key", "api-key"]:
        k = r.random()
        if k < 0.5:
            h[name] = r.choice(["Bearer sk-test", "key-a", "key-b", " padded\t", "", "x"])
        elif k < 0.7:
            h[name] = None
        elif k < 0.8:
            h[name] = r.choice([0, 5, [], ["x"], False])
        else:
            continue
    if r.random() < 0.1:
        h["X-Api-Key"] = "upper-case-key"
    return h


def _sse_body(r: random.Random) -> bytes:
    lines = []
    if r.random() < 0.1:
        lines.append("﻿")
    for _ in range(r.randint(0, 8)):
        k = r.random()
        if k < 0.25:
            usage = {"input_tokens": r.randint(0, 9000), "output_tokens": r.randint(0, 900),
                     "cache_read_input_tokens": r.choice([0, 12, None]),
                     "cache_creation": {"ephemeral_5m_input_tokens": r.choice([0, 5, 2.9, True]),
                                        "ephemeral_1h_input_tokens": r.choice([0, 3, None, "x"])}}
            lines.append("data: " + json.dumps({"type": "message_start", "message": {"usage": usage}}))
        elif k < 0.45:
            lines.append("data:" + json.dumps({"type": "message_delta", "usage": {"output_tokens": r.randint(0, 999)}}))
        elif k < 0.55:
            lines.append("data: [DONE]")
        elif k < 0.6:
            lines.append("data: {not json")
        elif k < 0.65:
            lines.append("data: " + json.dumps({"usage": {"cache_creation": {"a": float("nan")}}}))
        elif k < 0.68:
            lines.append("data: " + json.dumps({"usage": {"cache_creation": {"a": float("inf")}}}))
        elif k < 0.72:
            lines.append("event: ping")
        elif k < 0.76:
            lines.append("data: " + json.dumps({"usage": 5}))
        elif k < 0.8:
            lines.append("  data:   " + json.dumps(5))
        elif k < 0.85:
            lines.append("data: " + json.dumps({"message": {"usage": {"input_tokens": 3,
                                                                       "cache_creation_input_tokens": 9,
                                                                       "cache_creation": {"x": 1}}}}))
        elif k < 0.93:
            lines.append("data: " + json.dumps({"usage": {"output_tokens": float("nan"),
                                                          "rate": float("-inf"), "z": -0.0}}))
        else:
            lines.append("data: " + json.dumps({"usage": {"output_tokens": 1, "meta": {"k": "v"},
                                                          "list": [1], "nul": None}}))
    sep = r.choice(["\n", "\r\n", "\n\n", " ", "\x0c"])
    return sep.join(lines).encode("utf-8") + (b"\n" if r.random() < 0.5 else b"")


def _body(r: random.Random) -> bytes:
    k = r.random()
    if k < 0.4:
        return _sse_body(r)
    if k < 0.6:
        return json.dumps({"id": "m", "usage": {"input_tokens": r.randint(1, 500),
                                                "cache_creation": {"a": r.randint(0, 9)}}}).encode("utf-8")
    if k < 0.75:
        return _rand_text(r).encode("utf-8", "ignore")
    if k < 0.8:
        return bytes(r.randint(0, 255) for _ in range(r.randint(0, 80)))
    if k < 0.9:
        return b"\xff\xfe\x80\xc0" * r.randint(1, 30)
    return b""


def _long_tail_payload(r: random.Random) -> dict:
    """Last message is a user tool_result with long text (head/tail cap paths)."""
    blocks = []
    for _ in range(r.randint(1, 3)):
        k = r.random()
        if k < 0.35:
            content: Any = "".join(r.choice(PROSE) + r.choice(SEPARATORS) for _ in range(r.randint(100, 900)))
        elif k < 0.6:
            content = [{"type": "text", "text": "".join(r.choice(LOG_LINES) + "\n" for _ in range(r.randint(100, 700)))}]
        elif k < 0.8:
            # A long list in a text block: Python slices it and formats it with repr.
            content = [{"type": "text", "text": [r.choice(["a", "b", 1, None, 2.5]) for _ in range(r.randint(1500, 1800))]}]
        elif k < 0.9:
            # A large dict: len() passes, then slicing raises TypeError.
            content = [{"type": "text", "text": {f"k{i}": i for i in range(r.randint(0, 1700))}}]
        else:
            content = [{"type": "text", "text": r.choice([None, 5, 3, ""])}]
        blocks.append({"type": "tool_result", "tool_use_id": "t", "content": content})
    return {"model": "m", "messages": [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Bash", "input": {}}]},
        {"role": "user", "content": blocks},
    ]}


VOLATILE_LINES = [
    "cc_version=1.2.3", "Claude Code version: 9.9", "Current date: 2026-10-29",
    "current time = 12:00", "Today's date: 2026-01-02", "stamp 2026-10-29T12:00:00.123Z here",
    "normal line about the system", "", "   ", "You are a helpful assistant.",
]


def _system_value(r: random.Random) -> Any:
    k = r.random()
    if k < 0.5:
        return "\n".join(r.choice(VOLATILE_LINES) for _ in range(r.randint(0, 12)))
    if k < 0.85:
        return [{"type": r.choice(["text", "text", "image"]),
                 "text": r.choice(VOLATILE_LINES + [" ".join(r.choice(PROSE) for _ in range(5))])}
                for _ in range(r.randint(0, 4))]
    return r.choice([None, 5, [], "", [{"type": "text", "text": 5}], [{"type": "text"}]])


TOOL_NAMES = ["zeta", "alpha", "ccr_retrieve", "mcp__srv__tool", "a__b", "verylongname__x",
              "short", "Beta", "beta", "", "mcp__x", "__lead"]


def _tool_entry(r: random.Random) -> Any:
    name: Any = r.choice(TOOL_NAMES) if r.random() < 0.93 else r.choice([None, 5, ["x"], True])
    t: dict = {"name": name}
    if r.random() < 0.8:
        t["description"] = " ".join(r.choice(PROSE) for _ in range(r.randint(1, 12)))
    if r.random() < 0.6:
        t["input_schema"] = {"type": "object", "properties": {"q": {"type": "string"}},
                             "required": r.choice([["q"], []])}
    if r.random() < 0.05:
        return r.choice([7, None, "str"])
    return t


def _msg_text(r: random.Random) -> str:
    k = r.random()
    if k < 0.4:
        return " ".join(r.choice(PROSE) for _ in range(r.randint(20, 120)))
    if k < 0.7:
        return "\n".join(r.choice(LOG_LINES) for _ in range(r.randint(1, 40)))
    return r.choice(["", "short", "error: x failed in /srv/a.py:3"])


def _extras_content(r: random.Random) -> Any:
    k = r.random()
    if k < 0.45:
        return _msg_text(r)
    blocks: list = []
    for _ in range(r.randint(0, 5)):
        b = r.random()
        if b < 0.25:
            blocks.append({"type": "text", "text": _msg_text(r)})
        elif b < 0.4:
            blocks.append({"type": r.choice(["image", "document", "file"]), "source": {"data": "AAA"}})
        elif b < 0.55:
            blocks.append({"type": "thinking", "thinking": r.choice(["", "  ", "plan the change", None])})
        elif b < 0.62:
            blocks.append({"type": "redacted_thinking", "data": r.choice(["", "x", None])})
        elif b < 0.75:
            blocks.append(_tool_block(r, f"tu_{r.randint(0, 3)}"))
        elif b < 0.85:
            blocks.append(_result_block(r, f"tu_{r.randint(0, 3)}"))
        else:
            blocks.append(r.choice([None, 4, "bare"]))
    return blocks


def _extras_payload(r: random.Random) -> dict:
    n = r.randint(0, 12)
    msgs = []
    for _ in range(n):
        msg: dict = {"role": r.choice(["user", "assistant", "assistant", "user"]),
                     "content": _extras_content(r)}
        if r.random() < 0.03:
            msg = {"role": "user"}
        msgs.append(msg)
    p: dict = {"model": "m", "messages": msgs if r.random() < 0.92 else r.choice([None, "x", 5])}
    if r.random() < 0.7:
        p["system"] = _system_value(r)
    if r.random() < 0.6:
        p["tools"] = [_tool_entry(r) for _ in range(r.randint(0, 12))] if r.random() < 0.9 else r.choice([None, "t", []])
    if r.random() < 0.2:
        p["provider_extra"] = _rand_json(r, 0)
    return p


def _loop_session_steps(r: random.Random, session: Any) -> list:
    """Observe calls over a growing history, so decompress and relax trigger."""
    steps: list = []
    path = r.choice(PATHS[:-1])
    search = {"pattern": r.choice(PROSE)}
    kind = r.random()
    for i in range(r.randint(2, 5)):
        msgs = [{"role": "user", "content": "task"}]
        for j in range(i + 1):
            if kind < 0.35:
                name, inp = r.choice(["read_file", "Read", "view"]), {"file_path": path}
            elif kind < 0.7:
                name, inp = r.choice(["grep", "Glob", "ls"]), search
            else:
                name, inp = r.choice(["bash", "Bash"]), {"command": "make test"}
            tid = f"l{i}_{j}"
            msgs.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": tid, "name": name, "input": inp}]})
            msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid,
                                                       "content": r.choice(LOG_LINES)}]})
        msgs.append({"role": "user", "content": r.choice(["continue", "next step please"])})
        steps.append({"fn": "loopwatch.observe", "args": {"payload": {"messages": msgs},
                                                          "session_key": session}})
        if r.random() < 0.3:
            steps.append({"fn": r.choice(["loopwatch.is_relaxed", "loopwatch.is_decompress",
                                          "loopwatch.is_bypass_compression"]),
                          "args": {"session_key": session}})
        if r.random() < 0.15:
            steps.append({"fn": "loopwatch.tick_decompress", "args": {"session_key": session}})
    return steps


def _step(r: random.Random) -> dict:
    """One random step with realistic arguments."""
    names = [
        ("optimizer.optimize_text", 8), ("optimizer.count_tokens", 3), ("optimizer.inject_anti_yap", 2),
        ("optimizer._content_text", 3), ("optimizer._optimize_content", 3),
        ("optimizer._prune_middle_content", 4), ("optimizer._truncate_verbose_logs", 4),
        ("optimizer._first_sentence", 2), ("optimizer._extract_file_path", 2),
        ("optimizer._squash_exploratory_chains", 3), ("optimizer.extract_text", 3),
        ("optimizer.prune_history", 5), ("optimizer.optimize_payload", 8),
        ("optimizer.apply_anthropic_cache_alignment", 4),
        ("optimizer.apply_newest_tool_result_compaction", 4), ("optimizer.clamp_max_tokens", 3),
        ("optimizer.minify_tools_deterministic", 3), ("optimizer.derive_session_key", 2),
        ("optimizer.credential_fingerprint", 2), ("optimizer.compute_request_hash", 2),
        ("optimizer.canonical_json_dumps", 2), ("optimizer.measure_wire_tokens", 3),
        ("optimizer.extract_upstream_usage", 4), ("error_guard.is_error_payload", 4),
        ("error_guard.preserve_diagnostic_text", 3),
        ("error_guard.extract_stacktrace_blocks", 3), ("error_guard.preserve_stacktrace_blocks", 3),
        ("extras.strip_volatile_system", 3), ("extras.stabilize_tools", 3),
        ("extras.apply_defer_loading", 3), ("extras.apply_extractive_digest", 3),
        ("extras.strip_media_blocks", 2), ("extras.sanitize_thinking_blocks", 2),
        ("extras.fold_repeated_log_lines", 3), ("extras.apply_cache_and_structure_fixes", 3),
        ("extras.normalize_intent_text", 2), ("extras.compute_intent_hash", 2),
        ("loopwatch.clear_session", 1), ("loopwatch.is_bypass_compression", 1),
    ]
    total = sum(w for _, w in names)
    pick = r.uniform(0, total)
    fn = names[-1][0]
    for name, w in names:
        if pick < w:
            fn = name
            break
        pick -= w
    return {"fn": fn, "args": _args_for(r, fn)}


def _args_for(r: random.Random, fn: str) -> dict:
    if fn == "optimizer.optimize_text":
        return {"text": _rand_text(r) if r.random() < 0.9 else r.choice([None, 0, [], ["x"], 5, ""])}
    if fn == "optimizer.count_tokens":
        return {"text": _rand_text(r)}
    if fn == "optimizer.inject_anti_yap":
        return {"system": r.choice([None, _rand_text(r), [{"type": "text", "text": "a"}], 5, {}, [], "x"])}
    if fn in ("optimizer._content_text", "optimizer._optimize_content"):
        return {"content": _content(r)}
    if fn == "optimizer._prune_middle_content":
        text: Any = _rand_text(r) if r.random() < 0.9 else r.choice([None, "", 0, [], 5])
        return {"text": text, "log_truncation_enabled": r.random() < 0.8}
    if fn == "optimizer._truncate_verbose_logs":
        return {"text": "\n".join(r.choice(LOG_LINES) for _ in range(r.randint(0, 120)))}
    if fn == "optimizer._first_sentence":
        return {"text": _rand_text(r)}
    if fn == "optimizer._extract_file_path":
        return {"block": _tool_block(r, "tu_x")}
    if fn == "optimizer._squash_exploratory_chains":
        return {"messages": _messages(r, r.randint(0, 14))}
    if fn == "optimizer.extract_text":
        return {"payload": _payload(r)}
    if fn == "optimizer.prune_history":
        return {"messages": _messages(r, r.randint(0, 16)) if r.random() < 0.95 else r.choice([None, 5]),
                "log_truncation_enabled": r.random() < 0.8, "turn_squashing_enabled": r.random() < 0.8}
    if fn == "optimizer.optimize_payload":
        return {"payload": _payload(r), "compression_enabled": r.random() < 0.85,
                "anti_yap_enabled": r.random() < 0.85, "history_pruning_enabled": r.random() < 0.85,
                "log_truncation_enabled": r.random() < 0.85}
    if fn == "optimizer.apply_anthropic_cache_alignment":
        return {"payload": _payload(r), "provider": r.choice(["anthropic", "anthropic", "openai", "gemini"])}
    if fn == "optimizer.apply_newest_tool_result_compaction" and r.random() < 0.7:
        return {"payload": _long_tail_payload(r)}
    if fn in ("optimizer.apply_newest_tool_result_compaction", "optimizer.clamp_max_tokens",
              "optimizer.derive_session_key"):
        return {"payload": _payload(r)}
    if fn == "optimizer.minify_tools_deterministic":
        return {"tools": _tools(r) if r.random() < 0.9 else r.choice([None, "x", {}])}
    if fn == "optimizer.credential_fingerprint":
        return {"headers": _headers(r)}
    if fn == "optimizer.compute_request_hash":
        return {"payload": _payload(r), "provider": r.choice(["anthropic", "openai"]),
                "credential": r.choice(["", "k1", "fp"])}
    if fn == "optimizer.canonical_json_dumps":
        return {"data": _payload(r)}
    if fn == "optimizer.measure_wire_tokens":
        # Often no usable payload, so the body (or the binary estimate) decides.
        pay = _payload(r) if r.random() < 0.5 else r.choice([None, "str", 5, {"messages": []}])
        return {"body_b64": base64.b64encode(_body(r)).decode("ascii"), "payload": pay}
    if fn == "optimizer.extract_upstream_usage":
        return {"raw_body_b64": base64.b64encode(_body(r)).decode("ascii")}
    if fn == "error_guard.is_error_payload":
        return {"text": _rand_text(r), "tool_name": r.choice([None, "bash", "Read", "shell", "BashTool", "", "Bash"]),
                "is_error_flag": r.random() < 0.1}
    if fn == "error_guard.extract_stacktrace_blocks":
        return {"text": "\n".join(r.choice(LOG_LINES) for _ in range(r.randint(0, 60)))}
    if fn == "error_guard.preserve_stacktrace_blocks":
        return {"text": "\n".join(r.choice(LOG_LINES) for _ in range(r.randint(0, 80))),
                "ceiling": r.choice([0, 40, 300, 3000])}
    if fn == "extras.strip_volatile_system":
        return {"system": _system_value(r)}
    if fn == "extras.stabilize_tools":
        return {"tools": [_tool_entry(r) for _ in range(r.randint(0, 6))] if r.random() < 0.9 else r.choice([None, "x"])}
    if fn == "extras.apply_defer_loading":
        return {"tools": [_tool_entry(r) for _ in range(r.randint(0, 12))] if r.random() < 0.95 else r.choice([None, {}])}
    if fn in ("extras.apply_extractive_digest", "extras.strip_media_blocks",
              "extras.sanitize_thinking_blocks", "extras.apply_cache_and_structure_fixes"):
        return {"payload": _extras_payload(r)}
    if fn == "extras.fold_repeated_log_lines":
        lines = []
        for _ in range(r.randint(0, 30)):
            base = r.choice(LOG_LINES) + " " + str(r.randint(0, 999))
            lines.extend([base] * r.randint(1, 6)) if r.random() < 0.5 else lines.append(base)
            if r.random() < 0.2:
                lines.append(f"2026-10-29 12:0{r.randint(0, 9)}:01 tick {r.randint(0, 99)}")
        return {"text": "\n".join(lines)}
    if fn == "extras.normalize_intent_text":
        return {"text": r.choice([None, _msg_text(r), "  Mixed   CASE\tText  "])}
    if fn == "extras.compute_intent_hash":
        return {"payload": _extras_payload(r), "provider": r.choice(["anthropic", "openai"])}
    if fn in ("loopwatch.clear_session", "loopwatch.is_relaxed", "loopwatch.is_decompress",
              "loopwatch.tick_decompress", "loopwatch.is_bypass_compression"):
        return {"session_key": r.choice(["s1", "s2", None, ""])}
    if fn == "loopwatch.observe":
        return {"payload": _extras_payload(r), "session_key": r.choice(["s1", None, ""])}
    if fn == "error_guard.preserve_diagnostic_text":
        return {"text": "\n".join(r.choice(LOG_LINES) for _ in range(r.randint(0, 80))),
                "ceiling": r.choice([0, 10, 200, 3000, 50])}
    raise KeyError(fn)


# --- e2 generators and adapters: aggressive, maxsave, wirecrush -------------
# Inserted before _case(). Stateful pieces: the wirecrush seen set (_E2_SEEN),
# the CCR store and the loopwatch flags, all reset per case by RESET.

from engines import aggressive, ccr as _ccr, maxsave, wirecrush  # noqa: E402

_E2_SEEN: set = set()

GIT_WALL = (
    "# Committing changes with git\n"
    + "Only create commits when the user asks. Run git status and git diff first. " * 6
    + "\n# Creating pull requests\nUse gh pr create with a HEREDOC body. " * 3
    + "\n# Other common operations\nUse gh api for issues.\n"
)
GIT_SAFETY = "Git Safety Protocol: never force push, never amend unless asked.\n\n"
SYS_REMINDER = "<system-reminder>\nCurrent date: 2026-10-29. Keep it private.\n</system-reminder>\n"
OUTPUT_FMT = "\n\nOutput format guidelines: be terse. Prefer bullets.\nFollow house style.\n"
PLUGIN_BLOB = "<recommended_plugins>\n" + "plugin-catalog entry with long description text. " * 25 + "\n</recommended_plugins>\n"
MD_SECTIONS = "# Tone\nBe concise.\n# Safety\nRefuse harmful requests.\n# Other\nNothing relevant here.\n"
PROGRESS_LINES = ["Downloading pkg 45%", "⠋ resolving deps", "eta 0:00:03",
                  "=====>  fetching packages", "extracting archive", "ok"]
TABLE_LINES = [
    "CONTAINER ID   IMAGE    STATUS    PORTS    NAMES",
    "abc123   nginx    Up 2h    80/tcp   web",
    "def456   redis    Up 1h    6379     cache",
    "fed789   app      Exited   -        worker",
]


def _e2_text(r: random.Random, lo: int = 40, hi: int = 400, pool: list | None = None) -> str:
    """Realistic tool output: logs, JSON, grep, ANSI, CLI tables, tracebacks, prose."""
    if pool and r.random() < 0.25:
        return r.choice(pool)
    n = r.randint(lo, hi)
    k = r.random()
    if k < 0.12:
        text = "\n".join(f"INFO step {i} ok" for i in range(n))
    elif k < 0.22:
        text = json.dumps([{"id": i, "name": f"item{i}", "ok": i % 2 == 0, "tags": ["a", "b"]}
                           for i in range(max(2, n // 6))], indent=r.choice([None, 2]))
    elif k < 0.30:
        text = "\n".join(f"src/m{i % 7}.py:{i}: match value {i}" if i % 3 else "plain line" for i in range(n))
    elif k < 0.36:
        text = "".join(f"\x1b[31mERROR\x1b[0m step {i} failed\r\n" for i in range(n))
    elif k < 0.42:
        text = "\n".join(r.choice(PROGRESS_LINES) for _ in range(n))
    elif k < 0.48:
        text = "\n".join(TABLE_LINES * r.randint(1, 5))
    elif k < 0.54:
        text = "Traceback (most recent call last):\n" + "\n".join(
            f'  File "/srv/a{i}.py", line {i}, in f' for i in range(max(1, n // 8))) + "\nValueError: boom"
    elif k < 0.58:
        text = "data:image/png;base64," + "QUJD" * r.randint(100, 220)
    elif k < 0.63:
        text = "\n".join(f"tool output line {i}" for i in range(n)) + "\nCertainly! " * r.randint(0, 4)
    elif k < 0.68:
        text = "Context dump\n" + "x" * r.randint(900, 2500) + "\n[...truncated...]"
    elif k < 0.72:
        text = "[" + ",".join(str(i) for i in range(n)) + "]"
    elif k < 0.76:
        text = "\n".join(r.choice(LOG_LINES) for _ in range(n))
    elif k < 0.80:
        text = " ".join(r.choice(PROSE) for _ in range(n))
    elif k < 0.84:
        text = "\n".join(f"stamp 2026-10-29 12:00:{i % 60:02d} tick {i}" for i in range(n))
    elif k < 0.88:
        text = "\n".join(r.choice(["npm ERR! code 1", "exit code 2", "plain"]) for _ in range(n))
    elif k < 0.92:
        text = "x" * r.randint(800, 6000)
    elif k < 0.95:
        text = "\n".join(f"  \t-rw-r--r--   1 root   root   {i}   Oct 29   file{i}" for i in range(n))
    elif k < 0.97:
        text = "[" + ",".join(['"a\\ud800"'] * 14) + "]"  # JSON escape for a lone surrogate
    else:
        text = "\n".join(f"{i}|{i}|x" for i in range(n))
    if pool is not None and len(pool) < 40:
        pool.append(text)
    return text


def _e2_system_str(r: random.Random) -> str:
    return _e2_text(r, 20, 60) + "\n" + GIT_WALL + GIT_SAFETY + SYS_REMINDER + OUTPUT_FMT + MD_SECTIONS


def _e2_system(r: random.Random, pool: list) -> Any:
    k = r.random()
    if k < 0.3:
        return _e2_system_str(r)
    if k < 0.5:
        return [{"type": "text", "text": GIT_WALL + _e2_text(r, 20, 40)},
                {"type": "text", "text": SYS_REMINDER},
                {"type": "text", "text": "   "},
                {"type": "text", "text": None if r.random() < 0.1 else _e2_text(r, 5, 20)}]
    if k < 0.65:
        return [{"type": "text", "text": "Long prompt. " * 60}, {"type": "image"}]
    if k < 0.72:
        return "Long prompt. " * 1700  # over the 20000 char light trim
    return r.choice([None, "", 5, [], "x" * 300, [{"type": "text"}]])


def _e2_schema() -> dict:
    return {"type": "object", "title": "T", "description": "d",
            "properties": {"path": {"type": "string", "description": "The path. More."},
                           "items": {"type": "array", "items": {"type": "string", "description": "one. two"}}},
            "required": ["path"], "$schema": "x", "examples": [1]}


def _e2_tools(r: random.Random) -> list:
    out: list = []
    for i in range(r.randint(0, 10)):
        desc = " ".join(r.choice(PROSE) for _ in range(r.randint(1, 30)))
        if r.random() < 0.2:
            desc = GIT_WALL + desc
        name = "ccr_retrieve" if r.random() < 0.15 else f"tool_{i}"
        t: Any = {"name": name, "description": desc, "input_schema": _e2_schema()}
        if r.random() < 0.15:
            t = {"type": "function", "function": {"name": name, "description": desc, "parameters": _e2_schema()}}
        if r.random() < 0.04:
            t = r.choice([7, "str", None])
        out.append(t)
    return out


def _e2_result_block(r: random.Random, tid: str, pool: list) -> dict:
    shape = r.random()
    if shape < 0.55:
        content: Any = _e2_text(r, pool=pool)
    elif shape < 0.8:
        content = [{"type": "text", "text": _e2_text(r, pool=pool)} for _ in range(r.randint(1, 3))]
    elif shape < 0.87:
        content = [{"type": "image", "source": {"type": "base64", "data": "AAAA"}},
                   {"type": "text", "text": _e2_text(r, pool=pool)}]
    elif shape < 0.93:
        content = [{"type": "text", "text": r.choice([5, None, "", ["x"], "  "])}, {"type": "text", "text": _e2_text(r, pool=pool)}]
    else:
        content = None
    block: dict = {"type": "tool_result", "tool_use_id": tid, "content": content}
    if r.random() < 0.1:
        block["is_error"] = True
    return block


def _e2_messages(r: random.Random, pool: list, n: int) -> list:
    msgs: list = []
    path = r.choice(PATHS[:-1])
    for _ in range(n):
        k = r.random()
        if k < 0.42:
            tid = f"t{r.randint(0, 9)}"
            name = r.choice(["Read", "view", "Edit", "Bash", "Shell", "grep", "list_dir", "ccr_retrieve"])
            if name in ("Read", "view", "Edit"):
                inp: Any = {"file_path": path}
            elif name == "ccr_retrieve":
                inp = {"hash": r.choice(["0" * 16, "deadbeef" * 2])}
            elif name in ("Bash", "Shell"):
                inp = {"command": "ls -la"}
            else:
                inp = {"pattern": "x"}
            use: dict = {"type": "tool_use", "id": tid, "name": name, "input": inp}
            if r.random() < 0.02:
                use["id"] = ["bad"]  # unhashable id: Python raises in the names map
            content: list = [use]
            if r.random() < 0.3:
                content.append({"type": "text", "text": _e2_text(r, pool=pool)})
            msgs.append({"role": "assistant", "content": content})
        elif k < 0.7:
            msgs.append({"role": "user", "content": [_e2_result_block(r, f"t{r.randint(0, 9)}", pool)]})
        else:
            c: Any = _e2_text(r, pool=pool) if r.random() < 0.7 else [{"type": "text", "text": _e2_text(r, pool=pool)}]
            msgs.append({"role": r.choice(["user", "assistant"]), "content": c})
        if r.random() < 0.03:
            msgs.append({"content": "   "})
    return msgs


def _e2_anthropic(r: random.Random) -> dict:
    pool: list = []
    p: dict = {"model": "claude-test", "max_tokens": r.choice([1024, 4096]),
               "messages": _e2_messages(r, pool, r.randint(0, 22))}
    if r.random() < 0.9:
        p["system"] = _e2_system(r, pool)
    if r.random() < 0.85:
        p["tools"] = _e2_tools(r)
    if r.random() < 0.04:
        p["messages"] = r.choice([None, "x", {}])
    return p


def _e2_openai(r: random.Random) -> dict:
    pool: list = []
    msgs: list = [{"role": "system", "content": _e2_system_str(r)}]
    for i in range(r.randint(0, 20)):
        role = r.choice(["user", "assistant", "tool", "tool", "tool"])
        if role == "tool":
            m: dict = {"role": "tool", "tool_call_id": f"c{i}", "content": _e2_text(r, pool=pool)}
        else:
            m = {"role": role, "content": _e2_text(r, 40, 600, pool=pool)}
        if r.random() < 0.03:
            m["content"] = ["list"]
        msgs.append(m)
    p: dict = {"model": "m", "messages": msgs}
    if r.random() < 0.7:
        p["tools"] = [{"type": "function", "function": {"name": f"f{i}", "description": " ".join(
            r.choice(PROSE) for _ in range(10)), "parameters": _e2_schema()}} for i in range(r.randint(1, 6))]
    if r.random() < 0.1:
        p["messages"] = msgs[:1]
    return p


def _e2_responses(r: random.Random) -> dict:
    pool: list = []
    items: list = []
    for i in range(r.randint(0, 14)):
        k = r.random()
        if k < 0.2:
            items.append({"type": "function_call_output", "call_id": f"c{i}", "output": _e2_text(r, pool=pool)})
        elif k < 0.3:
            items.append({"type": "function_call_output", "call_id": f"c{i}",
                          "output": [{"type": "output_text", "text": _e2_text(r, pool=pool)}, {"type": "input_image"}]})
        elif k < 0.38:
            items.append({"type": "tool_result", "output": _e2_text(r, pool=pool)})
        elif k < 0.45:
            items.append({"role": "tool", "content": _e2_text(r, pool=pool)})
        elif k < 0.6:
            if r.random() < 0.5:
                c: Any = _e2_text(r, 200, 500, pool=pool)
            else:
                c = [{"type": "input_text", "text": PLUGIN_BLOB + _e2_text(r, 5, 80, pool=pool)},
                     {"type": "text", "text": _e2_text(r, pool=pool)}]
            items.append({"role": "user", "content": c})
        elif k < 0.7:
            if r.random() < 0.5:
                dc: Any = OUTPUT_FMT * 30 + MD_SECTIONS
            else:
                dc = [{"type": "input_text", "text": "You are Codex. " + "rule text. " * 60 + PLUGIN_BLOB}]
            items.append({"role": "developer", "content": dc})
        elif k < 0.8:
            items.append({"type": "additional_tools", "tools": [
                {"type": "namespace", "name": "ns", "tools": _e2_tools(r)},
                {"type": "function", "name": "exec", "description": "Run code. " * 400, "parameters": _e2_schema()},
            ]})
        else:
            items.append({"type": "message", "content": _e2_text(r, pool=pool)})
    p: dict = {"model": "m"}
    if r.random() < 0.6:
        p["instructions"] = _e2_system_str(r)
    p["input"] = items if r.random() < 0.9 else _e2_text(r, 200, 500)
    if r.random() < 0.5:
        p["tools"] = _e2_tools(r)
    return p


# Minimal protobuf encoder, independent of wirecrush's helpers.
def _pvi(n: int) -> bytes:
    out = bytearray()
    while n > 0x7F:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _pld(fn: int, b: bytes) -> bytes:
    return _pvi((fn << 3) | 2) + _pvi(len(b)) + b


def _pvf(fn: int, n: int) -> bytes:
    return _pvi((fn << 3) | 0) + _pvi(n)


def _pfile(r: random.Random) -> bytes:
    """A file read: a filename, then one {1: lineno, 2: text} wrapper per line."""
    parts = [_pld(1, r.choice([b"src/app/main.py", b"README.md"]))] if r.random() < 0.9 else []
    for i in range(r.randint(20, 520)):
        line = r.choice(CODE_LINES) + " " * r.randint(0, 40) + ("y" * r.randint(0, 120) if r.random() < 0.3 else "")
        parts.append(_pld(2, _pvf(1, i + 1) + _pld(2, line.encode("utf-8"))))
    return b"".join(parts)


def _ptext(r: random.Random) -> bytes:
    return _e2_text(r, 20, 300).encode("utf-8")


def _pmsg(r: random.Random, depth: int) -> bytes:
    parts: list = []
    for _ in range(r.randint(1, 6)):
        k = r.random()
        if k < 0.12 and depth < 3:
            parts.append(_pld(r.choice([1, 2, 3, 5, 9]), _pmsg(r, depth + 1)))
        elif k < 0.22:
            parts.append(_pld(29, _pld(1, b"skill-" + str(r.randint(0, 99)).encode())
                              + _pld(3, _ptext(r) if r.random() < 0.6 else b"short")))
        elif k < 0.3:
            parts.append(_pld(34, _pld(2, _pld(1, _ptext(r)))) if r.random() < 0.6 else _pld(34, _ptext(r)))
        elif k < 0.36:
            parts.append(_pld(14, _ptext(r)))
        elif k < 0.5:
            parts.append(_pld(r.randint(1, 9), _ptext(r)))
        elif k < 0.6:
            parts.append(_pvf(r.randint(1, 9), r.choice([0, 1, 300, 2 ** 40])))
        elif k < 0.64:
            parts.append(_pvi((r.randint(1, 9) << 3) | 1) + bytes(r.randint(0, 255) for _ in range(8)))
        elif k < 0.67:
            parts.append(_pvi((r.randint(1, 9) << 3) | 5) + bytes(r.randint(0, 255) for _ in range(4)))
        elif k < 0.7:
            # A 64-bit-plus field key: Python keeps the big int, Go declines to re-encode.
            parts.append(bytes([0x82] + [0x80] * 9 + [0x01]) + _pvi(3) + b"abc")
        elif k < 0.78:
            parts.append(_pfile(r))
        else:
            parts.append(_pld(r.randint(1, 9), _ptext(r)))
    return b"".join(parts)


def _e2_frames(r: random.Random) -> bytes:
    kind = r.random()
    if kind < 0.3:
        return _pfile(r)
    if kind < 0.6:
        return _pmsg(r, 0)
    out = b""
    for _ in range(r.randint(1, 3)):
        payload = _pmsg(r, 0) if r.random() < 0.8 else _pfile(r)
        flags = r.choice([0, 0, 0, 1, 2])
        if flags & 1:
            payload = gzip.compress(payload, mtime=0)
        out += bytes([flags]) + len(payload).to_bytes(4, "big") + payload
    return out


def _e2_json_body(r: random.Random) -> bytes:
    obj: Any = {"messages": [{"role": "user", "content": _e2_text(r, 30, 200)}],
                "meta": {"k": [_e2_text(r, 3, 30), 1, None]}}
    if r.random() < 0.3:
        obj = [_e2_text(r, 40, 100) for _ in range(r.randint(1, 16))]
    return json.dumps(obj, indent=r.choice([None, 2])).encode("utf-8")


def _b64_arg(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _e2_step(r: random.Random, session: Any) -> dict:
    k = r.random()
    if k < 0.14:
        return {"fn": "aggressive.apply_aggressive", "args": {
            "payload": _e2_anthropic(r), "session_key": session}}
    if k < 0.22:
        return {"fn": "aggressive.apply_aggressive_openai", "args": {"payload": _e2_openai(r)}}
    if k < 0.28:
        return {"fn": "aggressive.apply_aggressive_responses", "args": {"payload": _e2_responses(r)}}
    if k < 0.36:
        fn = r.choice(["maxsave.apply_tool_result_preprocess", "maxsave.apply_duplicate_tool_result_stubs",
                       "maxsave.apply_message_reminder_strip", "maxsave.drop_empty_text_blocks",
                       "maxsave.apply_age_elide", "maxsave.drop_unused_tool_schemas",
                       "maxsave.apply_maxsave", "maxsave.apply_maxsave_late", "aggressive.apply_stale_tool_compaction",
                       "aggressive.apply_pure_tool_result_ccr"])
        args: dict = {"payload": _e2_anthropic(r), "relax": r.random() < 0.2,
                      "lossless_only": r.random() < 0.3}
        return {"fn": fn, "args": args}
    if k < 0.44:
        fn = r.choice(["maxsave.strip_ansi", "maxsave.squeeze_ws", "maxsave.strip_base64_blobs",
                       "maxsave.drop_progress_lines", "maxsave.crush_cli_tables", "maxsave.compact_json_text",
                       "maxsave.preprocess_tool_text", "maxsave.strip_system_reminders",
                       "maxsave.shrink_bash_git_bloat", "aggressive.crush_text", "aggressive._ccr_wrap",
                       "aggressive._crush_log_lines", "aggressive._crush_json_array",
                       "aggressive._crush_grepish", "aggressive._crush_responses_output_text",
                       "aggressive.aggressive_trim_codex_developer"])
        text = _e2_text(r, 40, 500) if r.random() < 0.85 else r.choice(["", "  ", None, 7])
        return {"fn": fn, "args": {"text": text}}
    if k < 0.48:
        return {"fn": r.choice(["aggressive.aggressive_trim_system", "aggressive.aggressive_trim_system_hard",
                                "maxsave.strip_git_instruction_bloat"]),
                "args": {"system": _e2_system(r, [])}}
    if k < 0.52:
        return {"fn": r.choice(["aggressive.hard_minify_tools", "aggressive.hard_minify_codex_tools",
                                "aggressive.hard_minify_openai_tools", "maxsave.shrink_tool_description_walls"]),
                "args": {"tools": _e2_tools(r)}}
    if k < 0.66:
        body = _e2_frames(r)
        if r.random() < 0.15:
            body = body[: r.randint(0, len(body))]  # truncated frame or body
        fn = r.choice(["wirecrush.crush_connect_body", "wirecrush.crush_connect_body",
                       "wirecrush.crush_protobuf_message", "wirecrush.drain_crush_frames"])
        return {"fn": fn, "args": {"body_b64": _b64_arg(body), "aggressive": r.random() < 0.6,
                                   "use_seen": r.random() < 0.7}}
    if k < 0.74:
        content_type = r.choice(["application/connect+proto", "application/grpc", "application/json",
                                 "application/connect+json", "", "text/plain"])
        path = r.choice(["agent.v1.AgentService/Run", "aiserver.v1.Chat", "/api/chat", "analytics/trackEvents"])
        body = _e2_frames(r) if r.random() < 0.6 else _e2_json_body(r)
        return {"fn": "wirecrush.crush_wire_body", "args": {
            "body_b64": _b64_arg(body), "path": path, "content_type": content_type,
            "aggressive": r.random() < 0.6}}
    if k < 0.78:
        return {"fn": "wirecrush.crush_json_body", "args": {
            "body_b64": _b64_arg(_e2_json_body(r)), "aggressive": r.random() < 0.6}}
    if k < 0.82:
        return {"fn": "wirecrush.crush_json_value", "args": {
            "node": json.loads(_e2_json_body(r)), "aggressive": r.random() < 0.6}}
    return {"fn": "aggressive.apply_aggressive", "args": {"payload": _e2_anthropic(r), "session_key": None}}


def _e2_case(r: random.Random) -> list:
    """Stateful case: a shared session key and seen set, or one step."""
    session = r.choice(["e2s", None]) if r.random() < 0.5 else None
    if r.random() < 0.35:
        n = r.randint(2, 4)
        return [_e2_step(r, session) for _ in range(n)]
    return [_e2_step(r, session)]


def _ag_apply_openai(a):
    p = a["payload"]
    return [aggressive.apply_aggressive_openai(p), p]


def _ag_apply_responses(a):
    p = a["payload"]
    return [aggressive.apply_aggressive_responses(p), p]


def _ag_apply(a):
    p = a["payload"]
    return [aggressive.apply_aggressive(p, a.get("session_key")), p]


def _seen(a):
    return _E2_SEEN if a.get("use_seen") else None


def _b64s(b: bytes) -> str:
    return "b64:" + base64.b64encode(b).decode("ascii")


def _w_connect(a):
    new, saved = wirecrush.crush_connect_body(_b64(a, "body_b64"), aggressive=a["aggressive"], seen=_seen(a))
    return [_b64s(new), saved]


def _w_proto(a):
    new, saved = wirecrush.crush_protobuf_message(_b64(a, "body_b64"), aggressive=a["aggressive"], seen=_seen(a))
    return [_b64s(new), saved]


def _w_wire(a):
    new, cats, saved = wirecrush.crush_wire_body(
        _b64(a, "body_b64"), path=a["path"], content_type=a["content_type"], aggressive=a["aggressive"])
    return [_b64s(new), cats, saved]


def _w_drain(a):
    rest, frames = wirecrush.drain_crush_frames(_b64(a, "body_b64"), aggressive=a["aggressive"], seen=_seen(a))
    return [_b64s(rest), [_b64s(f) for f in frames]]


def _w_json_body(a):
    return wirecrush.crush_json_body(_b64(a, "body_b64"), aggressive=a["aggressive"])


def _w_json_value(a):
    return wirecrush.crush_json_value(a["node"], aggressive=a["aggressive"])


def _m_pair(fn):
    def run(a):
        p = a["payload"]
        return [fn(a, p), p]
    return run


def _m_strip_git(a):
    new, ch = maxsave.strip_git_instruction_bloat(a["system"])
    return [new, ch]


def _m_walls(a):
    t = a["tools"]
    return [maxsave.shrink_tool_description_walls(t), t]


def _m_drop_empty(a):
    p = a["payload"]
    maxsave.drop_empty_text_blocks(p)
    return p


E2_ADAPTERS = {
    "aggressive.apply_aggressive": _ag_apply,
    "aggressive.apply_aggressive_openai": _ag_apply_openai,
    "aggressive.apply_aggressive_responses": _ag_apply_responses,
    "aggressive.aggressive_trim_system": lambda a: aggressive.aggressive_trim_system(a["system"]),
    "aggressive.aggressive_trim_system_hard": lambda a: aggressive.aggressive_trim_system_hard(a["system"]),
    "aggressive.aggressive_trim_codex_developer": lambda a: aggressive.aggressive_trim_codex_developer(a["text"]),
    "aggressive.crush_text": lambda a: aggressive.crush_text(a["text"]),
    "aggressive._ccr_wrap": lambda a: aggressive._ccr_wrap(a["text"]),
    "aggressive._crush_log_lines": lambda a: aggressive._crush_log_lines(a["text"]),
    "aggressive._crush_json_array": lambda a: aggressive._crush_json_array(a["text"]),
    "aggressive._crush_grepish": lambda a: aggressive._crush_grepish(a["text"]),
    "aggressive._crush_responses_output_text": lambda a: aggressive._crush_responses_output_text(a["text"]),
    "aggressive.apply_pure_tool_result_ccr": _m_pair(lambda a, p: aggressive.apply_pure_tool_result_ccr(p, relax=a["relax"])),
    "aggressive.apply_stale_tool_compaction": _m_pair(lambda a, p: aggressive.apply_stale_tool_compaction(p)),
    "aggressive.hard_minify_tools": lambda a: aggressive.hard_minify_tools(a["tools"]),
    "aggressive.hard_minify_codex_tools": lambda a: aggressive.hard_minify_codex_tools(a["tools"]),
    "aggressive.hard_minify_openai_tools": lambda a: aggressive.hard_minify_openai_tools(a["tools"]),
    "maxsave.strip_ansi": lambda a: maxsave.strip_ansi(a["text"]),
    "maxsave.squeeze_ws": lambda a: maxsave.squeeze_ws(a["text"]),
    "maxsave.strip_base64_blobs": lambda a: maxsave.strip_base64_blobs(a["text"]),
    "maxsave.drop_progress_lines": lambda a: maxsave.drop_progress_lines(a["text"]),
    "maxsave.crush_cli_tables": lambda a: maxsave.crush_cli_tables(a["text"]),
    "maxsave.compact_json_text": lambda a: maxsave.compact_json_text(a["text"]),
    "maxsave.preprocess_tool_text": lambda a: maxsave.preprocess_tool_text(a["text"]),
    "maxsave.strip_system_reminders": lambda a: maxsave.strip_system_reminders(a["text"]),
    "maxsave.shrink_bash_git_bloat": lambda a: maxsave.shrink_bash_git_bloat(a["text"]),
    "maxsave.apply_tool_result_preprocess": _m_pair(lambda a, p: maxsave.apply_tool_result_preprocess(p)),
    "maxsave.apply_duplicate_tool_result_stubs": _m_pair(lambda a, p: maxsave.apply_duplicate_tool_result_stubs(p)),
    "maxsave.apply_message_reminder_strip": _m_pair(lambda a, p: maxsave.apply_message_reminder_strip(p)),
    "maxsave.apply_age_elide": _m_pair(lambda a, p: maxsave.apply_age_elide(p, relax=a["relax"])),
    "maxsave.drop_unused_tool_schemas": _m_pair(lambda a, p: maxsave.drop_unused_tool_schemas(p)),
    "maxsave.apply_maxsave": _m_pair(lambda a, p: maxsave.apply_maxsave(p, lossless_only=a["lossless_only"])),
    "maxsave.apply_maxsave_late": _m_pair(lambda a, p: maxsave.apply_maxsave_late(p, relax=a["relax"])),
    "maxsave.strip_git_instruction_bloat": _m_strip_git,
    "maxsave.shrink_tool_description_walls": _m_walls,
    "maxsave.drop_empty_text_blocks": _m_drop_empty,
    "wirecrush.crush_connect_body": _w_connect,
    "wirecrush.crush_protobuf_message": _w_proto,
    "wirecrush.crush_wire_body": _w_wire,
    "wirecrush.drain_crush_frames": _w_drain,
    "wirecrush.crush_json_body": _w_json_body,
    "wirecrush.crush_json_value": _w_json_value,
}
ADAPTERS.update(E2_ADAPTERS)


def _reset_ccr() -> None:
    _ccr._store = _ccr.CcrStore()


def _reset_e2() -> None:
    _E2_SEEN.clear()
    # Pin the debug-capture counters so no bench/runs files are written.
    wirecrush._DUMPED = 5
    wirecrush._AGENT_CAPTURES = 8


RESET.extend([_reset_ccr, _reset_e2])


def _case(r: random.Random, case_id: int, huge: bool) -> dict:
    """Build one case: a list of steps, the expected results, and the errors."""
    if huge:
        steps = [{"fn": "optimizer.apply_file_delta_tracking",
                  "args": {"payload": {"messages": [{"role": "user", "content": "x"}]},
                           "session_key": f"s{i}"}} for i in range(205)]
    elif r.random() < 0.08:
        steps = _loop_session_steps(r, r.choice(["s1", "s2"]))
    elif r.random() < 0.12:
        steps = _file_sequence(r, f"sess{r.randint(0, 2)}", r.randint(2, 4))
    elif r.random() < 0.45:
        steps = _e2_case(r)
    else:
        steps = [_step(r) for _ in range(r.randint(1, 3))]
        if r.random() < 0.08:
            steps.append({"fn": "optimizer.derive_session_key", "args": {"payload": _payload(r)}})
    return {"steps": steps}


def _run_case(case_id: int, steps: list) -> dict:
    for reset in RESET:
        reset()
    wants: list[str] = []
    errs: list[bool] = []
    for step in steps:
        fn = ADAPTERS[step["fn"]]
        try:
            value = fn(json.loads(json.dumps(step["args"], ensure_ascii=True, allow_nan=True)))
        except Exception:  # noqa: BLE001 - any Python exception is an error case
            wants.append("")
            errs.append(True)
            continue
        wants.append(canonical(value))
        errs.append(False)
    return {"id": case_id, "steps": steps, "want": wants, "err": errs}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    r = random.Random(args.seed)
    huge_at = args.count // 2
    with open(args.out, "w", encoding="utf-8") as fh:
        for n in range(args.count):
            case = _case(r, n, huge=(n == huge_at))
            line = _run_case(n, case["steps"])
            fh.write(json.dumps(line, ensure_ascii=True, allow_nan=True, separators=(",", ":")) + "\n")
    print(f"wrote {args.count} cases to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
