#!/usr/bin/env python3
"""HTTP-level parity between the Python reference and the Go Gatekeep server.

Both servers run against the same fake upstream (Anthropic and OpenAI style,
streaming and non-streaming, errors, gzip). For every scenario the harness
records what reached the upstream (method, path, query, forwarded headers,
exact body bytes), what the client received (status, headers, body bytes),
and the dashboard API answers. Each server gets its own fresh data directory.

Usage:
  python -I server/parity/http_parity.py --go-bin PATH [--py-dir backend]
  python -I server/parity/http_parity.py --self-check   # Python vs Python

Exit status is 0 only when every comparison matches or is listed in KNOWN.
Secrets: the API key used for the auth scenarios is random per run and is
never printed.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import http.client
import http.server
import json
import os
import random
import re
import secrets
import shutil
import socket
import string
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Known, intentional differences. Each entry needs a reason. A scenario that
# differs only in these fields is reported as KNOWN, not as a failure.
KNOWN: dict[str, str] = {
    "header:server": "uvicorn and Go net/http send different Server headers",
    "header:date": "wall-clock date",
    "header:transfer-encoding": "chunk framing may differ; body bytes are compared instead",
    "header:connection": "hop-by-hop",
    "header:content-length": ("Python sends Content-Length for a buffered SSE body; Go streams it chunked. "
                              "Body bytes are compared and identical; only framing differs"),
    "header:keep-alive": "hop-by-hop",
    "header:etag": "Go static file serving omits ETag; body bytes identical (cosmetic, cache validation only)",
    "header:cache-control": ("Intentional: Go sends no-cache on every served .html, including /dashboard/ "
                             "(Python's normpath quirk skips the index)"),
}

VOLATILE_KEYS = {
    "ts", "timestamp", "created_at", "updated_at", "started_at", "ended_at",
    "time", "latency_ms", "duration_ms", "elapsed_ms", "ms", "id", "request_id",
    "updated", "created", "last_seen", "fetched_at", "refreshed_at",
}
ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer",
              "upgrade", "proxy-authenticate", "proxy-authorization", "host",
              "accept-encoding", "date", "server"}


# ---------------------------------------------------------------- upstream

class Upstream:
    """Fake Anthropic/OpenAI upstream. Records every request it receives."""

    def __init__(self) -> None:
        self.log: list[dict] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args) -> None:  # keep stdout clean
                return

            def _handle(self) -> None:
                n = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(n) if n else b""
                path, _, query = self.path.partition("?")
                mode = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("mode", "json")
                rec = {
                    "method": self.command,
                    "path": path,
                    "query": query,
                    "headers": sorted(
                        (k.lower(), v) for k, v in self.headers.items()
                        if k.lower() not in HOP_BY_HOP and k.lower() != "content-length"
                    ),
                    "body": body,
                    "mode": mode,
                }
                with outer._lock:
                    outer.log.append(rec)
                outer._respond(self, mode, body)

            do_GET = do_POST = do_PUT = do_DELETE = _handle

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def take(self) -> list[dict]:
        with self._lock:
            out, self.log = self.log, []
        return out

    @staticmethod
    def _respond(h, mode: str, body: bytes) -> None:
        digest = hashlib.sha256(body).hexdigest()[:16]
        if mode == "stream":
            events = [
                b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"usage":{"input_tokens":10,"output_tokens":1}}}\n\n',
                b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n',
                b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" done"}}\n\n',
                b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            ]
            _send(h, 200, "text/event-stream", b"".join(events), {"x-upstream-body-sha256": digest}, chunked=True)
        elif mode == "openai_stream":
            events = [
                b'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n',
                b'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            _send(h, 200, "text/event-stream", b"".join(events), {"x-upstream-body-sha256": digest}, chunked=True)
        elif mode == "gzip":
            payload = json.dumps({"id": "msg_2", "type": "message", "content": [{"type": "text", "text": "gz"}]}).encode()
            _send(h, 200, "application/json", gzip.compress(payload), {"content-encoding": "gzip", "x-upstream-body-sha256": digest})
        elif mode == "err429":
            _send(h, 429, "application/json", b'{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}', {"retry-after": "1"})
        elif mode == "err500":
            _send(h, 500, "text/plain", b"boom", {})
        elif mode == "badjson":
            _send(h, 200, "application/json", b"not json at all", {})
        else:
            payload = json.dumps({"id": "msg_1", "type": "message", "role": "assistant",
                                  "content": [{"type": "text", "text": "ok"}],
                                  "usage": {"input_tokens": 10, "output_tokens": 1}}).encode()
            _send(h, 200, "application/json", payload, {"x-upstream-body-sha256": digest})


def _send(h, status: int, ctype: str, body: bytes, extra: dict, chunked: bool = False) -> None:
    h.send_response(status)
    h.send_header("content-type", ctype)
    for k, v in extra.items():
        h.send_header(k, v)
    if chunked:
        h.send_header("transfer-encoding", "chunked")
        h.end_headers()
        for i in range(0, len(body), 97):
            piece = body[i:i + 97]
            h.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
        h.wfile.write(b"0\r\n\r\n")
    else:
        h.send_header("content-length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)
    h.wfile.flush()


# ---------------------------------------------------------------- servers

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """One Gatekeep process (Python or Go) with a fresh data directory."""

    def __init__(self, label: str, cmd: list[str], cwd: str, env: dict, port: int, data: str) -> None:
        self.label = label
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self.data = data
        full_env = dict(os.environ)
        full_env.update(env)
        full_env.update({
            "GATEKEEP_HOST": "127.0.0.1",
            "GATEKEEP_PORT": str(port),
            "GATEKEEP_AGENT_PORT": "9478",  # fixed so agent_url matches across servers (listener is disabled)
            "GATEKEEP_DISABLE_AGENT": "1",
            "GATEKEEP_DATA_DIR": data,
            "GATEKEEP_PUBLIC_URL": self.base,
        })
        # Server output goes to a log file so a failed start is diagnosable.
        self.log_path = Path(data).parent / f"{label}-{port}.log"
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=cwd, env=full_env,
                                     stdout=self.log_file, stderr=subprocess.STDOUT)
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                st, _h, _b = self.request("GET", "/api/stats")
                if st == 200:
                    return
            except OSError:
                pass
            if self.proc.poll() is not None:
                break
            time.sleep(0.25)
        self.stop()
        raise RuntimeError(f"{label} did not answer /api/stats (log: {self.log_path})")

    def request(self, method: str, path: str, body: bytes | None = None,
                headers: dict | None = None) -> tuple[int, list[tuple[str, str]], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            resp = conn.getresponse()
            data = resp.read()
            return resp.status, [(k.lower(), v) for k, v in resp.getheaders()], data
        finally:
            conn.close()

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log_file.close()


# ---------------------------------------------------------------- workloads

WORDS_RNG = random.Random(7)
WORDS = ["".join(WORDS_RNG.choices(string.ascii_lowercase, k=WORDS_RNG.randint(2, 9))) for _ in range(800)]
CODE = "def handler(event, ctx):\n    for item in event['items']:\n        if item.get('ok'):\n            total += item['value']  # accumulate\n    return {'total': total}\n"
LOG = "2026-10-29T10:00:{s:02d}Z INFO worker[{n}] processed batch id={u} in 12ms status=200\n"


def conversation(tokens: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    target, size, msgs, i = tokens * 4, 0, [], 0
    while size < target:
        kind = i % 3
        if kind == 0:
            txt = " ".join(rng.choices(WORDS, k=120))
        elif kind == 1:
            txt = "```python\n" + CODE * 8 + "```"
        else:
            txt = "".join(LOG.format(s=j % 60, n=j % 7, u=uuid.UUID(int=rng.getrandbits(128)).hex[:8]) for j in range(60))
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": txt})
        size += len(txt)
        i += 1
    if msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": "continue"})
    return msgs


def anthropic_body(tokens: int, seed: int, stream: bool = True, extra: dict | None = None) -> bytes:
    msgs = conversation(tokens, seed)
    msgs[0]["content"] = f"{seed:08x} " + msgs[0]["content"]
    body = {"model": "claude-sonnet-4-5", "max_tokens": 64, "stream": stream,
            "system": "You are a coding agent.", "messages": msgs}
    body.update(extra or {})
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def openai_chat_body(tokens: int, seed: int) -> bytes:
    msgs = [{"role": "system", "content": "You are helpful. Certainly! Sure thing."}]
    msgs += conversation(tokens, seed)
    return json.dumps({"model": "gpt-4o", "stream": True, "messages": msgs}, ensure_ascii=False).encode()


def scenarios(api_key: str | None) -> list[dict]:
    big = anthropic_body(3000, 11)
    huge_line = json.dumps({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "x" * 120000}]}).encode()
    unicode_msg = json.dumps({"model": "m", "max_tokens": 8, "messages": [
        {"role": "user", "content": "Ik heb een fout: éè 你好 \U0001F600 \r\n\tTab\r\nCRLF   spaces  "}]}, ensure_ascii=False).encode()
    tool = json.dumps({"model": "m", "max_tokens": 8, "tools": [{"name": "bash", "description": "Run a shell command. " * 30,
                                                                "input_schema": {"type": "object", "properties": {"cmd": {"type": "string"}}}}],
                       "messages": [{"role": "user", "content": "list files"}]}).encode()
    gz_body = gzip.compress(big)
    rows = [
        {"name": "anthropic-small-json", "method": "POST", "path": "/v1/messages?mode=json",
         "body": anthropic_body(20, 1, stream=False), "headers": {"content-type": "application/json"}},
        {"name": "anthropic-large-stream", "method": "POST", "path": "/v1/messages?mode=stream", "body": big,
         "headers": {"content-type": "application/json"}},
        {"name": "anthropic-cache-repeat", "method": "POST", "path": "/v1/messages?mode=stream", "body": big,
         "headers": {"content-type": "application/json"}},
        {"name": "anthropic-gzip-request", "method": "POST", "path": "/v1/messages?mode=stream", "body": gz_body,
         "headers": {"content-type": "application/json", "content-encoding": "gzip"}},
        {"name": "anthropic-gzip-response", "method": "POST", "path": "/v1/messages?mode=gzip", "body": big,
         "headers": {"content-type": "application/json"}},
        {"name": "anthropic-huge-line", "method": "POST", "path": "/v1/messages?mode=json", "body": huge_line,
         "headers": {"content-type": "application/json"}},
        {"name": "anthropic-unicode-crlf", "method": "POST", "path": "/v1/messages?mode=json", "body": unicode_msg,
         "headers": {"content-type": "application/json"}},
        {"name": "anthropic-tools", "method": "POST", "path": "/v1/messages?mode=json", "body": tool,
         "headers": {"content-type": "application/json"}},
        {"name": "anthropic-upstream-429", "method": "POST", "path": "/v1/messages?mode=err429", "body": anthropic_body(50, 2, stream=False),
         "headers": {"content-type": "application/json"}},
        {"name": "anthropic-upstream-500", "method": "POST", "path": "/v1/messages?mode=err500", "body": anthropic_body(50, 3, stream=False),
         "headers": {"content-type": "application/json"}},
        {"name": "anthropic-upstream-badjson", "method": "POST", "path": "/v1/messages?mode=badjson", "body": anthropic_body(50, 4, stream=False),
         "headers": {"content-type": "application/json"}},
        {"name": "anthropic-invalid-body", "method": "POST", "path": "/v1/messages?mode=json", "body": b"{bad",
         "headers": {"content-type": "application/json"}},
        {"name": "anthropic-empty-body", "method": "POST", "path": "/v1/messages?mode=json", "body": b"",
         "headers": {"content-type": "application/json"}},
        {"name": "openai-chat-stream", "method": "POST", "path": "/v1/chat/completions?mode=openai_stream",
         "body": openai_chat_body(2000, 5), "headers": {"content-type": "application/json"}},
        {"name": "openai-responses", "method": "POST", "path": "/v1/responses?mode=json",
         "body": json.dumps({"model": "gpt-5", "input": "Hello there, could you please summarise this? " * 40}).encode(),
         "headers": {"content-type": "application/json"}},
        {"name": "models-list", "method": "GET", "path": "/v1/models?mode=json", "body": None, "headers": {}},
        {"name": "unknown-path", "method": "GET", "path": "/no/such/thing?mode=json", "body": None, "headers": {}},
        {"name": "root-html", "method": "GET", "path": "/", "body": None, "headers": {"accept": "text/html"}},
        {"name": "root-json", "method": "GET", "path": "/", "body": None, "headers": {"accept": "application/json"}},
        {"name": "dashboard-redirect", "method": "GET", "path": "/dashboard", "body": None, "headers": {}},
        {"name": "dashboard-index", "method": "GET", "path": "/dashboard/", "body": None, "headers": {}},
        {"name": "wire-script", "method": "GET", "path": "/install/wire-agents.ps1", "body": None, "headers": {}},
        {"name": "settings-bad-post", "method": "POST", "path": "/api/settings/aggressive_enabled", "body": b"{}",
         "headers": {"content-type": "application/json"}},
    ]
    if api_key:
        auth_body = anthropic_body(10, 9, stream=False)
        rows += [
            {"name": "auth-missing", "method": "POST", "path": "/v1/messages?mode=json", "body": auth_body,
             "headers": {"content-type": "application/json"}},
            {"name": "auth-wrong", "method": "POST", "path": "/v1/messages?mode=json", "body": auth_body,
             "headers": {"content-type": "application/json", "authorization": "Bearer wrong"}},
            {"name": "auth-bearer-ok", "method": "POST", "path": "/v1/messages?mode=json", "body": auth_body,
             "headers": {"content-type": "application/json", "authorization": f"bearer {api_key}"}},
            {"name": "auth-header-ok", "method": "POST", "path": "/v1/messages?mode=json", "body": auth_body,
             "headers": {"content-type": "application/json", "x-gatekeep-key": api_key}},
        ]
    return rows


def setting_matrix(setting_keys: list[str]) -> list[dict]:
    """Every boolean setting off and on, with a stream and a non-stream request."""
    rows = []
    for key in setting_keys:
        for value in (0, 1):
            rows.append({"name": f"setting-{key}={value}", "method": "POST",
                         "path": f"/api/settings/{key}", "body": json.dumps({"value": value}).encode(),
                         "headers": {"content-type": "application/json"}, "kind": "setting"})
            rows.append({"name": f"after-{key}={value}-stream", "method": "POST",
                         "path": "/v1/messages?mode=stream", "body": anthropic_body(400, 20 + value, stream=True),
                         "headers": {"content-type": "application/json"}})
            rows.append({"name": f"after-{key}={value}-json", "method": "POST",
                         "path": "/v1/messages?mode=json", "body": anthropic_body(400, 40 + value, stream=False),
                         "headers": {"content-type": "application/json"}})
    return rows


DASHBOARD = ["/api/stats?range=24h", "/api/stats?range=24h&include_noise=true", "/api/logs?limit=50",
             "/api/logs?limit=5&offset=2&range=24h", "/api/prices", "/api/settings", "/api/server-info"]


# ---------------------------------------------------------------- compare

IGNORED_HEADERS = {"server", "date", "transfer-encoding", "connection", "keep-alive"}


def is_volatile(key: str) -> bool:
    """Timing fields differ between any two runs; the rest must match exactly."""
    return key in VOLATILE_KEYS or "latency" in key or key.endswith("_ms") or "elapsed" in key


def normalise_json(obj, port_tokens: list[str]):
    if isinstance(obj, dict):
        return {k: ("<volatile>" if is_volatile(k) else normalise_json(v, port_tokens)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalise_json(v, port_tokens) for v in obj]
    if isinstance(obj, str):
        out = ISO_RE.sub("<ts>", obj)
        for tok in port_tokens:
            out = out.replace(tok, "<port>")
        return out
    return obj


def first_json_diff(a, b, path: str = "$") -> str | None:
    """Path and values of the first difference between two decoded JSON values."""
    if type(a) is not type(b):
        return f"{path}: {str(a)[:60]!r} != {str(b)[:60]!r}"
    if isinstance(a, dict):
        if list(a) != list(b):
            return f"{path}: keys {list(a)[:8]} != {list(b)[:8]}"
        for k in a:
            d = first_json_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_json_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {str(a)[:60]!r} != {str(b)[:60]!r}"


def compare_responses(a: tuple, b: tuple, port_a: str, port_b: str) -> list[str]:
    """Differences between two (status, headers, body) tuples. Empty when equal."""
    sa, ha, ba = a
    sb, hb, bb = b
    diffs = []
    if sa != sb:
        diffs.append(f"status {sa} != {sb}")
    # JSON bodies are compared after normalisation, so their length may legitimately differ.
    is_json = "application/json" in next((v for k, v in ha if k == "content-type"), "")
    skip = IGNORED_HEADERS | ({"content-length"} if is_json else set())
    fa = sorted((k, v.replace(port_a, "<port>")) for k, v in ha if k not in skip)
    fb = sorted((k, v.replace(port_b, "<port>")) for k, v in hb if k not in skip)
    if fa != fb:
        for name in sorted({k for k, _ in fa + fb}):
            va = [v for k, v in fa if k == name]
            vb = [v for k, v in fb if k == name]
            if va != vb:
                diffs.append(f"header {name}: {va} != {vb}")
    ctype = next((v for k, v in ha if k == "content-type"), "")
    if "application/json" in ctype and ba and bb:
        try:
            ja = normalise_json(json.loads(ba), [port_a])
            jb = normalise_json(json.loads(bb), [port_b])
            d = first_json_diff(ja, jb)
            if d:
                diffs.append(f"json body differs at {d}")
        except ValueError:
            if ba != bb:
                diffs.append("body bytes differ (not JSON)")
    elif ba.replace(port_a.encode(), b"<port>") != bb.replace(port_b.encode(), b"<port>"):
        diffs.append(f"body bytes differ ({len(ba)} vs {len(bb)} bytes)")
    return diffs


def compare_upstream(a: list[dict], b: list[dict]) -> list[str]:
    """What reached the upstream: path, query, forwarded headers and exact body bytes."""
    if len(a) != len(b):
        return [f"upstream request count {len(a)} != {len(b)}"]
    diffs = []
    for ra, rb in zip(a, b):
        for field in ("method", "path", "query"):
            if ra[field] != rb[field]:
                diffs.append(f"upstream {field} {ra[field]!r} != {rb[field]!r}")
        if ra["headers"] != rb["headers"]:
            only_a = sorted(set(ra["headers"]) - set(rb["headers"]))
            only_b = sorted(set(rb["headers"]) - set(ra["headers"]))
            diffs.append(f"upstream headers differ: only-py={only_a} only-other={only_b}")
        if ra["body"] != rb["body"]:
            diffs.append(
                f"upstream body differs: sha256 {hashlib.sha256(ra['body']).hexdigest()[:12]} != "
                f"{hashlib.sha256(rb['body']).hexdigest()[:12]} ({len(ra['body'])} vs {len(rb['body'])} bytes)")
    return diffs


# ---------------------------------------------------------------- driver

def run_rows(pair: tuple[Server, Server], upstream: Upstream, rows: list[dict],
             extra_headers: dict, report: list[dict]) -> None:
    py, other = pair
    for row in rows:
        headers = dict(row.get("headers") or {})
        if not row["name"].startswith("auth-"):
            headers.update(extra_headers)
        got = []
        for srv in (py, other):
            upstream.take()
            resp = srv.request(row["method"], row["path"], body=row["body"], headers=headers)
            got.append((resp, upstream.take()))
        (resp_a, up_a), (resp_b, up_b) = got
        diffs = compare_responses(resp_a, resp_b, py.base, other.base)
        diffs += compare_upstream(up_a, up_b)
        report.append({"name": row["name"], "diffs": diffs, "upstream_calls": len(up_a)})


def run_dashboard(pair: tuple[Server, Server], paths: list[str], report: list[dict]) -> None:
    py, other = pair
    for path in paths:
        diffs = compare_responses(py.request("GET", path), other.request("GET", path), py.base, other.base)
        report.append({"name": f"dashboard {path}", "diffs": diffs, "upstream_calls": 0})


def is_known(diff: str) -> bool:
    """A diff is KNOWN only when it is a header listed in KNOWN."""
    m = re.match(r"header ([a-z0-9-]+): ", diff)
    return bool(m) and f"header:{m.group(1)}" in KNOWN


def classify(report: list[dict]) -> tuple[int, int, int]:
    ok = known = bad = 0
    for item in report:
        if not item["diffs"]:
            ok += 1
        elif all(is_known(d) for d in item["diffs"]):
            known += 1
        else:
            bad += 1
    return ok, known, bad


def start_pair(py_cmd: list[str], py_dir: str, other_cmd: list[str], other_cwd: str, env: dict,
               tmp: Path, label: str, servers: list[Server], other_label: str) -> tuple[Server, Server]:
    py = Server("python", py_cmd, py_dir, env, free_port(), str(tmp / f"py-{label}"))
    servers.append(py)
    other = Server(other_label, other_cmd, other_cwd, env, free_port(), str(tmp / f"{other_label}-{label}"))
    servers.append(other)
    return py, other


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go-bin", help="built Go server binary")
    ap.add_argument("--py-dir", default=str(REPO / "backend"), help="Python reference backend directory")
    ap.add_argument("--self-check", action="store_true", help="run the Python server twice (harness sanity)")
    ap.add_argument("--keep", action="store_true", help="keep the temp data directories")
    args = ap.parse_args()
    if not args.self_check and not args.go_bin:
        ap.error("--go-bin or --self-check is required")

    upstream = Upstream()
    base_env = {"ANTHROPIC_BASE_URL": upstream.url, "OPENAI_BASE_URL": upstream.url}
    py_cmd = [sys.executable, "main.py"]  # no -I: the server imports its own modules from its directory
    if args.self_check:
        other_cmd, other_cwd, other_label = py_cmd, args.py_dir, "python-2"
        other_cwd_default = args.py_dir
    else:
        other_cmd, other_label = [str(Path(args.go_bin).resolve())], "go"
        other_cwd_default = str(REPO / "server")
    tmp = Path(tempfile.mkdtemp(prefix="gk-http-parity-"))
    servers: list[Server] = []
    report: list[dict] = []
    try:
        # Pass 1: open server (no API key). Every scenario, the settings matrix, the dashboard.
        pair = start_pair(py_cmd, args.py_dir, other_cmd, other_cwd_default, base_env,
                          tmp, "open", servers, other_label)
        run_rows(pair, upstream, scenarios(None), {}, report)
        status, _h, body = pair[0].request("GET", "/api/settings")
        keys = sorted(k for k, v in json.loads(body).items() if v in (0, 1)) if status == 200 else []
        run_rows(pair, upstream, setting_matrix(keys), {}, report)
        run_dashboard(pair, DASHBOARD, report)
        for s in servers:
            s.stop()
        servers.clear()

        # Pass 2: API key required. The key is random per run and never printed.
        api_key = secrets.token_urlsafe(24)
        pair = start_pair(py_cmd, args.py_dir, other_cmd, other_cwd_default,
                          dict(base_env, GATEKEEP_API_KEY=api_key), tmp, "auth", servers, other_label)
        run_rows(pair, upstream, [r for r in scenarios(api_key) if r["name"].startswith("auth-")], {}, report)
        run_rows(pair, upstream, [{"name": "auth-keyed-json", "method": "POST", "path": "/v1/messages?mode=json",
                                   "body": anthropic_body(10, 9, stream=False),
                                   "headers": {"content-type": "application/json"}}],
                 {"authorization": f"Bearer {api_key}"}, report)
        run_dashboard(pair, DASHBOARD, report)
    finally:
        for s in servers:
            s.stop()
        upstream.server.shutdown()
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    ok, known, bad = classify(report)
    for item in report:
        if item["diffs"]:
            print(f"DIFF {item['name']}: " + "; ".join(item["diffs"])[:400])
    print(f"cases={len(report)} identical={ok} known={known} differing={bad} "
          f"upstream_calls={sum(i['upstream_calls'] for i in report)}")
    if known:
        print("known differences (reasons):")
        for k, why in KNOWN.items():
            print(f"  {k}: {why}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
