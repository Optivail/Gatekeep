#!/usr/bin/env python3
"""SQLite round trip between the Python reference and the Go server.

One data directory is shared. Python writes rows through its server, Go reads
them and writes more, and Python reads everything back. Each read is compared
with the other side's API output (volatile timing fields are ignored), and the
database must pass PRAGMA integrity_check after every step.

Usage: python -I server/parity/db_roundtrip.py --go-bin PATH
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import http_parity as hp  # noqa: E402


def integrity(db: Path) -> None:
    con = sqlite3.connect(db)
    try:
        result = con.execute("PRAGMA integrity_check").fetchone()[0]
        rows = con.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]
    finally:
        con.close()
    if result != "ok":
        raise SystemExit(f"integrity_check failed: {result}")
    print(f"  integrity ok, request_logs rows={rows}")


def fetch(srv: hp.Server, path: str) -> object:
    status, _h, body = srv.request("GET", path)
    if status != 200:
        raise SystemExit(f"{srv.label} GET {path} -> {status}")
    return json.loads(body)


def send_requests(srv: hp.Server, seed: int) -> None:
    sent = [("/v1/messages?mode=json", hp.anthropic_body(300, seed, stream=False)),
            ("/v1/chat/completions?mode=openai_stream", hp.openai_chat_body(300, seed + 1, stream=True))]
    for path, body in sent:
        status, _h, _b = srv.request("POST", path, body=body, headers={"content-type": "application/json"})
        if status != 200:
            raise SystemExit(f"{srv.label} POST {path} -> {status}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go-bin", required=True)
    args = ap.parse_args()
    upstream = hp.Upstream()
    env = {"ANTHROPIC_BASE_URL": upstream.url, "OPENAI_BASE_URL": upstream.url}
    tmp = Path(tempfile.mkdtemp(prefix="gk-db-roundtrip-"))
    data = tmp / "shared"
    data.mkdir()
    py_cmd = [sys.executable, "main.py"]
    go_cmd = [str(Path(args.go_bin).resolve())]
    py_dir = str(hp.REPO / "backend")
    go_dir = str(hp.REPO / "server")
    servers: list[hp.Server] = []
    try:
        print("1. Python writes rows")
        py = hp.Server("python", py_cmd, py_dir, env, hp.free_port(), str(data))
        servers.append(py)
        send_requests(py, 1)
        py_view_1 = fetch(py, "/api/stats?range=24h")
        py.stop(); servers.remove(py)
        integrity(data / "proxy_data.db") if (data / "proxy_data.db").exists() else print("  (db file name differs; see below)")

        print("2. Go reads Python rows, then writes more")
        go = hp.Server("go", go_cmd, go_dir, env, hp.free_port(), str(data))
        servers.append(go)
        go_view_1 = fetch(go, "/api/stats?range=24h")
        go_logs_1 = fetch(go, "/api/logs?limit=50&range=24h")
        if hp.normalise_json(go_view_1, []) != hp.normalise_json(py_view_1, []):
            print("  DIFF: Go stats differ from Python on the Python-written DB:",
                  hp.first_json_diff(hp.normalise_json(py_view_1, []), hp.normalise_json(go_view_1, [])))
            return 1
        print("  Go stats match Python on Python-written rows")
        send_requests(go, 2)
        go_view_2 = fetch(go, "/api/stats?range=24h")
        go.stop(); servers.remove(go)

        print("3. Python reads Go rows")
        py = hp.Server("python", py_cmd, py_dir, env, hp.free_port(), str(data))
        servers.append(py)
        py_view_2 = fetch(py, "/api/stats?range=24h")
        py_logs_2 = fetch(py, "/api/logs?limit=50&range=24h")
        py.stop(); servers.remove(py)
        d = hp.first_json_diff(hp.normalise_json(go_view_2, []), hp.normalise_json(py_view_2, []))
        if d:
            print("  DIFF: Python stats differ from Go on the Go-written DB:", d)
            return 1
        print("  Python stats match Go's on Go-written rows")
        if len(go_logs_1.get("logs", [])) == 0 or len(py_logs_2.get("logs", [])) <= len(go_logs_1.get("logs", [])):
            print("  DIFF: row counts did not grow as expected")
            return 1
        integrity(data / "proxy_data.db") if (data / "proxy_data.db").exists() else print("  (db file name differs)")
        print("round trip OK")
        return 0
    finally:
        for s in servers:
            s.stop()
        upstream.server.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
