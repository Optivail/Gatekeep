# Gatekeep

<p align="center">
  <img src="docs/assets/readme-dashboard.png" alt="Gatekeep dashboard: live token deltas, save rate, and observed headroom for the selected session period" width="920" />
</p>

<p align="center"><em>Example session view (27 requests): 815k raw tokens reduced to 115k sent (85.9% save rate, ×7.1 observed headroom). Multipliers update live per request.</em></p>

**Self-hosted proxy that compresses AI coding CLI traffic.** Point Claude Code, Cursor, Codex, or Vibe at Gatekeep once. Same workflow, fewer tokens billed upstream.

## Features

- **High-Impact Token Savings:** Up to 85%+ token reduction on deep agent loops and heavy multi-turn contexts via wire crush and tool minification
- **Real-time metrics dashboard** tracking live token deltas, save rates, and dynamic capacity multipliers per request
- **Transparent pass-through** with sub-40ms average proxy overhead
- **Multi-provider routing:** Anthropic, Cursor ConnectRPC, ChatGPT Codex, Mistral/Vibe, OpenAI-compatible APIs
- **Zero workflow disruption:** wire once, run your CLIs normally
- Docker-first install with one-shot wire scripts for client machines
- Optional `GATEKEEP_API_KEY` for proxy authentication

## Requirements

- **Production:** Docker and Docker Compose
- **Development:** Go 1.27+, Node.js 22+ (frontend build); Python 3.12+ only for the parity checks

Default ports: **9477** (proxy + dashboard), **9478** (Cursor agent listener).

## Quick start

```bash
git clone <repo-url> gatekeep && cd gatekeep
cp .env.example .env
./install/install.sh
```

Open the dashboard at http://127.0.0.1:9477/dashboard/

On a remote server, set `GATEKEEP_PUBLIC_URL` and `GATEKEEP_AGENT_URL` in `.env` before starting. See [docs/INSTALL.md](docs/INSTALL.md).

## Wire coding CLIs

From a client machine (PowerShell):

```powershell
$g='http://YOUR-SERVER:9477'; $a='http://YOUR-SERVER:9478'; $f=Join-Path $env:TEMP 'gk-wire.ps1'; irm "$g/install/wire-agents.ps1" -OutFile $f; & $f -GatekeepUrl $g -AgentUrl $a
```

The dashboard Wiki tab shows the exact command for your deployment.

Manual details: [docs/AGENTS.md](docs/AGENTS.md)

## Development

The server is the Go backend in `server/` (the production build). The Python
backend in `backend/` is kept as the reference that the Go code is checked
against.

```bash
cd frontend && npm install && npm run build
cd ../server && go run .            # listens on 9477 by default
```

Set `GATEKEEP_DEBUG=1` for verbose proxy logging.

### Parity checks

```bash
cd server
go test -count=1 ./...                                   # unit tests
python -I parity/engine_cases.py --out /tmp/engines.jsonl --count 5000 --seed 1
GK_PARITY_CORPUS=/tmp/engines.jsonl go test -count=1 -run Parity ./engines   # engine byte parity
go build -o /tmp/gatekeep-go .
python -I parity/http_parity.py --go-bin /tmp/gatekeep-go   # HTTP parity vs the Python backend
```

The Python reference needs `pip install -r backend/requirements.txt`.

### Not yet ported to Go

- The Cursor agent listener on port 9478 (`backend/core/cursor_agent*.py`,
  `backend/engines/cursor_rewrite.py`). The Go server does not open that port,
  and requests routed to the cursor provider return 501.
- Requests the Go engines cannot reproduce byte for byte (a `python` fence in
  pruned history, zstd bodies, some Connect frames) are forwarded unmodified,
  so their upstream bytes differ from the Python backend.

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/INSTALL.md](docs/INSTALL.md) | Install, remote deploy, operations |
| [docs/AGENTS.md](docs/AGENTS.md) | CLI wiring reference |
| [.env.example](.env.example) | Environment variables |

## License

Copyright (c) 2026 Sealeria. Gatekeep is licensed under [AGPL-3.0-or-later](LICENSE).
