# syntax=docker/dockerfile:1

FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM golang:1.27 AS build
WORKDIR /src/server
COPY server/go.mod server/go.sum ./
RUN go mod download
COPY server/ ./
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gatekeep .

FROM debian:stable-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system gatekeep \
    && useradd --system --gid gatekeep --home-dir /var/lib/gatekeep --shell /usr/sbin/nologin gatekeep

WORKDIR /app

# The binary finds the repo root by looking for server/go.mod beside it, so
# that file must be present. Nothing under backend/ is needed at runtime.
COPY --from=build /out/gatekeep /app/server/gatekeep
COPY server/go.mod /app/server/go.mod
COPY install/wire-agents.ps1 /app/install/wire-agents.ps1
COPY --from=frontend /app/frontend/dist /app/frontend/dist

# A named volume copies this directory's owner on first use, so the non-root
# user can write the SQLite file. A bind mount must be chowned to this uid.
RUN mkdir -p /var/lib/gatekeep && chown gatekeep:gatekeep /var/lib/gatekeep

ENV GATEKEEP_HOST=0.0.0.0 \
    GATEKEEP_PORT=9477 \
    GATEKEEP_AGENT_HOST=0.0.0.0 \
    GATEKEEP_AGENT_PORT=9478 \
    GATEKEEP_DATA_DIR=/var/lib/gatekeep

EXPOSE 9477 9478

USER gatekeep

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:9477/api/stats" >/dev/null || exit 1

CMD ["/app/server/gatekeep"]
