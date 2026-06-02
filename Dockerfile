# Multi-stage build: Go 1.23 builder → distroless runtime
FROM golang:1.23-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG VERSION=dev
ARG COMMIT=unknown
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath \
    -ldflags="-s -w -X main.version=${VERSION} -X main.commit=${COMMIT} -X main.builtAt=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    -o /out/mcp-rag ./cmd/mcp-rag/

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/mcp-rag /usr/local/bin/mcp-rag
COPY --from=build /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=build /src/static /static
COPY --from=build /src/config.docker.yaml /etc/mcp-rag/config.yaml
EXPOSE 8060
USER nonroot:nonroot
ENTRYPOINT ["/usr/local/bin/mcp-rag", "serve", "--config", "/etc/mcp-rag/config.yaml"]
