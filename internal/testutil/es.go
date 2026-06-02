//go:build integration

// Package testutil provides shared test helpers for integration tests.
package testutil

import (
	"context"
	"fmt"
	"net/http"
	"os/exec"
	"sync"
	"testing"
	"time"
)

const (
	esImage       = "elasticsearch:8.16.1"
	esHostPort    = "9200"
	esName        = "mcp-rag-itest-es"
	esStartupWait = 60 * time.Second
)

const esURL = "http://localhost:" + esHostPort

var (
	esOnce     sync.Once
	esStartErr error
)

// StartES starts an ephemeral Elasticsearch container (one per test
// run — sync.Once per process, reuses existing container across
// packages via reuseIfHealthy). Returns the ES URL.
//
// Uses docker CLI directly (no testcontainers-go) to avoid pulling
// the ryuk sidecar from Docker Hub.
func StartES(t *testing.T, ctx context.Context) (string, error) {
	t.Helper()
	esOnce.Do(func() {
		esStartErr = startES(ctx)
	})
	return esURL, esStartErr
}

// StopES removes the ephemeral ES container.
func StopES() {
	exec.Command("docker", "rm", "-f", esName).Run()
}

func startES(ctx context.Context) error {
	// If the named container is already running and healthy (e.g. from a
	// previous package in the same test run), reuse it.
	if reuseIfHealthy(ctx) {
		return nil
	}

	// Remove any leftover stopped container with the same name.
	exec.Command("docker", "rm", "-f", esName).Run()

	runCmd := exec.CommandContext(ctx, "docker", "run",
		"-d",
		"--name", esName,
		"-p", esHostPort+":"+esHostPort,
		"-e", "discovery.type=single-node",
		"-e", "xpack.security.enabled=false",
		"-e", "ES_JAVA_OPTS=-Xms512m -Xmx512m",
		esImage,
	)
	out, err := runCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("docker run %s: %w\n%s", esImage, err, string(out))
	}

	if err := waitForES(ctx, esStartupWait); err != nil {
		exec.Command("docker", "rm", "-f", esName).Run()
		return fmt.Errorf("elasticsearch not ready at %s: %w", esURL, err)
	}

	return nil
}

func reuseIfHealthy(ctx context.Context) bool {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(esURL)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode < 500
}

func waitForES(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 3 * time.Second}
	for time.Now().Before(deadline) {
		resp, err := client.Get(esURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode < 500 {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(1 * time.Second):
		}
	}
	return fmt.Errorf("timed out after %v", timeout)
}
