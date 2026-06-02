//go:build integration

// Package testutil provides shared test helpers for integration tests.
package testutil

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/wait"
)

const (
	// esImage matches the go-elasticsearch v8.19.x client version.
	esImage = "docker.io/elasticsearch:8.19.6"
	esPort  = "9200/tcp"
)

var (
	esOnce     sync.Once
	esURL      string
	esStartErr error
)

// GetESURL returns the URL of an ephemeral Elasticsearch container.
// The container is started once per test process (sync.Once), so
// multiple packages calling this during the same go test run share
// a single container.
//
// Callers should skip the test if the returned error is non-nil
// (e.g., Docker unavailable in CI).
func GetESURL(ctx context.Context) (string, error) {
	esOnce.Do(func() {
		esURL, esStartErr = startESContainer(ctx)
	})
	return esURL, esStartErr
}

func startESContainer(ctx context.Context) (string, error) {
	req := testcontainers.ContainerRequest{
		Image: esImage,
		Env: map[string]string{
			"discovery.type":         "single-node",
			"xpack.security.enabled": "false",
			"ES_JAVA_OPTS":           "-Xms512m -Xmx512m",
		},
		ExposedPorts: []string{esPort},
		WaitingFor: wait.ForHTTP("/").
			WithPort(esPort).
			WithStartupTimeout(120*time.Second).
			WithStatusCodeMatcher(func(status int) bool {
				return status < 500
			}),
	}

	container, err := testcontainers.GenericContainer(ctx,
		testcontainers.GenericContainerRequest{
			ContainerRequest: req,
			Started:          true,
		})
	if err != nil {
		return "", fmt.Errorf("start elasticsearch container: %w", err)
	}

	host, err := container.Host(ctx)
	if err != nil {
		return "", fmt.Errorf("get es container host: %w", err)
	}

	mappedPort, err := container.MappedPort(ctx, "9200")
	if err != nil {
		return "", fmt.Errorf("get es container mapped port: %w", err)
	}

	return fmt.Sprintf("http://%s:%s", host, mappedPort.Port()), nil
}
