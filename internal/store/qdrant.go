package store

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"path"
	"strings"
	"time"

	"github.com/meistro57/meta-bridge/internal/chunker"
	"github.com/meistro57/meta-bridge/internal/claim"
	"github.com/meistro57/meta-bridge/internal/source"
)

const (
	CollectionClaims  = "mb_claims"
	CollectionChunks  = "mb_chunks"
	CollectionSources = "mb_sources"
)

type Client struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

type vectorsConfig struct {
	Size     int    `json:"size"`
	Distance string `json:"distance"`
}

type collectionRequest struct {
	Vectors vectorsConfig `json:"vectors"`
}

type Point struct {
	ID      uint64                 `json:"id"`
	Vector  []float64              `json:"vector"`
	Payload map[string]interface{} `json:"payload"`
}

type upsertRequest struct {
	Points []Point `json:"points"`
}

func NewClient(baseURL, apiKey string) *Client {
	url := strings.TrimSpace(baseURL)
	if url == "" {
		url = "http://localhost:6333"
	}
	url = strings.TrimRight(url, "/")

	return &Client{
		baseURL: url,
		apiKey:  apiKey,
		http:    &http.Client{Timeout: 30 * time.Second},
	}
}

func (c *Client) EnsureCollections(ctx context.Context, vectorSize int) error {
	if vectorSize <= 0 {
		return fmt.Errorf("invalid vector size: %d", vectorSize)
	}
	for _, name := range []string{CollectionSources, CollectionChunks, CollectionClaims} {
		if err := c.ensureCollection(ctx, name, vectorSize); err != nil {
			return err
		}
	}
	return nil
}

func (c *Client) UpsertSource(ctx context.Context, src source.Source, vector []float64) error {
	payload, err := toPayload(src)
	if err != nil {
		return err
	}
	payload["entity_type"] = "source"
	return c.upsertPoint(ctx, CollectionSources, Point{
		ID:      pointIDForKey("source:" + src.ID),
		Vector:  vector,
		Payload: payload,
	})
}

func (c *Client) UpsertChunk(ctx context.Context, sourceID string, ch chunker.Chunk, vector []float64) error {
	payload, err := toPayload(ch)
	if err != nil {
		return err
	}
	payload["entity_type"] = "chunk"
	payload["source_id"] = sourceID

	return c.upsertPoint(ctx, CollectionChunks, Point{
		ID:      pointIDForKey(chunkPointKey(sourceID, ch.Index)),
		Vector:  vector,
		Payload: payload,
	})
}

func (c *Client) UpsertClaim(ctx context.Context, cl claim.Claim, vector []float64) error {
	payload, err := toPayload(cl)
	if err != nil {
		return err
	}
	return c.UpsertClaimPayload(ctx, cl.ID, payload, vector)
}

func (c *Client) UpsertClaimPayload(ctx context.Context, claimID string, payload map[string]interface{}, vector []float64) error {
	if payload == nil {
		payload = map[string]interface{}{}
	}
	payload["entity_type"] = "claim"
	return c.upsertPoint(ctx, CollectionClaims, Point{
		ID:      pointIDForKey("claim:" + claimID),
		Vector:  vector,
		Payload: payload,
	})
}

func (c *Client) ensureCollection(ctx context.Context, name string, vectorSize int) error {
	_, status, err := c.request(ctx, http.MethodGet, "/collections/"+name, nil)
	if err != nil {
		return fmt.Errorf("check collection %s: %w", name, err)
	}
	if status == http.StatusOK {
		return nil
	}
	if status != http.StatusNotFound {
		return fmt.Errorf("check collection %s: unexpected status %d", name, status)
	}

	body := collectionRequest{Vectors: vectorsConfig{Size: vectorSize, Distance: "Cosine"}}
	_, status, err = c.request(ctx, http.MethodPut, "/collections/"+name, body)
	if err != nil {
		return fmt.Errorf("create collection %s: %w", name, err)
	}
	if status >= 400 {
		return fmt.Errorf("create collection %s: unexpected status %d", name, status)
	}
	return nil
}

func (c *Client) upsertPoint(ctx context.Context, collection string, pt Point) error {
	if len(pt.Vector) == 0 {
		return fmt.Errorf("vector is empty for point %d", pt.ID)
	}
	body := upsertRequest{Points: []Point{pt}}
	_, status, err := c.request(ctx, http.MethodPut, "/collections/"+collection+"/points?wait=true", body)
	if err != nil {
		return fmt.Errorf("upsert point %d in %s: %w", pt.ID, collection, err)
	}
	if status >= 400 {
		return fmt.Errorf("upsert point %d in %s: unexpected status %d", pt.ID, collection, status)
	}
	return nil
}

func (c *Client) request(ctx context.Context, method, endpoint string, body any) ([]byte, int, error) {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return nil, 0, fmt.Errorf("marshal request body: %w", err)
		}
		r = bytes.NewReader(b)
	}

	u := c.baseURL + endpoint
	httpReq, err := http.NewRequestWithContext(ctx, method, u, r)
	if err != nil {
		return nil, 0, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("api-key", c.apiKey)
	}

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, 0, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode >= 400 && resp.StatusCode != http.StatusNotFound {
		return raw, resp.StatusCode, fmt.Errorf("qdrant %s %s failed: %d %s", method, path.Clean(endpoint), resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	return raw, resp.StatusCode, nil
}

func chunkPointKey(sourceID string, idx int) string {
	return fmt.Sprintf("%s_chunk_%04d", sourceID, idx)
}

func pointIDForKey(key string) uint64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(key))
	return h.Sum64()
}

func toPayload(v any) (map[string]any, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}
	var out map[string]any
	if err := json.Unmarshal(b, &out); err != nil {
		return nil, fmt.Errorf("unmarshal payload: %w", err)
	}
	return out, nil
}
