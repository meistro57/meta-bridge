// Package llm provides a minimal OpenRouter client for batch (non-streaming)
// chat completions.
//
// This is deliberately NOT shared with kae/internal/llm. Meta Bridge and KAE
// are peer projects per ARCHITECTURE.md ("loose coupling"); if the interface
// stabilizes we can extract a shared module later.
package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const openRouterURL = "https://openrouter.ai/api/v1/chat/completions"

// Client talks to OpenRouter.
type Client struct {
	apiKey string
	http   *http.Client
}

// NewClient constructs an OpenRouter client. Timeout is 120s by default to
// accommodate reasoning models (DeepSeek R1) which can take a while.
func NewClient(apiKey string) *Client {
	return &Client{
		apiKey: apiKey,
		http:   &http.Client{Timeout: 120 * time.Second},
	}
}

// Message is one chat turn.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Request is an OpenRouter chat completion request.
type Request struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature,omitempty"`
	// ResponseFormat forces JSON output when set to {"type":"json_object"}.
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
}

// ResponseFormat constrains the model output.
type ResponseFormat struct {
	Type string `json:"type"` // "json_object"
}

// Response is the subset of the OpenRouter response we care about.
type Response struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
		Code    any    `json:"code"`
	} `json:"error,omitempty"`
}

// Complete sends a chat request and returns the assistant's message text.
func (c *Client) Complete(ctx context.Context, req Request) (string, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", openRouterURL, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	httpReq.Header.Set("HTTP-Referer", "https://github.com/meistro57/meta-bridge")
	httpReq.Header.Set("X-Title", "Meta Bridge")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("http do: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read body: %w", err)
	}

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("openrouter %d: %s", resp.StatusCode, string(raw))
	}

	var parsed Response
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", fmt.Errorf("unmarshal response: %w (body=%s)", err, string(raw))
	}
	if parsed.Error != nil {
		return "", fmt.Errorf("openrouter error: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("openrouter returned no choices (body=%s)", string(raw))
	}

	return parsed.Choices[0].Message.Content, nil
}
