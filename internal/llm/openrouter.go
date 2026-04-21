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
	"strings"
	"time"
)

const openRouterURL = "https://openrouter.ai/api/v1/chat/completions"
const defaultOllamaURL = "http://localhost:11434"
const defaultEmbeddingModel = "nomic-embed-text:latest"

// Client talks to OpenRouter.
type Client struct {
	apiKey    string
	ollamaURL string
	http      *http.Client
}

// NewClient constructs an OpenRouter client. Timeout is 120s by default to
// accommodate reasoning models (DeepSeek R1) which can take a while.
func NewClient(apiKey string) *Client {
	return &Client{
		apiKey:    apiKey,
		ollamaURL: defaultOllamaURL,
		http:      &http.Client{Timeout: 120 * time.Second},
	}
}

func (c *Client) SetOllamaURL(url string) {
	if strings.TrimSpace(url) == "" {
		return
	}
	c.ollamaURL = strings.TrimRight(url, "/")
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
	if strings.HasPrefix(req.Model, "ollama:") {
		return c.completeOllama(ctx, req)
	}
	return c.completeOpenRouter(ctx, req)
}

func (c *Client) completeOpenRouter(ctx context.Context, req Request) (string, error) {
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

type ollamaChatRequest struct {
	Model       string          `json:"model"`
	Messages    []Message       `json:"messages"`
	Stream      bool            `json:"stream"`
	Temperature float64         `json:"temperature,omitempty"`
	Format      json.RawMessage `json:"format,omitempty"`
}

type ollamaChatResponse struct {
	Message Message `json:"message"`
	Error   string  `json:"error,omitempty"`
}

func (c *Client) completeOllama(ctx context.Context, req Request) (string, error) {
	model := strings.TrimPrefix(req.Model, "ollama:")
	if strings.TrimSpace(model) == "" {
		return "", fmt.Errorf("ollama model cannot be empty (use ollama:<model>)")
	}

	chatReq := ollamaChatRequest{
		Model:       model,
		Messages:    req.Messages,
		Stream:      false,
		Temperature: req.Temperature,
	}
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
		chatReq.Format = json.RawMessage(`"json"`)
	}

	body, err := json.Marshal(chatReq)
	if err != nil {
		return "", fmt.Errorf("marshal ollama request: %w", err)
	}

	url := strings.TrimRight(c.ollamaURL, "/") + "/api/chat"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("new ollama request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("ollama http do: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read ollama body: %w", err)
	}
	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("ollama %d: %s", resp.StatusCode, string(raw))
	}

	var parsed ollamaChatResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", fmt.Errorf("unmarshal ollama response: %w (body=%s)", err, string(raw))
	}
	if strings.TrimSpace(parsed.Error) != "" {
		return "", fmt.Errorf("ollama error: %s", parsed.Error)
	}
	if strings.TrimSpace(parsed.Message.Content) == "" {
		return "", fmt.Errorf("ollama returned empty message (body=%s)", string(raw))
	}

	return parsed.Message.Content, nil
}

type embeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type embeddingResponse struct {
	Embedding []float64 `json:"embedding"`
}

func (c *Client) Embed(ctx context.Context, text string) ([]float64, error) {
	reqBody := embeddingRequest{
		Model:  defaultEmbeddingModel,
		Prompt: text,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal embedding request: %w", err)
	}

	url := strings.TrimRight(c.ollamaURL, "/") + "/api/embeddings"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("new embedding request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("embedding http do: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read embedding body: %w", err)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("ollama %d: %s", resp.StatusCode, string(raw))
	}

	var parsed embeddingResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("unmarshal embedding response: %w (body=%s)", err, string(raw))
	}
	if len(parsed.Embedding) == 0 {
		return nil, fmt.Errorf("ollama returned empty embedding")
	}

	return parsed.Embedding, nil
}
