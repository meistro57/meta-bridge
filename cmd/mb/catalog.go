// catalog.go — source catalog loader for the Meta Bridge ingest pipeline.
//
// sources.yaml lives at the repo root (same directory you run `mb` from).
// The catalog is checked first on every ingest; LLM metadata extraction is
// only the fallback for sources not listed here.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// CatalogEntry is one row in sources.yaml.
type CatalogEntry struct {
	ID          string `yaml:"id"`
	Title       string `yaml:"title"`
	Author      string `yaml:"author"`
	Year        int    `yaml:"year"`
	ChannelType string `yaml:"channel_type"`
	Tradition   string `yaml:"tradition"`
	Medium      string `yaml:"medium"`
	File        string `yaml:"file"`  // exact filename in incoming/
	Skip        bool   `yaml:"skip"`  // if true: log and abort ingest for this file
	Notes       string `yaml:"notes"` // human notes, logged but not stored
}

// catalog is the in-memory parsed catalog.
type catalog struct {
	Sources []CatalogEntry `yaml:"sources"`
}

// loadCatalog reads sources.yaml from the given path.
// Returns an empty catalog (no error) if the file doesn't exist, so the
// pipeline degrades gracefully to LLM extraction on a fresh checkout.
func loadCatalog(path string) (*catalog, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &catalog{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read catalog %s: %w", path, err)
	}
	var c catalog
	if err := yaml.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("parse catalog %s: %w", path, err)
	}
	return &c, nil
}

// lookup finds a CatalogEntry by matching the ingest file path against the
// catalog's File field (basename, case-insensitive, extension stripped).
// Returns (entry, true) on a hit, (nil, false) on a miss.
func (c *catalog) lookup(filePath string) (*CatalogEntry, bool) {
	base := bareBasename(filePath)
	for i := range c.Sources {
		if bareBasename(c.Sources[i].File) == base {
			return &c.Sources[i], true
		}
	}
	return nil, false
}

// bareBasename returns the filename without directory or extension, lowercased.
func bareBasename(p string) string {
	b := filepath.Base(p)
	b = strings.TrimSuffix(b, filepath.Ext(b))
	return strings.ToLower(b)
}

// toSourceMeta converts a CatalogEntry into the sourceMeta shape the ingest
// pipeline already knows how to consume.
func (e *CatalogEntry) toSourceMeta() sourceMeta {
	return sourceMeta{
		Title:       e.Title,
		Author:      e.Author,
		Year:        e.Year,
		ChannelType: e.ChannelType,
		Tradition:   e.Tradition,
		Medium:      e.Medium,
	}
}

// catalogPath returns the expected location of sources.yaml relative to the
// working directory, with an optional MB_CATALOG_PATH env override.
func catalogPath() string {
	if p := os.Getenv("MB_CATALOG_PATH"); p != "" {
		return p
	}
	return "sources.yaml"
}
