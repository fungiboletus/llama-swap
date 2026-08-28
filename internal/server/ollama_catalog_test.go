package server

import (
	"strings"
	"testing"
)

func TestOllamaCatalog_ParseTags(t *testing.T) {
	names, err := parseOllamaTags(strings.NewReader(`{
  "models": [
    {"name":" qwen3:8b "},
    {"name":"llama3.2:3b"}
  ]
}`))
	if err != nil {
		t.Fatalf("parse tags: %v", err)
	}
	if got, want := strings.Join(names, ","), "qwen3:8b,llama3.2:3b"; got != want {
		t.Errorf("names = %q, want %q", got, want)
	}
}

func TestOllamaCatalog_ParseTagsRejectsMalformedJSON(t *testing.T) {
	if _, err := parseOllamaTags(strings.NewReader(`{"models":`)); err == nil {
		t.Fatal("parse malformed tags succeeded")
	}
}
