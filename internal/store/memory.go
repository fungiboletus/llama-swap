package store

import (
	"context"
	"time"
)

// MemoryProfile is an OS-attributed memory observation for a model configuration.
type MemoryProfile struct {
	ModelID       string
	Fingerprint   string
	ObservedBytes uint64
	ObservedAt    time.Time
}

type MemoryProfileRepository interface {
	Load(ctx context.Context, modelID, fingerprint string) (MemoryProfile, bool, error)
	Save(ctx context.Context, profile MemoryProfile) error
}
