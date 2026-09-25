package sqlite

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/mostlygeek/llama-swap/internal/store"
)

type memoryProfileRepository struct{ db *sql.DB }

var _ store.MemoryProfileRepository = (*memoryProfileRepository)(nil)

func (r *memoryProfileRepository) Load(ctx context.Context, modelID, fingerprint string) (store.MemoryProfile, bool, error) {
	var observedBytes, observedAt int64
	err := r.db.QueryRowContext(ctx, `
		SELECT observed_bytes, observed_at FROM memory_profiles
		WHERE model_id = ? AND fingerprint = ?`, modelID, fingerprint).Scan(&observedBytes, &observedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return store.MemoryProfile{}, false, nil
	}
	if err != nil {
		return store.MemoryProfile{}, false, fmt.Errorf("load memory profile: %w", err)
	}
	if observedBytes < 0 {
		return store.MemoryProfile{}, false, fmt.Errorf("load memory profile: invalid negative byte count")
	}
	return store.MemoryProfile{ModelID: modelID, Fingerprint: fingerprint, ObservedBytes: uint64(observedBytes), ObservedAt: time.Unix(observedAt, 0)}, true, nil
}

func (r *memoryProfileRepository) Save(ctx context.Context, profile store.MemoryProfile) error {
	if profile.ModelID == "" || profile.Fingerprint == "" {
		return errors.New("memory profile model ID and fingerprint are required")
	}
	if profile.ObservedBytes > math.MaxInt64 {
		return errors.New("memory profile byte count exceeds SQLite integer range")
	}
	if profile.ObservedAt.IsZero() {
		profile.ObservedAt = time.Now()
	}
	_, err := r.db.ExecContext(ctx, `
		INSERT INTO memory_profiles (model_id, fingerprint, observed_bytes, observed_at)
		VALUES (?, ?, ?, ?)
		ON CONFLICT(model_id, fingerprint) DO UPDATE SET
			observed_bytes = excluded.observed_bytes,
			observed_at = excluded.observed_at`,
		profile.ModelID, profile.Fingerprint, int64(profile.ObservedBytes), profile.ObservedAt.Unix())
	if err != nil {
		return fmt.Errorf("save memory profile: %w", err)
	}
	return nil
}
