-- +goose Up
CREATE TABLE memory_profiles (
    model_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    observed_bytes INTEGER NOT NULL,
    observed_at INTEGER NOT NULL,
    PRIMARY KEY (model_id, fingerprint)
);

-- +goose Down
DROP TABLE memory_profiles;
