-- +goose Up
-- The fork used versions 2-4 for memory, authenticated user, and source.
-- Those databases skip upstream's version 3 cache migration, so add it here.
CREATE TABLE IF NOT EXISTS cache (
    key         TEXT PRIMARY KEY,
    data        BLOB NOT NULL,
    ttl_seconds INTEGER NOT NULL DEFAULT 0,
    ts_created  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cache_expiry
    ON cache (ts_created)
    WHERE ttl_seconds > 0;

-- +goose Down
-- Preserve the cache table; it may predate this compatibility migration.
SELECT 1;
