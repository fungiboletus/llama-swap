// Package sqlite implements store.Store on top of embedded SQLite databases.
// The schemas are managed by goose migrations embedded in migrations/ (the
// activity database) and capture_migrations/ (the captures database) and
// applied when a store is opened.
package sqlite

import (
	"context"
	"database/sql"
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"strings"

	"github.com/pressly/goose/v3"
	_ "modernc.org/sqlite"

	"github.com/mostlygeek/llama-swap/internal/store"
)

//go:embed migrations/*.sql
var migrationFS embed.FS

//go:embed capture_migrations/*.sql
var captureMigrationFS embed.FS

// Store is a store.Store backed by two SQLite databases: one for activity and
// cache rows, and a separate one for request/response captures, so large
// capture blobs never share a file with the activity log.
type Store struct {
	db         *sql.DB
	capturesDB *sql.DB
	inMemory   bool
	activity   *activityRepository
	cache      *cacheRepository
	captures   *captureRepository
	memory     *memoryProfileRepository
}

var _ store.Store = (*Store)(nil)

// New opens a SQLite store and applies pending migrations. path is the
// activity database and capturesPath the captures database; an empty path
// keeps that database in memory, lost on Close.
//
// The two databases are independent: activity rows name their capture by
// capture ID, and a missing capture is not an error.
func New(path, capturesPath string) (*Store, error) {
	ctx := context.Background()

	db, diskFile, err := openDB(ctx, path, migrationFS, "migrations")
	if err != nil {
		return nil, err
	}
	capturesDB, _, err := openDB(ctx, capturesPath, captureMigrationFS, "capture_migrations")
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("captures: %w", err)
	}

	fail := func(err error) (*Store, error) {
		capturesDB.Close()
		db.Close()
		return nil, err
	}

	captures, err := newCaptureRepository(ctx, capturesDB)
	if err != nil {
		return fail(err)
	}
	s := &Store{
		db:         db,
		capturesDB: capturesDB,
		inMemory:   !diskFile,
		activity:   &activityRepository{db: db},
		cache:      &cacheRepository{db: db},
		captures:   captures,
		memory:     &memoryProfileRepository{db: db},
	}
	// Drop cache rows that expired while the process was not running, so a
	// long-lived database does not accumulate them.
	if err := s.cache.Prune(ctx); err != nil {
		return fail(err)
	}
	return s, nil
}

// openDB opens the SQLite database at path, or an in-memory one when path is
// empty, and applies the migrations in dir of migrations. diskFile reports
// whether the database is on disk.
func openDB(ctx context.Context, path string, migrations fs.FS, dir string) (db *sql.DB, diskFile bool, err error) {
	dsn := strings.TrimSpace(path)
	diskFile = dsn != ""
	if dsn == "" {
		dsn = ":memory:"
	}

	db, err = sql.Open("sqlite", dsn)
	if err != nil {
		return nil, false, fmt.Errorf("open sqlite store: %w", err)
	}
	// A single connection keeps an in-memory database alive (each new
	// connection to ":memory:" would be a fresh empty database) and
	// serializes writers for on-disk files.
	db.SetMaxOpenConns(1)

	if _, err := db.ExecContext(ctx, `PRAGMA busy_timeout = 5000`); err != nil {
		db.Close()
		return nil, false, fmt.Errorf("configure sqlite store: %w", err)
	}
	if diskFile {
		var mode string
		if err := db.QueryRowContext(ctx, `PRAGMA journal_mode = WAL`).Scan(&mode); err != nil {
			db.Close()
			return nil, false, fmt.Errorf("enable sqlite store WAL mode: %w", err)
		}
		if !strings.EqualFold(mode, "wal") {
			db.Close()
			return nil, false, fmt.Errorf("enable sqlite store WAL mode: got %q", mode)
		}
	}
	if err := runMigrations(ctx, db, migrations, dir); err != nil {
		db.Close()
		return nil, false, err
	}
	return db, diskFile, nil
}

// repairLegacyForkMigrations fixes activity databases created by fork builds
// that used goose versions 4 (memory profiles) and 5 (cache) before upstream
// claimed those numbers for task/route and capture_id columns. The fork's
// migrations now live at 00100+ and are idempotent, so forgetting the stale
// 4/5 records lets upstream's real 4/5 run. Databases that already have the
// upstream columns are left alone.
func repairLegacyForkMigrations(ctx context.Context, db *sql.DB) error {
	var n int
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='goose_db_version'`).Scan(&n); err != nil {
		return fmt.Errorf("inspect migration table: %w", err)
	}
	if n == 0 {
		return nil
	}
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM pragma_table_info('activity') WHERE name='task'`).Scan(&n); err != nil {
		return fmt.Errorf("inspect activity schema: %w", err)
	}
	if n > 0 {
		return nil
	}
	if err := db.QueryRowContext(ctx, `SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='activity'`).Scan(&n); err != nil {
		return fmt.Errorf("inspect activity schema: %w", err)
	}
	if n == 0 {
		return nil
	}
	if _, err := db.ExecContext(ctx, `DELETE FROM goose_db_version WHERE version_id IN (4, 5)`); err != nil {
		return fmt.Errorf("repair legacy fork migrations: %w", err)
	}
	return nil
}

func runMigrations(ctx context.Context, db *sql.DB, migrationsFS fs.FS, dir string) error {
	migrations, err := fs.Sub(migrationsFS, dir)
	if err != nil {
		return fmt.Errorf("sqlite store migrations: %w", err)
	}
	if err := repairLegacyForkMigrations(ctx, db); err != nil {
		return err
	}
	// Fork migrations live at 00100+ so upstream numbers stay free; allow
	// out-of-order so new upstream migrations still apply beneath them.
	provider, err := goose.NewProvider(goose.DialectSQLite3, db, migrations, goose.WithAllowOutofOrder(true))
	if err != nil {
		return fmt.Errorf("sqlite store migrations: %w", err)
	}
	if _, err := provider.Up(ctx); err != nil {
		return fmt.Errorf("sqlite store migrations up: %w", err)
	}
	// Upstream version 3 and the fork's version 3 have different schemas.
	// Apply the fork's user column when opening an upstream database, while
	// preserving databases where it was already created before this refactor.
	rows, err := db.QueryContext(ctx, `PRAGMA table_info(activity)`)
	if err != nil {
		return fmt.Errorf("inspect activity schema: %w", err)
	}
	hasAuthenticatedUser, hasActivityTable := false, false
	for rows.Next() {
		hasActivityTable = true
		var cid, notNull, primaryKey int
		var name, columnType string
		var defaultValue sql.NullString
		if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &primaryKey); err != nil {
			rows.Close()
			return fmt.Errorf("inspect activity column: %w", err)
		}
		hasAuthenticatedUser = hasAuthenticatedUser || name == "authenticated_user"
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return fmt.Errorf("inspect activity columns: %w", err)
	}
	rows.Close()
	// The captures database has no activity table and needs no fix-up.
	if hasActivityTable && !hasAuthenticatedUser {
		if _, err := db.ExecContext(ctx, `ALTER TABLE activity ADD COLUMN authenticated_user TEXT NOT NULL DEFAULT ''`); err != nil {
			return fmt.Errorf("add authenticated user column: %w", err)
		}
	}
	return nil
}

// Activity implements store.Store.
func (s *Store) Activity() store.ActivityRepository {
	return s.activity
}

// Cache implements store.Store.
func (s *Store) Cache() store.CacheRepository {
	return s.cache
}

// MemoryProfiles implements store.Store.
func (s *Store) MemoryProfiles() store.MemoryProfileRepository {
	return s.memory
}

// Captures implements store.Store.
func (s *Store) Captures() store.CaptureRepository {
	return s.captures
}

// IsInMemory implements store.Store.
func (s *Store) IsInMemory() bool {
	return s.inMemory
}

// Close implements store.Store.
func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return errors.Join(s.db.Close(), s.capturesDB.Close())
}
