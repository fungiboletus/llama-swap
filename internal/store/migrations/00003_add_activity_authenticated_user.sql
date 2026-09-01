-- +goose Up
ALTER TABLE activity ADD COLUMN authenticated_user TEXT NOT NULL DEFAULT '';

-- +goose Down
ALTER TABLE activity DROP COLUMN authenticated_user;
