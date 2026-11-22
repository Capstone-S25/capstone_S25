-- 005_password_resets.sql
-- Self-service "Forgot password?": one-time emailed links. Only the SHA-256 of
-- each link's token is stored. Same table as the incidents database's Alembic
-- migration f4c1a7d92e36. Safe to run more than once.
CREATE TABLE IF NOT EXISTS password_resets (
    reset_id    BIGSERIAL PRIMARY KEY,
    token_hash  BYTEA NOT NULL UNIQUE,
    user_id     BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL,
    used_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS password_resets_user_idx ON password_resets (user_id);
