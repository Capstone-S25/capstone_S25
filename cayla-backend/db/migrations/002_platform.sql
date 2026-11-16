-- 002_platform.sql
-- Cayla's tables: users, sessions, locations, cameras, event_types, rules,
-- rule_cameras, settings, audit_log.
-- Written to work on a fresh DB *and* on the old schema from the deck
-- (CREATE ... IF NOT EXISTS + ADD COLUMN IF NOT EXISTS). Run after 001.

-- ---------------------------------------------------------------- users
CREATE TABLE IF NOT EXISTS users (
    user_id        BIGSERIAL PRIMARY KEY,
    name           TEXT NOT NULL,
    email          TEXT NOT NULL,
    role           TEXT NOT NULL,
    is_active      BOOLEAN NOT NULL DEFAULT TRUE
);
-- Columns added by the new design (old table had cognito_sub instead of a password).
ALTER TABLE users ADD COLUMN IF NOT EXISTS staff_id      TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS date_of_birth DATE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at    TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at    TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_by    BIGINT REFERENCES users(user_id);
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
ALTER TABLE users ADD CONSTRAINT users_role_check CHECK (role IN ('admin', 'staff'));
CREATE UNIQUE INDEX IF NOT EXISTS users_email_uq    ON users (lower(email));
CREATE UNIQUE INDEX IF NOT EXISTS users_staff_id_uq ON users (staff_id) WHERE staff_id IS NOT NULL;

-- ------------------------------------------------------------- sessions
-- Not on the "Database Changes" sheet: added so logout and deactivation
-- take effect immediately (a plain JWT stays valid until it expires).
CREATE TABLE IF NOT EXISTS sessions (
    session_id   BIGSERIAL PRIMARY KEY,
    token_hash   BYTEA NOT NULL UNIQUE,          -- sha256 of the token; raw token never stored
    user_id      BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ NOT NULL,
    revoked_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions (user_id);

-- ------------------------------------------------------------ locations
CREATE TABLE IF NOT EXISTS locations (
    location_id  BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS locations_name_uq ON locations (lower(name));

-- -------------------------------------------------------------- cameras
CREATE TABLE IF NOT EXISTS cameras (
    camera_id    BIGSERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    source_type  TEXT NOT NULL,                  -- 'usb' | 'rtsp' | 'http' (from the deck)
    device_ref   TEXT NOT NULL,                  -- device name or RTSP URL
    is_active    BOOLEAN NOT NULL DEFAULT TRUE
);
-- Every camera belongs to one location. Nullable only so the old rows can be
-- migrated; once every camera has a location, run:
--   ALTER TABLE cameras ALTER COLUMN location_id SET NOT NULL;
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS location_id BIGINT REFERENCES locations(location_id);
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS created_at  TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS updated_at  TIMESTAMPTZ NOT NULL DEFAULT now();
CREATE INDEX IF NOT EXISTS cameras_location_idx ON cameras (location_id);

-- ---------------------------------------------------------- event_types
CREATE TABLE IF NOT EXISTS event_types (
    event_type_id BIGSERIAL PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE   -- "retire" = set false, never delete
);
CREATE UNIQUE INDEX IF NOT EXISTS event_types_name_uq ON event_types (lower(name));

-- ---------------------------------------------------------------- rules
CREATE TABLE IF NOT EXISTS rules (
    rule_id         BIGSERIAL PRIMARY KEY,
    location_id     BIGINT NOT NULL REFERENCES locations(location_id),
    event_type_id   BIGINT NOT NULL REFERENCES event_types(event_type_id),
    days            SMALLINT[] NOT NULL,          -- ISO weekdays, 1 = Mon ... 7 = Sun
    start_time      TIME NOT NULL,
    end_time        TIME NOT NULL,                -- end < start means it runs past midnight;
                                                  -- start = end means all day
    severity        TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    min_confidence  NUMERIC(4,3) NOT NULL CHECK (min_confidence BETWEEN 0 AND 1),
                                                  -- no DEFAULT on purpose: "no global default"
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by      BIGINT REFERENCES users(user_id),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by      BIGINT REFERENCES users(user_id),
    CONSTRAINT rules_days_valid CHECK (
        cardinality(days) BETWEEN 1 AND 7 AND days <@ ARRAY[1,2,3,4,5,6,7]::SMALLINT[]
    )
);
CREATE INDEX IF NOT EXISTS rules_event_type_idx ON rules (event_type_id) WHERE is_active;

CREATE TABLE IF NOT EXISTS rule_cameras (
    rule_id    BIGINT NOT NULL REFERENCES rules(rule_id) ON DELETE CASCADE,
    camera_id  BIGINT NOT NULL REFERENCES cameras(camera_id) ON DELETE CASCADE,
    PRIMARY KEY (rule_id, camera_id)
);
CREATE INDEX IF NOT EXISTS rule_cameras_camera_idx ON rule_cameras (camera_id);

-- ------------------------------------------------------------- settings
-- Exactly one row (id = 1). Default values are placeholders: confirm with the team.
CREATE TABLE IF NOT EXISTS settings (
    id                       SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    t_pre_seconds            INTEGER NOT NULL DEFAULT 10  CHECK (t_pre_seconds  >= 0),
    t_post_seconds           INTEGER NOT NULL DEFAULT 10  CHECK (t_post_seconds >= 0),
    t_quiet_seconds          INTEGER NOT NULL DEFAULT 30  CHECK (t_quiet_seconds >= 0),
    segment_retention_days   INTEGER NOT NULL DEFAULT 7   CHECK (segment_retention_days  > 0),
    incident_retention_days  INTEGER NOT NULL DEFAULT 90  CHECK (incident_retention_days > 0),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by               BIGINT REFERENCES users(user_id)
);
INSERT INTO settings (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- ------------------------------------------------------------ audit_log
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id      BIGSERIAL PRIMARY KEY,
    occurred_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    actor_user_id BIGINT REFERENCES users(user_id),   -- NULL = system / worker
    action        TEXT NOT NULL,                      -- 'create' | 'update' | 'delete' | 'review' | 'settle' ...
    entity_type   TEXT NOT NULL,                      -- 'user' | 'rule' | 'settings' | 'review' ...
    entity_id     TEXT NOT NULL,
    old_values    JSONB,                              -- only the fields that changed
    new_values    JSONB
);
CREATE INDEX IF NOT EXISTS audit_entity_idx   ON audit_log (entity_type, entity_id);
CREATE INDEX IF NOT EXISTS audit_occurred_idx ON audit_log (occurred_at DESC);
CREATE INDEX IF NOT EXISTS audit_actor_idx    ON audit_log (actor_user_id);
