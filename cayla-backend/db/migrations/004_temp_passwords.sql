-- 004_temp_passwords.sql
-- Staff invites: the admin adds a staff member, the system emails a temporary
-- password, and the staff member must choose their own at first login.
-- Safe to run more than once.
ALTER TABLE users ADD COLUMN IF NOT EXISTS must_change_password    BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS temp_password_expires_at TIMESTAMPTZ;
