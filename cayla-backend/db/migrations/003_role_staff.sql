-- 003_role_staff.sql
-- Renames the 'member' role to 'staff'. Safe to run more than once.
-- Only needed on a database created before this change; fresh databases get
-- 'staff' straight from 002.
BEGIN;
ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check;
UPDATE users SET role = 'staff' WHERE role = 'member';
ALTER TABLE users ADD CONSTRAINT users_role_check CHECK (role IN ('admin', 'staff'));
COMMIT;
