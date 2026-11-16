-- 001_renames.sql
-- Renames from the "Database Changes" sheet. Safe to run on a fresh database:
-- every statement is a no-op if the old table/column doesn't exist.
--
-- !! Agree this with Yee Mon and whoever owns the Python workers first !!
-- The recording/decoder/event-poller workers query `sources` and `anomaly_events`
-- by name, so their SQL must change in the same commit.

DO $$
BEGIN
    -- sources -> cameras
    IF to_regclass('public.sources') IS NOT NULL AND to_regclass('public.cameras') IS NULL THEN
        ALTER TABLE sources RENAME TO cameras;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_name = 'cameras' AND column_name = 'source_id') THEN
        ALTER TABLE cameras RENAME COLUMN source_id TO camera_id;
    END IF;

    -- anomaly_events -> incidents (Person B's table; column renames left to them)
    IF to_regclass('public.anomaly_events') IS NOT NULL AND to_regclass('public.incidents') IS NULL THEN
        ALTER TABLE anomaly_events RENAME TO incidents;
    END IF;
END $$;
