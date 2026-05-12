-- Counselor-only JSON: custom week titles + private per-task notes (not exposed to student app).
-- Run in Supabase SQL Editor if not using full supabase_migration.sql.
ALTER TABLE public.students
  ADD COLUMN IF NOT EXISTS counselor_weekly_extras jsonb DEFAULT '{}'::jsonb;

COMMENT ON COLUMN public.students.counselor_weekly_extras IS
  'Counselor-only: { "week_labels": { "<week_start_ms>": "title" }, "task_notes": { "<week>_<task_id>": "text" } }';
