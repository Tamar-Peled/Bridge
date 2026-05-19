-- ═══════════════════════════════════════════════════════════════════════════
-- NOTE: If you are running this for the first time, prefer the consolidated
-- backend/sql/fix_everything.sql which patches tasks AND meeting_notes AND
-- students/reports in one go. This file is kept for historical reference.
--
-- tasks — fix missing columns + (optional) RLS that breaks saves/deletes.
-- Run in: Supabase Dashboard → SQL Editor → New Query.
--
-- Why this file exists
-- --------------------
-- The backend writes the following columns on /tasks/{id}/select and
-- /tasks/{id}/deselect:
--      selected (bool), confidence_score (int), selected_at (timestamptz),
--      done (bool)
--
-- The original schema only had: id, student_id, text, selected, done,
-- created_at. `selected_at` was added in code but never added to the table,
-- so every "save task to weekly list" call returns an error like
-- "column \"selected_at\" of relation \"tasks\" does not exist" and the
-- task silently never persists. The counselor weekly view also reads
-- `selected_at` to bucket tasks per week.
--
-- DELETE failures: if RLS is enabled on `tasks` without a DELETE policy,
-- the backend still returns success but the row never goes away. The
-- script below disables RLS on `tasks` (matching the rest of the schema
-- per supabase_migration.sql §7) so the backend's service_role / anon
-- key can delete normally. If you prefer to keep RLS on, replace the
-- "disable RLS" block with explicit DELETE policies.
-- ═══════════════════════════════════════════════════════════════════════════

-- 1) Ensure all columns the backend writes exist on public.tasks ────────────
ALTER TABLE public.tasks
  ADD COLUMN IF NOT EXISTS selected         boolean      DEFAULT false,
  ADD COLUMN IF NOT EXISTS done             boolean      DEFAULT false,
  ADD COLUMN IF NOT EXISTS selected_at      timestamptz  DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS confidence_score integer      DEFAULT NULL
    CHECK (confidence_score IS NULL OR (confidence_score BETWEEN 1 AND 5)),
  ADD COLUMN IF NOT EXISTS created_at       timestamptz  DEFAULT now();

-- 2) Helpful indexes for the counselor weekly view + the student list ───────
CREATE INDEX IF NOT EXISTS idx_tasks_student_selected
  ON public.tasks (student_id, selected);

CREATE INDEX IF NOT EXISTS idx_tasks_selected_at
  ON public.tasks (selected_at);

-- 3) Make sure RLS does not silently swallow DELETE / UPDATE ───────────────
-- The rest of the project disables RLS on these tables and goes through the
-- FastAPI backend (which uses the service_role key). If RLS was ever
-- re-enabled on `tasks` without a delete/update policy, the API will get
-- "0 rows affected" instead of an error and the UI will look as if the
-- operation succeeded even though nothing changed.
ALTER TABLE public.tasks DISABLE ROW LEVEL SECURITY;

-- 4) Backfill: any task that already has selected = true but no
--    selected_at gets the row's created_at as a safe default so the
--    counselor weekly view can still bucket them. ────────────────────────────
UPDATE public.tasks
   SET selected_at = COALESCE(selected_at, created_at, now())
 WHERE selected = true
   AND selected_at IS NULL;

-- 5) Verify ─────────────────────────────────────────────────────────────────
SELECT column_name, data_type, is_nullable, column_default
  FROM information_schema.columns
 WHERE table_schema = 'public' AND table_name = 'tasks'
 ORDER BY ordinal_position;
