-- ═══════════════════════════════════════════════════════════════════════════
-- BRIDGE — single "fix everything" SQL.
-- Run this once in: Supabase Dashboard → SQL Editor → New Query.
--
-- It is idempotent (uses IF NOT EXISTS everywhere) and brings the schema in
-- line with what backend/main.py actually writes. Running it on a project
-- that is already up-to-date is a no-op.
-- ═══════════════════════════════════════════════════════════════════════════

-- 0) RLS: backend writes must not be silently ignored ───────────────────────
-- The backend now uses SUPABASE_SERVICE_KEY first. If a deployment is still
-- configured with the anon key, RLS can make PostgREST return HTTP 200 with
-- data=[] for UPDATE/DELETE. These DISABLE statements make the app work even
-- before Render is reconfigured. If you prefer policies, re-enable RLS and add
-- explicit INSERT/UPDATE/DELETE policies for the role your backend uses.
ALTER TABLE IF EXISTS public.students DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.tasks DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.meeting_notes DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.reports DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.student_documents DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.student_notes DISABLE ROW LEVEL SECURITY;


-- 1) tasks: full counselor weekly flow + delete/update RLS off ──────────────
-- Without `selected_at` the PATCH /tasks/{id}/select call errors out and
-- nothing is persisted. The counselor weekly view also reads selected_at
-- to bucket tasks into the right week.
ALTER TABLE public.tasks
  ADD COLUMN IF NOT EXISTS selected         boolean      DEFAULT false,
  ADD COLUMN IF NOT EXISTS done             boolean      DEFAULT false,
  ADD COLUMN IF NOT EXISTS selected_at      timestamptz  DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS confidence_score integer      DEFAULT NULL
    CHECK (confidence_score IS NULL OR (confidence_score BETWEEN 1 AND 5)),
  ADD COLUMN IF NOT EXISTS created_at       timestamptz  DEFAULT now();

ALTER TABLE public.tasks DISABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS idx_tasks_student_selected
  ON public.tasks (student_id, selected);

CREATE INDEX IF NOT EXISTS idx_tasks_selected_at
  ON public.tasks (selected_at);

-- Backfill: rows that already have selected=true but no selected_at get a
-- safe default so the weekly accordion can still place them.
UPDATE public.tasks
   SET selected_at = COALESCE(selected_at, created_at, now())
 WHERE selected = true AND selected_at IS NULL;


-- 2) meeting_notes: link to counselor weekly accordion + counselor metadata ─
-- Without `week_start_ms` every save of a session note tied to a specific
-- week fails with PGRST204 ("column not in schema cache") and shows up as
-- "Internal Server Error" in the UI.
ALTER TABLE public.meeting_notes
  ADD COLUMN IF NOT EXISTS week_start_ms   bigint,
  ADD COLUMN IF NOT EXISTS note_type       text   DEFAULT 'session',
  ADD COLUMN IF NOT EXISTS attachments     jsonb  DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS is_ai_generated boolean DEFAULT false,
  ADD COLUMN IF NOT EXISTS edit_status     text,
  ADD COLUMN IF NOT EXISTS ai_insights     text,
  ADD COLUMN IF NOT EXISTS file_urls       text[] DEFAULT '{}';

ALTER TABLE public.meeting_notes DISABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS idx_meeting_notes_student_week
  ON public.meeting_notes (student_id, week_start_ms)
  WHERE week_start_ms IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_meeting_notes_student
  ON public.meeting_notes (student_id);

CREATE INDEX IF NOT EXISTS idx_meeting_notes_created
  ON public.meeting_notes (created_at DESC);


-- 3) students: counselor weekly summary text + week labels / per-task notes ─
ALTER TABLE public.students
  ADD COLUMN IF NOT EXISTS description                text   DEFAULT '',
  ADD COLUMN IF NOT EXISTS photo                      text   DEFAULT '',
  ADD COLUMN IF NOT EXISTS general_files              jsonb  DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS key_points                 jsonb  DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS weekly_counselor_summaries jsonb  DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS counselor_weekly_extras    jsonb  DEFAULT '{}'::jsonb;

ALTER TABLE public.students DISABLE ROW LEVEL SECURITY;


-- 4) reports: task FK + audio + confidence + indexes ────────────────────────
ALTER TABLE public.reports
  ADD COLUMN IF NOT EXISTS task_id          uuid    REFERENCES public.tasks(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS audio_url        text    DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS confidence_score integer DEFAULT NULL
    CHECK (confidence_score IS NULL OR (confidence_score BETWEEN 1 AND 5));

ALTER TABLE public.reports DISABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS idx_reports_task_id
  ON public.reports (task_id);

CREATE INDEX IF NOT EXISTS idx_reports_student_id
  ON public.reports (student_id);


-- 5) Verify — list of columns the backend touches; eyeball that nothing is
--    NULL/missing in the output. ───────────────────────────────────────────
SELECT table_name, column_name, data_type
  FROM information_schema.columns
 WHERE table_schema = 'public'
   AND table_name IN ('tasks', 'meeting_notes', 'students', 'reports')
 ORDER BY table_name, ordinal_position;
