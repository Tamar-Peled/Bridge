-- ============================================================
-- BRIDGE – Supabase Migration
-- Run this in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- RLS: backend writes must not be silently ignored. The backend should use
-- SUPABASE_SERVICE_KEY, but disabling RLS here also protects older deployments
-- that still have SUPABASE_KEY set to the anon key.
ALTER TABLE IF EXISTS public.students DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.tasks DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.meeting_notes DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.reports DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.student_documents DISABLE ROW LEVEL SECURITY;
ALTER TABLE IF EXISTS public.student_notes DISABLE ROW LEVEL SECURITY;

-- ── 1. students: add missing columns ────────────────────────
ALTER TABLE public.students
  ADD COLUMN IF NOT EXISTS description  text          DEFAULT '',
  ADD COLUMN IF NOT EXISTS photo        text          DEFAULT '',   -- public Storage URL (legacy rows may still hold base64 until migrated)
  ADD COLUMN IF NOT EXISTS general_files jsonb       DEFAULT '[]'::jsonb;  -- counselor file cabinet (not shown to student app)

-- ── 2. reports: add task_id FK + audio + confidence ─────────
ALTER TABLE public.reports
  ADD COLUMN IF NOT EXISTS task_id          uuid          REFERENCES public.tasks(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS audio_url        text          DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS confidence_score integer       DEFAULT NULL
    CHECK (confidence_score IS NULL OR (confidence_score >= 1 AND confidence_score <= 5));

-- ── 3. tasks: full counselor weekly-flow schema ─────────────
-- The backend writes these on /tasks/{id}/select and /tasks/{id}/deselect.
-- Missing any of them causes saves/deletes to silently fail at the DB layer.
ALTER TABLE public.tasks
  ADD COLUMN IF NOT EXISTS selected         boolean      DEFAULT false,
  ADD COLUMN IF NOT EXISTS done             boolean      DEFAULT false,
  ADD COLUMN IF NOT EXISTS selected_at      timestamptz  DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS confidence_score integer      DEFAULT NULL
    CHECK (confidence_score IS NULL OR (confidence_score BETWEEN 1 AND 5)),
  ADD COLUMN IF NOT EXISTS created_at       timestamptz  DEFAULT now();

-- RLS off so the backend (service_role / anon) can UPDATE & DELETE.
-- If you re-enable RLS, add explicit UPDATE + DELETE policies for tasks.
ALTER TABLE public.tasks DISABLE ROW LEVEL SECURITY;

CREATE INDEX IF NOT EXISTS idx_tasks_student_selected
  ON public.tasks (student_id, selected);

CREATE INDEX IF NOT EXISTS idx_tasks_selected_at
  ON public.tasks (selected_at);

-- ── 4. Useful index: reports by task_id ─────────────────────
CREATE INDEX IF NOT EXISTS idx_reports_task_id
  ON public.reports(task_id);

CREATE INDEX IF NOT EXISTS idx_reports_student_id
  ON public.reports(student_id);

CREATE INDEX IF NOT EXISTS idx_tasks_student_id
  ON public.tasks(student_id);

-- ── 5. Storage bucket for audio recordings ──────────────────
-- Run this from Dashboard → Storage → New Bucket, OR via SQL:
INSERT INTO storage.buckets (id, name, public)
  VALUES ('audio', 'audio', true)
  ON CONFLICT (id) DO NOTHING;

-- ── 6. Storage policies for audio bucket ────────────────────
-- Allow all reads (public bucket)
CREATE POLICY "Public audio read" ON storage.objects
  FOR SELECT USING (bucket_id = 'audio');

-- Allow authenticated inserts (or use service_role from backend)
CREATE POLICY "Allow audio upload" ON storage.objects
  FOR INSERT WITH CHECK (bucket_id = 'audio');

-- ── 7. RLS – DISABLE for now (since frontend hits API directly)
-- Your tables are already RLS disabled (seen in screenshots).
-- The backend uses service_role key → bypasses RLS anyway.
-- If you ever enable RLS, add policies here.

-- ── 8. meeting_notes: session vs insight + attachments ─────
ALTER TABLE public.meeting_notes
  ADD COLUMN IF NOT EXISTS note_type   text DEFAULT 'session',
  ADD COLUMN IF NOT EXISTS attachments jsonb DEFAULT '[]'::jsonb;

-- ── 8b. students: counselor weekly meeting summary map ─────
ALTER TABLE public.students
  ADD COLUMN IF NOT EXISTS weekly_counselor_summaries jsonb DEFAULT '{}'::jsonb;

-- ── 8b2. students: counselor-only weekly labels + private task notes (jsonb) ──
ALTER TABLE public.students
  ADD COLUMN IF NOT EXISTS counselor_weekly_extras jsonb DEFAULT '{}'::jsonb;

-- ── 8c. meeting_notes: link session note to counselor week (epoch ms) ──
ALTER TABLE public.meeting_notes
  ADD COLUMN IF NOT EXISTS week_start_ms bigint;

-- ── 9. Verify columns exist ─────────────────────────────────
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name IN ('students','tasks','reports','logs','meeting_notes')
  AND table_schema = 'public'
ORDER BY table_name, column_name;
