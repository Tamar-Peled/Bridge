-- ============================================================
-- BRIDGE – Supabase Migration
-- Run this in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- ── 1. students: add missing columns ────────────────────────
ALTER TABLE public.students
  ADD COLUMN IF NOT EXISTS description  text          DEFAULT '',
  ADD COLUMN IF NOT EXISTS photo        text          DEFAULT '';   -- base64 or storage URL

-- ── 2. reports: add task_id FK + audio + confidence ─────────
ALTER TABLE public.reports
  ADD COLUMN IF NOT EXISTS task_id          uuid          REFERENCES public.tasks(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS audio_url        text          DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS confidence_score integer       DEFAULT NULL
    CHECK (confidence_score IS NULL OR (confidence_score >= 1 AND confidence_score <= 5));

-- ── 3. tasks: add pre-task confidence question ──────────────
ALTER TABLE public.tasks
  ADD COLUMN IF NOT EXISTS confidence_score integer DEFAULT NULL
    CHECK (confidence_score IS NULL OR (confidence_score >= 1 AND confidence_score <= 5));

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

-- ── 8. Verify columns exist ─────────────────────────────────
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name IN ('students','tasks','reports','logs')
  AND table_schema = 'public'
ORDER BY table_name, column_name;
