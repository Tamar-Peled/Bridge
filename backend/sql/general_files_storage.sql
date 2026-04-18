-- ═══════════════════════════════════════════════════════════════════════════
-- General documents (מסמכים) — column + Storage checklist
-- The API stores file metadata in public.students.general_files (jsonb array).
-- Each element may be { "name", "mime", "url" } after upload to Storage, or
-- legacy { "name", "mime", "data" } (base64/data URL) for older rows.
--
-- There is NO separate student_files table required by the current backend.
-- ═══════════════════════════════════════════════════════════════════════════

ALTER TABLE public.students
  ADD COLUMN IF NOT EXISTS general_files jsonb DEFAULT '[]'::jsonb;

COMMENT ON COLUMN public.students.general_files IS 'Counselor file cabinet: array of {name, mime, url} or {name, mime, data}';

-- meeting_notes: expected by POST/PATCH /meeting-notes (run meeting_notes_schema.sql if missing)
-- student_documents: preferred for מסמכים uploads (run student_documents_schema.sql)

-- ─── Supabase Storage (Dashboard) ─────────────────────────────────────────
-- 1. Create bucket: student-files (or set env SUPABASE_STORAGE_BUCKET to your name).
-- 2. For public read links: set bucket to Public, or use a signed-URL policy.
-- 3. INSERT policy for authenticated/service role uploads as required by your setup.
-- The FastAPI server uses SUPABASE_KEY (service role) to upload; ensure the key
-- has storage.objects insert/update permissions on that bucket.
