-- ═══════════════════════════════════════════════════════════════════════════
-- meeting_notes — counselor session summaries (run in Supabase SQL Editor)
-- Fixes 500s when POST/PATCH expect columns that are missing.
--
-- Canonical columns (requested):
--   id, student_id, summary_text, ai_insights, file_urls (text[]), created_at
-- Legacy columns kept for backward compatibility with existing app versions:
--   content, attachments (jsonb), is_ai_generated, edit_status, note_type
--
-- File URLs: the FastAPI backend fills file_urls with http(s) links found in
-- attachments[]. When you move to Supabase Storage, upload there and store
-- public URLs in attachments + they will sync into file_urls automatically.
-- ═══════════════════════════════════════════════════════════════════════════

-- 1) Create table if it does not exist (fresh projects)

CREATE TABLE IF NOT EXISTS public.meeting_notes (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id uuid NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  summary_text text,
  content text,
  ai_insights text,
  file_urls text[] DEFAULT '{}',
  attachments jsonb DEFAULT '[]'::jsonb,
  is_ai_generated boolean DEFAULT false,
  edit_status text,
  note_type text DEFAULT 'session',
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_meeting_notes_student ON public.meeting_notes(student_id);
CREATE INDEX IF NOT EXISTS idx_meeting_notes_created ON public.meeting_notes(created_at DESC);

-- 3) Backfill summary_text from legacy content
UPDATE public.meeting_notes SET summary_text = COALESCE(NULLIF(trim(summary_text),''), content)
  WHERE summary_text IS NULL OR trim(summary_text) = '';

