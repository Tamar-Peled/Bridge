-- ═══════════════════════════════════════════════════════════════════════════
-- student_documents — counselor "מסמכים" (general files) with stable IDs
-- Run in Supabase SQL Editor after students table exists.
-- The API syncs each row into students.general_files for backward compatibility.
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS public.student_documents (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id uuid NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  file_name text NOT NULL,
  file_url text NOT NULL,
  mime text,
  created_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_student_documents_student ON public.student_documents(student_id);
CREATE INDEX IF NOT EXISTS idx_student_documents_created ON public.student_documents(created_at DESC);

COMMENT ON TABLE public.student_documents IS 'General counselor documents per student; uploaded to Storage, URL stored here';
