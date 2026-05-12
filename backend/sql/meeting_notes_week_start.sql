-- Links a meeting note to the counselor weekly accordion (Sunday week start, epoch ms).
ALTER TABLE public.meeting_notes
  ADD COLUMN IF NOT EXISTS week_start_ms bigint;

CREATE INDEX IF NOT EXISTS idx_meeting_notes_student_week
  ON public.meeting_notes(student_id, week_start_ms)
  WHERE week_start_ms IS NOT NULL;
