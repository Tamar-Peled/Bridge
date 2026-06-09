-- Daily student check-ins (emoji + optional text + optional voice)
CREATE TABLE IF NOT EXISTS public.daily_checkins (
  id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id    uuid        NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  checkin_date  date        NOT NULL,
  mood          text,
  text          text,
  audio_url     text,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (student_id, checkin_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_checkins_student_date
  ON public.daily_checkins (student_id, checkin_date DESC);

ALTER TABLE public.daily_checkins DISABLE ROW LEVEL SECURITY;
