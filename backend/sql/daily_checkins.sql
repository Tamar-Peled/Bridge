-- Daily student check-ins (emoji + optional text + optional voice)
CREATE TABLE IF NOT EXISTS public.daily_checkins (
  id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  student_id    uuid        NOT NULL REFERENCES public.students(id) ON DELETE CASCADE,
  checkin_date  date        NOT NULL,
  day_of_week   text,
  week_start    date,
  week_number   integer,
  mood          text,
  text          text,
  audio_url     text,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (student_id, checkin_date)
);

ALTER TABLE public.daily_checkins
  ADD COLUMN IF NOT EXISTS day_of_week text,
  ADD COLUMN IF NOT EXISTS week_start    date,
  ADD COLUMN IF NOT EXISTS week_number   integer;

CREATE INDEX IF NOT EXISTS idx_daily_checkins_student_date
  ON public.daily_checkins (student_id, checkin_date DESC);

CREATE INDEX IF NOT EXISTS idx_daily_checkins_student_week
  ON public.daily_checkins (student_id, week_start DESC);

ALTER TABLE public.daily_checkins DISABLE ROW LEVEL SECURITY;
