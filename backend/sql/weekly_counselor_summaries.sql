-- Counselor weekly meeting summary text (per calendar week), keyed in JSON by week start (ms as string).
-- Run in Supabase SQL Editor if PATCH fails with unknown column.

ALTER TABLE public.students
  ADD COLUMN IF NOT EXISTS weekly_counselor_summaries jsonb DEFAULT '{}'::jsonb;
