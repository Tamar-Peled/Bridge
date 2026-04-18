-- Key points (נקודות חשובות) — counselor-facing structured bullets per student
-- Run in Supabase SQL Editor after meeting_notes_schema.sql

ALTER TABLE public.students
  ADD COLUMN IF NOT EXISTS key_points jsonb DEFAULT '[]'::jsonb;

COMMENT ON COLUMN public.students.key_points IS 'Array of { "text": string, "at": ISO8601 string } — counselor key points';
