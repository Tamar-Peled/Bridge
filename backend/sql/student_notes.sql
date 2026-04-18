-- ═══════════════════════════════════════════════════════════════════════════
-- student_notes — private notes for students (4-digit code identity)
-- Run this entire script in the Supabase SQL Editor (once).
--
-- Security model:
--   • The BRIDGE FastAPI backend uses SUPABASE_KEY (service_role). It bypasses RLS.
--   • Students do NOT use Supabase Auth; they call only your API. The backend
--     validates the code against public.students before read/write.
--   • RLS is enabled with NO policies so anon/authenticated roles cannot read or
--     write this table via PostgREST — only the service role (backend) can.
--   • Do NOT expose the service_role key in frontend or mobile apps.
-- ═══════════════════════════════════════════════════════════════════════════

create table if not exists public.student_notes (
  id uuid primary key default gen_random_uuid(),
  student_code text not null,
  content text not null,
  created_at timestamptz not null default now()
);

create index if not exists idx_student_notes_student_code_created_at
  on public.student_notes (student_code, created_at desc);

comment on table public.student_notes is 'Private notes per student login code; accessed via BRIDGE API only.';

alter table public.student_notes enable row level security;

-- Intentionally no policies: RLS blocks direct client access; service_role bypasses RLS.
