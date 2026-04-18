-- ═══════════════════════════════════════════════════════════════════════════
-- Supabase Storage — bucket `student-files` (must match SUPABASE_STORAGE_BUCKET)
--
-- 1) Backend uploads: set SUPABASE_KEY to the service_role secret
--    (Dashboard → Project Settings → API). That JWT bypasses RLS on storage.objects,
--    so uploads work without permissive policies — this is the preferred fix.
--
-- 2) If you must use another client key, add policies below. Public SELECT lets
--    anyone read objects in this bucket when the bucket is "Public" in the UI.
--
-- Run in Supabase SQL Editor. Adjust policy names if you already created some.
-- ═══════════════════════════════════════════════════════════════════════════

-- Optional: confirm the bucket id matches exactly (case-sensitive)
-- SELECT id, public FROM storage.buckets WHERE id = 'student-files';

-- Public read (objects in this bucket)
DROP POLICY IF EXISTS "student_files_public_read" ON storage.objects;
CREATE POLICY "student_files_public_read"
  ON storage.objects
  FOR SELECT
  TO public
  USING (bucket_id = 'student-files');

-- Authenticated users: upload/update/delete within this bucket (if you use Supabase Auth clients)
DROP POLICY IF EXISTS "student_files_authenticated_insert" ON storage.objects;
CREATE POLICY "student_files_authenticated_insert"
  ON storage.objects
  FOR INSERT
  TO authenticated
  WITH CHECK (bucket_id = 'student-files');

DROP POLICY IF EXISTS "student_files_authenticated_update" ON storage.objects;
CREATE POLICY "student_files_authenticated_update"
  ON storage.objects
  FOR UPDATE
  TO authenticated
  USING (bucket_id = 'student-files')
  WITH CHECK (bucket_id = 'student-files');

DROP POLICY IF EXISTS "student_files_authenticated_delete" ON storage.objects;
CREATE POLICY "student_files_authenticated_delete"
  ON storage.objects
  FOR DELETE
  TO authenticated
  USING (bucket_id = 'student-files');

-- NOT RECOMMENDED: allow anon INSERT for debugging only (exposes bucket to abuse)
-- DROP POLICY IF EXISTS "student_files_anon_insert_debug" ON storage.objects;
-- CREATE POLICY "student_files_anon_insert_debug"
--   ON storage.objects
--   FOR INSERT
--   TO anon
--   WITH CHECK (bucket_id = 'student-files');
