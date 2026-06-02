-- ═══════════════════════════════════════════════════════════════════════════
-- Supabase Storage — bucket `student-photos` (profile avatars)
--
-- Backend uploads with SUPABASE_KEY = service_role (recommended).
-- Bucket must be Public so <img src="https://...supabase.co/storage/..."> works.
--
-- Run once in Supabase Dashboard → SQL Editor.
-- Env (optional): SUPABASE_PHOTOS_BUCKET=student-photos
-- ═══════════════════════════════════════════════════════════════════════════

INSERT INTO storage.buckets (id, name, public)
VALUES ('student-photos', 'student-photos', true)
ON CONFLICT (id) DO UPDATE SET public = true;

DROP POLICY IF EXISTS "student_photos_public_read" ON storage.objects;
CREATE POLICY "student_photos_public_read"
  ON storage.objects
  FOR SELECT
  TO public
  USING (bucket_id = 'student-photos');

DROP POLICY IF EXISTS "student_photos_authenticated_insert" ON storage.objects;
CREATE POLICY "student_photos_authenticated_insert"
  ON storage.objects
  FOR INSERT
  TO authenticated
  WITH CHECK (bucket_id = 'student-photos');

DROP POLICY IF EXISTS "student_photos_authenticated_update" ON storage.objects;
CREATE POLICY "student_photos_authenticated_update"
  ON storage.objects
  FOR UPDATE
  TO authenticated
  USING (bucket_id = 'student-photos')
  WITH CHECK (bucket_id = 'student-photos');

DROP POLICY IF EXISTS "student_photos_authenticated_delete" ON storage.objects;
CREATE POLICY "student_photos_authenticated_delete"
  ON storage.objects
  FOR DELETE
  TO authenticated
  USING (bucket_id = 'student-photos');
