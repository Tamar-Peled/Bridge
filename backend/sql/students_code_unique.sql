-- Step 1: Deduplicate student login codes (run this first)
-- Keeps the earliest student per code; assigns the next free 4-digit code to the rest.

DO $$
DECLARE
  rec RECORD;
  candidate text;
  i int;
BEGIN
  FOR rec IN
    SELECT s.id, s.code AS old_code
    FROM public.students s
    INNER JOIN (
      SELECT id,
             ROW_NUMBER() OVER (
               PARTITION BY code
               ORDER BY created_at ASC NULLS LAST, id ASC
             ) AS rn
      FROM public.students
      WHERE code IS NOT NULL AND btrim(code) <> ''
    ) ranked ON ranked.id = s.id AND ranked.rn > 1
    ORDER BY s.id
  LOOP
    candidate := NULL;
    FOR i IN 0..9999 LOOP
      candidate := lpad(i::text, 4, '0');
      IF NOT EXISTS (
        SELECT 1 FROM public.students WHERE code = candidate
      ) THEN
        EXIT;
      END IF;
    END LOOP;
    IF candidate IS NULL THEN
      RAISE EXCEPTION 'No free 4-digit student code available';
    END IF;
    RAISE NOTICE 'Reassigned student % from code % to %', rec.id, rec.old_code, candidate;
    UPDATE public.students SET code = candidate WHERE id = rec.id;
  END LOOP;
END $$;

-- Step 2: Verify no duplicates remain (should return 0 rows)
SELECT code, COUNT(*) AS cnt, array_agg(id ORDER BY created_at NULLS LAST, id) AS student_ids
FROM public.students
WHERE code IS NOT NULL AND btrim(code) <> ''
GROUP BY code
HAVING COUNT(*) > 1;

-- Step 3: Create unique index (run after step 1 succeeds)
CREATE UNIQUE INDEX IF NOT EXISTS idx_students_code_unique
  ON public.students (code)
  WHERE code IS NOT NULL AND code <> '';
