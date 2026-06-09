"""Dedupe student codes and create unique index when DATABASE_URL is set."""
import os
import sys
from collections import defaultdict

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")

if not url or not key:
    raise SystemExit("Missing SUPABASE_URL or SUPABASE_KEY")

db = create_client(url, key)


def fetch_students():
    res = db.table("students").select("id,code,created_at,name").execute()
    return res.data or []


def dedupe_via_api():
    students = fetch_students()
    by_code: dict[str, list] = defaultdict(list)
    for s in students:
        code = str(s.get("code") or "").strip()
        if code:
            by_code[code].append(s)

    used = {code for code, grp in by_code.items() if grp}
    changes = []

    print("=== BEFORE ===")
    for code, grp in sorted(by_code.items()):
        if len(grp) > 1:
            print(f"  DUPLICATE code={code} count={len(grp)}")
            for s in grp:
                print(f"    - {s.get('name')} ({s['id']}) created_at={s.get('created_at')}")

    for code, grp in by_code.items():
        if len(grp) <= 1:
            continue
        grp.sort(key=lambda s: (s.get("created_at") or "", str(s.get("id") or "")))
        keeper = grp[0]
        print(f"Keeping code {code} for {keeper.get('name')} ({keeper['id']})")
        for s in grp[1:]:
            new_code = None
            for i in range(10000):
                candidate = f"{i:04d}"
                if candidate not in used:
                    new_code = candidate
                    break
            if not new_code:
                raise RuntimeError("No free 4-digit code")
            used.add(new_code)
            upd = db.table("students").update({"code": new_code}).eq("id", s["id"]).execute()
            if not upd.data:
                raise RuntimeError(f"Failed to update student {s['id']}")
            changes.append((s.get("name"), s["id"], code, new_code))
            print(f"  Reassigned {s.get('name')} ({s['id']}): {code} -> {new_code}")

    print("=== AFTER ===")
    students = fetch_students()
    by_code = defaultdict(list)
    for s in students:
        code = str(s.get("code") or "").strip()
        if code:
            by_code[code].append(s)
    dups = {c: g for c, g in by_code.items() if len(g) > 1}
    if dups:
        print("STILL DUPLICATES:", dups)
        return False
    print("No duplicate codes remain.")
    print(f"Changes made: {len(changes)}")
    return True


def create_index_via_postgres():
    if not db_url:
        return False
    try:
        import psycopg2
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary", "-q"])
        import psycopg2

    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_students_code_unique
          ON public.students (code)
          WHERE code IS NOT NULL AND code <> '';
        """
    )
    cur.close()
    conn.close()
    print("Unique index created (or already existed).")
    return True


if __name__ == "__main__":
    ok = dedupe_via_api()
    if not ok:
        raise SystemExit(1)
    if create_index_via_postgres():
        print("Done.")
    else:
        print("Dedupe complete. Set DATABASE_URL to create index via postgres, or run CREATE INDEX in Supabase SQL Editor.")
