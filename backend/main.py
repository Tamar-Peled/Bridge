"""
BRIDGE – FastAPI Backend v3
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime
import os
import io
import re
import json
import uuid
import base64
from difflib import SequenceMatcher

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from openai import BadRequestError
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

api_key      = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not api_key:      raise ValueError("Missing OPENAI_API_KEY")
if not supabase_url or not supabase_key: raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")

openai_client = OpenAI(api_key=api_key)

app: FastAPI = FastAPI(title="BRIDGE Backend v3")
db: Client   = create_client(supabase_url, supabase_key)


def _supabase_jwt_role(key: str) -> Optional[str]:
    """Decode JWT payload role claim (no signature verify). anon vs service_role matters for Storage RLS."""
    try:
        parts = (key or "").split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        pad = "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode((payload_b64 + pad).encode("ascii")))
        return str(payload.get("role") or "")
    except Exception:
        return None


_jwt_role = _supabase_jwt_role(supabase_key)
if _jwt_role == "anon":
    print(
        "[BRIDGE] WARNING: SUPABASE_KEY decodes as role=anon. Storage uploads will usually fail with RLS. "
        "Use the service_role secret from Supabase Dashboard → Project Settings → API (never expose it to the browser)."
    )
elif _jwt_role and _jwt_role != "service_role":
    print(f"[BRIDGE] INFO: SUPABASE_KEY JWT role={_jwt_role!r} (expected service_role for full Storage access).")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://bridge-git-main-tamar-peleds-projects.vercel.app",
        "https://bridge-five-tau.vercel.app",
    ],
    # Live Server, Vite, etc. on any localhost port
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════ MODELS ═══════════════════════════════════

class StudentCreate(BaseModel):
    name:             str
    grade:            str
    reason:           str
    status:           str  = "בתהליך"
    code:             Optional[str] = None   
    description:      str  = ""
    photo:            str  = ""
    engagement_level: str  = "medium"

class StudentPatch(BaseModel):
    name:             Optional[str] = None
    grade:            Optional[str] = None
    reason:           Optional[str] = None
    status:           Optional[str] = None
    description:      Optional[str] = None
    photo:            Optional[str] = None
    engagement_level: Optional[str] = None
    general_files:    Optional[List[dict]] = None  # [{name, mime, data}] counselor-only file cabinet
    key_points:       Optional[List[dict]] = None  # [{text, at}] counselor "נקודות חשובות"

class TaskCreate(BaseModel):
    student_id:       str
    text:             str

class TaskSelect(BaseModel):
    """Payload when student selects a task into their weekly list"""
    confidence_score: int = Field(..., ge=1, le=5)

class ReportCreate(BaseModel):
    student_id:       str
    task_id:          Optional[str] = None
    task_name:        Optional[str] = None
    mood:             str
    text:             Optional[str] = None
    audio_url:        Optional[str] = None
    confidence_score: Optional[int] = Field(None, ge=1, le=5)

class ReportPatch(BaseModel):
    mood:             Optional[str] = None
    text:             Optional[str] = None
    audio_url:        Optional[str] = None
    confidence_score: Optional[int] = Field(None, ge=1, le=5)


class StudentNoteCreate(BaseModel):
    """Private notes for students (identified by 4-digit code, validated server-side)."""
    student_code: str = Field(..., min_length=4, max_length=4)
    content: str = Field(..., min_length=1, max_length=20000)


class StudentNotePatch(BaseModel):
    student_code: str = Field(..., min_length=4, max_length=4)
    content: str = Field(..., min_length=1, max_length=20000)


class MeetingNoteCreate(BaseModel):
    content: str
    ai_insights: Optional[str] = None
    is_ai_generated: bool = False
    edit_status: Optional[Literal["manual", "ai_generated", "ai_edited"]] = None
    note_type: Literal["session", "insight"] = "session"
    attachments: Optional[List[dict]] = None  # [{name, mime, data}] data = data URL or base64


class MeetingNotePatch(BaseModel):
    content: Optional[str] = None
    ai_insights: Optional[str] = None
    is_ai_generated: Optional[bool] = None
    edit_status: Optional[Literal["manual", "ai_generated", "ai_edited"]] = None
    note_type: Optional[Literal["session", "insight"]] = None
    attachments: Optional[List[dict]] = None

class AnalysisResult(BaseModel):
    insights:       List[str]
    alert_level:    Literal["low", "medium", "high"]
    possible_cause: str
    recommendations: List[str]
    suggested_tasks: List[str]

# ═══════════════════ AI SETUP ═════════════════════════════════

class AnalysisResultV4(BaseModel):
    sentiment_summary: str


class KeyPointsDraftResult(BaseModel):
    """Concise Hebrew bullet points for counselor follow-up."""
    points: List[str] = Field(..., min_length=1, max_length=12)


class StudentFileUpload(BaseModel):
    """Upload binary from data URL or raw base64; stored in Supabase Storage."""
    name: str = Field(..., min_length=1, max_length=255)
    mime: str = "application/octet-stream"
    data: str = Field(..., min_length=1, description="data:image/...;base64,... or raw base64")


class StudentDocumentRename(BaseModel):
    file_name: str = Field(..., min_length=1, max_length=255)


llm_v4 = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=api_key)
structured_llm_v4 = llm_v4.with_structured_output(AnalysisResultV4)

prompt_v4 = ChatPromptTemplate.from_messages([
    ("system",
     "את/ה 'Concise Counselor Assistant' — עוזר/ת ליועצת חינוכית בצורה עניינית וקצרה. "
     "החזר/י עברית מקצועית, רגישה ולא שיפוטית, עם תשומת לב לניואנסים (סלנג, אירוניה, ניסוחים מרומזים) ולמשמעות של אימוג'ים. "
     "אל תאבחני/ן אבחנות רפואיות. "
     "החזיר/י JSON בלבד בהתאם לסכמה."),
    ("human", """נתונים (בעברית):

1) תיאור ראשוני של היועצת/ה על התלמיד/ה:
{student_description}

2) הדיווח האחרון של התלמיד/ה (אימוג'י + טקסט אם קיים):
{latest_report}

3) משימות שבועיות שהוקצו אך לא הושלמו (דגל אדום):
{weekly_assigned_not_done}

4) משימות שהושלמו ויש עליהן משוב/דיווח (עם טקסט משוב):
{completed_with_feedback}

כללי עדיפות נתונים:
- לנתח אך ורק את (3) ו-(4).
- להתייחס ל-(2) רק כדי להבין מצב רגשי נוכחי.
- להתעלם לחלוטין ממשימות שלא הוקצו לשבוע (Mission Bank).

דרישות פלט:
- sentiment_summary: סיכום מקצועי וקצר בעברית (עד 3–4 משפטים).
- להתמקד למה משימות שהוקצו לא בוצעו (ככל שניתן להסיק מהדיווחים) ומה המצב הרגשי הנוכחי.
- לא לכלול פעולות מומלצות בטקסט.
"""),
])

chain_v4 = prompt_v4 | structured_llm_v4

llm_key_points = ChatOpenAI(model="gpt-4o", temperature=0.25, api_key=api_key)
structured_llm_key_points = llm_key_points.with_structured_output(KeyPointsDraftResult)

prompt_key_points = ChatPromptTemplate.from_messages([
    (
        "system",
        "את/ה עוזר/ת מקצועי/ת ליועצת חינוכית. "
        "על בסיס הנתונים שלהלן, הפק/י **נקודות חשובות** קצרות ומעשיות בעברית — כדי שהיועצת תזכור מה דורש מעקב. "
        "כל נקודה חייבת להיות **קצרה מאוד**: לכל היותר 1–2 שורות טקסט, ולרוב משפט אחד בלבד (עד ~100 תווים). "
        "אל תכלול/י פסקאות ארוכות או רשימות משנה בתוך נקודה אחת. "
        "התמקד/י במגמות, סיכונים, הישגים, דפוסי דיווח, וביצוע משימות — לא אבחונים רפואיים. "
        "החזיר/י JSON בלבד לפי הסכמה.",
    ),
    (
        "human",
        """נתונים (עברית):

[תיאור יועצת על התלמיד/ה]
{counselor_description}

[סיכומי פגישות אחרונים]
{meeting_notes_history}

[דיווחי תלמיד — מצב רוח וטקסט]
{student_reports}

[משימות — שבועי / בנק / בוצע / לא בוצע]
{mission_history}

הפק/י בין 4 ל-8 נקודות חשובות (points) כמערך מחרוזות בעברית בלבד — כל מחרוזת נקודה קצרה (משפט אחד עד שני משפטים קצרים).""",
    ),
])

chain_key_points = prompt_key_points | structured_llm_key_points


class PracticalTaskRecommendation(BaseModel):
    recommended_task_title: str
    reasoning: str


llm_task_rec = ChatOpenAI(model="gpt-4o", temperature=0.25, api_key=api_key)
structured_llm_task_rec = llm_task_rec.with_structured_output(PracticalTaskRecommendation)

prompt_task_rec = ChatPromptTemplate.from_messages([
    ("system",
     "את/ה פסיכולוג/ית חינוכי/ת בכיר/ה. "
     "עליך להמליץ על משימה אחת בלבד שהיא **מעשית מאוד**, **קונקרטית**, ושניתן לבצע **היום** (לא ניסוחים מופשטים כמו 'לשפר ניהול לחץ').\n"
     "\n"
     "כללים למשימה:\n"
     "- recommended_task_title חייב להיות משפט קצר ובר-ביצוע בעברית (פועל בתחילה), לדוגמה:\n"
     "  'להיפגש עם חברה אחר הצהריים', 'ללכת לפעולה בצופים', 'לכתוב 3 דברים טובים שקרו היום', 'לצאת להליכה של 10 דקות'.\n"
     "- התאימ/י למצב הרגשי וההקשר: אם יש בדידות/בדידות משתמעת — הציע/י פעולה חברתית קטנה; אם יש לחץ/חרדה/עומס — הציע/י משימה קלה, קצרה, ברת-השגה.\n"
     "- אם קיימת התאמה מצוינת לכותרת קיימת בבנק המשימות, אפשר להשתמש **בדיוק** באותה כותרת (מילה במילה) אם היא כבר מעשית.\n"
     "- אם הבנק ריק או שאין התאמה טובה, צור/י כותרת חדשה מעשית (בעברית) שמתאימה לצורך הנוכחי.\n"
     "\n"
     "פלט:\n"
     "- recommended_task_title: בעברית בלבד.\n"
     "- reasoning: משפט אחד או שניים קצרים בעברית שמסבירים ליועצת למה דווקא הפעולה הזו מתאימה כרגע.\n"
     "החזיר/י JSON בלבד בהתאם לסכמה."),
    ("human", """נתונים (בעברית, השתמש/י בהכל):

[תיאור ראשוני של היועצת]
{counselor_description}

[היסטוריית סיכומי פגישות]
{meeting_notes_history}

[דיווחי תלמיד — טקסט + תמלולי Whisper (מופיעים בשדה text); audio_url מציין שדיווח קולי קיים]
{student_reports}

[היסטוריית משימות — בנק מול שבועי, בוצע/לא בוצע, ביטחון; מצב רוח אחרון מקושר כשקיים]
{mission_history}

[בנק משימות נוכחי (לא שבועי) — עשוי להיות ריק]
{mission_bank_titles}
"""),
])

chain_task_rec = prompt_task_rec | structured_llm_task_rec

# ═══════════════════ HELPERS ══════════════════════════════════

def strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _decode_upload_data(data: str) -> tuple[bytes, str]:
    s = (data or "").strip()
    if not s:
        raise HTTPException(status_code=400, detail="נתוני קובץ חסרים")
    if s.startswith("data:"):
        m = re.match(r"data:([^;]+);base64,(.+)", s, re.DOTALL)
        if not m:
            raise HTTPException(status_code=400, detail="פורמט data URL לא תקין")
        mime = m.group(1).strip()
        b64 = m.group(2).strip()
        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail="פענוח base64 נכשל") from e
        return raw, mime
    try:
        raw = base64.b64decode(s)
    except Exception as e:
        raise HTTPException(status_code=400, detail="פענוח base64 נכשל") from e
    return raw, "application/octet-stream"


def _safe_storage_filename(name: str) -> str:
    base = re.sub(r"[^a-zA-Z0-9._\u0590-\u05FF\-]", "_", (name or "file").strip())[:120]
    return base or "file"


def _general_files_as_list(raw) -> List[dict]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _storage_upload_student_file(student_id: str, body: StudentFileUpload) -> dict:
    """
    Upload to Supabase Storage (bucket student-files by default).
    Returns {name, mime, url}.
    """
    chk = db.table("students").select("id").eq("id", student_id).execute()
    if not chk.data:
        raise HTTPException(status_code=404, detail="תלמיד לא נמצא")
    try:
        raw, detected_mime = _decode_upload_data(body.data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if len(raw) > 4 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="הקובץ גדול מדי (מקס׳ 4MB)")
    mime = (body.mime or detected_mime or "application/octet-stream").strip()
    bucket = (os.getenv("SUPABASE_STORAGE_BUCKET") or "student-files").strip()
    safe_name = _safe_storage_filename(body.name)
    path = f"{student_id}/{uuid.uuid4().hex}_{safe_name}"
    file_opts = {"content-type": mime, "upsert": "true"}
    try:
        db.storage.from_(bucket).upload(path, raw, file_opts)
    except Exception as e:
        err = str(e)
        hint = ""
        if _supabase_jwt_role(supabase_key) == "anon":
            hint = (
                " סיבה סבירה: המפתח ב-SUPABASE_KEY הוא anon (מפתח ציבורי). "
                "החליפי ל-service_role מתפריט Project Settings → API ב-Supabase (בשרת בלבד)."
            )
        raise HTTPException(
            status_code=500,
            detail=(
                f"העלאה ל-Storage נכשלה (bucket={bucket!r}): {err}.{hint} "
                f"אם השם שונה מ-student-files, הגדירי SUPABASE_STORAGE_BUCKET. "
                "הריצי גם את sql/storage_policies_student_files.sql אם עדיין חסום ע״י RLS."
            ),
        ) from e
    pub = db.storage.from_(bucket).get_public_url(path)
    url = pub if isinstance(pub, str) else str(getattr(pub, "data", pub) or pub)
    return {"name": body.name, "mime": mime, "url": url}


def compute_engagement(tasks: list) -> str:
    if not tasks: return "לא ידוע"
    done = sum(1 for t in tasks if t.get("done"))
    r = done / len(tasks)
    return "גבוהה" if r >= 0.7 else "בינונית" if r >= 0.4 else "נמוכה"

def format_date_hebrew(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        days = ["שני","שלישי","רביעי","חמישי","שישי","שבת","ראשון"]
        return f"יום {days[dt.weekday()]} {dt.day}.{dt.month}.{dt.year}"
    except Exception:
        return iso

# ═══════════════════ HEALTH ═══════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Accepts an audio file upload and returns Hebrew transcription text using Whisper.
    """
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="קובץ ריק")

        buf = io.BytesIO(data)
        buf.name = file.filename or "audio.webm"

        tr = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=buf,
            language="he",
        )
        text = (getattr(tr, "text", None) or "").strip()
        return {"text": text}
    except BadRequestError as e:
        # Usually invalid/empty audio or unsupported encoding
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transcribe-image")
async def transcribe_image(file: UploadFile = File(...)):
    """
    OCR / text extraction from an image (JPG/PNG/WebP/GIF) for meeting summaries.
    Returns Hebrew-friendly transcription; empty text if none detected.
    """
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="קובץ ריק")
        if len(data) > 4 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="התמונה גדולה מדי (מקס׳ 4MB)")

        mime = (file.content_type or "").split(";")[0].strip().lower() or "image/jpeg"
        if not mime.startswith("image/"):
            raise HTTPException(status_code=400, detail="נדרש קובץ תמונה (JPG/PNG וכו׳)")

        b64 = base64.b64encode(data).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"

        r = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                (
                    "system",
                    "את/ה מומחה לתמלול טקסט מתמונות (כולל כתב יד). "
                    "החזר/י רק את הטקסט הנראה, בעברית או בשפת המקור כפי שמופיע. "
                    "שמור/י שורות ריקות בין פסקאות כשמתאים. "
                    "אם אין בכלל טקסט קריא בתמונה, השב/י במדויק את המחרוזת: __NO_TEXT__",
                ),
                (
                    "user",
                    [
                        {"type": "text", "text": "תמלל/י את כל הטקסט הנראה בתמונה."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                ),
            ],
            max_tokens=2048,
        )
        raw = (r.choices[0].message.content or "").strip()
        if not raw or raw == "__NO_TEXT__" or "__NO_TEXT__" in raw:
            return {"text": "", "has_text": False}
        return {"text": raw, "has_text": True}
    except HTTPException:
        raise
    except BadRequestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract-document-text")
async def extract_document_text(file: UploadFile = File(...)):
    """Extract plain text from PDF or Word (DOCX) for meeting summaries."""
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="קובץ ריק")
        if len(data) > 6 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="הקובץ גדול מדי (מקס׳ 6MB)")

        name = (file.filename or "").lower()
        ctype = (file.content_type or "").lower()

        text = ""
        if name.endswith(".pdf") or "pdf" in ctype:
            try:
                from pypdf import PdfReader
            except ImportError as e:
                raise HTTPException(status_code=500, detail="שרת ללא תמיכה ב-PDF") from e
            try:
                reader = PdfReader(io.BytesIO(data))
                parts = []
                for page in reader.pages:
                    parts.append(page.extract_text() or "")
                text = "\n".join(parts)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"לא ניתן לקרוא PDF: {e}") from e
        elif name.endswith(".docx") or "wordprocessingml" in ctype or "officedocument.wordprocessingml" in ctype:
            try:
                from docx import Document
            except ImportError as e:
                raise HTTPException(status_code=500, detail="שרת ללא תמיכה ב-Word") from e
            try:
                doc = Document(io.BytesIO(data))
                text = "\n".join(p.text for p in doc.paragraphs if p.text)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"לא ניתן לקרוא Word: {e}") from e
        else:
            raise HTTPException(
                status_code=400,
                detail="פורמט נתמך: PDF או Word (DOCX)",
            )

        text = (text or "").strip()
        return {"text": text, "has_text": bool(text)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ═══════════════════ STUDENTS ═════════════════════════════════

@app.get("/students")
def get_students():
    res = db.table("students").select("*").order("created_at").execute()
    return res.data

# ⚠️ ROUTE ORDER: /students/login BEFORE /students/{id}
@app.get("/students/login/{code}")
def student_login_by_code(code: str):
    """Legacy: login by code. Still works if counselor assigned a code."""
    res = db.table("students").select("id,name,grade").eq("code", code).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="קוד לא נמצא")
    return res.data[0]

@app.get("/students/{student_id}")
def get_student(student_id: str):
    """
    NEW: Student identifies by UUID (no code needed).
    Returns only safe public fields — no description, no photo.
    """
    res = db.table("students").select("id,name,grade").eq("id", student_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="תלמיד לא נמצא")
    return res.data[0]

@app.post("/students")
def create_student(student: StudentCreate):
    res = db.table("students").insert(student.model_dump()).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="שגיאה ביצירת תלמיד")
    return res.data[0]

@app.patch("/students/{student_id}")
def patch_student(student_id: str, data: StudentPatch):
    payload = strip_none(data.model_dump())
    if not payload:
        raise HTTPException(status_code=400, detail="אין שדות לעדכון")
    res = db.table("students").update(payload).eq("id", student_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="תלמיד לא נמצא")
    return res.data[0]


@app.post("/students/{student_id}/files/upload")
def upload_student_file(student_id: str, body: StudentFileUpload):
    """
    Upload a file to Supabase Storage; returns {name, mime, url} (client merges into general_files).
    Prefer POST /students/{id}/documents/upload so the server merges without huge PATCH bodies.
    """
    return _storage_upload_student_file(student_id, body)


@app.get("/students/{student_id}/documents")
def list_student_documents(student_id: str):
    """List general documents: rows from student_documents plus legacy entries from students.general_files."""
    chk = db.table("students").select("id, general_files").eq("id", student_id).execute()
    if not chk.data:
        raise HTTPException(status_code=404, detail="תלמיד לא נמצא")
    try:
        res = (
            db.table("student_documents")
            .select("*")
            .eq("student_id", student_id)
            .order("created_at", desc=True)
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "טבלת student_documents חסרה או לא נגישה. הריצי backend/sql/student_documents_schema.sql. "
                f"{e!s}"
            ),
        ) from e
    out: List[dict] = []
    for r in rows:
        out.append(
            {
                "id": r.get("id"),
                "student_id": r.get("student_id"),
                "file_name": r.get("file_name"),
                "file_url": r.get("file_url"),
                "mime": r.get("mime"),
                "created_at": r.get("created_at"),
                "legacy": False,
            }
        )
    urls_in_table = {str(x.get("file_url") or "").strip() for x in out if x.get("file_url")}
    doc_ids_in_table = {str(x.get("id")) for x in out if x.get("id")}
    gf = _general_files_as_list(chk.data[0].get("general_files"))
    for i, a in enumerate(gf):
        did = str(a.get("document_id") or "").strip()
        u = str(a.get("url") or "").strip()
        if not u and str(a.get("data", "")).startswith("http"):
            u = str(a.get("data", "")).strip()
        if did and did in doc_ids_in_table:
            continue
        if u and u in urls_in_table:
            continue
        embedded = a.get("data")
        if u:
            out.append(
                {
                    "id": None,
                    "legacy_index": i,
                    "student_id": student_id,
                    "file_name": a.get("name") or "קובץ",
                    "file_url": u,
                    "mime": a.get("mime") or "application/octet-stream",
                    "created_at": None,
                    "legacy": True,
                    "embedded_data": None,
                }
            )
        else:
            out.append(
                {
                    "id": None,
                    "legacy_index": i,
                    "student_id": student_id,
                    "file_name": a.get("name") or "קובץ",
                    "file_url": "",
                    "mime": a.get("mime") or "application/octet-stream",
                    "created_at": None,
                    "legacy": True,
                    "embedded_data": str(embedded or ""),
                }
            )
    return out


@app.post("/students/{student_id}/documents/upload")
def upload_student_document(student_id: str, body: StudentFileUpload):
    """
    Upload to Storage (same bucket/logic as meeting file commits), insert student_documents,
    append compact entry to students.general_files on the server (avoids huge client PATCH).
    """
    up = _storage_upload_student_file(student_id, body)
    url = up["url"]
    mime = up["mime"]
    name = up["name"]
    try:
        ins = (
            db.table("student_documents")
            .insert(
                {
                    "student_id": student_id,
                    "file_name": name,
                    "file_url": url,
                    "mime": mime,
                }
            )
            .execute()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "שמירת מסמך בטבלה נכשלה. הריצי backend/sql/student_documents_schema.sql. "
                f"{e!s}"
            ),
        ) from e
    if not ins.data:
        raise HTTPException(status_code=500, detail="שמירת מסמך בטבלה נכשלה")
    row = ins.data[0]
    doc_id = row["id"]
    st = db.table("students").select("general_files").eq("id", student_id).execute()
    if not st.data:
        raise HTTPException(status_code=404, detail="תלמיד לא נמצא")
    gf = _general_files_as_list(st.data[0].get("general_files"))
    gf.append(
        {
            "name": name,
            "mime": mime,
            "url": url,
            "document_id": str(doc_id),
        }
    )
    db.table("students").update({"general_files": gf}).eq("id", student_id).execute()
    return {"document": row, "general_files": gf, **up}


@app.patch("/students/{student_id}/documents/{document_id}")
def rename_student_document(student_id: str, document_id: str, body: StudentDocumentRename):
    nm = (body.file_name or "").strip()
    if not nm:
        raise HTTPException(status_code=400, detail="שם קובץ חסר")
    res = (
        db.table("student_documents")
        .update({"file_name": nm})
        .eq("id", document_id)
        .eq("student_id", student_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="מסמך לא נמצא")
    st = db.table("students").select("general_files").eq("id", student_id).execute()
    if st.data:
        gf = _general_files_as_list(st.data[0].get("general_files"))
        for a in gf:
            if str(a.get("document_id") or "") == str(document_id):
                a["name"] = nm
                break
        db.table("students").update({"general_files": gf}).eq("id", student_id).execute()
    return res.data[0]


@app.delete("/students/{student_id}/documents/{document_id}")
def delete_student_document(student_id: str, document_id: str):
    cur = (
        db.table("student_documents")
        .select("file_url")
        .eq("id", document_id)
        .eq("student_id", student_id)
        .execute()
    )
    if not cur.data:
        raise HTTPException(status_code=404, detail="מסמך לא נמצא")
    url = str(cur.data[0].get("file_url") or "").strip()
    db.table("student_documents").delete().eq("id", document_id).execute()
    st = db.table("students").select("general_files").eq("id", student_id).execute()
    if not st.data:
        return {"deleted": True, "general_files": []}
    gf = _general_files_as_list(st.data[0].get("general_files"))
    sid = str(document_id)
    gf = [
        a
        for a in gf
        if not (
            str(a.get("document_id") or "") == sid
            or (url and str(a.get("url") or "").strip() == url)
        )
    ]
    db.table("students").update({"general_files": gf}).eq("id", student_id).execute()
    return {"deleted": True, "general_files": gf}


@app.delete("/students/{student_id}")
def delete_student(student_id: str):
    db.table("students").delete().eq("id", student_id).execute()
    return {"deleted": True}

# ═══════════════════ TASKS ════════════════════════════════════

@app.get("/tasks/{student_id}")
def get_tasks(student_id: str):
    res = (
        db.table("tasks").select("*")
        .eq("student_id", student_id)
        .order("created_at")
        .execute()
    )
    return res.data

@app.post("/tasks")
def create_task(task: TaskCreate):
    """Counselor creates a task (goes into task bank, selected=false)."""
    payload = task.model_dump()
    payload["selected"] = False
    res = db.table("tasks").insert(payload).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="שגיאה ביצירת משימה")
    return res.data[0]

@app.patch("/tasks/{task_id}/select")
def select_task(task_id: str, body: TaskSelect):
    """
    NEW: Student selects a task into their weekly list.
    Saves confidence score + marks selected=true.
    """
    now = datetime.utcnow().isoformat()
    res = (
        db.table("tasks")
        .update({
            "selected":         True,
            "confidence_score": body.confidence_score,
            "selected_at":      now,
        })
        .eq("id", task_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="משימה לא נמצאה")
    return res.data[0]

@app.patch("/tasks/{task_id}/deselect")
def deselect_task(task_id: str):
    """
    NEW: Student removes a task from their weekly list.
    Resets selected, confidence, done back to defaults.
    """
    res = (
        db.table("tasks")
        .update({
            "selected":         False,
            "confidence_score": None,
            "selected_at":      None,
            "done":             False,
        })
        .eq("id", task_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="משימה לא נמצאה")
    return res.data[0]

@app.patch("/tasks/{task_id}/done")
def mark_task_done(task_id: str):
    """Toggle task done/undone."""
    cur = db.table("tasks").select("done").eq("id", task_id).execute()
    if not cur.data:
        raise HTTPException(status_code=404, detail="משימה לא נמצאה")
    new_val = not cur.data[0]["done"]
    res = db.table("tasks").update({"done": new_val}).eq("id", task_id).execute()
    return res.data[0]

@app.delete("/tasks/{task_id}")
def delete_task(task_id: str):
    db.table("tasks").delete().eq("id", task_id).execute()
    return {"deleted": True}

# ═══════════════════ REPORTS ══════════════════════════════════

@app.get("/reports/{student_id}")
def get_reports(student_id: str):
    res = (
        db.table("reports").select("*")
        .eq("student_id", student_id)
        .order("created_at", desc=True)
        .execute()
    )
    return res.data

@app.post("/reports")
def create_report(report: ReportCreate):
    payload = strip_none(report.model_dump())
    res = db.table("reports").insert(payload).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="שגיאה בשמירת דיווח")
    db.table("logs").insert({
        "student_id": report.student_id,
        "text": f"דיווח נשלח: {report.mood}"
    }).execute()
    return res.data[0]

@app.patch("/reports/{report_id}")
def patch_report(report_id: str, data: ReportPatch):
    payload = strip_none(data.model_dump())
    if not payload:
        raise HTTPException(status_code=400, detail="אין שדות לעדכון")
    res = db.table("reports").update(payload).eq("id", report_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="דיווח לא נמצא")
    return res.data[0]


def _student_row_by_login_code(code: str) -> Optional[dict]:
    """Resolve 4-char login code to a students row (must match public.students.code)."""
    c = (code or "").strip()
    if len(c) != 4:
        return None
    res = db.table("students").select("id,code").eq("code", c).execute()
    if not res.data:
        return None
    return res.data[0]


@app.post("/notes")
def create_student_note(body: StudentNoteCreate):
    st = _student_row_by_login_code(body.student_code)
    if not st:
        raise HTTPException(status_code=404, detail="קוד לא נמצא")
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="אין תוכן")
    res = (
        db.table("student_notes")
        .insert({"student_code": st["code"], "content": content})
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=500, detail="שגיאה בשמירת פתק")
    return res.data[0]


@app.get("/notes/{student_code}")
def list_student_notes(student_code: str):
    st = _student_row_by_login_code(student_code)
    if not st:
        raise HTTPException(status_code=404, detail="קוד לא נמצא")
    res = (
        db.table("student_notes")
        .select("*")
        .eq("student_code", st["code"])
        .order("created_at", desc=True)
        .execute()
    )
    return res.data or []


@app.patch("/notes/{note_id}")
def patch_student_note(note_id: str, body: StudentNotePatch):
    st = _student_row_by_login_code(body.student_code)
    if not st:
        raise HTTPException(status_code=404, detail="קוד לא נמצא")
    content = (body.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="אין תוכן")
    cur = (
        db.table("student_notes")
        .select("id,student_code")
        .eq("id", note_id)
        .execute()
    )
    if not cur.data:
        raise HTTPException(status_code=404, detail="פתק לא נמצא")
    if cur.data[0]["student_code"] != st["code"]:
        raise HTTPException(status_code=403, detail="אין הרשאה")
    res = (
        db.table("student_notes")
        .update({"content": content})
        .eq("id", note_id)
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=500, detail="שגיאה בעדכון פתק")
    return res.data[0]


@app.delete("/notes/{note_id}")
def delete_student_note(
    note_id: str,
    student_code: str = Query(..., min_length=4, max_length=4),
):
    st = _student_row_by_login_code(student_code)
    if not st:
        raise HTTPException(status_code=404, detail="קוד לא נמצא")
    cur = (
        db.table("student_notes")
        .select("id,student_code")
        .eq("id", note_id)
        .execute()
    )
    if not cur.data:
        raise HTTPException(status_code=404, detail="פתק לא נמצא")
    if cur.data[0]["student_code"] != st["code"]:
        raise HTTPException(status_code=403, detail="אין הרשאה")
    db.table("student_notes").delete().eq("id", note_id).execute()
    return {"deleted": True}


@app.patch("/reports/{student_id}/mark-seen")
def mark_reports_seen(student_id: str):
    db.table("reports").update({"is_new": False}).eq("student_id", student_id).execute()
    return {"updated": True}

# ═══════════════════ LOGS ═════════════════════════════════════

@app.get("/logs/{student_id}")
def get_logs(student_id: str):
    res = (
        db.table("logs").select("*")
        .eq("student_id", student_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )
    return res.data

def _meeting_summary_text(row: Optional[dict]) -> str:
    if not row:
        return ""
    return (row.get("summary_text") or row.get("content") or "").strip()


def _http_urls_from_attachments(items: Optional[List[dict]]) -> List[str]:
    out: List[str] = []
    for a in items or []:
        if not isinstance(a, dict):
            continue
        s = str(a.get("data") or a.get("url") or "").strip()
        if s.startswith("http://") or s.startswith("https://"):
            out.append(s)
    return out


def _meeting_note_row_out(row: dict) -> dict:
    eff_status = row.get("edit_status") or (
        "ai_generated" if row.get("is_ai_generated") else "manual"
    )
    att = row.get("attachments")
    if att is None:
        att = []
    body = _meeting_summary_text(row)
    urls = row.get("file_urls")
    if urls is None:
        urls = []
    if not isinstance(urls, list):
        urls = []
    return {
        "id": row.get("id"),
        "student_id": row.get("student_id"),
        "content": body,
        "summary_text": body,
        "ai_insights": (row.get("ai_insights") or "").strip(),
        "file_urls": urls,
        "created_at": row.get("created_at"),
        "is_ai_generated": row.get("is_ai_generated", False),
        "edit_status": eff_status,
        "note_type": row.get("note_type") or "session",
        "attachments": att,
    }


def _meeting_note_insert_payload(student_id: str, note: MeetingNoteCreate) -> dict:
    content = (note.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="אין טקסט לשמירה")
    status = note.edit_status or ("ai_generated" if note.is_ai_generated else "manual")
    attachments = note.attachments if note.attachments is not None else []
    ai_ins = (note.ai_insights or "").strip() if note.ai_insights is not None else ""
    return {
        "student_id": student_id,
        "summary_text": content,
        "content": content,
        "ai_insights": ai_ins,
        "file_urls": _http_urls_from_attachments(attachments),
        "is_ai_generated": bool(note.is_ai_generated),
        "edit_status": status,
        "note_type": note.note_type,
        "attachments": attachments,
    }


def _meeting_note_patch_payload(data: MeetingNotePatch) -> dict:
    raw = strip_none(data.model_dump())
    if not raw:
        return {}
    if "content" in raw:
        raw["summary_text"] = (raw.pop("content") or "").strip()
    if "attachments" in raw:
        raw["file_urls"] = _http_urls_from_attachments(raw.get("attachments"))
    return raw


@app.get("/meeting-notes/{student_id}")
def get_meeting_notes(student_id: str):
    """
    Uses select('*') so missing optional columns in older DBs do not break PostgREST.
    If you get 500 here, ensure table public.meeting_notes exists and migrations ran.
    """
    try:
        res = (
            db.table("meeting_notes")
            .select("*")
            .eq("student_id", student_id)
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "שגיאת מסד נתונים בטעינת סיכומי פגישות. "
                "ודאי שקיימת טבלת meeting_notes ועמודות note_type, attachments, "
                f"או בדקי את מפתח ה-Supabase. פרטים: {e!s}"
            ),
        ) from e
    rows = res.data or []
    return [_meeting_note_row_out(r) for r in rows]

@app.post("/meeting-notes/{student_id}")
def add_meeting_note(student_id: str, note: MeetingNoteCreate):
    payload = _meeting_note_insert_payload(student_id, note)
    res = db.table("meeting_notes").insert(payload).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="שגיאה בשמירת סיכום")
    return _meeting_note_row_out(res.data[0])


@app.patch("/meeting-notes/{note_id}")
def patch_meeting_note(note_id: str, data: MeetingNotePatch):
    payload = _meeting_note_patch_payload(data)
    if not payload:
        raise HTTPException(status_code=400, detail="אין עדכונים")
    res = db.table("meeting_notes").update(payload).eq("id", note_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="סיכום לא נמצא")
    return _meeting_note_row_out(res.data[0])

# ═══════════════════ AI ═══════════════════════════════════════

@app.post("/analyze-student/{student_id}")
def analyze_student(student_id: str):
    try:
        student_res = db.table("students").select("*").eq("id", student_id).execute()
        if not student_res.data:
            raise HTTPException(status_code=404, detail="תלמיד לא נמצא")
        student = student_res.data[0]

        tasks_res   = db.table("tasks").select("*").eq("student_id", student_id).order("created_at").execute()
        reports_res = db.table("reports").select("*").eq("student_id", student_id).order("created_at", desc=True).execute()
        tasks, reports = tasks_res.data, reports_res.data

        # Index reports by task reference, prefer newest first (reports already desc)
        rep_by_task_id = {}
        rep_by_task_name = {}
        for r in reports:
            if r.get("task_id") and r.get("task_id") not in rep_by_task_id:
                rep_by_task_id[r["task_id"]] = r
            if r.get("task_name") and r.get("task_name") not in rep_by_task_name:
                rep_by_task_name[r["task_name"]] = r

        def rep_for_task(t: dict):
            return rep_by_task_id.get(t.get("id")) or rep_by_task_name.get(t.get("text"))

        # Data priority rules:
        # - Analyze ONLY weekly assigned missions, never mission bank
        weekly_tasks = [t for t in tasks if t.get("selected")]
        weekly_assigned_not_done = [t for t in weekly_tasks if not t.get("done")]
        completed_with_feedback = []
        for t in weekly_tasks:
            if not t.get("done"):
                continue
            rep = rep_for_task(t)
            # "feedback" = report text; mood-only is not treated as feedback
            if rep and (rep.get("text") or "").strip():
                completed_with_feedback.append((t, rep))

        def fmt_weekly_not_done(t: dict) -> str:
            d = format_date_hebrew(t.get("created_at", ""))
            cf = f" | ביטחון: {t['confidence_score']}/5" if t.get("confidence_score") else ""
            return f"{d} | {t.get('text','')} | לא הושלמה{cf}"

        def fmt_completed_feedback(t: dict, rep: dict) -> str:
            d = format_date_hebrew(rep.get("created_at") or t.get("created_at", ""))
            mood = rep.get("mood") or "לא דווח"
            txt = (rep.get("text") or "").strip()
            return f"{d} | {t.get('text','')} | {mood} | משוב: {txt}"

        latest_report = reports[0] if reports else None
        latest_mood = latest_report.get("mood") if latest_report else None
        latest_text = latest_report.get("text") if latest_report else None
        latest_report_str = (
            f"{latest_mood or 'לא דווח'}"
            + (f" — {latest_text.strip()}" if latest_text and latest_text.strip() else "")
        )
        result = chain_v4.invoke({
            "student_description": student.get("description", "לא צוין"),
            "latest_report": latest_report_str,
            "weekly_assigned_not_done": "\n".join(fmt_weekly_not_done(t) for t in weekly_assigned_not_done) or "אין",
            "completed_with_feedback": "\n".join(fmt_completed_feedback(t, rep) for (t, rep) in completed_with_feedback) or "אין",
        })
        return result.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _counselor_full_context(student_id: str) -> dict:
    """Shared narrative context for mission recommendation + key-points AI."""
    student_res = db.table("students").select("description").eq("id", student_id).execute()
    if not student_res.data:
        raise HTTPException(status_code=404, detail="תלמיד לא נמצא")
    counselor_description = (student_res.data[0].get("description") or "").strip() or "לא צוין"

    notes_res = (
        db.table("meeting_notes")
        .select("*")
        .eq("student_id", student_id)
        .order("created_at", desc=True)
        .limit(80)
        .execute()
    )
    notes = notes_res.data or []
    note_lines = []
    for n in notes:
        d = format_date_hebrew(n.get("created_at", ""))
        status = n.get("edit_status") or ("ai_generated" if n.get("is_ai_generated") else "manual")
        if status == "manual":
            src = "ידני"
        elif status == "ai_edited":
            src = "AI (נערך)"
        else:
            src = "AI"
        txt = _meeting_summary_text(n)
        if txt:
            note_lines.append(f"- {d} | {src} | {txt}")
    meeting_notes_history = "\n".join(note_lines) if note_lines else "אין סיכומי פגישות"

    reports_res = (
        db.table("reports")
        .select("created_at,mood,text,audio_url,task_id,task_name")
        .eq("student_id", student_id)
        .order("created_at", desc=True)
        .limit(120)
        .execute()
    )
    reports = reports_res.data or []

    def report_line(r: dict) -> str:
        d = format_date_hebrew(r.get("created_at", ""))
        mood = r.get("mood") or "לא דווח"
        txt = (r.get("text") or "").strip()
        audio = "כן" if r.get("audio_url") else "לא"
        tid = r.get("task_id") or "—"
        tname = r.get("task_name") or "—"
        body = f"{mood}" + (f" — {txt}" if txt else "")
        return f"- {d} | משימה: {tname} (id:{tid}) | {body} | הקלטה: {audio}"

    student_reports = "\n".join(report_line(r) for r in reports) if reports else "אין דיווחים"

    tasks_res = (
        db.table("tasks").select("*")
        .eq("student_id", student_id)
        .order("created_at")
        .execute()
    )
    tasks = tasks_res.data or []
    bank = [t for t in tasks if not t.get("selected")]

    mood_by_task_id = {}
    mood_by_task_name = {}
    for r in reports:
        if r.get("task_id") and r.get("task_id") not in mood_by_task_id:
            mood_by_task_id[r["task_id"]] = r.get("mood") or "לא דווח"
        if r.get("task_name") and r.get("task_name") not in mood_by_task_name:
            mood_by_task_name[r["task_name"]] = r.get("mood") or "לא דווח"

    hist_lines = []
    for t in tasks:
        d = format_date_hebrew(t.get("created_at", ""))
        bucket = "שבועי" if t.get("selected") else "בנק"
        done = "בוצעה" if t.get("done") else "לא בוצעה"
        cf = f" | ביטחון: {t['confidence_score']}/5" if t.get("confidence_score") else ""
        mood = mood_by_task_id.get(t.get("id")) or mood_by_task_name.get(t.get("text")) or "לא דווח"
        hist_lines.append(f"- {d} | {bucket} | {t.get('text','')} | {done}{cf} | דיווח אחרון: {mood}")

    mission_history = "\n".join(hist_lines) if hist_lines else "אין היסטוריית משימות"

    bank_texts = [t.get("text", "").strip() for t in bank if (t.get("text") or "").strip()]
    mission_bank_titles = "\n".join(f"- {b}" for b in bank_texts) if bank_texts else "(ריק — אין משימות בבנק כרגע)"

    return {
        "counselor_description": counselor_description,
        "meeting_notes_history": meeting_notes_history,
        "student_reports": student_reports,
        "mission_history": mission_history,
        "mission_bank_titles": mission_bank_titles,
        "_bank_texts": bank_texts,
    }


@app.post("/get-ai-task-recommendation/{student_id}")
def get_ai_task_recommendation(student_id: str):
    """
    Context-aware mission recommendation:
    - Uses counselor description + meeting notes + full reports + mission/task history.
    - Can return an exact mission bank title OR synthesize a new mission when appropriate.
    """
    try:
        ctx = _counselor_full_context(student_id)
        bank_texts = ctx.pop("_bank_texts")
        result = chain_task_rec.invoke({
            "counselor_description": ctx["counselor_description"],
            "meeting_notes_history": ctx["meeting_notes_history"],
            "student_reports": ctx["student_reports"],
            "mission_history": ctx["mission_history"],
            "mission_bank_titles": ctx["mission_bank_titles"],
        })
        title = (result.recommended_task_title or "").strip()
        reasoning = (result.reasoning or "").strip()
        if not title:
            raise HTTPException(status_code=500, detail="לא התקבלה הצעה")

        bank_set = set(bank_texts)

        def best_bank_match(candidate: str) -> Optional[str]:
            if not bank_texts or not candidate:
                return None
            if candidate in bank_set:
                return candidate
            # substring / containment
            hit = next((b for b in bank_texts if candidate in b or b in candidate), None)
            if hit:
                return hit
            # fuzzy best match
            best = None
            best_score = 0.0
            for b in bank_texts:
                s = SequenceMatcher(None, candidate, b).ratio()
                if s > best_score:
                    best_score = s
                    best = b
            if best_score >= 0.72:
                return best
            return None

        # If the model picked something very close to an existing bank mission, snap to exact bank text.
        # Otherwise keep the concrete synthesized Hebrew action as-is.
        if bank_texts:
            m = best_bank_match(title)
            if m:
                title = m

        return {"recommended_task_title": title, "reasoning": reasoning}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/key-points-draft/{student_id}")
def draft_key_points(student_id: str):
    """
    Generate concise Hebrew key points from meeting summaries, tasks, and reports.
    Counselor saves results via PATCH /students with key_points JSON array.
    """
    try:
        ctx = _counselor_full_context(student_id)
        ctx.pop("_bank_texts", None)
        result = chain_key_points.invoke({
            "counselor_description": ctx["counselor_description"],
            "meeting_notes_history": ctx["meeting_notes_history"],
            "student_reports": ctx["student_reports"],
            "mission_history": ctx["mission_history"],
        })
        def _tighten_key_point_line(s: str) -> str:
            s = (s or "").strip()
            if not s:
                return ""
            s = re.sub(r"\s+", " ", s)
            if len(s) <= 160:
                return s
            return s[:158].rstrip() + "…"

        pts_out: List[str] = []
        for p in result.points or []:
            raw = str(p).strip()
            if not raw:
                continue
            lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
            if not lines:
                continue
            # At most 2 short lines merged into one bullet
            chunk = " · ".join(lines[:2]) if len(lines) > 1 else lines[0]
            line = _tighten_key_point_line(chunk)
            if line:
                pts_out.append(line)
        if not pts_out:
            raise HTTPException(status_code=500, detail="לא התקבלו נקודות מהמודל")
        return {"points": pts_out[:12]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
