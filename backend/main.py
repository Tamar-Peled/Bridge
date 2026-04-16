"""
BRIDGE – FastAPI Backend v3
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

api_key      = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not api_key:      raise ValueError("Missing OPENAI_API_KEY")
if not supabase_url or not supabase_key: raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")

app: FastAPI = FastAPI(title="BRIDGE Backend v3")
db: Client   = create_client(supabase_url, supabase_key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "https://bridge-git-main-tamar-peleds-projects.vercel.app",
        "https://bridge-cf8nltqje-tamar-peleds-projects.vercel.app"
    ],
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

class MeetingNoteCreate(BaseModel):
    content: str
    is_ai_generated: bool = False

class AnalysisResult(BaseModel):
    insights:       List[str]
    alert_level:    Literal["low", "medium", "high"]
    possible_cause: str
    recommendations: List[str]
    suggested_tasks: List[str]

# ═══════════════════ AI SETUP ═════════════════════════════════

class AnalysisResultV4(BaseModel):
    sentiment_summary: str


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

# ═══════════════════ HELPERS ══════════════════════════════════

def strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}

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

@app.get("/meeting-notes/{student_id}")
def get_meeting_notes(student_id: str):
    res = (
        db.table("meeting_notes").select("id,student_id,content,created_at,is_ai_generated")
        .eq("student_id", student_id)
        .order("created_at", desc=True)
        .limit(100)
        .execute()
    )
    return res.data or []

@app.post("/meeting-notes/{student_id}")
def add_meeting_note(student_id: str, note: MeetingNoteCreate):
    content = (note.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="אין טקסט לשמירה")
    res = db.table("meeting_notes").insert({
        "student_id": student_id,
        "content": content,
        "is_ai_generated": bool(note.is_ai_generated),
    }).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="שגיאה בשמירת סיכום")
    row = res.data[0]
    return {
        "id": row.get("id"),
        "student_id": row.get("student_id"),
        "content": row.get("content"),
        "created_at": row.get("created_at"),
        "is_ai_generated": row.get("is_ai_generated", False),
    }

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
