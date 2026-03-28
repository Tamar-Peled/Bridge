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

# ─── KEYS ────────────────────────────────────────────────────────
api_key      = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY")

# ─── CLIENTS ─────────────────────────────────────────────────────
app: FastAPI = FastAPI(title="BRIDGE Counselor AI Backend")
db: Client   = create_client(supabase_url, supabase_key)

# ─── CORS ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── CRUD MODELS ─────────────────────────────────────────────────
class StudentCreate(BaseModel):
    name: str
    grade: str
    reason: str
    status: str = "יציב"
    code: str
    engagement_level: str = "medium"

class TaskCreate(BaseModel):
    student_id: str
    text: str

class ReportCreate(BaseModel):
    student_id: str
    mood: str
    text: Optional[str] = None
    task_name: Optional[str] = None

# ─── AI OUTPUT MODEL ─────────────────────────────────────────────
class AnalysisResult(BaseModel):
    insights: List[str] = Field(
        description="2-3 תובנות קצרות וממוקדות ליועצת, בעברית"
    )
    alert_level: Literal["low", "medium", "high"] = Field(
        description="רמת דחיפות לתשומת לב היועצת"
    )
    possible_cause: str = Field(
        description="סיבה אפשרית אחת עיקרית, בעברית"
    )
    recommendations: List[str] = Field(
        description="3 צעדים מעשיים ליועצת, בעברית"
    )
    suggested_tasks: List[str] = Field(
        description="2-3 משימות מוצעות לתלמיד בעברית, קצרות וברות ביצוע"
    )

# ─── AI SETUP ────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4, api_key=api_key)
structured_llm = llm.with_structured_output(AnalysisResult)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """אתה עוזר AI מקצועי לייעוץ חינוכי. תפקידך לסייע ליועצות בית ספר לנתח את המצב הרגשי וההתנהגותי של תלמידים.
        
חשוב מאוד: כל התשובות שלך חייבות להיות בעברית בלבד."""
    ),
    (
        "human",
        """נתח את נתוני התלמיד הבאים וזהה דפוסים התנהגותיים.

התמקד ב:
- תדירות ביצוע המשימות (האם מבצע באופן עקבי? האם יש ימים שבהם לא מבצע?)
- מגמות במצב הרוח לאורך זמן
- קשר בין מצב הרוח לביצוע משימות
- שינויים פתאומיים או אי-עקביות
- הרקע שהיועצת סיפקה על התלמיד

נתוני התלמיד:
שם: {name}
כיתה: {grade}
סטטוס: {status}
סיבת פנייה (רקע מהיועצת): {reason}
רמת מעורבות כללית: {engagement_level}
הערות התלמיד מהדיווחים: {reflection}

משימות ודיווחים לאורך זמן:
{recent_tasks}

החזר:
- תובנות ממוקדות ומעשיות ליועצת
- סיבה אפשרית אחת עיקרית
- 3 המלצות מעשיות ליועצת
- 2-3 משימות מוצעות לתלמיד שמתאימות למצבו

הכל בעברית בלבד."""
    ),
])

chain = prompt | structured_llm

# ─── HELPERS ─────────────────────────────────────────────────────
def format_date_hebrew(iso_string: str) -> str:
    """ממיר תאריך ISO לפורמט עברי קריא"""
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        days = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
        day_name = days[dt.weekday()]
        return f"יום {day_name} {dt.day}.{dt.month}.{dt.year}"
    except:
        return iso_string

def compute_engagement(tasks: list) -> str:
    """מחשב רמת מעורבות לפי אחוז ביצוע משימות"""
    if not tasks:
        return "לא ידוע"
    done = sum(1 for t in tasks if t.get("done"))
    ratio = done / len(tasks)
    if ratio >= 0.7:
        return "גבוהה"
    elif ratio >= 0.4:
        return "בינונית"
    else:
        return "נמוכה"

# ─── HEALTH ──────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ─── STUDENTS ────────────────────────────────────────────────────
@app.get("/students")
def get_students():
    """מחזיר את כל התלמידים"""
    res = db.table("students").select("*").order("created_at").execute()
    return res.data

@app.post("/students")
def create_student(student: StudentCreate):
    """מוסיף תלמיד חדש"""
    res = db.table("students").insert(student.model_dump()).execute()
    return res.data[0]

@app.delete("/students/{student_id}")
def delete_student(student_id: str):
    """מוחק תלמיד (CASCADE מוחק גם משימות ודיווחים)"""
    db.table("students").delete().eq("id", student_id).execute()
    return {"deleted": True}

# ─── LOGIN (תלמיד) ───────────────────────────────────────────────
@app.get("/students/login/{code}")
def student_login(code: str):
    """תלמיד נכנס עם קוד — מחזיר רק שדות בטוחים"""
    res = db.table("students").select("id, name, grade").eq("code", code).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="קוד לא נמצא")
    return res.data[0]

# ─── TASKS ───────────────────────────────────────────────────────
@app.get("/tasks/{student_id}")
def get_tasks(student_id: str):
    """מחזיר את כל המשימות של תלמיד"""
    res = db.table("tasks").select("*").eq("student_id", student_id).order("created_at").execute()
    return res.data

@app.post("/tasks")
def create_task(task: TaskCreate):
    """מוסיף משימה לתלמיד"""
    res = db.table("tasks").insert(task.model_dump()).execute()
    return res.data[0]

@app.patch("/tasks/{task_id}/done")
def mark_task_done(task_id: str):
    """מסמן משימה כבוצעה"""
    res = db.table("tasks").update({"done": True}).eq("id", task_id).execute()
    return res.data[0]

@app.delete("/tasks/{task_id}")
def delete_task(task_id: str):
    """מוחק משימה"""
    db.table("tasks").delete().eq("id", task_id).execute()
    return {"deleted": True}

# ─── REPORTS ─────────────────────────────────────────────────────
@app.get("/reports/{student_id}")
def get_reports(student_id: str):
    """מחזיר את כל הדיווחים של תלמיד"""
    res = db.table("reports").select("*").eq("student_id", student_id).order("created_at", desc=True).execute()
    return res.data

@app.post("/reports")
def create_report(report: ReportCreate):
    """תלמיד שולח דיווח"""
    res = db.table("reports").insert(report.model_dump()).execute()
    db.table("logs").insert({
        "student_id": report.student_id,
        "text": f"דיווח נשלח: {report.mood}"
    }).execute()
    return res.data[0]

@app.patch("/reports/{student_id}/mark-seen")
def mark_reports_seen(student_id: str):
    """מסמן דיווחים כנקראו"""
    db.table("reports").update({"is_new": False}).eq("student_id", student_id).execute()
    return {"updated": True}

# ─── LOGS ────────────────────────────────────────────────────────
@app.get("/logs/{student_id}")
def get_logs(student_id: str):
    """מחזיר היסטוריה של תלמיד"""
    res = db.table("logs").select("*").eq("student_id", student_id).order("created_at", desc=True).limit(20).execute()
    return res.data

# ─── AI ANALYSIS ─────────────────────────────────────────────────
@app.post("/analyze-student/{student_id}")
def analyze_student(student_id: str):
    """
    שולף את כל נתוני התלמיד מהDB,
    מחשב engagement_level אמיתי,
    ושולח לAI לניתוח מעמיק.
    """
    try:
        # שלב 1 — שליפת כל הנתונים מהDB
        student_res = db.table("students").select("*").eq("id", student_id).execute()
        if not student_res.data:
            raise HTTPException(status_code=404, detail="תלמיד לא נמצא")
        student = student_res.data[0]

        tasks_res   = db.table("tasks").select("*").eq("student_id", student_id).order("created_at").execute()
        reports_res = db.table("reports").select("*").eq("student_id", student_id).order("created_at").execute()

        tasks   = tasks_res.data
        reports = reports_res.data

        # שלב 2 — חישוב engagement_level אמיתי
        engagement = compute_engagement(tasks)

        # שלב 3 — בניית מפת דיווחים לפי שם משימה
        # כדי לקשר בין משימה לדיווח הרלוונטי
        reports_by_task = {}
        for r in reports:
            if r.get("task_name"):
                reports_by_task[r["task_name"]] = r.get("mood", "לא דווח")

        # שלב 4 — בניית רשימת משימות עם תאריכים ומצב רוח אמיתיים
        recent_tasks_formatted = []
        for t in tasks:
            date_str = format_date_hebrew(t.get("created_at", ""))
            mood = reports_by_task.get(t["text"], "לא דווח")
            status = "בוצעה ✓" if t["done"] else "לא בוצעה ✗"
            recent_tasks_formatted.append(
                f"{date_str} | משימה: {t['text']} | {status} | מצב רוח: {mood}"
            )

        # שלב 5 — איסוף הערות חופשיות מהדיווחים
        reflection = " | ".join(
            r["text"] for r in reports if r.get("text")
        ) or "אין הערות"

        # שלב 6 — הרצת ה-AI
        result = chain.invoke({
            "name":             student["name"],
            "grade":            student["grade"],
            "status":           student["status"],
            "reason":           student.get("reason", "לא צוין"),
            "engagement_level": engagement,
            "reflection":       reflection,
            "recent_tasks":     "\n".join(recent_tasks_formatted) or "אין משימות עדיין",
        })

        return result.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
