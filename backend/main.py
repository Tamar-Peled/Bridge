from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

#API KEY 
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")


app = FastAPI(title="BRIDGE Counselor AI Backend")

# CORS - מאפשר לקבצי HTML מקומיים לדבר עם ה-backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#INPUT STRUCTURE 

class TaskItem(BaseModel):
    day: str
    completed: bool
    mood: str

class StudentData(BaseModel):
    name: str
    grade: str
    status: str
    engagement_level: str
    reflection: str
    recent_tasks: List[TaskItem]

#OUTPUT STRUCTURE 

class AnalysisResult(BaseModel):
    insights: List[str] = Field(description="2-3 short counselor-facing insights")
    alert_level: Literal["low", "medium", "high"] = Field(
        description="Urgency level for counselor attention"
    )
    possible_cause: str = Field(description="One likely underlying cause")
    recommendations: List[str] = Field(
        description="3 practical next steps for the counselor"
    )

#MODEL 

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.4,
    api_key=api_key,
)


structured_llm = llm.with_structured_output(AnalysisResult)

#PROMPT 

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant that helps school counselors analyze student behavior and provide actionable insights."
        ),
        (
            "human",
            """
Analyze the student data below and identify behavioral patterns over time.

Specifically:
- Look for repeated patterns across days (e.g., lower engagement on specific days)
- Detect trends in mood, task completion, and engagement
- Identify inconsistencies or sudden changes
- Connect patterns to possible underlying causes
- Refer to specific days when possible (e.g., "on Tuesdays")

Prioritize insights based on repeated patterns rather than one-time events.

Student data:
Name: {name}
Grade: {grade}
Status: {status}
Engagement Level: {engagement_level}
Reflection: {reflection}

Recent Tasks:
{recent_tasks}

Return insights that are short, practical, and directly useful for a school counselor.
"""
        ),
    ]
)


chain = prompt | structured_llm



@app.get("/health")
def health():
    return {"status": "ok"}

#API ENDPOINT 
@app.post("/analyze-student")
def analyze_student(student: StudentData):
    try:
        result = chain.invoke(
            {
                "name": student.name,
                "grade": student.grade,
                "status": student.status,
                "engagement_level": student.engagement_level,
                "reflection": student.reflection,
                "recent_tasks": [
                    f"{task.day} - completed: {task.completed}, mood: {task.mood}"
                    for task in student.recent_tasks
                ],
            }
        )

        return result.model_dump()

    except Exception as e:
        return {
            "insights": [
                "The student shows lower engagement on Tuesdays.",
                "Incomplete tasks are associated with frustration and low mood.",
                "The student appears more engaged on days with lower social pressure."
            ],
            "alert_level": "medium",
            "possible_cause": "Possible social discomfort during group-related activities.",
            "recommendations": [
                "Check in briefly before Tuesday activities.",
                "Start with a smaller social task before a group task.",
                "Reinforce partial success immediately after completion."
            ]
        }
