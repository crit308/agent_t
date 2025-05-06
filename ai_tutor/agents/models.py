from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

# ---------------------------------------------------------------------------
# Lesson-related models (simplified versions kept for type-hints & JSON parsing)
# ---------------------------------------------------------------------------
class LessonContent(BaseModel):
    """Simplified text chunk returned by the teacher/explanation tools."""
    title: str
    segment_index: int
    is_last_segment: bool
    topic: Optional[str] = None
    text: str
    total_segments: Optional[int] = None


class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer_index: int
    explanation: str
    difficulty: str
    related_section: str


class Quiz(BaseModel):
    title: str
    description: str
    lesson_title: str
    questions: List[QuizQuestion]
    passing_score: int
    total_points: int
    estimated_completion_time_minutes: int


class LearningObjective(BaseModel):
    title: str
    description: str
    priority: int


class LessonSection(BaseModel):
    title: str
    objectives: List[LearningObjective]
    estimated_duration_minutes: int
    concepts_to_cover: List[str]
    prerequisites: List[str] = Field(default_factory=list)
    is_optional: bool


class LessonPlan(BaseModel):
    title: str
    description: str
    target_audience: str
    prerequisites: List[str]
    sections: List[LessonSection]
    total_estimated_duration_minutes: int
    additional_resources: List[str] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Quiz interaction models
# ---------------------------------------------------------------------------
class QuizUserAnswer(BaseModel):
    question_index: int
    selected_option_index: int
    time_taken_seconds: Optional[int] = None


class QuizUserAnswers(BaseModel):
    quiz_title: str
    user_answers: List[QuizUserAnswer]
    total_time_taken_seconds: Optional[int] = None


class QuizFeedbackItem(BaseModel):
    question_index: int
    question_text: str
    user_selected_option: str
    is_correct: bool
    correct_option: str
    explanation: str
    improvement_suggestion: str


class QuizFeedback(BaseModel):
    quiz_title: str
    total_questions: int
    correct_answers: int
    score_percentage: float
    passed: bool
    total_time_taken_seconds: int
    overall_feedback: str
    suggested_study_topics: List[str]
    next_steps: List[str]

# ---------------------------------------------------------------------------
# Session analysis models (high-level)
# ---------------------------------------------------------------------------
class LearningInsight(BaseModel):
    topic: str
    observation: str
    strength: bool
    recommendation: str


class TeachingInsight(BaseModel):
    approach: str
    effectiveness: str
    evidence: str
    suggestion: str


class SessionAnalysis(BaseModel):
    session_id: str
    session_duration_seconds: Optional[int] = None
    overall_effectiveness: float
    strengths: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)
    lesson_plan_quality: Optional[float] = None
    lesson_plan_insights: List[str] = Field(default_factory=list)
    content_quality: Optional[float] = None
    content_insights: List[str] = Field(default_factory=list)
    quiz_quality: Optional[float] = None
    quiz_insights: List[str] = Field(default_factory=list)
    student_performance: Optional[float] = None
    learning_insights: List[LearningInsight] = Field(default_factory=list)
    teaching_effectiveness: Optional[float] = None
    teaching_insights: List[TeachingInsight] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    recommended_adjustments: List[str] = Field(default_factory=list)
    suggested_resources: List[str] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Planner-related models
# ---------------------------------------------------------------------------
class FocusObjective(BaseModel):
    topic: str
    learning_goal: str
    priority: int
    relevant_concepts: List[str] = Field(default_factory=list)
    suggested_approach: Optional[str] = None
    target_mastery: float
    initial_difficulty: Optional[str] = None


class ActionSpec(BaseModel):
    agent: Literal["teacher", "quiz_creator", "explanation_checker"]
    params: Dict[str, Optional[str]]
    success_criteria: str
    max_steps: int


class PlannerOutput(BaseModel):
    objective: FocusObjective
    next_action: ActionSpec

# ---------------------------------------------------------------------------
# Misc agent results
# ---------------------------------------------------------------------------
class ExplanationResult(BaseModel):
    status: Literal["delivered", "failed", "skipped"]
    details: Optional[str] = None


class QuizCreationResult(BaseModel):
    status: Literal["created", "failed"]
    quiz: Optional[Quiz] = None
    question: Optional[QuizQuestion] = None
    details: Optional[str] = None 