from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json

# Custom JSON encoder to control floating point precision
class PrecisionControlEncoder(json.JSONEncoder):
    """Custom JSON encoder that ensures floating point values don't exceed 8 decimal places."""
    
    def __init__(self, *args, **kwargs):
        # Remove our custom parameter if present
        self.max_decimals = kwargs.pop('max_decimals', 8)
        super().__init__(*args, **kwargs)
    
    def encode(self, obj):
        if isinstance(obj, float):
            # First round the value to the desired precision
            rounded_val = round(obj, self.max_decimals)
            # Then format to ensure exact precision control
            formatted_val = float(f"{rounded_val:.{self.max_decimals}f}")
            return json.JSONEncoder.encode(self, formatted_val)
        return super().encode(obj)
    
    def iterencode(self, obj, _one_shot=False):
        # Special handling for top-level objects
        if isinstance(obj, dict):
            # Process the dictionary before encoding
            obj = self._process_dict(obj)
        elif isinstance(obj, list):
            # Process the list before encoding
            obj = self._process_list(obj)
        
        # Use the standard iterencode with our processed object
        return super().iterencode(obj, _one_shot)
    
    def _process_dict(self, d):
        """Process all items in a dictionary to control float precision."""
        result = {}
        for k, v in d.items():
            if isinstance(v, float):
                # First round the value properly
                rounded_val = round(v, self.max_decimals)
                # Format with exact decimal places
                result[k] = float(f"{rounded_val:.{self.max_decimals}f}")
            elif isinstance(v, dict):
                result[k] = self._process_dict(v)
            elif isinstance(v, list):
                result[k] = self._process_list(v)
            else:
                result[k] = v
        return result
    
    def _process_list(self, lst):
        """Process all items in a list to control float precision."""
        result = []
        for item in lst:
            if isinstance(item, float):
                # First round the value properly
                rounded_val = round(item, self.max_decimals)
                # Format with exact decimal places
                result.append(float(f"{rounded_val:.{self.max_decimals}f}"))
            elif isinstance(item, dict):
                result.append(self._process_dict(item))
            elif isinstance(item, list):
                result.append(self._process_list(item))
            else:
                result.append(item)
        return result


# --- REMOVED/COMMENTED OUT COMPLEX MODELS ---
# class ExplanationContent(BaseModel):
#     """Content explaining a concept or topic."""
#     topic: str = Field(description="The topic being explained")
#     explanation: str = Field(description="A clear, detailed explanation of the topic")
#     examples: List[str] = Field(description="Examples that illustrate the topic")


# class MiniQuizInfo(BaseModel):
#     """Information needed to display a mini-quiz in the practice phase."""
#     related_section_title: str = Field(description="The title of the section this quiz relates to.")
#     related_topic: str = Field(description="The specific topic within the section this quiz relates to.")
#     quiz_question: 'QuizQuestion' = Field(description="The actual quiz question.")


# class UserSummaryPromptInfo(BaseModel):
#     """Information needed to prompt the user for a summary."""
#     section_title: str = Field(description="The title of the section the summary relates to.")
#     topic: str = Field(description="The specific topic the user should summarize.")


# class Exercise(BaseModel):
#     """An exercise for the student to complete."""
#     question: str = Field(description="The exercise question or prompt")
#     difficulty_level: str = Field(description="Easy, Medium, or Hard")
#     answer: str = Field(description="The answer or solution to the exercise")
#     explanation: str = Field(description="Explanation of how to solve the exercise")


# class SectionContent(BaseModel):
#     """The full content for a section of the lesson."""
#     title: str = Field(description="The title of the section")
#     introduction: str = Field(description="Introduction to the section")
#     explanations: List[ExplanationContent] = Field(description="Explanations of key concepts")
#     exercises: List[Exercise] = Field(description="Exercises for practice")
#     summary: str = Field(description="A summary of key points from the section")


# --- MODIFIED LessonContent ---
class LessonContent(BaseModel):
    """The simplified lesson content created by the teacher agent."""
    title: str = Field(description="The title of the lesson")
    segment_index: int = Field(description="The 0-based index of this explanation segment within the topic.")
    is_last_segment: bool = Field(description="Indicates if this is the last segment for the current topic.")
    topic: Optional[str] = Field(None, description="Specific topic being explained in this chunk")
    text: str = Field(description="The full text content of the lesson")


class QuizQuestion(BaseModel):
    """A single quiz question with options and explanation."""
    question: str = Field(description="The question text")
    options: List[str] = Field(description="Multiple choice options (4 options recommended)")
    correct_answer_index: int = Field(description="Index (0-based) of the correct answer in the options list")
    explanation: str = Field(description="Explanation of why the correct answer is correct")
    difficulty: str = Field(description="Easy, Medium, or Hard")
    related_section: str = Field(description="Title of the lesson section this question relates to")


class Quiz(BaseModel):
    """A complete quiz generated based on lesson content."""
    title: str = Field(description="Title of the quiz")
    description: str = Field(description="Brief description of the quiz content and purpose")
    lesson_title: str = Field(description="Title of the lesson this quiz is based on")
    questions: List[QuizQuestion] = Field(description="List of quiz questions")
    passing_score: int = Field(description="Minimum number of correct answers to pass the quiz")
    total_points: int = Field(description="Total possible points for the quiz")
    estimated_completion_time_minutes: int = Field(description="Estimated time to complete the quiz in minutes")


class LearningObjective(BaseModel):
    """Represents a specific learning objective within the lesson plan."""
    title: str = Field(description="The title of the learning objective")
    description: str = Field(description="A detailed description of what the student should learn")
    priority: int = Field(description="Priority from 1-5, with 5 being highest priority")


class LessonSection(BaseModel):
    """Represents a section of the lesson plan."""
    title: str = Field(description="The title of the section")
    objectives: List[LearningObjective] = Field(description="The learning objectives for this section")
    estimated_duration_minutes: int = Field(description="Estimated time in minutes to complete this section")
    concepts_to_cover: List[str] = Field(description="Key concepts that should be covered in this section")
    prerequisites: List[str] = Field(default_factory=list, description="List of concept or section titles that should be understood before starting this section.")
    is_optional: bool = Field(description="Indicates if this section is optional or supplementary content.")


class LessonPlan(BaseModel):
    """The complete lesson plan generated by the planner agent."""
    title: str = Field(description="The overall title of the lesson plan")
    description: str = Field(description="A short description of what the student will learn")
    target_audience: str = Field(description="Who this lesson is designed for (e.g., 'Beginners to machine learning')")
    prerequisites: List[str] = Field(description="Prerequisites that students should know before taking this lesson")
    sections: List[LessonSection] = Field(description="The sections that make up this lesson plan")
    total_estimated_duration_minutes: int = Field(description="Total estimated time to complete the lesson")
    additional_resources: List[str] = Field(default_factory=list, description="Additional resources that might help students")


class QuizUserAnswer(BaseModel):
    """A single user answer to a quiz question."""
    question_index: int = Field(description="Index of the question in the quiz questions list")
    selected_option_index: int = Field(description="Index of the option selected by the user")
    time_taken_seconds: Optional[int] = Field(default=None, description="Time taken to answer the question in seconds")


class QuizUserAnswers(BaseModel):
    """Collection of user answers to a quiz."""
    quiz_title: str = Field(description="Title of the quiz being answered")
    user_answers: List[QuizUserAnswer] = Field(description="List of user answers to quiz questions")
    total_time_taken_seconds: Optional[int] = Field(default=None, description="Total time taken to complete the quiz in seconds")


class QuizFeedbackItem(BaseModel):
    """Feedback for a single user answer."""
    question_index: int = Field(description="Index of the question in the quiz questions list")
    question_text: str = Field(description="The text of the question")
    user_selected_option: str = Field(description="The option selected by the user")
    is_correct: bool = Field(description="Whether the user's answer is correct")
    correct_option: str = Field(description="The correct option")
    explanation: str = Field(description="Explanation of the correct answer")
    improvement_suggestion: str = Field(description="Suggestion for improving understanding if the answer was incorrect")


class QuizFeedback(BaseModel):
    """Complete feedback for a user's quiz answers."""
    quiz_title: str = Field(description="Title of the quiz")
    total_questions: int = Field(description="Total number of questions in the quiz")
    correct_answers: int = Field(description="Number of questions answered correctly")
    score_percentage: float = Field(description="Percentage score (0-100)")
    passed: bool = Field(description="Whether the user passed the quiz based on passing score")
    total_time_taken_seconds: int = Field(description="Total time taken to complete the quiz in seconds")
    feedback_items: List[QuizFeedbackItem] = Field(description="Feedback for each answer")
    overall_feedback: str = Field(description="Overall feedback on the quiz performance")
    suggested_study_topics: List[str] = Field(description="Topics suggested for further study")
    next_steps: List[str] = Field(description="Recommended next steps for learning")


class LearningInsight(BaseModel):
    """A specific insight about the learning session."""
    topic: str = Field(description="The topic or area this insight relates to")
    observation: str = Field(description="What was observed during the session")
    strength: bool = Field(description="Whether this is a strength (True) or area for improvement (False)")
    recommendation: str = Field(description="Recommendation based on this insight")


class TeachingInsight(BaseModel):
    """A specific insight about the teaching approach."""
    approach: str = Field(description="The teaching approach or method used")
    effectiveness: str = Field(description="How effective this approach was")
    evidence: str = Field(description="Evidence from the session supporting this assessment")
    suggestion: str = Field(description="Suggestion for improvement or continuation")


class SessionAnalysis(BaseModel):
    """Complete analysis of a teaching session workflow."""
    session_id: str = Field(description="Unique identifier for the teaching session")
    session_duration_seconds: int = Field(description="Total duration of the session in seconds")
    
    # Overall session assessment
    overall_effectiveness: float = Field(description="Overall effectiveness score (0-100)")
    strengths: List[str] = Field(description="Key strengths of the session")
    improvement_areas: List[str] = Field(description="Areas that need improvement")
    
    # Lesson plan assessment
    lesson_plan_quality: float = Field(description="Quality score for the lesson plan (0-100)")
    lesson_plan_insights: List[str] = Field(description="Insights about the lesson plan")
    
    # Teaching content assessment
    content_quality: float = Field(description="Quality score for the teaching content (0-100)")
    content_insights: List[str] = Field(description="Insights about the teaching content")
    
    # Quiz assessment 
    quiz_quality: float = Field(description="Quality score for the quiz (0-100)")
    quiz_insights: List[str] = Field(description="Insights about the quiz")
    
    # Student performance assessment
    student_performance: float = Field(description="Student performance score (0-100)")
    learning_insights: List[LearningInsight] = Field(description="Detailed insights about student learning")
    
    # Teaching methodology assessment
    teaching_effectiveness: float = Field(description="Score for teaching effectiveness (0-100)")
    teaching_insights: List[TeachingInsight] = Field(description="Detailed insights about teaching methodology")
    
    # Recommendations
    recommendations: List[str] = Field(description="Actionable recommendations for future sessions")
    recommended_adjustments: List[str] = Field(description="Specific adjustments for the next session")
    suggested_resources: List[str] = Field(description="Additional resources to address gaps")

# Forward reference resolution for ExplanationContent
# And MiniQuizInfo which references QuizQuestion
# LessonContent.model_rebuild()
# MiniQuizInfo.model_rebuild() 