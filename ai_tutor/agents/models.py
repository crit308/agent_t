from pydantic import BaseModel, Field
from typing import List


class ExplanationContent(BaseModel):
    """Content explaining a concept or topic."""
    topic: str = Field(description="The topic being explained")
    explanation: str = Field(description="A clear, detailed explanation of the topic")
    examples: List[str] = Field(description="Examples that illustrate the topic")


class Exercise(BaseModel):
    """An exercise for the student to complete."""
    question: str = Field(description="The exercise question or prompt")
    difficulty_level: str = Field(description="Easy, Medium, or Hard")
    answer: str = Field(description="The answer or solution to the exercise")
    explanation: str = Field(description="Explanation of how to solve the exercise")


class SectionContent(BaseModel):
    """The full content for a section of the lesson."""
    title: str = Field(description="The title of the section")
    introduction: str = Field(description="Introduction to the section")
    explanations: List[ExplanationContent] = Field(description="Explanations of key concepts")
    exercises: List[Exercise] = Field(description="Exercises for practice")
    summary: str = Field(description="A summary of key points from the section")


class LessonContent(BaseModel):
    """The complete lesson content created by the teacher agent."""
    title: str = Field(description="The title of the lesson")
    introduction: str = Field(description="Introduction to the overall lesson")
    sections: List[SectionContent] = Field(description="Content for each section of the lesson")
    conclusion: str = Field(description="Conclusion summarizing the lesson")
    next_steps: List[str] = Field(description="Suggested next steps for continued learning")


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