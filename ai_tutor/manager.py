import asyncio
import os
from typing import List, Optional
import openai

from agents import Runner, trace, gen_trace_id, set_tracing_export_api_key

from ai_tutor.tools.file_upload import FileUploadManager, upload_document
from ai_tutor.agents.planner_agent import create_planner_agent, LessonPlan
from ai_tutor.agents.teacher_agent import generate_lesson_content
from ai_tutor.agents.models import LessonContent, Quiz


class AITutorManager:
    """Main manager class for the AI Tutor system."""
    
    def __init__(self, api_key: str):
        """Initialize the AI Tutor manager with the OpenAI API key."""
        self.api_key = api_key
        
        # Set OpenAI API key globally - this is important for tracing
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            # Explicitly set the tracing API key
            set_tracing_export_api_key(api_key)
            print(f"Set OPENAI_API_KEY environment variable for API and tracing")
            
            # Initialize the OpenAI client for direct API calls if needed
            self.client = openai.OpenAI(api_key=api_key)
            
        self.file_upload_manager = FileUploadManager(api_key)
        self.vector_store_id = None
        self.lesson_plan = None
        self.lesson_content = None
        self.quiz = None
        self.file_paths = []
    
    async def upload_documents(self, file_paths: List[str]) -> str:
        """Upload documents to be used by the AI tutor."""
        results = []
        self.file_paths = file_paths  # Store file paths for reference
        
        for file_path in file_paths:
            try:
                uploaded_file = self.file_upload_manager.upload_and_process_file(file_path)
                results.append(f"Successfully uploaded {uploaded_file.filename}")
                
                # Store the vector store ID for later use
                self.vector_store_id = uploaded_file.vector_store_id
            except Exception as e:
                results.append(f"Error uploading {file_path}: {str(e)}")
        
        return "\n".join(results)
    
    async def generate_lesson_plan(self) -> Optional[LessonPlan]:
        """Generate a lesson plan based on the uploaded documents."""
        if not self.vector_store_id:
            raise ValueError("No documents have been uploaded yet")
        
        # Create and configure trace
        trace_id = gen_trace_id()
        print(f"Generating lesson plan with trace ID: {trace_id}")
        print(f"View trace at: https://platform.openai.com/traces/{trace_id}")
        
        # Ensure API key is set for tracing before creating the trace
        if self.api_key:
            set_tracing_export_api_key(self.api_key)
        
        with trace("Generating lesson plan", trace_id=trace_id):
            # Create the planner agent
            planner_agent = create_planner_agent(self.vector_store_id, self.api_key)
            
            # Create a prompt that focuses on document analysis
            prompt = """
            Create a comprehensive lesson plan based on the documents that have been uploaded.
            
            Use the file_search tool to explore and understand the content of the documents.
            Start by searching for general topics, then dive deeper into specific areas.
            
            Focus on creating a structured plan that covers all important concepts in the material.
            """
            
            # Run the planner agent to generate a lesson plan
            result = await Runner.run(
                planner_agent, 
                prompt
            )
            
            # Store and return the lesson plan
            self.lesson_plan = result.final_output_as(LessonPlan)
            return self.lesson_plan
    
    async def generate_lesson(self) -> Optional[LessonContent]:
        """Generate a lesson based on the lesson plan."""
        if not self.lesson_plan:
            raise ValueError("No lesson plan has been generated yet")
        
        if not self.vector_store_id:
            raise ValueError("No documents have been uploaded yet")
        
        trace_id = gen_trace_id()
        print(f"Generating lesson content with trace ID: {trace_id}")
        print(f"View trace at: https://platform.openai.com/traces/{trace_id}")
        
        # Ensure API key is set for tracing before creating the trace
        if self.api_key:
            set_tracing_export_api_key(self.api_key)
        
        with trace("Generating lesson content", trace_id=trace_id):
            try:
                # Generate the lesson content
                self.lesson_content = await generate_lesson_content(
                    self.lesson_plan, 
                    self.vector_store_id,
                    self.api_key
                )
                
                # Only proceed with quiz generation if we have valid lesson content with sections
                if self.lesson_content and hasattr(self.lesson_content, 'sections') and len(self.lesson_content.sections) > 0:
                    # Create a separate trace for quiz generation
                    quiz_trace_id = gen_trace_id()
                    print(f"Generating quiz with trace ID: {quiz_trace_id}")
                    print(f"View trace at: https://platform.openai.com/traces/{quiz_trace_id}")
                    
                    with trace("Generating quiz", trace_id=quiz_trace_id):
                        # Import and call quiz creator directly instead of relying on handoff
                        from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent, generate_quiz
                        try:
                            self.quiz = await generate_quiz(self.lesson_content, self.api_key)
                            print(f"Successfully generated quiz with {len(self.quiz.questions) if hasattr(self.quiz, 'questions') else 0} questions")
                        except Exception as quiz_error:
                            print(f"Error generating quiz: {quiz_error}")
                            self.quiz = None
                else:
                    print("Skipping quiz generation as lesson content has no sections")
                    self.quiz = None
                
                # Return the lesson content
                return self.lesson_content
                
            except Exception as e:
                print(f"Error generating lesson: {e}")
                # Return a placeholder if something went wrong
                return LessonContent(
                    title=self.lesson_plan.title,
                    introduction="Error generating complete lesson content.",
                    sections=[],
                    conclusion="An error occurred during lesson generation.",
                    next_steps=[]
                )
    
    async def run_full_workflow(self, file_paths: List[str]) -> dict:
        """Run the full AI tutor workflow from document upload to lesson and quiz generation."""
        # Upload documents
        await self.upload_documents(file_paths)
        
        # Generate lesson plan
        await self.generate_lesson_plan()
        
        # Generate lesson content and quiz
        lesson_content = await self.generate_lesson()
        
        # Prepare the result dictionary
        result = {
            "lesson_plan": self.lesson_plan,
            "lesson_content": lesson_content,
        }
        
        # Only include the quiz if it was successfully generated
        if self.quiz is not None:
            result["quiz"] = self.quiz
        else:
            print("No quiz was generated. Omitting quiz from results.")
        
        return result 