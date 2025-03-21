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
                
                # The handoff process may have resulted in a Quiz object being returned
                # Check if we got a Quiz instead of LessonContent and handle appropriately
                if isinstance(self.lesson_content, Quiz):
                    # Store the quiz but return the actual lesson content (which was generated first)
                    self.quiz = self.lesson_content
                    # Get a new instance of the lesson content
                    # We need to call the function again or retrieve the lesson content another way
                    raise TypeError("Handoff returned a Quiz object instead of LessonContent")
                
                return self.lesson_content
            except TypeError as te:
                # If we got a Quiz instead of LessonContent, handle it gracefully
                if "Handoff returned a Quiz object instead of LessonContent" in str(te):
                    print("Successfully generated quiz through handoff. Retrieving lesson content...")
                    # Note: At this point, we need to somehow retrieve the originally generated lesson content
                    # This is a placeholder - the actual implementation depends on how the handoff is structured
                    return LessonContent(
                        title=self.lesson_plan.title,
                        introduction="Lesson content not available due to handoff process.",
                        sections=[],
                        conclusion="Please check the trace for complete lesson content.",
                        next_steps=[]
                    )
                else:
                    raise
            except Exception as e:
                # If the error is about missing sections attribute, it's likely a Quiz object
                if "'Quiz' object has no attribute 'sections'" in str(e):
                    print("Received a Quiz object from the handoff process.")
                    # Store the quiz if it's in a variable we can access
                    if hasattr(e, 'obj') and isinstance(e.obj, Quiz):
                        self.quiz = e.obj
                    
                    # Return a placeholder lesson content
                    return LessonContent(
                        title=self.lesson_plan.title,
                        introduction="Lesson content was generated but a Quiz was returned from the handoff.",
                        sections=[],
                        conclusion="Please check the trace for complete lesson content.",
                        next_steps=[]
                    )
                else:
                    # Re-raise other exceptions
                    raise
    
    async def run_full_workflow(self, file_paths: List[str]) -> dict:
        """Run the full AI tutor workflow from document upload to lesson and quiz generation."""
        # Upload documents
        await self.upload_documents(file_paths)
        
        # Generate lesson plan
        await self.generate_lesson_plan()
        
        # Generate lesson content
        lesson_content = await self.generate_lesson()
        
        # The quiz is automatically generated through the handoff mechanism
        # We can check the trace to see the quiz output
        
        return {
            "lesson_plan": self.lesson_plan,
            "lesson_content": lesson_content,
            # Note: The quiz is generated through handoff, so we don't directly capture it here
            # If we need to access the quiz later, we should add a method to retrieve it from the handoff result
            "quiz": None  # Set to None to avoid attempts to access quiz.sections
        } 