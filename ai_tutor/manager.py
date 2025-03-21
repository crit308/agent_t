import asyncio
import os
from typing import List, Optional, Any
import openai
import threading

from agents import Runner, trace, gen_trace_id, set_tracing_export_api_key

from ai_tutor.tools.file_upload import FileUploadManager, upload_document
from ai_tutor.agents.planner_agent import create_planner_agent, LessonPlan
from ai_tutor.agents.teacher_agent import generate_lesson_content
from ai_tutor.agents.analyzer_agent import analyze_documents
from ai_tutor.agents.models import LessonContent, Quiz


class AITutorManager:
    """Main manager class for the AI Tutor system."""
    
    def __init__(self, api_key: str, auto_analyze: bool = False):
        """Initialize the AI Tutor manager with the OpenAI API key.
        
        Args:
            api_key: The OpenAI API key to use for all agents
            auto_analyze: Whether to automatically run document analysis when documents are uploaded
        """
        self.api_key = api_key
        self.auto_analyze = auto_analyze
        
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
        self.document_analysis = None
        self.file_paths = []
        self._analysis_task = None
        self._analysis_complete = asyncio.Event()
    
    async def _run_analyzer_in_background(self) -> None:
        """Run the analyzer agent in background and set the document_analysis property."""
        try:
            self.document_analysis = await analyze_documents(self.vector_store_id, self.api_key)
            # Get number of files and key concepts if available
            file_count = len(getattr(self.document_analysis, "file_names", [])) 
            concept_count = len(getattr(self.document_analysis, "key_concepts", []))
            print(f"Background document analysis complete. Found {file_count} files and {concept_count} key concepts.")
        except Exception as e:
            print(f"Error in background document analysis: {str(e)}")
            self.document_analysis = None
        finally:
            # Signal that analysis is complete
            self._analysis_complete.set()
    
    def start_background_analysis(self) -> None:
        """Start analyzing documents in the background."""
        if not self.vector_store_id:
            print("Cannot start background analysis: No vector store ID available")
            return
        
        # Create a new event loop for the background task
        loop = asyncio.new_event_loop()
        
        # Create a function that will run in the background thread
        def run_analyzer_in_thread():
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_analyzer_in_background())
            finally:
                loop.close()
        
        # Start a new thread to run the analysis
        analysis_thread = threading.Thread(target=run_analyzer_in_thread, daemon=True)
        analysis_thread.start()
        
        print(f"Started background document analysis for vector store {self.vector_store_id}")
    
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
        
        # If auto-analyze is enabled and we have a vector store ID, start the analyzer agent in background
        if self.auto_analyze and self.vector_store_id:
            # Reset the completion event
            self._analysis_complete = asyncio.Event()
            
            # Start background analysis
            self.start_background_analysis()
            results.append("Document analysis started in background.")
        
        return "\n".join(results)
    
    async def wait_for_analysis(self, timeout: float = None) -> Optional[str]:
        """Wait for the background analysis to complete and return the results."""
        if self._analysis_complete is None:
            print("No analysis is currently running")
            return self.document_analysis
        
        try:
            # Wait for the analysis to complete or timeout
            await asyncio.wait_for(self._analysis_complete.wait(), timeout)
            return self.document_analysis
        except asyncio.TimeoutError:
            print("Waiting for analysis timed out")
            return None
    
    async def analyze_documents(self, run_in_background: bool = False) -> Optional[str]:
        """Analyze the uploaded documents using the analyzer agent.
        
        Args:
            run_in_background: If True, analysis runs in background and function returns immediately
        
        Returns:
            DocumentAnalysis object if run_in_background is False, None otherwise
        """
        if not self.vector_store_id:
            raise ValueError("No documents have been uploaded yet")
        
        if run_in_background:
            # Start analysis in background and return immediately
            self._analysis_complete = asyncio.Event()
            self.start_background_analysis()
            return None
        
        # Create and configure trace
        trace_id = gen_trace_id()
        print(f"Analyzing documents with trace ID: {trace_id}")
        print(f"View trace at: https://platform.openai.com/traces/{trace_id}")
        
        # Ensure API key is set for tracing before creating the trace
        if self.api_key:
            set_tracing_export_api_key(self.api_key)
        
        with trace("Document analysis", trace_id=trace_id):
            # Run the analyzer on the vector store
            self.document_analysis = await analyze_documents(self.vector_store_id, self.api_key)
            return self.document_analysis
    
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
            
            # If we have document analysis results, enhance the prompt with key concepts
            if self.document_analysis:
                key_concepts = []
                # Try to get key concepts from the document analysis result
                if hasattr(self.document_analysis, "key_concepts"):
                    key_concepts = self.document_analysis.key_concepts[:10]  # Use top 10 concepts
                elif isinstance(self.document_analysis, str) and "KEY CONCEPTS:" in self.document_analysis:
                    # Try to extract key concepts from the text
                    try:
                        concepts_section = self.document_analysis.split("KEY CONCEPTS:")[1].split("CONCEPT DETAILS:")[0]
                        key_concepts = [c.strip() for c in concepts_section.strip().split("\n") if c.strip()][:10]
                    except Exception:
                        pass
                
                if key_concepts:
                    concepts_str = ", ".join(key_concepts)
                    prompt += f"""
                    
                    Based on preliminary analysis, these key concepts were identified:
                    {concepts_str}
                    
                    Consider incorporating these key concepts into your lesson plan where appropriate.
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
    
    async def run_full_workflow(self, file_paths: List[str], run_analyzer: bool = False, analyzer_in_background: bool = True) -> dict:
        """Run the full AI tutor workflow from document upload to lesson and quiz generation."""
        # Upload documents
        await self.upload_documents(file_paths)
        
        # Run analyzer if requested and not already run via auto-analyze
        if run_analyzer and not self.document_analysis:
            if analyzer_in_background:
                # Start analyzer in background
                await self.analyze_documents(run_in_background=True)
                print("Document analysis started in background.")
            else:
                # Run analyzer and wait for completion
                self.document_analysis = await self.analyze_documents()
                if self.document_analysis:
                    print(f"Document analysis completed. Found {len(self.document_analysis.file_names)} files and {len(self.document_analysis.key_concepts)} key concepts.")
        
        # Generate lesson plan
        await self.generate_lesson_plan()
        
        # If analysis was running in background, try to get results now
        if run_analyzer and analyzer_in_background:
            # Try to wait for analysis, but with a timeout to not block too long
            analysis_result = await self.wait_for_analysis(timeout=0.1)
            if analysis_result:
                print("Background analysis results are now available.")
        
        # Generate lesson content and quiz
        lesson_content = await self.generate_lesson()
        
        # Prepare the result dictionary
        result = {
            "lesson_plan": self.lesson_plan,
            "lesson_content": lesson_content,
        }
        
        # Include document analysis if available
        if self.document_analysis is not None:
            result["document_analysis"] = self.document_analysis
        
        # Only include the quiz if it was successfully generated
        if self.quiz is not None:
            result["quiz"] = self.quiz
        else:
            print("No quiz was generated. Omitting quiz from results.")
        
        return result 