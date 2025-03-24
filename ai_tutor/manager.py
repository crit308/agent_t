import asyncio
import os
from typing import List, Optional, Any
import openai
import threading

from agents import Runner, trace, gen_trace_id, set_tracing_export_api_key

from ai_tutor.tools.file_upload import FileUploadManager, upload_document
from ai_tutor.agents.planner_agent import create_planner_agent
from ai_tutor.agents.teacher_agent import generate_lesson_content, create_teacher_agent
from ai_tutor.agents.analyzer_agent import analyze_documents
from ai_tutor.agents.models import LessonContent, Quiz, LessonPlan, LessonSection, LearningObjective


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
            
            IMPORTANT: You MUST generate and output a complete LessonPlan object BEFORE 
            attempting to hand off to the teacher agent.
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
            
            # Add instruction to hand off to teacher agent after generating the plan
            prompt += """
            
            CRITICAL WORKFLOW:
            1. YOU MUST FIRST create and output a complete LessonPlan object
            2. ONLY AFTER that, use the transfer_to_lesson_teacher tool to hand off to the Teacher agent
            """
            
            # Run the planner agent to generate a lesson plan
            result = await Runner.run(
                planner_agent, 
                prompt
            )
            
            # Track what kind of output we got to handle multiple handoffs
            output_is_lesson_plan = False
            
            # Try to extract the LessonPlan
            try:
                self.lesson_plan = result.final_output_as(LessonPlan)
                output_is_lesson_plan = True
                
                # Basic validation to ensure the lesson plan has sections
                if not self.lesson_plan.sections or len(self.lesson_plan.sections) == 0:
                    print("WARNING: Generated lesson plan has no sections. This may cause issues with the teacher agent.")
                
                print(f"Successfully generated lesson plan: {self.lesson_plan.title}")
                print(f"  Sections: {len(self.lesson_plan.sections)}")
                print(f"  Total duration: {self.lesson_plan.total_estimated_duration_minutes} minutes")
                
            except Exception as e:
                if isinstance(result.final_output, Quiz):
                    # We got a Quiz object, which means we went through the full handoff chain
                    print("Full handoff chain completed: Planner → Teacher → Quiz Creator")
                    self.quiz = result.final_output
                    print(f"Successfully captured Quiz from full handoff chain: {self.quiz.title}")
                    print(f"Quiz has {len(self.quiz.questions)} questions")
                    output_is_lesson_plan = False
                elif isinstance(result.final_output, LessonContent):
                    # We got a LessonContent object, which means we went through the first handoff
                    print("Teacher handoff completed but no Quiz Creator handoff occurred")
                    self.lesson_content = result.final_output
                    print(f"Successfully captured LessonContent from handoff: {self.lesson_content.title}")
                    output_is_lesson_plan = False
                else:
                    print(f"Error extracting LessonPlan: {e}")
                    print(f"Final output type: {type(result.final_output)}")
                    print("The planner agent did not generate a valid lesson plan before attempting handoff.")
                    raise ValueError("Failed to generate a valid lesson plan. The agent may have attempted to hand off prematurely.")
            
            # If we received a lesson plan (no handoff completed), we're done
            if output_is_lesson_plan:
                return self.lesson_plan
            
            # Check for handoffs and capture outputs
            if result.last_agent:
                print(f"Last agent in chain: {result.last_agent.name}")
                
                # If we reached the Teacher agent
                if result.last_agent.name == "Lesson Teacher":
                    # Check if we already have the lesson content (should have been captured above)
                    if not self.lesson_content and isinstance(result.final_output, LessonContent):
                        self.lesson_content = result.final_output
                        print("Successfully captured LessonContent from handoff")
                    elif not self.lesson_content and isinstance(result.final_output, dict):
                        try:
                            self.lesson_content = LessonContent(**result.final_output)
                            print("Successfully captured LessonContent from handoff dictionary")
                        except Exception as e:
                            print(f"Error converting dictionary to LessonContent: {e}")
                
                # If we reached the Quiz Creator agent
                elif result.last_agent.name == "Quiz Creator":
                    # Check if we already have the quiz (should have been captured above)
                    if not self.quiz and isinstance(result.final_output, Quiz):
                        self.quiz = result.final_output
                        print("Successfully captured Quiz from handoff")
                    elif not self.quiz and isinstance(result.final_output, dict):
                        try:
                            self.quiz = Quiz(**result.final_output)
                            print("Successfully captured Quiz from handoff dictionary")
                        except Exception as e:
                            print(f"Error converting dictionary to Quiz: {e}")
            
            # If we don't have a lesson plan by this point, we need to create a synthetic one
            if not self.lesson_plan:
                # Try to reconstruct the lesson plan from lesson content if available
                if self.lesson_content:
                    print("Reconstructing lesson plan from lesson content")
                    
                    # Create sections based on the lesson content
                    sections = []
                    for section in self.lesson_content.sections:
                        learning_objectives = []
                        for explanation in section.explanations:
                            # Extract concepts from the explanation
                            key_concepts = [explanation.topic]
                            if explanation.examples:
                                # Add example topics as concepts
                                key_concepts.extend([ex.split(":")[0] for ex in explanation.examples if ":" in ex])
                            
                            # Create learning objective
                            learning_objectives.append(LearningObjective(
                                description=f"Understand {explanation.topic}",
                                key_concepts=key_concepts
                            ))
                        
                        # Create lesson section
                        sections.append(LessonSection(
                            title=section.title,
                            description=section.introduction,
                            learning_objectives=learning_objectives,
                            estimated_duration_minutes=30  # Default estimate
                        ))
                    
                    # Create synthesized lesson plan
                    self.lesson_plan = LessonPlan(
                        title=self.lesson_content.title,
                        description=self.lesson_content.introduction,
                        target_audience="Learners interested in this topic",
                        prerequisites=["Basic understanding of the subject"],
                        sections=sections,
                        total_estimated_duration_minutes=len(sections) * 30,  # Rough estimate
                        additional_resources=[]
                    )
                    
                    print(f"Successfully reconstructed lesson plan: {self.lesson_plan.title}")
                    print(f"  Sections: {len(self.lesson_plan.sections)}")
                # If we have a Quiz but no lesson plan or lesson content, create a minimal lesson plan based on the quiz
                elif self.quiz:
                    print("Creating minimal lesson plan from quiz")
                    
                    # Extract a title from the quiz title
                    quiz_based_title = self.quiz.title
                    if quiz_based_title.startswith("Quiz:"):
                        quiz_based_title = quiz_based_title[5:].strip()
                    
                    # Create sections based on question topics
                    topic_dict = {}
                    for question in self.quiz.questions:
                        section_title = question.related_section
                        if section_title not in topic_dict:
                            topic_dict[section_title] = []
                        topic_dict[section_title].append(question.question)
                    
                    sections = []
                    for section_title, questions in topic_dict.items():
                        # Create a learning objective for each question
                        learning_objectives = []
                        for question in questions:
                            # Extract a brief topic from the question
                            topic = question.split('?')[0]
                            if len(topic) > 50:
                                topic = topic[:47] + "..."
                            
                            learning_objectives.append(LearningObjective(
                                description=f"Understand {topic}",
                                key_concepts=[topic]
                            ))
                        
                        # Create a section with these learning objectives
                        sections.append(LessonSection(
                            title=section_title,
                            description=f"This section covers topics related to {section_title}",
                            learning_objectives=learning_objectives,
                            estimated_duration_minutes=len(questions) * 10  # Rough estimate
                        ))
                    
                    # Create synthesized lesson plan
                    self.lesson_plan = LessonPlan(
                        title=quiz_based_title,
                        description=self.quiz.description,
                        target_audience="Learners interested in this topic",
                        prerequisites=["Basic understanding of the subject"],
                        sections=sections,
                        total_estimated_duration_minutes=sum(section.estimated_duration_minutes for section in sections),
                        additional_resources=[]
                    )
                    
                    print(f"Successfully created minimal lesson plan from quiz: {self.lesson_plan.title}")
                    print(f"  Sections: {len(self.lesson_plan.sections)}")
                else:
                    # If we have neither lesson plan nor lesson content nor quiz, this is an error
                    raise ValueError("Failed to generate or reconstruct a valid lesson plan")
            
            # At this point, self.lesson_plan should be a valid LessonPlan object
            return self.lesson_plan
    
    async def generate_lesson(self) -> Optional[LessonContent]:
        """Generate a lesson based on the lesson plan."""
        if not self.lesson_plan:
            raise ValueError("No lesson plan has been generated yet")
        
        if not self.vector_store_id:
            raise ValueError("No documents have been uploaded yet")
        
        # If we already have lesson content from the handoff, return it
        if self.lesson_content:
            print("Using lesson content that was already generated via handoff")
            return self.lesson_content
        
        trace_id = gen_trace_id()
        print(f"Generating lesson content with trace ID: {trace_id}")
        print(f"View trace at: https://platform.openai.com/traces/{trace_id}")
        
        # Ensure API key is set for tracing before creating the trace
        if self.api_key:
            set_tracing_export_api_key(self.api_key)
        
        with trace("Generating lesson content", trace_id=trace_id):
            try:
                # Create the teacher agent
                teacher_agent = create_teacher_agent(self.vector_store_id, self.api_key)
                
                # Format the lesson plan as a string for the teacher agent
                prompt = f"""
                Generate comprehensive lesson content based on this lesson plan.
                
                For each section, use the file_search tool to gather accurate information before creating content.
                
                IMPORTANT: You MUST first generate a complete LessonContent object BEFORE 
                attempting to hand off to the quiz creator agent.
                
                CRITICAL WORKFLOW:
                1. YOU MUST FIRST create and output a complete LessonContent object
                2. ONLY AFTER that, use the transfer_to_quiz_creator tool to hand off to the Quiz Creator agent
                """
                
                # Run the teacher agent to generate a lesson content
                result = await Runner.run(
                    teacher_agent, 
                    prompt
                )
                
                # Verify that we have valid lesson content
                try:
                    self.lesson_content = result.final_output_as(LessonContent)
                    
                    # Basic validation to ensure the lesson content has sections
                    if not self.lesson_content.sections or len(self.lesson_content.sections) == 0:
                        print("WARNING: Generated lesson content has no sections. This may cause issues with the quiz creator agent.")
                    
                    print(f"Successfully generated lesson content: {self.lesson_content.title}")
                    print(f"  Sections: {len(self.lesson_content.sections)}")
                    
                except Exception as e:
                    print(f"Error extracting LessonContent: {e}")
                    print("The teacher agent did not generate valid lesson content before attempting handoff.")
                    raise ValueError("Failed to generate valid lesson content. The agent may have attempted to hand off prematurely.")
                
                # Check if there was a handoff that completed
                if result.last_agent and result.last_agent.name == "Quiz Creator":
                    print(f"Handoff to {result.last_agent.name} was successful")
                    
                    # If the quiz creator agent provided a final output, capture it
                    if isinstance(result.final_output, Quiz):
                        self.quiz = result.final_output
                        print("Successfully captured Quiz from handoff")
                    elif isinstance(result.final_output, dict):
                        try:
                            self.quiz = Quiz(**result.final_output)
                            print("Successfully captured Quiz from handoff dictionary")
                        except Exception as e:
                            print(f"Error converting dictionary to Quiz: {e}")
                    else:
                        print(f"Quiz creator agent handoff completed but didn't provide Quiz. Output type: {type(result.final_output)}")
                else:
                    # If there was no handoff or it didn't reach the quiz agent, generate the quiz directly
                    if self.lesson_content and hasattr(self.lesson_content, 'sections') and len(self.lesson_content.sections) > 0:
                        # Create a separate trace for quiz generation
                        quiz_trace_id = gen_trace_id()
                        print(f"No handoff to quiz creator detected. Generating quiz directly with trace ID: {quiz_trace_id}")
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
        """Run the full AI tutor workflow, from document upload to quiz generation.
        
        Args:
            file_paths: List of file paths to process
            run_analyzer: Whether to run document analysis
            analyzer_in_background: If running analysis, whether to do it in background
            
        Returns:
            Dictionary with the results of each step
        """
        results = {}
        
        # Step 1: Upload documents
        upload_result = await self.upload_documents(file_paths)
        results["upload"] = upload_result
        
        # Step 2 (optional): Analyze documents
        if run_analyzer:
            analysis = await self.analyze_documents(run_in_background=analyzer_in_background)
            if not analyzer_in_background and analysis:
                results["analysis"] = "Analysis completed"
            else:
                results["analysis"] = "Analysis started in background"
        
        # Step 3: Generate lesson plan
        try:
            lesson_plan = await self.generate_lesson_plan()
            # Make sure we're storing a LessonPlan object in the results, not a Quiz
            if isinstance(lesson_plan, LessonPlan):
                results["lesson_plan"] = lesson_plan
            else:
                # If lesson_plan is not a LessonPlan but we have self.lesson_plan, use that
                if self.lesson_plan and isinstance(self.lesson_plan, LessonPlan):
                    results["lesson_plan"] = self.lesson_plan
                else:
                    # Create a minimal LessonPlan as fallback
                    default_title = "Generated Content"
                    if self.lesson_content:
                        default_title = self.lesson_content.title
                    elif self.quiz:
                        default_title = self.quiz.title.replace("Quiz: ", "")
                    
                    results["lesson_plan"] = LessonPlan(
                        title=default_title,
                        description="Lesson plan generated from handoff process",
                        target_audience="Learners interested in this topic",
                        prerequisites=["Basic understanding of the subject"],
                        sections=[],
                        total_estimated_duration_minutes=60,
                        additional_resources=[]
                    )
        except Exception as e:
            print(f"Error in lesson plan generation: {e}")
            # Create a default lesson plan for the results
            results["lesson_plan"] = LessonPlan(
                title="Error in Lesson Plan Generation",
                description="There was an error generating the lesson plan",
                target_audience="Learners interested in this topic",
                prerequisites=["Basic understanding of the subject"],
                sections=[],
                total_estimated_duration_minutes=60,
                additional_resources=[]
            )
        
        # Step 4: Only generate lesson content if it wasn't created via handoff
        if not self.lesson_content:
            try:
                lesson_content = await self.generate_lesson()
                results["lesson_content"] = lesson_content
            except Exception as e:
                print(f"Error generating lesson content: {e}")
                results["lesson_content"] = LessonContent(
                    title=results["lesson_plan"].title,
                    introduction="Error generating lesson content",
                    sections=[],
                    conclusion="An error occurred during content generation",
                    next_steps=[]
                )
        else:
            # Lesson content was already generated via handoff
            results["lesson_content"] = self.lesson_content
        
        # Add quiz information to results if available
        if self.quiz:
            results["quiz"] = self.quiz
            results["quiz_info"] = f"Generated quiz: {self.quiz.title} with {len(self.quiz.questions)} questions"
        else:
            results["quiz_info"] = "No quiz was generated"
        
        # Return all results
        return results 