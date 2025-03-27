import asyncio
import os
from typing import List, Optional, Any
from pydantic import BaseModel  # Import BaseModel if not already imported
import threading
import uuid
import time

from agents import Runner, trace, gen_trace_id, set_tracing_export_api_key, RunConfig

from ai_tutor.tools.file_upload import FileUploadManager, upload_document
from ai_tutor.agents.planner_agent import create_planner_agent
from ai_tutor.agents.teacher_agent import generate_lesson_content, create_teacher_agent, create_teacher_agent_without_handoffs
from ai_tutor.agents.analyzer_agent import analyze_documents
from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent, generate_quiz, create_quiz_creator_agent_with_teacher_handoff
from ai_tutor.agents.quiz_teacher_agent import create_quiz_teacher_agent, generate_quiz_feedback
from ai_tutor.agents.session_analyzer_agent import create_session_analyzer_agent, analyze_teaching_session
from ai_tutor.agents.models import LessonContent, Quiz, LessonPlan, LessonSection, LearningObjective, QuizUserAnswers, QuizFeedback, QuizUserAnswer, SectionContent, ExplanationContent, Exercise, SessionAnalysis
from ai_tutor.context import TutorContext  # Import the new context model


class AITutorManager:
    """Main manager class for the AI Tutor system."""
    
    def __init__(self, auto_analyze: bool = False, output_logger=None):
        """Initialize the AI Tutor manager with the OpenAI API key.
        
        Args:
            auto_analyze: Whether to automatically run document analysis when documents are uploaded
            output_logger: Optional logger for capturing agent outputs
        """
        self.auto_analyze = auto_analyze
        self.file_upload_manager = FileUploadManager()
        self.lesson_plan = None
        self.lesson_content = None
        self.quiz = None
        self.quiz_feedback = None
        self.session_analysis = None
        self._analysis_task = None
        self._analysis_complete = asyncio.Event()
        self._current_trace_id = None
        self._session_start_time = time.time()
        
        # Initialize or set the output logger
        if output_logger:
            self.output_logger = output_logger
        else:
            # Import the logger here to avoid circular imports
            from ai_tutor.output_logger import get_logger
            self.output_logger = get_logger()
        
        # Initialize TutorContext
        self.context = TutorContext(
            session_id=f"session_{uuid.uuid4().hex[:8]}",
            vector_store_id=None, # Will be set after upload
            uploaded_file_paths=[]
        )
        self.document_analysis: Optional[str] = None # Store analysis result here
        print(f"Initialized AI Tutor Manager with session ID: {self.context.session_id}")
    
    async def _run_analyzer_in_background(self) -> None:
        """Run the analyzer agent in background and set the document_analysis property."""
        try:
            # Run the analyzer on the vector store
            try:
                # RunConfig will be passed inside analyze_documents
                self.document_analysis = await analyze_documents(self.context.vector_store_id, context=self.context) # Removed api_key arg
                return self.document_analysis
            except Exception as e:
                print(f"ERROR: Analyzer agent failed: {e}")
                if hasattr(self, 'output_logger') and self.output_logger:
                    self.output_logger.log_error("Analyzer Agent", e)
                # We're already in a try-except block, so we don't need to re-raise
                self.document_analysis = None
        except Exception as e:
            print(f"Error in background document analysis: {str(e)}")
            self.document_analysis = None
        finally:
            # Signal that analysis is complete
            self._analysis_complete.set()
    
    def start_background_analysis(self) -> None:
        """Start analyzing documents in the background."""
        if not self.context.vector_store_id:
            print("Cannot start background analysis: No vector store ID available.")
            return
        
        # Use asyncio.create_task instead of threading
        if self._analysis_task is None or self._analysis_task.done():
             print("Creating new asyncio task for background analysis.")
             # Ensure the event loop is running if this is called from sync code? No, manager methods are async.
             self._analysis_task = asyncio.create_task(self._run_analyzer_in_background())
             print(f"Started background document analysis task for vector store {self.context.vector_store_id}")
        else:
            print("Background analysis task already running.")
    
    async def upload_documents(self, file_paths: List[str]) -> str:
        """Upload documents to be used by the AI tutor."""
        results = []
        self.context.uploaded_file_paths = file_paths # Store file paths in context
        
        for file_path in file_paths:
            try:
                uploaded_file = self.file_upload_manager.upload_and_process_file(file_path)
                results.append(f"Successfully uploaded {uploaded_file.filename}")
                
                # Store the vector store ID for later use
                self.context.vector_store_id = uploaded_file.vector_store_id
            except Exception as e:
                results.append(f"Error uploading {file_path}: {str(e)}")
        
        # If auto-analyze is enabled and we have a vector store ID, start background analysis
        if self.auto_analyze and self.context.vector_store_id:
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
        if not self.context.vector_store_id:
            raise ValueError("No documents have been uploaded yet")
        
        if run_in_background:
            # Start analysis in background and return immediately
            self._analysis_complete = asyncio.Event()
            self.start_background_analysis()
            return None
        
        # No need for outer trace() if Runner.run uses RunConfig
        # with trace("Document analysis"): # Trace ID generated automatically
        print(f"Analyzing documents...")
        # Run the analyzer on the vector store
        try:
            # RunConfig will be passed inside analyze_documents
            self.document_analysis = await analyze_documents(self.context.vector_store_id, context=self.context)
            return self.document_analysis
        except Exception as e:
            print(f"ERROR: Analyzer agent failed: {e}")
            if hasattr(self, 'output_logger') and self.output_logger:
                self.output_logger.log_error("Analyzer Agent", e)
            # Re-raise for now
            raise
    
    async def generate_lesson_plan(self) -> Optional[LessonPlan]:
        """Generate a lesson plan based on the uploaded documents.
        
        This method may trigger a handoff chain from planner → teacher → quiz creator,
        potentially resulting in a Quiz as the final output instead of a LessonPlan.
        """
        if not self.context.vector_store_id:
            raise ValueError("No documents have been uploaded yet")
        
        # Check if the Knowledge Base file exists before starting the planner agent
        if not os.path.exists("Knowledge Base"):
            print("Waiting for Document Analyzer to create 'Knowledge Base' file...")
            # Wait for the file to be created with a timeout
            start_time = time.time()
            timeout = 120  # 2 minutes timeout
            while not os.path.exists("Knowledge Base"):
                await asyncio.sleep(1)
                if time.time() - start_time > timeout:
                    raise TimeoutError("Timed out waiting for Document Analyzer to create 'Knowledge Base' file. Please ensure the analyzer has completed.")
            print("'Knowledge Base' file created. Starting Planner Agent...")
        
        # No need for outer trace() if Runner.run uses RunConfig
        # with trace("Generating lesson plan"):
        print(f"Generating lesson plan...")

        # Create the planner agent
        planner_agent = create_planner_agent(self.context.vector_store_id) # No api_key

        # Create a prompt that focuses on document analysis and EXPLICITLY instructs to hand off
        prompt = """
        Create a comprehensive lesson plan based on the documents that have been uploaded.
        
        The lesson plan should include:
        1. Clear learning objectives
        2. Main sections with key points
        3. A structured approach that builds knowledge progressively
        
        Be sure to:
        - Focus on the most important concepts
        - Create a logical flow between sections
        - Include practical exercises or examples where relevant
        - Consider the target audience's needs
        
        When you're done creating the lesson plan, please hand off to the 
        Teacher Agent so they can develop the complete lesson content.
        """

        print("Running planner agent with explicit handoff instructions...")
        # Configure Tracing via RunConfig
        run_config = RunConfig(
            workflow_name="AI Tutor - Lesson Planning",
            group_id=self.context.session_id # Link traces within the same session
        )
        try:
            result = await Runner.run(
                planner_agent, 
                prompt,
                run_config=run_config, # Pass RunConfig
                context=self.context # Pass the context
            )
        except Exception as e: # Catch specific SDK exceptions later
            print(f"ERROR: Planner agent failed: {e}")
            if hasattr(self, 'output_logger') and self.output_logger:
                self.output_logger.log_error("Planner Agent", e)
            # Decide how to handle - raise, return None, create minimal plan?
            raise # Re-raise for now
            
        # Extract LessonPlan/Quiz from result FIRST
        # NOTE: When handoffs occur, result.final_output will contain the output
        # of the LAST agent in the chain (e.g., Quiz), not necessarily the
        # output type requested by the *first* agent (LessonPlan).
        final_output_obj = None
        output_is_lesson_plan = False # Initialize flag
        handoff_chain_completed = False # Initialize flag

        if result and hasattr(result, 'final_output'):
            final_output_obj = result.final_output
            print(f"DEBUG: Extracted final_output from RunResult. Type: {type(final_output_obj)}")
        else:
            print(f"Warning: Planner agent RunResult has no 'final_output'. Result: {result}")

        # Check the type of the actual final output to determine workflow path
        if isinstance(final_output_obj, LessonPlan):
            print("DEBUG: Final output IS a LessonPlan.")
            self.lesson_plan = final_output_obj
            output_is_lesson_plan = True
            # No need to set handoff_chain_completed here
        elif isinstance(final_output_obj, Quiz):
            print("DEBUG: Final output IS a Quiz.")
            self.quiz = final_output_obj
            print(f"Got Quiz directly from handoff chain: {self.quiz.title}")
            handoff_chain_completed = True
            
            # Create a synthetic lesson plan so the CLI doesn't get a None result
            # IMPORTANT: Ensure all required fields of LessonPlan are provided,
            # even if using default values like an empty list here.
            if not self.lesson_plan:
                print("Creating synthetic lesson plan from Quiz for consistent API")
                self.lesson_plan = LessonPlan(
                    title=f"Lesson Plan for {self.quiz.title}",
                    description=f"Auto-generated lesson plan based on quiz: {self.quiz.description}",
                    target_audience="Learners",
                    prerequisites=[],
                    total_estimated_duration_minutes=60,
                    sections=[
                        LessonSection(
                            title=f"Section on {self.quiz.title}",
                            estimated_duration_minutes=60,
                            objectives=[
                                LearningObjective(
                                    title="Complete the quiz",
                                    description="Successfully complete the associated quiz",
                                    priority=5
                                )
                            ],
                            concepts_to_cover=["Quiz topics"]
                        )
                    ],
                    additional_resources=[]
                )
        elif isinstance(final_output_obj, LessonContent):
            print("DEBUG: Final output IS a LessonContent.")
            self.lesson_content = final_output_obj
            print(f"Got lesson content directly from handoff: {self.lesson_content.title}")
            
            # Create a synthetic lesson plan based on the lesson content
            print("Creating synthetic lesson plan from LessonContent for consistent API")
            self.lesson_plan = LessonPlan(
                title=f"Lesson Plan for {self.lesson_content.title}",
                description=f"Auto-generated lesson plan based on lesson content",
                target_audience="Learners",
                prerequisites=[],
                total_estimated_duration_minutes=60,
                sections=[
                    LessonSection(
                        title=section.title,
                        estimated_duration_minutes=30,
                        objectives=[
                            LearningObjective(
                                title=f"Learn {section.title}",
                                description=f"Understand {section.title}",
                                priority=3
                            )
                        ],
                        concepts_to_cover=["Content topics"]
                    ) for section in self.lesson_content.sections
                ],
                additional_resources=[]
            )
        else:
            # Handle unexpected or missing output type
            print(f"ERROR: Final output was None or unexpected type: {type(final_output_obj)}. Cannot proceed directly.")
            if hasattr(result, 'model_dump'):
                print(f"Result: {result.model_dump()}")
            else:
                print(f"Result: {result}")
                
            # Create a minimal fallback lesson plan as a last resort
            print("Creating minimal fallback lesson plan")
            self.lesson_plan = LessonPlan(
                title="Fallback Lesson Plan",
                description="A minimal lesson plan created when no proper output was available",
                target_audience="General audience",
                prerequisites=[],
                total_estimated_duration_minutes=30,
                sections=[
                    LessonSection(
                        title="Introduction",
                        estimated_duration_minutes=30,
                        objectives=[
                            LearningObjective(
                                title="Learn basics",
                                description="Understand fundamental concepts",
                                priority=5
                            )
                        ],
                        concepts_to_cover=["Basic concepts"]
                    )
                ],
                additional_resources=[]
            )
                
        # Always return self.lesson_plan, which is now guaranteed to be a valid LessonPlan object
        return self.lesson_plan
    
    async def generate_lesson_content(self) -> Optional[LessonContent]:
        """Generate detailed lesson content based on the lesson plan.
        
        This may trigger a handoff to the quiz creator agent.
        """
        if not self.lesson_plan:
            raise ValueError("No lesson plan has been generated yet")
        
        # No need for outer trace() if Runner.run uses RunConfig
        # with trace("Generating lesson content"):
        print(f"Generating lesson content...")

        # IMPORTANT: Check if we already have a quiz from the handoff chain
        # If so, create lesson content directly without calling the teacher agent
        # This prevents unnecessary teacher agent calls when we already have a quiz
        if self.quiz and hasattr(self.quiz, 'questions') and len(self.quiz.questions) > 0:
            print("Quiz already exists - creating lesson content directly from quiz without calling teacher agent")
            self.lesson_content = self._create_lesson_content_from_quiz(self.quiz)
            print("Successfully generated LessonContent from existing quiz")
            return self.lesson_content
        
        # Otherwise, use the teacher agent with handoff to quiz creator
        print("Using teacher agent with handoff to quiz creator")
        teacher_agent = create_teacher_agent(self.context.vector_store_id) # No api_key
        
        # Generate the lesson content
        try:
            # RunConfig will be passed inside generate_lesson_content
            lesson_content_result = await generate_lesson_content(teacher_agent, self.lesson_plan, context=self.context)
        except Exception as e:
            print(f"ERROR: Teacher agent failed: {e}")
            if hasattr(self, 'output_logger') and self.output_logger:
                self.output_logger.log_error("Teacher Agent", e)
            # Re-raise for now, but consider creating a minimal lesson content
            raise
        
        # Store the result
        if isinstance(lesson_content_result, LessonContent):
            self.lesson_content = lesson_content_result
        elif isinstance(lesson_content_result, Quiz):
            self.quiz = lesson_content_result
            # Create synthetic lesson content from the quiz
            self.lesson_content = self._create_lesson_content_from_quiz(self.quiz)
        else:
            print(f"Warning: Teacher agent returned unexpected type: {type(lesson_content_result)}")
            
        return self.lesson_content
    
    async def generate_quiz(self) -> Optional[Quiz]:
        """Generate a quiz based on the lesson content."""
        if not self.lesson_content:
            print("Cannot generate quiz: Lesson content has not been generated yet")
            return None
        
        # Determine if we should use teacher handoff
        if self.quiz_feedback is not None:
            # If we have quiz feedback, we're regenerating after grading
            # So don't enable teacher handoff to avoid circular references
            enable_teacher_handoff = False
            print("Using quiz creator without teacher handoff since we have quiz feedback")
        else:
            # Normal flow - allow quiz creator to hand off back to teacher
            enable_teacher_handoff = True
        
        # No need for outer trace() if Runner.run uses RunConfig
        # with trace("Generating quiz"):
        print(f"Generating quiz...")

        try:
            from ai_tutor.agents.quiz_creator_agent import generate_quiz
            # RunConfig will be passed inside generate_quiz
            self.quiz = await generate_quiz(
                self.lesson_content, 
                enable_teacher_handoff=enable_teacher_handoff,
                context=self.context
            ) # No api_key
            print(f"Successfully generated quiz: {self.quiz.title}")
            print(f"  Questions: {len(self.quiz.questions)}")
            print(f"  Passing score: {self.quiz.passing_score}/{self.quiz.total_points}")
            return self.quiz
        except Exception as e:
            print(f"ERROR: Quiz creator agent failed: {e}")
            if hasattr(self, 'output_logger') and self.output_logger:
                self.output_logger.log_error("Quiz Creator Agent", e)
            # Already has fallback error handling
            print(f"Error generating quiz: {str(e)}")
            return None
    
    async def run_full_workflow_with_quiz_teacher(self, file_paths: List[str]) -> dict:
        """Run the full workflow from document upload to quiz creation and feedback.
        
        Args:
            file_paths: A list of file paths to upload for the lesson
            
        Returns:
            A dictionary containing the lesson plan, lesson content, quiz, 
            user's answers, and quiz feedback
        """
        # Record the session start time
        self._session_start_time = time.time()
        
        print("Running full workflow with quiz teacher...")
        
        # 1. Upload documents
        print("\n1. Uploading documents...")
        for file_path in file_paths:
            await self.upload_file(file_path)
        
        # Track workflow completion
        full_handoff_completed = False
        lesson_content_generated = False
        
        # 2. Generate the lesson plan (which may trigger handoff to teacher)
        try:
            print("\n2. Generating lesson plan...")
            self.lesson_plan = await self.generate_lesson_plan()
            
            # Check if we got a complete handoff chain with quiz
            if self.quiz is not None and hasattr(self.quiz, 'questions') and len(self.quiz.questions) > 0:
                print("Full handoff chain completed successfully! (Planner → Teacher → Quiz Creator)")
                print(f"Quiz was generated automatically through the handoff chain: {self.quiz.title}")
                print(f"Quiz has {len(self.quiz.questions)} questions")
                full_handoff_completed = True
            elif not self.lesson_plan:
                raise ValueError("Failed to generate a lesson plan")
        except Exception as e:
            print(f"Error generating lesson plan: {e}")
            # Create a minimal lesson plan
            self.lesson_plan = LessonPlan(
                title="Test Lesson Plan",
                description="This is a test lesson plan.",
                target_audience="Beginner learners",
                prerequisites=["Basic reading skills"],
                sections=[
                    LessonSection(
                        title="Introduction",
                        objectives=[
                            LearningObjective(
                                title="Learn basics",
                                description="Understand the fundamental concepts",
                                priority=5
                            )
                        ],
                        estimated_duration_minutes=15,
                        concepts_to_cover=["Basic concept"]
                    )
                ],
                total_estimated_duration_minutes=15,
                additional_resources=[]
            )
        
        # 3. Generate the lesson content
        try:
            print("\n3. Creating lesson content...")
            self.lesson_content = await self.generate_lesson_content()
            
            if self.lesson_content and hasattr(self.lesson_content, 'sections'):
                lesson_content_generated = True
            else:
                raise ValueError("Failed to generate valid lesson content with sections")
        except Exception as e:
            print(f"Error generating lesson content: {e}")
            # Create minimal lesson content
            # IMPORTANT: Make sure to use the correct field names as defined in the models.py file
            # For ExplanationContent model: use 'topic' (not 'title') and 'explanation' (not 'content')
            # For Exercise model: include required 'difficulty_level' and 'explanation' fields
            # Validation errors can occur if required fields are missing or field names don't match
            from ai_tutor.agents.models import LessonContent, SectionContent, ExplanationContent, Exercise
            self.lesson_content = LessonContent(
                title=self.lesson_plan.title,
                introduction="This is automatically generated test content.",
                sections=[
                    SectionContent(
                        title="Test Section",
                        introduction="Introduction to test section",
                        explanations=[
                            ExplanationContent(
                                topic="Test Explanation",
                                explanation="This is a test explanation.",
                                examples=["Example 1"]
                            )
                        ],
                        exercises=[
                            Exercise(
                                question="Test exercise question?",
                                answer="Test exercise answer.",
                                difficulty_level="Easy",
                                explanation="This is a test exercise explanation."
                            )
                        ],
                        summary="Summary of test section"
                    )
                ],
                conclusion="This concludes the test content.",
                next_steps=["Review the material"]
            )
        
        # Only generate quiz if we don't already have one from the handoff chain
        if not full_handoff_completed:
            # 4. Generate the quiz (without teacher handoff to prevent duplicate workflow)
            if not self.quiz or not hasattr(self.quiz, 'questions') or len(self.quiz.questions) == 0:
                try:
                    print("\n4. Creating quiz...")
                    self.quiz = await self.generate_quiz()
                    if not self.quiz or not hasattr(self.quiz, 'questions'):
                        raise ValueError("Failed to generate valid quiz with questions")
                except Exception as e:
                    print(f"Error generating quiz: {e}")
                    # Create a minimal quiz with at least one question
                    from ai_tutor.agents.models import Quiz, QuizQuestion
                    self.quiz = Quiz(
                        title=f"Quiz on {self.lesson_content.title}",
                        description="This is a test quiz.",
                        lesson_title=self.lesson_content.title,
                        questions=[
                            QuizQuestion(
                                question="What is the purpose of this quiz?",
                                options=["To test knowledge", "To test the quiz teacher", "To fail", "None of the above"],
                                correct_answer_index=1,
                                explanation="This quiz is designed to test the quiz teacher agent functionality.",
                                difficulty="Easy",
                                related_section="Test Section"
                            )
                        ],
                        passing_score=1,
                        total_points=1,
                        estimated_completion_time_minutes=5
                    )
            else:
                print(f"\n4. Using existing quiz: {self.quiz.title} ({len(self.quiz.questions)} questions)")
        else:
            print("\n4. Quiz already generated via handoff chain - skipping generation")
        
        # 5. Create sample quiz answers for demonstration
        try:
            user_answers = await self.create_sample_quiz_answers()
            if not user_answers:
                raise ValueError("Failed to create sample quiz answers")
        except Exception as e:
            print(f"Error creating sample answers: {e}")
            # Create minimal user answers
            from ai_tutor.agents.models import QuizUserAnswers, QuizUserAnswer
            user_answers = QuizUserAnswers(
                quiz_title=self.quiz.title,
                user_answers=[
                    QuizUserAnswer(
                        question_index=0,
                        selected_option_index=1,
                        time_taken_seconds=30
                    )
                ],
                total_time_taken_seconds=30
            )
        
        # Generate feedback on the answers using the quiz teacher
        try:
            print("\n5. Generating quiz feedback using quiz teacher agent...")
            # This directly uses the quiz teacher without handoff chain
            self.quiz_feedback = await self.submit_quiz_answers(user_answers)
            if not self.quiz_feedback:
                raise ValueError("Failed to generate quiz feedback")
        except Exception as e:
            print(f"Error generating feedback: {e}")
            # Create minimal feedback
            from ai_tutor.agents.models import QuizFeedback, QuizFeedbackItem
            self.quiz_feedback = QuizFeedback(
                quiz_title=self.quiz.title,
                total_questions=len(self.quiz.questions),
                correct_answers=1,
                score_percentage=100.0,
                passed=True,
                total_time_taken_seconds=user_answers.total_time_taken_seconds,
                feedback_items=[
                    QuizFeedbackItem(
                        question_index=0,
                        question_text=self.quiz.questions[0].question,
                        user_selected_option=self.quiz.questions[0].options[1],
                        is_correct=True,
                        correct_option=self.quiz.questions[0].options[1],
                        explanation=self.quiz.questions[0].explanation,
                        improvement_suggestion=""
                    )
                ],
                overall_feedback="Good job on the test quiz!",
                suggested_study_topics=["None needed"],
                next_steps=["Continue to the next lesson"]
            )
        
        # Prepare the output for the demo - handle case if lesson content wasn't created
        if not hasattr(self, 'lesson_content') or self.lesson_content is None:
            print("Creating placeholder lesson content for results display")
            self.lesson_content = self.quiz  # Use quiz for display purposes
            
        return {
            "lesson_plan": self.lesson_plan,
            "lesson_content": self.lesson_content,
            "quiz": self.quiz,
            "user_answers": user_answers,
            "quiz_feedback": self.quiz_feedback
        }

    async def submit_quiz_answers(self, user_answers: QuizUserAnswers) -> QuizFeedback:
        """Submit user answers to the quiz and get feedback."""
        if not self.quiz:
            raise ValueError("No quiz has been generated yet")
        
        # Log the raw user answers if we have a logger
        if hasattr(self, 'output_logger') and self.output_logger:
            self.output_logger.log_raw_user_answers(user_answers)
        
        # No need for outer trace() if Runner.run uses RunConfig
        # with trace("Quiz feedback generation"):
        print(f"Processing quiz answers...")
        print("Using quiz teacher agent directly (without handoff chain) to avoid duplicate workflows")
        
        # The generate_quiz_feedback function will use a fresh agent without handoffs
        try:
            # RunConfig will be passed inside generate_quiz_feedback
            self.quiz_feedback = await generate_quiz_feedback(self.quiz, user_answers, context=self.context) # Pass context
            return self.quiz_feedback
        except Exception as e:
            print(f"ERROR: Quiz teacher agent failed: {e}")
            if hasattr(self, 'output_logger') and self.output_logger:
                self.output_logger.log_error("Quiz Teacher Agent", e)
            # Re-raise for now, consider creating minimal feedback
            raise
            
    async def create_sample_quiz_answers(self) -> QuizUserAnswers:
        """Create sample quiz answers for testing purposes.
        
        This is useful for demonstrating the quiz feedback functionality.
        
        Returns:
            QuizUserAnswers object with sample answers
        """
        if not self.quiz or not self.quiz.questions:
            raise ValueError("No quiz with questions is available.")
        
        import random
        from ai_tutor.agents.models import QuizUserAnswers, QuizUserAnswer
        
        # Create answers for each question
        user_answers = []
        for i, question in enumerate(self.quiz.questions):
            # Randomly select an option (with 70% chance of being correct)
            if random.random() < 0.7:
                selected_option = question.correct_answer_index
            else:
                # Choose an incorrect option
                options = list(range(len(question.options)))
                options.remove(question.correct_answer_index)
                selected_option = random.choice(options) if options else 0
            
            # Random time between 10 and 60 seconds
            time_taken = random.randint(10, 60)
            
            user_answers.append(QuizUserAnswer(
                question_index=i,
                selected_option_index=selected_option,
                time_taken_seconds=time_taken
            ))
        
        # Calculate total time
        total_time = sum(answer.time_taken_seconds for answer in user_answers)
        
        return QuizUserAnswers(
            quiz_title=self.quiz.title,
            user_answers=user_answers,
            total_time_taken_seconds=total_time
        )
        
    async def upload_file(self, file_path: str) -> str:
        """Upload a single file to be used by the AI tutor.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            Result message
        """
        try:
            # Add to file paths for reference
            if file_path not in self.context.uploaded_file_paths:
                self.context.uploaded_file_paths.append(file_path)
                
            # Upload the file using the file upload manager
            uploaded_file = self.file_upload_manager.upload_and_process_file(file_path)
            
            # Store the vector store ID for later use
            self.context.vector_store_id = uploaded_file.vector_store_id
            
            return f"Successfully uploaded {uploaded_file.filename}"
        except Exception as e:
            return f"Error uploading {file_path}: {str(e)}"

    async def analyze_session(self, session_duration_seconds: int = None) -> SessionAnalysis:
        """Analyze the teaching session performance."""
        # Validate prerequisites
        if not self.lesson_plan:
            raise ValueError("No lesson plan has been generated yet")
        
        if not self.quiz_feedback:
            raise ValueError("No quiz feedback available - a completed quiz is required for analysis")
        
        # If session_duration_seconds is not provided, calculate it
        if session_duration_seconds is None:
            # Get the current time
            session_end_time = time.time()
            
            # If we have a start time, use it to calculate duration
            if hasattr(self, '_session_start_time') and self._session_start_time:
                session_duration_seconds = int(session_end_time - self._session_start_time)
            
        # No need for outer trace() if Runner.run uses RunConfig
        # with trace("Session analysis"):
        print(f"Analyzing teaching session...")
    
        try:
            # Create a simple QuizUserAnswers object from the quiz feedback
            user_answers = self.create_user_answers_from_feedback()
            
            # Get the raw agent outputs from the output logger if available
            raw_agent_outputs = {}
            if hasattr(self, 'output_logger') and self.output_logger:
                # Get the raw outputs for all the important agents
                raw_agent_outputs = self.output_logger.get_captured_outputs()
                
            # Create the session analyzer agent and analyze the session
            try:
                # RunConfig will be passed inside analyze_teaching_session
                self.session_analysis = await analyze_teaching_session(
                    document_analysis=self.document_analysis,
                    lesson_plan=self.lesson_plan,
                    lesson_content=self.lesson_content,
                    quiz=self.quiz,
                    user_answers=user_answers,  # Use the reconstructed user answers
                    quiz_feedback=self.quiz_feedback,
                    session_duration_seconds=session_duration_seconds,
                    raw_agent_outputs=raw_agent_outputs,  # Pass the raw outputs to the analyzer
                    context=self.context  # Pass context
                )
                
                return self.session_analysis
            except Exception as e:
                print(f"ERROR: Session analyzer agent failed: {e}")
                if hasattr(self, 'output_logger') and self.output_logger:
                    self.output_logger.log_error("Session Analyzer Agent", e)
                print(f"Error analyzing session: {str(e)}")
                return None
        except Exception as e:
            print(f"Error analyzing session: {str(e)}")
            return None

    def create_user_answers_from_feedback(self) -> QuizUserAnswers:
        """Create QuizUserAnswers object from quiz feedback.
        
        Returns:
            A QuizUserAnswers object reconstructed from feedback data
        """
        if not self.quiz_feedback:
            return None
        
        # Create a simple QuizUserAnswers object from the quiz feedback
        user_answers = QuizUserAnswers(
            quiz_title=self.quiz_feedback.quiz_title,
            total_time_taken_seconds=self.quiz_feedback.total_time_taken_seconds or 0,
            user_answers=[]
        )
        
        # Process feedback items to create user answers
        for item in self.quiz_feedback.feedback_items:
            if not hasattr(item, 'question_index') or item.question_index is None:
                # Skip items that don't have question index
                continue
                
            # Ensure the question index is valid
            if item.question_index < len(self.quiz.questions):
                question = self.quiz.questions[item.question_index]
                # Find the selected option index
                selected_option_index = 0  # Default to first option if not found
                
                # Try to find the matching option
                for i, option in enumerate(question.options):
                    if option == item.user_selected_option:
                        selected_option_index = i
                        break
                
                user_answer = QuizUserAnswer(
                    question_index=item.question_index,
                    selected_option_index=selected_option_index,
                    time_taken_seconds=0  # We don't have this information from the feedback
                )
                user_answers.user_answers.append(user_answer)
        
        return user_answers

    def _create_lesson_content_from_quiz(self, quiz: Quiz) -> LessonContent:
        """Create synthetic lesson content from a quiz object.
        
        Args:
            quiz: The quiz to create lesson content from
            
        Returns:
            A LessonContent object derived from the quiz
        """
        from ai_tutor.agents.models import LessonContent, SectionContent, ExplanationContent
        
        # Extract section titles from quiz questions
        section_titles = set()
        for question in quiz.questions:
            if hasattr(question, 'related_section') and question.related_section:
                section_titles.add(question.related_section)
        
        # If no sections found, use a default one
        if not section_titles:
            section_titles = ["Main Concepts"]
        
        # Create sections based on question topics
        sections = []
        for section_title in section_titles:
            # Find all questions for this section
            section_questions = [q for q in quiz.questions 
                               if hasattr(q, 'related_section') and q.related_section == section_title]
            
            # Create explanations from questions
            explanations = []
            for question in section_questions[:3]:  # Limit to first 3 questions per section
                # IMPORTANT: The ExplanationContent model requires specific fields:
                # - 'topic' (not 'title') for the explanation topic
                # - 'explanation' (not 'content') for the actual explanation text
                # - 'examples' list for examples
                # Make sure field names exactly match the model definition to prevent validation errors
                explanations.append(ExplanationContent(
                    topic=question.question.split('?')[0][:50] + "...",
                    explanation=f"This explains the concept of {question.question.split('?')[0][:30]}...",
                    examples=[f"Example: {question.options[question.correct_answer_index]}"]
                ))
            
            # Create section content
            sections.append(SectionContent(
                title=section_title,
                introduction=f"Introduction to {section_title}",
                explanations=explanations,
                exercises=[],
                summary=f"Summary of {section_title}"
            ))
        
        # Create lesson content from the quiz
        return LessonContent(
            title=quiz.lesson_title or "Lesson from Quiz",
            introduction=f"This lesson content was derived from the quiz '{quiz.title}'.",
            sections=sections,
            conclusion="See the quiz for assessment of learning objectives.",
            next_steps=[]
        )
