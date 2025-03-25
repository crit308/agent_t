import asyncio
import os
from typing import List, Optional, Any
import openai
import threading
import uuid

from agents import Runner, trace, gen_trace_id, set_tracing_export_api_key

from ai_tutor.tools.file_upload import FileUploadManager, upload_document
from ai_tutor.agents.planner_agent import create_planner_agent
from ai_tutor.agents.teacher_agent import generate_lesson_content, create_teacher_agent
from ai_tutor.agents.analyzer_agent import analyze_documents
from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent, generate_quiz, create_quiz_creator_agent_with_teacher_handoff
from ai_tutor.agents.quiz_teacher_agent import create_quiz_teacher_agent, generate_quiz_feedback
from ai_tutor.agents.models import LessonContent, Quiz, LessonPlan, LessonSection, LearningObjective, QuizUserAnswers, QuizFeedback, QuizUserAnswer, SectionContent, ExplanationContent, Exercise


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
        self.quiz_feedback = None
        self.file_paths = []
        self._analysis_task = None
        self._analysis_complete = asyncio.Event()
        self._current_trace_id = None
    
    def get_trace_id(self) -> Optional[str]:
        """Get the current trace ID for OpenAI tracing.
        
        Returns:
            The current trace ID or None if no trace is active
        """
        # For now, we'll generate a new trace ID each time
        # A more robust implementation would track the active trace ID
        return gen_trace_id()
    
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
        """Generate a lesson plan based on the uploaded documents.
        
        This method may trigger a handoff chain from planner → teacher → quiz creator,
        potentially resulting in a Quiz as the final output instead of a LessonPlan.
        """
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
            handoff_chain_completed = False
            
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
                    handoff_chain_completed = True
                    
                    # We also need to create a synthetic lesson plan
                    lesson_plan_title = self.quiz.title
                    if "quiz" in lesson_plan_title.lower():
                        lesson_plan_title = lesson_plan_title.replace("Quiz", "").replace("quiz", "").strip()
                    
                    # Create a minimal lesson plan for continuity
                    self.lesson_plan = LessonPlan(
                        title=lesson_plan_title,
                        description=self.quiz.description,
                        target_audience="Learners interested in this topic",
                        prerequisites=["Basic understanding of the subject"],
                        sections=[
                            LessonSection(
                                title="Main Topics",
                                objectives=[
                                    LearningObjective(
                                        title="Understanding Key Concepts",
                                        description="Master the fundamental concepts in this topic",
                                        priority=5
                                    )
                                ],
                                estimated_duration_minutes=30,
                                concepts_to_cover=["Key concepts from the quiz"]
                            )
                        ],
                        total_estimated_duration_minutes=30,
                        additional_resources=[]
                    )
                    
                elif isinstance(result.final_output, LessonContent):
                    print("Teacher handoff completed but no Quiz Creator handoff occurred")
                    self.lesson_content = result.final_output
                    print(f"Successfully captured LessonContent from handoff: {self.lesson_content.title}")
                    
                    # Create a synthetic lesson plan from the lesson content
                    print("Creating a lesson plan from the lesson content")
                    sections = []
                    for section in self.lesson_content.sections:
                        learning_objectives = []
                        for explanation in section.explanations:
                            # Create learning objective
                            learning_objectives.append(LearningObjective(
                                title="Learning " + explanation.topic,
                                description=f"Understand {explanation.topic}",
                                priority=5
                            ))
                        
                        # Create lesson section
                        sections.append(LessonSection(
                            title=section.title,
                            objectives=learning_objectives,
                            estimated_duration_minutes=30,  # Default estimate
                            concepts_to_cover=[obj.title for obj in learning_objectives]
                        ))
                    
                    # Create synthesized lesson plan
                    self.lesson_plan = LessonPlan(
                        title=self.lesson_content.title,
                        description=self.lesson_content.introduction[:100] + "...",  # Use introduction as description
                        target_audience="Learners interested in this topic",
                        prerequisites=["Basic understanding of the subject"],
                        sections=sections,
                        total_estimated_duration_minutes=len(sections) * 30,  # Rough estimate
                        additional_resources=[]
                    )
                    
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
            if result.last_agent and not handoff_chain_completed:
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
                        handoff_chain_completed = True
                    elif not self.quiz and isinstance(result.final_output, dict):
                        try:
                            self.quiz = Quiz(**result.final_output)
                            print("Successfully captured Quiz from handoff dictionary")
                            handoff_chain_completed = True
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
                                title="Learning " + explanation.topic,
                                description=f"Understand {explanation.topic}",
                                priority=5
                            ))
                        
                        # Create lesson section
                        sections.append(LessonSection(
                            title=section.title,
                            objectives=learning_objectives,
                            estimated_duration_minutes=30,  # Default estimate
                            concepts_to_cover=[obj.title for obj in learning_objectives]
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
                                title="Understanding " + topic[:20] + "...",
                                description=f"Understand {topic}",
                                priority=5
                            ))
                        
                        # Create a section with these learning objectives
                        sections.append(LessonSection(
                            title=section_title,
                            objectives=learning_objectives,
                            estimated_duration_minutes=len(questions) * 10,  # Rough estimate
                            concepts_to_cover=[obj.description for obj in learning_objectives]
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
    
    async def generate_lesson_content(self) -> Optional[LessonContent]:
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
                
                DO NOT SKIP the handoff to the quiz creator. After you've generated the lesson content,
                you MUST use the transfer_to_quiz_creator tool to pass the content to the Quiz Creator.
                """
                
                # Run the teacher agent to generate a lesson content
                result = await Runner.run(
                    teacher_agent, 
                    prompt
                )
                
                # Check the type of the final output first
                if isinstance(result.final_output, Quiz):
                    print("Teacher agent handoff completed the full chain to Quiz Creator")
                    print("Storing Quiz and creating a synthetic LessonContent")
                    # Store the quiz
                    self.quiz = result.final_output
                    # Create a minimal lesson content based on the quiz
                    self.lesson_content = LessonContent(
                        title=self.lesson_plan.title,
                        introduction=f"Content for {self.lesson_plan.title}.",
                        sections=[
                            SectionContent(
                                title="Topic Overview",
                                introduction="Overview of the main topics covered in this lesson.",
                                explanations=[
                                    ExplanationContent(
                                        topic="Key Concepts",
                                        explanation="The key concepts covered in this lesson.",
                                        examples=["Example 1", "Example 2"]
                                    )
                                ],
                                exercises=[
                                    Exercise(
                                        question="Test your understanding",
                                        difficulty_level="Medium",
                                        answer="This is a sample answer.",
                                        explanation="This is an explanation of the answer."
                                    )
                                ],
                                summary="Summary of the key points covered in this section."
                            )
                        ],
                        conclusion="Conclusion of the lesson.",
                        next_steps=["Next step 1", "Next step 2"]
                    )
                elif isinstance(result.final_output, LessonContent):
                    print("Received LessonContent from teacher agent")
                    self.lesson_content = result.final_output
                    
                    # Check if we need to create a quiz - only if one wasn't already created in a previous workflow
                    if not self.quiz:
                        print("No quiz was created via handoff. Generating quiz separately...")
                        # Generate the quiz separately
                        quiz_trace_id = gen_trace_id()
                        print(f"Generating quiz with trace ID: {quiz_trace_id}")
                        
                        with trace("Generating quiz", trace_id=quiz_trace_id):
                            # Import and call quiz creator directly instead of relying on handoff
                            from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent, generate_quiz
                            try:
                                # Use enable_teacher_handoff=False to prevent another handoff chain
                                self.quiz = await generate_quiz(self.lesson_content, self.api_key, enable_teacher_handoff=False)
                                print(f"Successfully generated quiz with {len(self.quiz.questions) if hasattr(self.quiz, 'questions') else 0} questions")
                            except Exception as quiz_error:
                                print(f"Error generating quiz: {quiz_error}")
                                self.quiz = None
                    else:
                        print(f"Using existing quiz from previous workflow: {self.quiz.title} ({len(self.quiz.questions)} questions)")
                else:
                    print(f"Unexpected output type from teacher agent: {type(result.final_output)}")
                    # Create a default lesson content
                    self.lesson_content = LessonContent(
                        title=self.lesson_plan.title,
                        introduction=f"Content for {self.lesson_plan.title}.",
                        sections=[],
                        conclusion="No detailed content available.",
                        next_steps=[]
                    )
                
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
    
    async def submit_quiz_answers(self, user_answers: QuizUserAnswers) -> QuizFeedback:
        """Submit user answers to the quiz and get feedback from the quiz teacher agent.
        
        Args:
            user_answers: The user's answers to the quiz questions
            
        Returns:
            QuizFeedback object containing detailed feedback on each answer
        """
        if not self.quiz:
            raise ValueError("No quiz is available. Generate a quiz first.")
        
        # Create and configure trace
        trace_id = gen_trace_id()
        print(f"Processing quiz answers with trace ID: {trace_id}")
        print(f"View trace at: https://platform.openai.com/traces/{trace_id}")
        
        # Ensure API key is set for tracing before creating the trace
        if self.api_key:
            set_tracing_export_api_key(self.api_key)
        
        print("Using quiz teacher agent directly (without handoff chain) to avoid duplicate workflows")
        with trace("Quiz feedback generation", trace_id=trace_id):
            # Generate feedback using the quiz teacher agent directly without enabling handoffs
            # This prevents triggering another handoff chain
            self.quiz_feedback = await generate_quiz_feedback(self.quiz, user_answers, self.api_key)
            return self.quiz_feedback
    
    async def create_sample_quiz_answers(self) -> QuizUserAnswers:
        """Create sample quiz answers for testing purposes.
        
        This is useful for demonstrating the quiz feedback functionality.
        
        Returns:
            QuizUserAnswers object with sample answers
        """
        if not self.quiz or not self.quiz.questions:
            raise ValueError("No quiz with questions is available.")
        
        import random
        
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
    
    async def update_quiz_creator_agent_with_teacher_handoff(self):
        """Update the quiz creator agent to include handoff to quiz teacher.
        
        This method modifies the quiz creator agent to hand off to the quiz teacher
        agent after creating a quiz.
        
        Returns:
            Updated quiz creator agent
        """
        # Create the quiz creator agent
        quiz_creator_agent = create_quiz_creator_agent(self.api_key)
        
        # Create the quiz teacher agent to hand off to
        quiz_teacher_agent = create_quiz_teacher_agent(self.api_key)
        
        # Define an on_handoff function that runs when quiz creator hands off to quiz teacher
        def on_handoff_to_quiz_teacher(ctx: RunContextWrapper[any], quiz: Quiz) -> None:
            print(f"Handoff triggered with quiz: {quiz.title}")
            print(f"Quiz has {len(quiz.questions)} questions")
            self.quiz = quiz  # Store the quiz for later use
        
        # Add handoff to the quiz creator agent
        quiz_creator_agent.handoffs.append(
            handoff(
                agent=quiz_teacher_agent,
                on_handoff=on_handoff_to_quiz_teacher,
                input_type=Quiz,
                input_filter=None,  # No need for a filter here
                tool_description_override="Transfer to the Quiz Teacher agent who will evaluate user responses and provide feedback. Provide the complete Quiz object as input."
            )
        )
        
        return quiz_creator_agent

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
                lesson_content = await self.generate_lesson_content()
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

    async def generate_quiz(self, enable_teacher_handoff: bool = False) -> Optional[Quiz]:
        """Generate a quiz based on the lesson content.
        
        Args:
            enable_teacher_handoff: Whether to enable handoff to the quiz teacher agent
            
        Returns:
            Quiz object or None if generation fails
        """
        if not self.lesson_content:
            raise ValueError("No lesson content has been generated yet")
        
        # If we already have a quiz from a previous handoff, return it
        if self.quiz and hasattr(self.quiz, 'questions') and len(self.quiz.questions) > 0:
            print("Using quiz that was already generated via handoff")
            print(f"Quiz has {len(self.quiz.questions)} questions")
            return self.quiz
        
        trace_id = gen_trace_id()
        print(f"Generating quiz with trace ID: {trace_id}")
        print(f"View trace at: https://platform.openai.com/traces/{trace_id}")
        
        # Ensure API key is set for tracing before creating the trace
        if self.api_key:
            set_tracing_export_api_key(self.api_key)
        
        with trace("Generating quiz", trace_id=trace_id):
            try:
                # Generate the quiz using the quiz creator agent
                quiz_result = await generate_quiz(
                    self.lesson_content, 
                    self.api_key, 
                    enable_teacher_handoff=enable_teacher_handoff
                )
                
                # Store the quiz for later reference
                if quiz_result and hasattr(quiz_result, 'questions'):
                    # Keep the existing quiz if it has questions but the new one doesn't
                    if len(quiz_result.questions) == 0 and self.quiz and hasattr(self.quiz, 'questions') and len(self.quiz.questions) > 0:
                        print(f"WARNING: New quiz has no questions, keeping existing quiz with {len(self.quiz.questions)} questions.")
                    else:
                        self.quiz = quiz_result
                        
                        # Basic validation to ensure the quiz has questions
                        if not hasattr(self.quiz, 'questions') or len(self.quiz.questions) == 0:
                            print("WARNING: Generated quiz has no questions.")
                        else:
                            print(f"Successfully generated quiz: {self.quiz.title}")
                            print(f"  Questions: {len(self.quiz.questions)}")
                            print(f"  Passing score: {self.quiz.passing_score}/{self.quiz.total_points}")
                elif quiz_result:
                    # We got something, but it doesn't have questions
                    print("WARNING: Quiz result doesn't have questions attribute")
                    self.quiz = quiz_result
                else:
                    print("WARNING: Quiz generation returned None")
                
                # If we still don't have a quiz with questions at this point, create a minimal one
                if not self.quiz or not hasattr(self.quiz, 'questions') or len(self.quiz.questions) == 0:
                    # Create a minimal quiz with a test question
                    from ai_tutor.agents.models import Quiz, QuizQuestion
                    self.quiz = Quiz(
                        title=f"Quiz on {self.lesson_content.title}",
                        description="Quiz for testing knowledge on the lesson content.",
                        lesson_title=self.lesson_content.title,
                        questions=[
                            QuizQuestion(
                                question="What is the main topic of this lesson?",
                                options=[
                                    f"Understanding {self.lesson_content.title}", 
                                    "Learning general knowledge", 
                                    "Testing the quiz system", 
                                    "None of the above"
                                ],
                                correct_answer_index=0,
                                explanation=f"This quiz is about {self.lesson_content.title}.",
                                difficulty="Easy",
                                related_section="Introduction"
                            )
                        ],
                        passing_score=1,
                        total_points=1,
                        estimated_completion_time_minutes=5
                    )
                    print(f"Created minimal quiz with 1 question since quiz generation didn't produce questions")
                
                return self.quiz
                
            except Exception as e:
                print(f"Error generating quiz: {e}")
                # Return a minimal quiz if something went wrong
                self.quiz = Quiz(
                    title=f"Quiz on {self.lesson_content.title}",
                    description="Error generating complete quiz.",
                    lesson_title=self.lesson_content.title,
                    questions=[
                        QuizQuestion(
                            question="What is the main topic of this lesson?",
                            options=[
                                f"Understanding {self.lesson_content.title}", 
                                "Learning general knowledge", 
                                "Testing the quiz system", 
                                "None of the above"
                            ],
                            correct_answer_index=0,
                            explanation=f"This quiz is about {self.lesson_content.title}.",
                            difficulty="Easy",
                            related_section="Introduction"
                        )
                    ],
                    passing_score=1,
                    total_points=1,
                    estimated_completion_time_minutes=5
                )
                return self.quiz

    async def run_full_workflow_with_quiz_teacher(self, file_paths: List[str]) -> dict:
        """Run the full workflow from document upload to quiz creation and feedback.
        
        Args:
            file_paths: List of paths to documents to upload
            
        Returns:
            Dictionary with lesson plan, lesson content, quiz, user answers, and quiz feedback
        """
        print("Running full workflow with quiz teacher...")
        
        # 1. Upload documents
        print("\n1. Uploading documents...")
        for file_path in file_paths:
            await self.upload_file(file_path)
        
        # Track whether we've completed a full handoff chain
        full_handoff_completed = False
        
        # 2. Generate the lesson plan (which may trigger handoffs)
        try:
            print("\n2. Generating lesson plan...")
            self.lesson_plan = await self.generate_lesson_plan()
            
            # Check if we got a complete handoff chain
            if self.quiz is not None and hasattr(self.quiz, 'questions') and len(self.quiz.questions) > 0:
                print("Full handoff chain detected (planner → teacher → quiz creator)")
                print(f"Using quiz that was generated during handoff chain: {self.quiz.title}")
                print(f"Quiz has {len(self.quiz.questions)} questions")
                full_handoff_completed = True
            elif not self.lesson_plan:
                raise ValueError("Failed to generate a lesson plan")
        except Exception as e:
            print(f"Error generating lesson plan: {e}")
            # Create a minimal lesson plan
            from ai_tutor.agents.models import LessonPlan, LessonSection, LearningObjective
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
        
        # Only generate lesson content and quiz if they weren't already created by the handoff chain
        if not full_handoff_completed:
            # Save the current state of quiz before generating lesson content
            previous_quiz = self.quiz
            
            # 3. Generate the lesson content from the lesson plan
            try:
                print("\n3. Creating lesson content...")
                self.lesson_content = await self.generate_lesson_content()
                
                # If a new quiz was created during lesson content generation but we had a previous quiz,
                # restore the previous quiz to avoid overwriting with a different one
                if previous_quiz is not None and self.quiz != previous_quiz and hasattr(previous_quiz, 'questions') and len(previous_quiz.questions) > 0:
                    print(f"Restoring original quiz from first workflow: {previous_quiz.title} ({len(previous_quiz.questions)} questions)")
                    self.quiz = previous_quiz
                
                if not self.lesson_content or not hasattr(self.lesson_content, 'sections'):
                    raise ValueError("Failed to generate valid lesson content with sections")
            except Exception as e:
                print(f"Error generating lesson content: {e}")
                # Create minimal lesson content
                from ai_tutor.agents.models import LessonContent, SectionContent, ExplanationContent, Exercise
                self.lesson_content = LessonContent(
                    title=self.lesson_plan.title,
                    introduction="This is automatically generated test content.",
                    sections=[
                        SectionContent(
                            title="Test Section",
                            explanations=[
                                ExplanationContent(
                                    title="Test Explanation",
                                    content="This is a test explanation.",
                                    examples=["Example 1"]
                                )
                            ],
                            exercises=[
                                Exercise(
                                    question="Test exercise question?",
                                    answer="Test exercise answer."
                                )
                            ]
                        )
                    ],
                    conclusion="This concludes the test content.",
                    next_steps=["Review the material"]
                )
            
            # 4. Generate the quiz with teacher handoff disabled to prevent duplicate workflow
            # Only generate quiz if we don't already have one
            if not self.quiz or not hasattr(self.quiz, 'questions') or len(self.quiz.questions) == 0:
                try:
                    print("\n4. Creating quiz...")
                    # Use enable_teacher_handoff=False to prevent triggering another workflow
                    self.quiz = await self.generate_quiz(enable_teacher_handoff=False)
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
            print("\n3. Lesson content already created via handoff chain")
            print("\n4. Quiz already generated via handoff chain")

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
        
        # Generate feedback on the answers
        try:
            print("Generating quiz feedback using quiz teacher agent...")
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

    async def upload_file(self, file_path: str) -> str:
        """Upload a single file to be used by the AI tutor.
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            Result message
        """
        try:
            # Add to file paths for reference
            if file_path not in self.file_paths:
                self.file_paths.append(file_path)
                
            # Upload the file using the file upload manager
            uploaded_file = self.file_upload_manager.upload_and_process_file(file_path)
            
            # Store the vector store ID for later use
            self.vector_store_id = uploaded_file.vector_store_id
            
            return f"Successfully uploaded {uploaded_file.filename}"
        except Exception as e:
            return f"Error uploading {file_path}: {str(e)}" 