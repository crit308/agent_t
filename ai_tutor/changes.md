1. Update Data Models (ai_tutor/agents/models.py)

We need a new model for the Planner's output (FocusObjective) and potentially result models for agents called as tools.

--- a/ai_tutor/agents/models.py
+++ b/ai_tutor/agents/models.py
@@ -4,7 +4,7 @@
 
 # Custom JSON encoder to control floating point precision
 class PrecisionControlEncoder(json.JSONEncoder):
-    """Custom JSON encoder that ensures floating point values don't exceed 8 decimal places."""
+    """DEPRECATED: Use RoundingModelWrapper. Custom JSON encoder for float precision."""
     
     def __init__(self, *args, **kwargs):
         # Remove our custom parameter if present
@@ -78,6 +78,21 @@
     text: str = Field(description="The full text content of the lesson")
 
 
+# --- NEW: Focus Objective from Planner ---
+class FocusObjective(BaseModel):
+    """The next learning focus identified by the Planner Agent."""
+    topic: str = Field(description="The primary topic or concept to focus on.")
+    learning_goal: str = Field(description="A specific, measurable goal for this focus area (e.g., 'Understand local vs global scope', 'Solve quadratic equations').")
+    priority: int = Field(description="Priority from 1-5 (5=highest) based on prerequisites or user need.")
+    relevant_concepts: List[str] = Field(default_factory=list, description="List of related concepts from the knowledge base.")
+    suggested_approach: Optional[str] = Field(None, description="Optional hint for the Orchestrator (e.g., 'Needs examples', 'Requires practice quiz').")
+
+# --- NEW: Potential Result Models for Agents as Tools ---
+class ExplanationResult(BaseModel):
+    """Result returned by the Teacher agent tool."""
+    status: Literal["delivered", "failed", "skipped"]
+    details: Optional[str] = None
+
 class QuizQuestion(BaseModel):
     """A single quiz question with options and explanation."""
     question: str = Field(description="The question text")
@@ -87,6 +102,12 @@
     difficulty: str = Field(description="Easy, Medium, or Hard")
     related_section: str = Field(description="Title of the lesson section this question relates to")
 
+# --- NEW: Potential Result Models for Agents as Tools ---
+class QuizCreationResult(BaseModel):
+    """Result returned by the Quiz Creator agent tool."""
+    status: Literal["created", "failed"]
+    quiz: Optional[Quiz] = None # Could contain the full quiz if multiple questions requested
+    question: Optional[QuizQuestion] = None # Could contain a single question
+    details: Optional[str] = None
 
 class Quiz(BaseModel):
     """A complete quiz generated based on lesson content."""


2. Update Tutor Context (ai_tutor/context.py)

Add the current_focus_objective field.

--- a/ai_tutor/context.py
+++ b/ai_tutor/context.py
@@ -15,7 +15,7 @@
 
 # Use TYPE_CHECKING to prevent runtime circular imports for type hints
 if TYPE_CHECKING:
-    from ai_tutor.agents.models import LessonPlan, QuizQuestion, LearningObjective
+    from ai_tutor.agents.models import LessonPlan, QuizQuestion, LearningObjective, FocusObjective # Import FocusObjective
     from ai_tutor.agents.analyzer_agent import AnalysisResult
 
 class UserConceptMastery(BaseModel):
@@ -59,6 +59,7 @@
     knowledge_base_path: Optional[str] = None # Add path to KB file
     lesson_plan: Optional['LessonPlan'] = None # Use forward reference
     current_quiz_question: Optional['QuizQuestion'] = None # Use forward reference
+    current_focus_objective: Optional['FocusObjective'] = None # NEW: Store the current focus from Planner
     user_model_state: UserModelState = Field(default_factory=UserModelState)
     last_interaction_summary: Optional[str] = None # What did the tutor just do? What did user respond?
     current_teaching_topic: Optional[str] = None # Which topic is the Teacher actively explaining?
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Diff
IGNORE_WHEN_COPYING_END

3. Refactor Planner Agent (ai_tutor/agents/planner_agent.py)

Change output type and instructions to focus identification.

--- a/ai_tutor/agents/planner_agent.py
+++ b/ai_tutor/agents/planner_agent.py
@@ -7,7 +7,7 @@
 from agents.models.openai_provider import OpenAIProvider
 from agents.run_context import RunContextWrapper
 
-from ai_tutor.agents.models import LearningObjective, LessonSection, LessonPlan, QuizQuestion
+from ai_tutor.agents.models import FocusObjective # Import FocusObjective, remove LessonPlan etc.
 from ai_tutor.agents.utils import RoundingModelWrapper
 from ai_tutor.context import TutorContext
 import os
@@ -50,7 +50,7 @@
         return error_msg
 # -----------------------------------------------
 
-def create_planner_agent(vector_store_id: str) -> Agent[TutorContext]:
+def create_planner_agent(vector_store_id: str) -> Agent[TutorContext, FocusObjective]: # Change return type annotation
     """Creates a planner agent that can search through files and create a lesson plan."""
     
     # Create a FileSearchTool that can search the vector store containing the uploaded documents
@@ -68,36 +68,32 @@
     provider: OpenAIProvider = OpenAIProvider()
     base_model = provider.get_model("o3-mini")
 
-    # Create the planner agent with access to the file search tool
-    planner_agent = Agent[TutorContext](
+    # Create the planner agent focusing on identifying the next focus
+    planner_agent = Agent[TutorContext, FocusObjective]( # Change agent generic type
         name="Lesson Planner",
-        instructions="""You are an expert curriculum designer. Your task is to create a well-structured lesson plan based on analyzed documents.
+        instructions="""You are an expert learning strategist. Your task is to determine the user's **next learning focus** based on the analyzed documents and potentially their current progress (provided in the prompt context).
 
         AVAILABLE INFORMATION:
         - You have a `read_knowledge_base` tool to get the document analysis summary stored in the database.
         - You have a `file_search` tool to look up specific details within the source documents (vector store).
+        - The prompt may contain information about the user's current state (`UserModelState` summary).
 
         YOUR WORKFLOW **MUST** BE:
-        1.  **Read Knowledge Base ONCE:** Call the `read_knowledge_base` tool *exactly one time* at the beginning to get the document analysis summary.
+        1.  **Read Knowledge Base ONCE:** Call the `read_knowledge_base` tool *exactly one time* at the beginning to get the document analysis summary (key concepts, terms, etc.).
         2.  **Confirm KB Received & Analyze Summary:** Once you have the Knowledge Base summary from the tool, **DO NOT call `read_knowledge_base` again**. Analyze the KB and any provided user state summary.
-        3.  **Use `file_search` ONLY if Necessary:** If, *after analyzing the full KB summary*, you lack specific details (like examples or steps) needed for a section, use `file_search` sparingly to find that information in the source documents. Do NOT use `file_search` for information already present in the KB summary.
-        4.  **Create Lesson Plan:** Synthesize information from the KB analysis and any necessary `file_search` results to create a complete `LessonPlan` object.
-        - For each `LessonSection`, you MUST include:
-          * Clear learning objectives for each section
-          * Logical sequence of sections
-          * Appropriate time durations for each section
-          * Consideration of prerequisites
-          * Target audience
-          * `prerequisites`: A list of concept/section titles that must be understood *before* this section. Leave empty if none.
-          * `is_optional`: A boolean indicating if the section covers core material (False) or is supplementary/advanced (True). Infer this based on the content's nature (e.g., introductory sections are rarely optional).
-          * Ensure `concepts_to_cover` clearly relates to the `objectives` for that section.
+        3.  **Identify Next Focus:** Determine the single most important topic or concept the user should learn next. Consider prerequisites implied by the KB structure and the user's current state (e.g., last completed topic, identified struggles).
+        4.  **Define Learning Goal:** Formulate a clear, specific learning goal for this focus topic.
+        5.  **Use `file_search` Sparingly:** If needed to clarify the goal or identify crucial related concepts for the chosen focus topic, use `file_search`.
 
-        STEP 4: OUTPUT
-        - Output the lesson plan as a complete structured LessonPlan object.
+        OUTPUT:
+        - Output the determined focus as a structured `FocusObjective` object. This object MUST contain:
+            * `topic`: The main topic to focus on.
+            * `learning_goal`: The specific objective for this topic.
+            * `priority`: An estimated priority (1-5).
+            * `relevant_concepts`: Key concepts from the KB related to this topic.
+            * `suggested_approach`: (Optional) A hint for the Orchestrator.
 
         CRITICAL REMINDERS:
         - **You MUST call `read_knowledge_base` only ONCE at the very start.**
-        - DO NOT call any handoff tools. Your only output should be the LessonPlan object.
+        - Your only output MUST be a single `FocusObjective` object. Do NOT create a full `LessonPlan`.
         """,
         tools=planner_tools,
-        output_type=LessonPlan,
+        output_type=FocusObjective, # Change output type
         model=RoundingModelWrapper(base_model),
     )
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Diff
IGNORE_WHEN_COPYING_END

4. Refactor Teacher Agent (ai_tutor/agents/teacher_agent.py)

Make it callable as a tool and return a structured result.

--- a/ai_tutor/agents/teacher_agent.py
+++ b/ai_tutor/agents/teacher_agent.py
@@ -9,21 +9,23 @@
 from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
 from agents.run_context import RunContextWrapper
 
-from ai_tutor.agents.models import LessonPlan, LessonSection, LessonContent
+# Import models needed for the *new* Teacher agent tool
+from ai_tutor.agents.models import ExplanationResult # Import result type
 from typing import List, Callable, Optional, Any, Dict, TYPE_CHECKING, Union
-from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
+# Remove old imports no longer used
+# from ai_tutor.agents.models import LessonPlan, LessonSection, LessonContent
+# from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent
 from ai_tutor.agents.utils import process_handoff_data, RoundingModelWrapper
 
 # Import the new tools
-from ai_tutor.tools.teacher_tools import present_explanation, ask_checking_question
+# Remove imports for tools that are now internal logic or outputs
+# from ai_tutor.tools.teacher_tools import present_explanation, ask_checking_question
 # Import necessary response types for the agent's output_type
-from ai_tutor.api_models import ExplanationResponse, QuestionResponse, MessageResponse, ErrorResponse # Add ErrorResponse if needed
+# Remove API response imports - the agent tool returns ExplanationResult
+# from ai_tutor.api_models import ExplanationResponse, QuestionResponse, MessageResponse, ErrorResponse
 
 if TYPE_CHECKING:
     from ai_tutor.context import TutorContext
-
-
-# The Teacher Agent's final output for a *single interaction turn* will be
-# one of the responses generated by its tools.
-TeacherInteractionOutput = Union[
-    ExplanationResponse,
-    QuestionResponse,
-    # MessageResponse, # Uncomment if you add more tools like prompt_for_summary
-    # ErrorResponse # Uncomment if the agent needs to signal an internal error
-]
 
 def create_interactive_teacher_agent(vector_store_id: str) -> Agent['TutorContext']:
     """Creates an INTERACTIVE Teacher Agent."""
@@ -31,7 +33,7 @@
     provider: OpenAIProvider = OpenAIProvider()
     # Maybe use a slightly more capable model for interactive logic
     base_model = provider.get_model("gpt-4o-2024-08-06")
-
+    
     file_search_tool = FileSearchTool(
         vector_store_ids=[vector_store_id],
         max_num_results=3 # Fewer results might be needed for focused segment explanation
@@ -39,49 +41,39 @@
 
     # Define the tools the *teacher itself* can use
     teacher_tools = [
-        file_search_tool,
-        present_explanation,
-        ask_checking_question,
-        # prompt_for_summary # Add if implemented
+        file_search_tool, # Still needs file search for content generation
+        # The agent's task *is* to explain, so 'present_explanation' is not a tool it calls, but its output goal.
+        # Similarly, 'ask_checking_question' is an *action* it takes, represented by its output, not a tool it calls.
     ]
 
-    teacher_agent = Agent['TutorContext'](
+    teacher_agent = Agent['TutorContext', ExplanationResult]( # Agent now returns ExplanationResult
         name="InteractiveLessonTeacher",
         # Instructions focus on the interactive loop for *one topic*
         instructions="""
-        You are an interactive AI teacher responsible for explaining a specific topic segment by segment and checking user understanding.
+        You are an expert AI teacher responsible for explaining a specific topic or concept clearly.
 
         YOUR CONTEXT:
-        - You will be given the `current_teaching_topic` and the `current_topic_segment_index` to focus on via the RunContext.
-        - You may also receive the user's last response if you previously asked a question (`pending_interaction_type` will be set).
-        - Use the `file_search` tool *only* if necessary to get specific details for the *current segment* you need to explain.
+        - You will be given instructions via the prompt about **what specific topic/concept to explain** and potentially **what aspect to focus on** (e.g., "explain 'variable scope' focusing on local vs global", "provide an example of closures").
+        - Use the `file_search` tool *only* if necessary to retrieve specific information or examples related to the requested explanation from the provided documents.
 
-        YOUR INTERACTIVE WORKFLOW FOR A TOPIC:
-        1.  **Check for Pending Interaction:** If the context indicates you were waiting for a user response (e.g., to a checking question), evaluate that response first.
-        2.  **Decide Next Micro-Step:** Based on the current segment index and any user response evaluation:
-            *   If starting a topic (segment 0) or continuing after successful interaction: Explain the *next* segment.
-            *   If user struggled with a previous check: Re-explain the *previous* segment differently or provide an example.
-            *   After explaining a few segments: Ask a checking question.
-            *   If the explanation for the topic is complete: Indicate this (e.g., set `is_last_segment=True` in `present_explanation`).
-        3.  **Execute Micro-Step using Tools:**
-            *   **To Explain:** Call the `present_explanation` tool with the text for the *current segment*, the `segment_index`, and whether it's the `is_last_segment` for this topic. **This ends your turn.**
-            *   **To Check Understanding:** Call the `ask_checking_question` tool with a relevant `QuizQuestion` object focusing on the recently explained content. **This ends your turn.**
-            *   **(Optional) To Prompt Summary:** Call `prompt_for_summary` tool. **This ends your turn.**
-        4.  **Your final action MUST be a call to ONE of the available tools (`present_explanation`, `ask_checking_question`).** The result of that tool call will be the output relayed to the user.
+        YOUR TASK:
+        1.  Understand the specific explanation request from the prompt.
+        2.  If necessary, use `file_search` to gather precise details or examples.
+        3.  Synthesize the information into a clear, concise, and accurate explanation tailored to the request.
+        4.  Format your final output ONLY as an `ExplanationResult` object.
+            *   Set `status` to "delivered".
+            *   Put the full explanation text in the `details` field.
 
         **CRITICAL:**
-        - Focus ONLY on the `current_teaching_topic` and `current_topic_segment_index`.
-        - Your goal is to take ONE step in the interactive loop (explain or ask).
-        - **You MUST end your turn by calling exactly ONE interaction tool (`present_explanation`, `ask_checking_question`). Do not call multiple interaction tools in a single turn.**
+        - Focus ONLY on generating the requested explanation.
+        - Do NOT ask follow-up questions or try to manage a conversation.
+        - Your final output MUST be a single, valid `ExplanationResult` JSON object.
         """,
         tools=teacher_tools,
-        # The output type IS the *result* of one of its interaction tools
-        output_type=TeacherInteractionOutput,
+        output_type=ExplanationResult, # Set the output type
         model=base_model,
-        tool_use_behavior="stop_on_first_tool", # Force stop after one tool call
+        # tool_use_behavior="stop_on_first_tool", # Keep default or adjust as needed
         # No handoffs needed FROM the teacher in this model
     )
     return teacher_agent
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Diff
IGNORE_WHEN_COPYING_END

5. Refactor Quiz Creator Agent (ai_tutor/agents/quiz_creator_agent.py)

Make it callable as a tool and return a structured result.

--- a/ai_tutor/agents/quiz_creator_agent.py
+++ b/ai_tutor/agents/quiz_creator_agent.py
@@ -7,9 +7,10 @@
 from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
 
 from ai_tutor.agents.models import LessonContent, Quiz
+# Import new result model and QuizQuestion if needed for single question output
+from ai_tutor.agents.models import QuizCreationResult, QuizQuestion
 from ai_tutor.agents.utils import process_handoff_data, RoundingModelWrapper
-from ai_tutor.agents.quiz_teacher_agent import create_quiz_teacher_agent
-
+# Remove import for quiz teacher handoff
 
 def quiz_to_teacher_handoff_filter(handoff_data: HandoffInputData) -> HandoffInputData:
     """Process handoff data from quiz creator to quiz teacher."""
@@ -22,7 +23,7 @@
         print(f"ERROR in quiz_to_teacher_handoff_filter: {e}")
         return handoff_data  # Fallback
 
-
+# Keep this function if you still need a standalone quiz creator elsewhere, but it won't be used by the Orchestrator tool.
 def create_quiz_creator_agent(api_key: str = None):
     """Create a basic quiz creator agent without handoff capability."""
     
@@ -72,7 +73,7 @@
          YOUR OUTPUT MUST BE ONLY A VALID QUIZ OBJECT.
         """,
         output_type=Quiz,
-        model=RoundingModelWrapper(base_model),
+        model=RoundingModelWrapper(base_model), # Consider using gpt-4o if o3-mini struggles
     )
     
     return quiz_creator_agent
@@ -90,9 +91,6 @@
     else:
         print(f"Using OPENAI_API_KEY from environment for quiz creator agent")
     
-    # Create the quiz teacher agent to hand off to
-    quiz_teacher_agent = create_quiz_teacher_agent(api_key)
-    
     # Define an on_handoff function for when quiz creator hands off to quiz teacher
     async def on_handoff_to_quiz_teacher(ctx: RunContextWrapper[any], quiz: Quiz) -> None:
         print(f"Handoff triggered from quiz creator to quiz teacher: {quiz.title}")
@@ -100,14 +98,14 @@
     
     # Instantiate the base model provider and get the base model
     provider: ModelProvider = OpenAIProvider()
-    base_model = provider.get_model("o3-mini")
+    base_model = provider.get_model("gpt-4o") # Use gpt-4o for potentially better structured output
     
     # Create the quiz creator agent
-    quiz_creator_agent = Agent(
+    quiz_creator_agent = Agent[Any, QuizCreationResult]( # Agent now returns QuizCreationResult
         name="Quiz Creator",
         instructions=prompt_with_handoff_instructions("""
         You are an expert educational assessment designer specialized in creating effective quizzes.
-        Your task is to create a comprehensive quiz based on the lesson content provided to you.
+        Your task is to create quiz questions based on the specific instructions provided in the prompt (e.g., topic, number of questions, difficulty).
         
         Guidelines for creating effective quiz questions:
         1. Create a mix of easy, medium, and hard questions that cover all key concepts from the lesson
@@ -118,8 +116,8 @@
         6. Target approximately 2-3 questions per lesson section
         
         CRITICAL REQUIREMENTS:
-        1. You MUST create at least 5 questions for the quiz, even if the lesson content is minimal
-        2. Each question MUST have exactly 4 multiple-choice options
+        1. Follow the instructions in the prompt regarding the number of questions. If asked for ONE question, create only ONE.
+        2. Each multiple-choice question MUST have exactly 4 options unless specified otherwise.
         3. Set an appropriate passing score (typically 70% of total points)
         4. Ensure total_points equals the number of questions
         5. Set a reasonable estimated_completion_time_minutes (typically 1-2 minutes per question)
@@ -138,23 +136,22 @@
           * passing_score: Integer (minimum points to pass)
           * total_points: Integer (total possible points)
           * estimated_completion_time_minutes: Integer (estimated time to complete)
-        
-        YOUR OUTPUT MUST BE ONLY A VALID QUIZ OBJECT.
-        
-        After generating the quiz, use the transfer_to_quiz_teacher tool to hand off to the Quiz Teacher agent
-        which will evaluate user responses and provide feedback.
+
+        FORMAT REQUIREMENTS FOR YOUR OUTPUT:
+        - Your final output MUST be a single, valid `QuizCreationResult` JSON object.
+        - If successful, set `status` to "created".
+        - If you created a single question, put the `QuizQuestion` object in the `question` field.
+        - If you created multiple questions, put the full `Quiz` object in the `quiz` field.
+        - If failed, set `status` to "failed" and provide details.
+
+        Do NOT ask follow-up questions or manage conversation flow. Just create the requested quiz/question(s).
         """),
         handoffs=[
-            handoff(
-                agent=quiz_teacher_agent,
-                on_handoff=on_handoff_to_quiz_teacher,
-                input_type=Quiz,
-                input_filter=quiz_to_teacher_handoff_filter,
-                tool_description_override="Transfer to the Quiz Teacher agent who will evaluate user responses and provide feedback. Provide the complete Quiz object as input."
-            )
+            # Remove handoff - Orchestrator calls Quiz Teacher tool directly if needed
         ],
-        output_type=Quiz,
-        model=RoundingModelWrapper(base_model),
+        output_type=QuizCreationResult, # Change output type
+        model=base_model, # Use the base model directly
+        # model=RoundingModelWrapper(base_model), # Use wrapper if needed
     )
     
     return quiz_creator_agent
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Diff
IGNORE_WHEN_COPYING_END

6. Refactor Orchestrator Tools (ai_tutor/tools/orchestrator_tools.py)

Remove deprecated tools, add new tools to call other agents.

--- a/ai_tutor/tools/orchestrator_tools.py
+++ b/ai_tutor/tools/orchestrator_tools.py
@@ -9,14 +9,17 @@
 from ai_tutor.context import TutorContext, UserConceptMastery, UserModelState
 
 # --- Import necessary models ---
-from ai_tutor.agents.models import LessonPlan, QuizQuestion, QuizFeedbackItem, LessonContent, Quiz, LessonSection, LearningObjective
+# Models needed by the tools themselves or for type hints
+from ai_tutor.agents.models import FocusObjective, QuizQuestion, QuizFeedbackItem, ExplanationResult, QuizCreationResult # Import new models
+# from ai_tutor.agents.models import LessonPlan, LessonContent, Quiz, LessonSection, LearningObjective # Remove unused models
 
 # Import API response models for potentially formatting tool output
 from ai_tutor.api_models import (
     ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse
 )
 
+
 # --- Orchestrator Tool Implementations ---
 
 @function_tool
@@ -24,91 +27,13 @@
     ctx: RunContextWrapper[TutorContext],
     topic: str
 ) -> Union[QuizQuestion, str]: # Return Question object or error string
-    """Generates a single multiple-choice question for the given topic using the Quiz Creator agent."""
-    print(f"[Tool] Requesting mini-quiz for topic '{topic}'")
-    
-    if not topic or not isinstance(topic, str):
-        return "Error: Invalid topic provided for quiz."
+    """DEPRECATED: Use call_quiz_creator_agent instead. Generates a single multiple-choice question."""
+    return "Error: This tool is deprecated. Use call_quiz_creator_agent to invoke the quiz creator."
 
-    try:
-        # --- Import and Create Agent *Inside* ---
-        from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent # Import here
-        # ----------------------------------------
-        run_config = RunConfig(workflow_name="Orchestrator_QuizCall", group_id=ctx.context.session_id)
-
-        # Get user's mastery level for difficulty adjustment
-        mastery_level = 0.0
-        if topic in ctx.context.user_model_state.concepts:
-            mastery_level = ctx.context.user_model_state.concepts[topic].mastery_level
-
-        # Get confusion points if any
-        confusion_points = []
-        if topic in ctx.context.user_model_state.concepts:
-            confusion_points = ctx.context.user_model_state.concepts[topic].confusion_points
-
-        # Construct the prompt
-        confusion_points_str = "\nFocus on these areas of confusion:\n- " + "\n- ".join(confusion_points) if confusion_points else ""
-        difficulty_guidance = "Make the question more challenging." if mastery_level > 0.7 else "Keep the question straightforward." if mastery_level < 0.3 else ""
-
-        quiz_prompt = f"""
-        Create a single multiple-choice question to test understanding of: {topic}
-        
-        Current mastery level: {mastery_level:.2f}
-        {difficulty_guidance}
-        {confusion_points_str}
-        
-        Format your response as a Quiz object with a single question that includes:
-        - question text
-        - multiple choice options
-        - correct answer index
-        - explanation for the correct answer
-        - related_section: '{topic}'
-        """
-
-        result = await Runner.run(
-            quiz_creator,
-            quiz_prompt,
-            context=ctx.context,
-            run_config=run_config
-        )
-
-        quiz_output = result.final_output_as(Quiz)
-        if quiz_output and quiz_output.questions:
-            first_question = quiz_output.questions[0]
-            print(f"[Tool] Got mini-quiz for '{topic}': {first_question.question[:50]}...")
-            ctx.context.current_quiz_question = first_question
-            return first_question
-        else:
-            return f"Error: Quiz creator failed to generate question for topic '{topic}'."
-
-    except Exception as e:
-        error_msg = f"Error in call_quiz_creator_mini: {str(e)}"
-        print(f"[Tool] {error_msg}")
-        return error_msg
-
-
 @function_tool
 async def call_quiz_teacher_evaluate(ctx: RunContextWrapper[TutorContext], user_answer_index: int) -> Union[QuizFeedbackItem, str]:
     """Evaluates the user's answer to the current question using the Quiz Teacher logic (via helper function)."""
-    print(f"[Tool] Evaluating user answer index '{user_answer_index}' for current question.")
-    
+    print(f"[Tool call_quiz_teacher_evaluate] Evaluating user answer index '{user_answer_index}'.")
     try:
         # --- Import evaluation function *Inside* ---
         from ai_tutor.agents.quiz_teacher_agent import evaluate_single_answer # Import helper here
@@ -138,12 +63,12 @@
             return feedback_item
         else:
             # The evaluate_single_answer helper should ideally return feedback or raise
-            error_msg = f"Error: Evaluation helper function returned None for question on topic '{getattr(question_to_evaluate, 'related_section', 'N/A')}'."
+            error_msg = f"Error: Evaluation failed for question on topic '{getattr(question_to_evaluate, 'related_section', 'N/A')}'."
             print(f"[Tool] {error_msg}")
             return error_msg # Return error string
             
     except Exception as e:
-        error_msg = f"Error in call_quiz_teacher_evaluate: {str(e)}"
+        error_msg = f"Exception in call_quiz_teacher_evaluate: {str(e)}"
         print(f"[Tool] {error_msg}")
         # Optionally clear pending state even on error? Depends on desired flow.
         # ctx.context.user_model_state.pending_interaction_type = None 
@@ -151,61 +76,10 @@
         return error_msg # Return error string
 
 @function_tool
-def determine_next_learning_step(ctx: RunContextWrapper[TutorContext], lesson_plan: LessonPlan = None) -> Dict[str, Any]:
-    """Determines the next learning step based on the current state and lesson plan."""
-    user_state = ctx.context.user_model_state
-    current_topic = ctx.context.current_teaching_topic
-
-    print(f"[Tool determine_next_learning_step] Called. Current topic from context: '{current_topic}'")
-
-    if not lesson_plan or not lesson_plan.sections:
-        print("[Tool determine_next_learning_step] Lesson plan not found or empty.")
-        return {"error": "Lesson plan not found or empty.", "next_topic": None}
-
-    current_section_index = -1
-    if current_topic:
-        try:
-            current_section_index = next(i for i, sec in enumerate(lesson_plan.sections) if sec.title == current_topic)
-        except StopIteration:
-            print(f"[Tool determine_next_learning_step] Warning: Current topic '{current_topic}' not found in plan. Resetting.")
-            current_topic = None # Reset if current topic is invalid
-            current_section_index = -1 # Ensure index is reset too
-
-    next_section_index = -1
-    next_section = None
-
-    # If there's a current topic, check if objectives are met before moving on
-    if current_section_index != -1:
-        print(f"[Tool determine_next_learning_step] Current topic '{current_topic}' found at index {current_section_index}.")
-        current_section = lesson_plan.sections[current_section_index]
-        # Basic check: Have all objectives for this section been 'mastered'?
-        # TODO: Enhance this check - map concept mastery to objectives if possible.
-        all_objectives_mastered = len(user_state.mastered_objectives_current_section) >= len(current_section.objectives)
-
-        if all_objectives_mastered:
-            print(f"[Tool determine_next_learning_step] Objectives for '{current_topic}' seem complete. Determining next section.")
-            if current_section_index + 1 < len(lesson_plan.sections):
-                next_section_index = current_section_index + 1
-            else:
-                next_section_index = -2 # Signal end of lesson
-        else:
-            # **** CHANGE ****
-            # If objectives are NOT met, the decision to stay or re-assess
-            # should be made by the Orchestrator based on interaction outcomes.
-            # This tool should signal that the current topic is NOT complete.
-            print(f"[Tool determine_next_learning_step] Objectives for '{current_topic}' not yet met. Signaling to stay on topic.")
-            # Return current topic details, but Orchestrator should know not to reset segment index
-            # based on its own logic.
-            ctx.context.user_model_state.current_section_objectives = current_section.objectives # Ensure objectives are set
-            return {"next_topic": current_topic, "objectives": [o.model_dump() for o in current_section.objectives], "status": "topic_incomplete"}
-            # **** END CHANGE ****
-    else:
-        # No current topic, start from the beginning
-        print("[Tool determine_next_learning_step] Starting from the first section.")
-        next_section_index = 0
-
-    if next_section_index >= 0:
-        next_section = lesson_plan.sections[next_section_index]
-        next_topic_title = next_section.title
-
+def determine_next_learning_step(ctx: RunContextWrapper[TutorContext]) -> Dict[str, Any]:
+    """DEPRECATED: The Planner agent now determines the next focus. Use call_planner_agent."""
+    return {"error": "This tool is deprecated. Use call_planner_agent to get the next focus."}
+
+
+@function_tool
+async def update_user_model(
+    ctx: RunContextWrapper[TutorContext],
+    topic: str,
+    outcome: Literal['correct', 'incorrect', 'mastered', 'struggled', 'explained'],
+    confusion_point: Optional[str] = None,
+    last_accessed: Optional[str] = None,
+    mastered_objective_title: Optional[str] = None, # Optional: Mark an objective as mastered
+) -> str:
+    """Updates the user model state with interaction outcomes and temporal data."""
+    print(f"[Tool update_user_model] Updating '{topic}' with outcome '{outcome}'")
+
+    # Ensure context and user model state exist
+    if not ctx.context or not ctx.context.user_model_state:
+        return "Error: TutorContext or UserModelState not found."
+
+    if not topic or not isinstance(topic, str):
+        return "Error: Invalid topic provided for user model update."
+
+    # Initialize concept if needed
+    if topic not in ctx.context.user_model_state.concepts:
+        ctx.context.user_model_state.concepts[topic] = UserConceptMastery()
+
+    concept_state = ctx.context.user_model_state.concepts[topic]
+    concept_state.last_interaction_outcome = outcome
+
+    # Update last_accessed with ISO 8601 timestamp
+    concept_state.last_accessed = last_accessed or datetime.now().isoformat()
+
+    # Add confusion point if provided
+    if confusion_point and confusion_point not in concept_state.confusion_points:
+        concept_state.confusion_points.append(confusion_point)
+
+    # Update attempts and mastery for evaluative outcomes
+    if outcome in ['correct', 'incorrect', 'mastered', 'struggled']:
+        concept_state.attempts += 1
+
+        # Adjust mastery level based on outcome
+        if outcome in ['correct', 'mastered']:
+            concept_state.mastery_level = min(1.0, concept_state.mastery_level + 0.2)
+        elif outcome in ['incorrect', 'struggled']:
+            concept_state.mastery_level = max(0.0, concept_state.mastery_level - 0.1)
+            
+            # Adjust learning pace if struggling
+            if len(concept_state.confusion_points) > 2:
+                ctx.context.user_model_state.learning_pace_factor = max(0.5, 
+                    ctx.context.user_model_state.learning_pace_factor - 0.1)
+
+    # Update mastered objectives if provided
+    if mastered_objective_title and mastered_objective_title not in ctx.context.user_model_state.mastered_objectives_current_section:
+         ctx.context.user_model_state.mastered_objectives_current_section.append(mastered_objective_title)
+         print(f"[Tool] Marked objective '{mastered_objective_title}' as mastered for current section.")
+
+    print(f"[Tool] Updated '{topic}' - Mastery: {concept_state.mastery_level:.2f}, "
+          f"Pace: {ctx.context.user_model_state.learning_pace_factor:.2f}")
+    return f"User model updated for {topic}."
+
+@function_tool
+def update_explanation_progress(ctx: RunContextWrapper[TutorContext], segment_index: int) -> str:
+    """DEPRECATED: The Orchestrator manages micro-steps directly."""
+    return "Error: This tool is deprecated. Orchestrator manages micro-steps."
+
+@function_tool
+async def get_user_model_status(ctx: RunContextWrapper[TutorContext], topic: Optional[str] = None) -> Dict[str, Any]:
+    """Retrieves detailed user model state, optionally for a specific topic."""
+    print(f"[Tool get_user_model_status] Retrieving status for topic '{topic}'")
+
+    if not ctx.context.user_model_state:
+        return {"error": "No user model state found in context."}
+
+    state = ctx.context.user_model_state
+
+    if topic:
+        if topic not in state.concepts:
+            return {
+                "topic": topic,
+                "exists": False,
+                "message": "Topic not found in user model."
+            }
+            
+        concept = state.concepts[topic]
+        return {
+            "topic": topic,
+            "exists": True,
+            "mastery_level": concept.mastery_level,
+            "attempts": concept.attempts,
+            "last_outcome": concept.last_interaction_outcome,
+            "confusion_points": concept.confusion_points,
+            "last_accessed": concept.last_accessed
+        }
+    
+    # Return full state summary if no specific topic requested
+    # Use model_dump for serializable output
+    return state.model_dump(mode='json')
+
+@function_tool
+async def reflect_on_interaction(
+    ctx: RunContextWrapper[TutorContext],
+    topic: str,
+    interaction_summary: str, # e.g., "User answered checking question incorrectly."
+    user_response: Optional[str] = None, # The actual user answer/input
+    feedback_provided: Optional[QuizFeedbackItem] = None # Feedback from teacher tool if available
+) -> Dict[str, Any]:
+    """
+    Analyzes the last interaction for a given topic, identifies potential reasons for user difficulty,
+    and suggests adaptive next steps for the Orchestrator.
+    """
+    print(f"[Tool reflect_on_interaction] Called for topic '{topic}'. Summary: {interaction_summary}")
+
+    # Basic reflection logic (can be enhanced, e.g., calling another LLM for deeper analysis)
+    suggestions = []
+    analysis = f"Reflection on interaction regarding '{topic}': {interaction_summary}. "
+
+    if feedback_provided and not feedback_provided.is_correct:
+        analysis += f"User incorrectly selected '{feedback_provided.user_selected_option}' when the correct answer was '{feedback_provided.correct_option}'. "
+        analysis += f"Explanation: {feedback_provided.explanation}. "
+        suggestions.append(f"Re-explain the core concept using the provided explanation: '{feedback_provided.explanation}'.")
+        if feedback_provided.improvement_suggestion:
+            suggestions.append(f"Focus on the improvement suggestion: '{feedback_provided.improvement_suggestion}'.")
+        suggestions.append(f"Try asking a slightly different checking question on the same concept.")
+    elif "incorrect" in interaction_summary.lower() or "struggled" in interaction_summary.lower():
+        suggestions.append(f"Consider re-explaining the last segment of '{topic}' using a different approach or analogy.")
+        suggestions.append(f"Ask a simpler checking question focused on the specific confusion points for '{topic}'.")
+    else: # If interaction was positive or just an explanation
+        analysis += "Interaction seems positive or neutral."
+        suggestions.append("Proceed with the next logical step in the micro-plan (e.g., next segment, checking question).")
+
+
+    print(f"[Tool reflect_on_interaction] Analysis: {analysis}. Suggestions: {suggestions}")
+    return {"analysis": analysis, "suggested_next_steps": suggestions}
+
+# --- NEW Tools to Call Other Agents ---
+
+@function_tool
+async def call_planner_agent(
+    ctx: RunContextWrapper[TutorContext],
+    user_state_summary: Optional[str] = None # Optional summary of user state
+) -> Union[FocusObjective, str]:
+    """Calls the Planner Agent to determine the next learning focus objective."""
+    print("[Tool call_planner_agent] Calling Planner Agent...")
+    try:
+        # --- Import and Create Agent *Inside* ---
+        from ai_tutor.agents.planner_agent import create_planner_agent # Import here
+        # ----------------------------------------
+        if not ctx.context.vector_store_id:
+            return "Error: Vector store ID not found in context for Planner."
+
+        planner_agent = create_planner_agent(ctx.context.vector_store_id)
+        run_config = RunConfig(workflow_name="Orchestrator_PlannerCall", group_id=ctx.context.session_id)
+
+        # Construct prompt for the planner
+        planner_prompt = f"""
+        Determine the next learning focus for the user.
+        First, call `read_knowledge_base` to understand the material's structure and concepts.
+        Analyze the knowledge base.
+        {f'Consider the user state: {user_state_summary}' if user_state_summary else 'Assume the user is starting or has just completed the previous focus.'}
+        Identify the single most important topic or concept for the user to focus on next.
+        Output your decision ONLY as a FocusObjective object.
+        """
+
+        result = await Runner.run(
+            planner_agent,
+            planner_prompt,
+            context=ctx.context,
+            run_config=run_config
+        )
+
+        focus_objective = result.final_output_as(FocusObjective)
+        if focus_objective:
+            print(f"[Tool call_planner_agent] Planner returned focus: {focus_objective.topic}")
+            # Store the new focus in context
+            ctx.context.current_focus_objective = focus_objective
+            return focus_objective
+        else:
+            return "Error: Planner agent did not return a valid FocusObjective."
+
+    except Exception as e:
+        error_msg = f"Error calling Planner Agent: {str(e)}\n{traceback.format_exc()}"
+        print(f"[Tool] {error_msg}")
+        return error_msg
+
+@function_tool
+async def call_teacher_agent(
+    ctx: RunContextWrapper[TutorContext],
+    topic: str,
+    explanation_details: str # e.g., "Explain the concept generally", "Provide an example", "Focus on the difference between X and Y"
+) -> Union[ExplanationResult, str]:
+    """Calls the Teacher Agent to provide an explanation for a specific topic/detail."""
+    print(f"[Tool call_teacher_agent] Requesting explanation for '{topic}': {explanation_details}")
+    try:
+        # --- Import and Create Agent *Inside* ---
+        from ai_tutor.agents.teacher_agent import create_interactive_teacher_agent # Naming needs update based on actual refactor
+        # ----------------------------------------
+        if not ctx.context.vector_store_id:
+            return "Error: Vector store ID not found in context for Teacher."
+
+        teacher_agent = create_interactive_teacher_agent(ctx.context.vector_store_id) # TODO: Update function name if changed
+        run_config = RunConfig(workflow_name="Orchestrator_TeacherCall", group_id=ctx.context.session_id)
+
+        teacher_prompt = f"""
+        Explain the topic: '{topic}'.
+        Specific instructions for this explanation: {explanation_details}.
+        Use the file_search tool if needed to find specific information or examples from the documents.
+        Format your response ONLY as an ExplanationResult object containing the explanation text in the 'details' field.
+        """
+
+        result = await Runner.run(
+            teacher_agent,
+            teacher_prompt,
+            context=ctx.context,
+            run_config=run_config
+        )
+
+        explanation_result = result.final_output_as(ExplanationResult)
+        if explanation_result and explanation_result.status == "delivered":
+            print(f"[Tool call_teacher_agent] Teacher delivered explanation for '{topic}'.")
+            return explanation_result
+        else:
+            details = getattr(explanation_result, 'details', 'No details provided.')
+            return f"Error: Teacher agent failed or skipped explanation for '{topic}'. Status: {getattr(explanation_result, 'status', 'unknown')}. Details: {details}"
+
+    except Exception as e:
+        error_msg = f"Error calling Teacher Agent: {str(e)}\n{traceback.format_exc()}"
+        print(f"[Tool] {error_msg}")
+        return error_msg
+
+@function_tool
+async def call_quiz_creator_agent(
+    ctx: RunContextWrapper[TutorContext],
+    topic: str,
+    instructions: str # e.g., "Create one medium difficulty question", "Create a 3-question quiz covering key concepts"
+) -> Union[QuizCreationResult, str]:
+    """Calls the Quiz Creator Agent to generate one or more quiz questions."""
+    print(f"[Tool call_quiz_creator_agent] Requesting quiz creation for '{topic}': {instructions}")
+    try:
+        # --- Import and Create Agent *Inside* ---
+        from ai_tutor.agents.quiz_creator_agent import create_quiz_creator_agent # Or the new tool function name
+        # ----------------------------------------
+
+        # Assuming create_quiz_creator_agent is the function that returns the agent instance
+        quiz_creator_agent = create_quiz_creator_agent() # TODO: Update function name if changed. Pass API key if needed.
+        run_config = RunConfig(workflow_name="Orchestrator_QuizCreatorCall", group_id=ctx.context.session_id)
+
+        quiz_creator_prompt = f"""
+        Create quiz questions based on the following instructions:
+        Topic: '{topic}'
+        Instructions: {instructions}
+        Format your response ONLY as a QuizCreationResult object. Include the created question(s) in the appropriate field ('question' or 'quiz').
+        """
+
+        result = await Runner.run(
+            quiz_creator_agent,
+            quiz_creator_prompt,
+            context=ctx.context,
+            run_config=run_config
+        )
+
+        quiz_creation_result = result.final_output_as(QuizCreationResult)
+        if quiz_creation_result and quiz_creation_result.status == "created":
+            question_count = 1 if quiz_creation_result.question else len(quiz_creation_result.quiz.questions) if quiz_creation_result.quiz else 0
+            print(f"[Tool call_quiz_creator_agent] Quiz Creator created {question_count} question(s) for '{topic}'.")
+            # Store the created question if it's a single one for evaluation
+            if quiz_creation_result.question:
+                 ctx.context.current_quiz_question = quiz_creation_result.question
+            return quiz_creation_result
+        else:
+            details = getattr(quiz_creation_result, 'details', 'No details provided.')
+            return f"Error: Quiz Creator agent failed for '{topic}'. Status: {getattr(quiz_creation_result, 'status', 'unknown')}. Details: {details}"
+
+    except Exception as e:
+        error_msg = f"Error calling Quiz Creator Agent: {str(e)}\n{traceback.format_exc()}"
+        print(f"[Tool] {error_msg}")
+        return error_msg
 
 
 # Ensure all tools intended for the orchestrator are exported or available
 __all__ = [
-    'call_quiz_creator_mini',
+    # --- NEW ---
+    'call_planner_agent',
+    'call_teacher_agent',
+    'call_quiz_creator_agent',
+    # --- Kept ---
     'call_quiz_teacher_evaluate',
     'reflect_on_interaction',
-    'determine_next_learning_step',
     'update_user_model',
     'get_user_model_status',
-    'update_explanation_progress',
+    # --- Removed ---
+    # 'call_quiz_creator_mini',
+    # 'determine_next_learning_step',
+    # 'update_explanation_progress',
 ]
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Diff
IGNORE_WHEN_COPYING_END

7. Refactor Orchestrator Agent (ai_tutor/agents/orchestrator_agent.py)

Major rewrite of instructions and update tools list.

--- a/ai_tutor/agents/orchestrator_agent.py
+++ b/ai_tutor/agents/orchestrator_agent.py
@@ -10,10 +10,11 @@
     from ai_tutor.context import TutorContext # Use the enhanced context
 # Import tool functions (assuming they will exist in a separate file)
 from ai_tutor.tools.orchestrator_tools import (
-    call_quiz_creator_mini,
     call_quiz_teacher_evaluate,
-    determine_next_learning_step,
+    call_planner_agent, # NEW
+    call_teacher_agent, # NEW
+    call_quiz_creator_agent, # NEW
     update_user_model,
     get_user_model_status,
-    update_explanation_progress,
     reflect_on_interaction,
 )
 
@@ -35,11 +36,12 @@
     base_model = provider.get_model("gpt-4o-2024-08-06")  # Using exact model version
 
     orchestrator_tools = [
-        call_quiz_creator_mini,
+        call_planner_agent, # Determines next focus
+        call_teacher_agent, # Explains a concept/topic
+        call_quiz_creator_agent, # Creates checking questions/quizzes
         call_quiz_teacher_evaluate,
-        determine_next_learning_step,
+        # call_quiz_creator_mini, # Replaced by call_quiz_creator_agent
         reflect_on_interaction,
         update_user_model,
         get_user_model_status,
-        update_explanation_progress,
     ]
 
     orchestrator_agent = Agent['TutorContext'](
@@ -47,67 +49,73 @@
         instructions="""
         You are the central conductor of an AI tutoring session. Your primary goal is to guide the user towards mastering specific learning objectives identified by the Planner Agent.
 
+        CONTEXT:
+        - You operate based on the `current_focus_objective` provided in the `TutorContext`. This objective (topic, goal) is set by the Planner Agent.
+        - If `current_focus_objective` is missing, your FIRST action MUST be to call `call_planner_agent` to get the initial focus.
+        - You manage the user's learning state via `UserModelState` using tools like `get_user_model_status` and `update_user_model`.
+        - You interact with specialist agents (Teacher, Quiz Creator) using the `call_teacher_agent` and `call_quiz_creator_agent` tools.
+        - You evaluate user answers to checking questions using `call_quiz_teacher_evaluate`.
+        - `reflect_on_interaction` helps you analyze difficulties and adapt your strategy.
+        - User's last input/action is provided in the prompt.
+        - `tutor_context.user_model_state.pending_interaction_type` indicates if you are waiting for a user response (e.g., to a 'checking_question').
+
         **Core Responsibilities:**
-        1.  **State Assessment & Objective Tracking:** Analyze user input, `UserModelState` (mastery, confusion points related to `current_section_objectives`), and interaction history.
-        2.  **Dynamic Task Management:** Decompose complex user requests into smaller, manageable steps.
-        3.  **Adaptive Guidance:** Decide the next micro-step (explain, re-explain, check understanding, move topic) based on assessment.
-        4.  **Reflection & Adjustment:** After significant interactions (e.g., incorrect answers, confusion), reflect on the effectiveness and adjust the strategy.
-
-        CONTEXT:
-        - You have access to the overall `LessonPlan` via the context object.
-        - You can read and update the `UserModelState` via tools (`get_user_model_status`, `update_user_model`). This state tracks concept mastery, pace, style, notes, current topic, segment index, pending interactions, *current_section_objectives*, and *mastered_objectives_current_section*.
-        - You know the user's last input/action provided in the prompt.
-        - `tutor_context.user_model_state.pending_interaction_type` tells you if the Teacher agent is waiting for a user response (e.g., to a 'checking_question').
+        1.  **Ensure Focus:** If no `current_focus_objective`, call `call_planner_agent`.
+        2.  **Micro-Planning:** Based on the `current_focus_objective` and `UserModelState`, devise a short sequence of steps (e.g., Explain -> Check -> Example).
+        3.  **Execute Step:** Call the appropriate agent tool (`call_teacher_agent` for explanations/examples, `call_quiz_creator_agent` for checks). Provide specific instructions to the tool.
+        4.  **Process User Input/Agent Results:** Handle user answers (using `call_quiz_teacher_evaluate`) or results from agent tools. Update `UserModelState` using `update_user_model`.
+        5.  **Evaluate Objective:** Assess if the `current_focus_objective`'s `learning_goal` has been met based on interactions and mastery levels. Use `reflect_on_interaction` if the user struggles.
+        6.  **Loop or Advance:**
+            *   If the objective is NOT met, determine the next micro-step (re-explain, different example, different question) and go back to step 3.
+            *   If the objective IS met, call `call_planner_agent` to get the *next* focus objective. If the planner indicates completion, end the session.
 
         CORE WORKFLOW:
-        1.  **Assess Current State:** Check user input (`interaction_input.type`, `interaction_input.data`), `UserModelState` (`pending_interaction_type`, `current_teaching_topic`, `current_topic_segment_index`), interaction history.
+        1.  **Check Focus:** Is `tutor_context.current_focus_objective` set?
+            *   **NO:** Call `call_planner_agent`. **END TURN** (The context will be updated, next interaction will proceed).
+            *   **YES:** Proceed to step 2.
+        2.  **Assess Interaction State:** Check `UserModelState` (`pending_interaction_type`).
         3.  **Handle Pending Interaction:**
             *   If `pending_interaction_type` is 'checking_question':
                 - Use `call_quiz_teacher_evaluate` with the user's answer and details from `pending_interaction_details`.
-                - Based on the feedback (correct/incorrect): Decide next step (continue explanation, ask again, re-explain). Update mastery/confusion via `update_user_model`. If the answer demonstrates mastery of an objective, update `mastered_objectives_current_section` via `update_user_model`.
-                - **If incorrect:** Call `reflect_on_interaction` to analyze *why* the user struggled and get suggestions for the next step (e.g., re-explain differently, use analogy). **Log this reflection.**
-                - **Clear pending state** (tool handles this or manage carefully).
+                - Update state via `update_user_model` based on feedback (correct/incorrect).
+                - If incorrect, call `reflect_on_interaction`.
                 -> **END TURN**
-        3.  **Handle New User Input (No Pending Interaction):**
+        4.  **Handle New User Input / Decide Next Micro-Step (No Pending Interaction):**
+            *   Analyze user input (question, request, feedback).
             *   If user asked a complex question or made a request requiring multiple steps (e.g., "Compare X and Y", "Give me a harder problem"):
                 - **Decompose the request:** Plan the micro-steps needed.
-                - Initiate the *first step* of your decomposed plan (e.g., signal teaching for X).
+                - Execute the *first step* by calling the appropriate agent tool (e.g., `call_teacher_agent` to explain X).
                 -> **END TURN**
-            *   If user asked a simple question: Answer briefly with `MessageResponse` or signal teaching for a relevant segment.
+            *   If user asked a simple clarification related to the current focus: Call `call_teacher_agent` with specific instructions.
                 -> **END TURN**
-            *   If user provided feedback or other input: Update `session_summary_notes` via `update_user_model`.
-                -> **Decide next step based on state (like no input).**
-        4.  **Determine Next Step (if no user input or pending interaction):**
-            *   **Check Current Topic Progress:** Is `tutor_context.current_teaching_topic` set?
-                *   **YES (Topic Active):**
-                    - Was the last interaction successful (e.g., `last_interaction_outcome` is 'explained' or 'correct')?
-                    - Was the last explanation *not* the final segment for the topic (check `is_last_segment` from the previous `ExplanationResponse` if available, or assume False initially)?
-                    - **If YES to both:**
-                        + Use the `update_explanation_progress` tool to increment the `current_topic_segment_index`. **Log this increment.**
-                        + Signal teaching (via `MessageResponse` with `message_type='initiate_teaching'`) for the **new segment index** of the **current topic**. **Ensure `current_topic_segment_index` is updated in context *before* returning.** -> **END TURN**
-                    - **If NO (e.g., user struggled, or last segment was final):**
-                        + **Check Objectives:** Are the `current_section_objectives` met (compare `mastered_objectives_current_section`)?
-                            *   **If YES:** Call `determine_next_learning_step` to get the *next* topic/section. **Log the transition.** If a next topic exists, signal teaching for segment 0. If lesson complete, send completion message. -> **END TURN**
-                            *   **If NO (Objectives not met & cannot proceed to next segment):** Decide the *micro-step* towards the *next unmet objective*. Use `reflect_on_interaction` results if needed. Decide whether to re-explain (signal teaching for *same/alternative* segment), provide a hint (`MessageResponse`), or ask a checking question (`call_quiz_creator_mini`). **Log your decision and reasoning.** -> **END TURN**
-                *   **NO (No current topic):**
-                    + Call `determine_next_learning_step` to determine the starting topic/section. **Log this.**
-                    + Signal teaching for segment 0 of the new topic. -> **END TURN**
-        5.  **Select Action & Update State:**
-            *   **Initiate Teaching:** Signal this via a `MessageResponse` with `message_type='initiate_teaching'`. Set `tutor_context.current_teaching_topic` and `current_topic_segment_index` correctly **before** returning this signal. The external loop invokes the `InteractiveLessonTeacher`.
-            *   **Quiz/Feedback/Message/Error:** Use appropriate tools/responses. Your output *is* the tool's output or the formulated response.
-            *   **State Updates:** Use `update_user_model`, `update_explanation_progress`, etc., strategically *before* deciding the final action. **Log significant state changes.**
+            *   If no specific user input, determine the next micro-step based on the current focus objective and user state:
+                - Has the topic been explained yet? If no -> Call `call_teacher_agent`.
+                - Was the last step an explanation? -> Call `call_quiz_creator_agent` to create a checking question.
+                - Was the last step an incorrect answer? -> Use `reflect_on_interaction` suggestions. Maybe call `call_teacher_agent` to re-explain or provide an example.
+                - Is mastery level high enough for this focus objective? -> If yes, call `call_planner_agent` for next focus. If no, decide next practice/reinforcement step (e.g., harder question via `call_quiz_creator_agent`).
+            *   Execute the chosen micro-step by calling the appropriate agent tool.
+            -> **END TURN**
+
+        OBJECTIVE EVALUATION:
+        - After each relevant interaction (e.g., correct answer to checking question, successful completion of an exercise if implemented), evaluate if the `current_focus_objective.learning_goal` seems to be met. Check `UserModelState.concepts[topic].mastery_level`.
+        - If met: Call `call_planner_agent` to get the next focus. If the planner returns a completion signal, output a final success message.
+        - If not met: Continue the micro-step loop (Step 4).
 
         PRINCIPLES:
-        - **Be Adaptive, Reflective & Objective-Focused.** Prioritize achieving learning objectives.
-        - Your primary role is **decision-making and state management**. You decide *what* should happen next (explain, quiz, evaluate, move on) and prepare the context for the next agent (often the Teacher).
+        - **Focus-Driven:** Always work towards the `current_focus_objective`.
+        - **Adaptive & Reflective:** Use user state and reflection to adjust micro-steps.
+        - **Agent Orchestration:** You call other agents (Planner, Teacher, Quiz Creator) as tools to perform specific tasks.
+        - **State Management:** Keep `UserModelState` updated via tools.
         - Ensure your final output strictly adheres to the required JSON format (`TutorInteractionResponse`).
-        - Ensure context (`current_teaching_topic`, `current_topic_segment_index`, `current_section_objectives`) is correctly set before signaling `initiate_teaching`.
+
+        Your final output for each turn will typically be the direct result passed back from the tool you called (e.g., the feedback item from `call_quiz_teacher_evaluate`, or potentially a message you construct if signaling completion).
         """,
         tools=orchestrator_tools,
         output_type=TutorInteractionResponse,
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Diff
IGNORE_WHEN_COPYING_END

8. Refactor API Interaction Endpoint (ai_tutor/routers/tutor.py)

Adjust the /interact endpoint to use the new Orchestrator logic. Remove direct Teacher agent calls.

--- a/ai_tutor/routers/tutor.py
+++ b/ai_tutor/routers/tutor.py
@@ -10,7 +10,7 @@
 from ai_tutor.session_manager import SessionManager
 from ai_tutor.tools.file_upload import FileUploadManager
 from ai_tutor.agents.analyzer_agent import analyze_documents
-from ai_tutor.agents.planner_agent import create_planner_agent
+# from ai_tutor.agents.planner_agent import create_planner_agent # Planner is called via Orchestrator tool now
 from ai_tutor.agents.session_analyzer_agent import analyze_teaching_session
 from ai_tutor.agents.orchestrator_agent import create_orchestrator_agent
 from ai_tutor.agents.teacher_agent import create_interactive_teacher_agent
@@ -124,44 +124,54 @@
     """
     logger = get_session_logger(session_id)
     # Get user from request state
-    user: User = request.state.user # Get user from request state populated by verify_token dependency
+    user: User = request.state.user
 
     # Get vector store ID and analysis result from context
     vector_store_id = tutor_context.vector_store_id
-    analysis_result = tutor_context.analysis_result
-    folder_id = tutor_context.folder_id # Get folder_id from context
+    # analysis_result = tutor_context.analysis_result # No longer directly needed by this endpoint
+    # folder_id = tutor_context.folder_id # No longer directly needed by this endpoint
 
     # Initial checks are good to keep, ensure data looks okay before agent creation
     if not vector_store_id:
         raise HTTPException(status_code=400, detail="Documents must be uploaded first.")
-    if not folder_id:
-        raise HTTPException(status_code=400, detail="Knowledge base file path not found or file missing.")
-
+    # Remove KB check, planner tool handles reading it
     # --- Wrap the main logic in a try...except block ---
     try:
-        print(f"[Debug /plan] Creating planner agent for vs_id: {vector_store_id}") # Add log
-        planner_agent: Agent[TutorContext] = create_planner_agent(vector_store_id)
+        # --- Orchestrator now calls Planner via a tool ---
+        # print(f"[Debug /plan] Creating planner agent for vs_id: {vector_store_id}") # Add log
+        # planner_agent: Agent[TutorContext] = create_planner_agent(vector_store_id)
 
         # Pass the full TutorContext to the Runner
         run_config = RunConfig(
-            workflow_name="AI Tutor API - Planning",
+            workflow_name="AI Tutor API - Get Initial Focus", # Name reflects new purpose
             group_id=str(session_id) # Convert UUID to string
         )
 
-        print(f"[Debug /plan] Starting Runner.run for planner agent...") # Add log
-
-        # Prompt tells planner to use its tools (read_knowledge_base and file_search)
-        plan_prompt = """
-        Create a lesson plan. First, use the `read_knowledge_base` tool to understand the document analysis. Then, use the `file_search` tool to clarify details as needed. Finally, generate the `LessonPlan` object based on both sources of information. Follow your detailed agent instructions.
+        # We need the orchestrator to call the planner tool
+        orchestrator_agent = create_orchestrator_agent() # Assuming it doesn't need vs_id directly anymore
+        
+        # Prompt for orchestrator to get initial focus
+        orchestrator_prompt = "The session is starting. Call the `call_planner_agent` tool to determine the initial learning focus objective."
+        
+        print(f"[Debug /plan] Running Orchestrator to get initial focus...") # Add log
+        print(f"[Debug /plan] Orchestrator prompt:\n{orchestrator_prompt}") # Log start of prompt
+
+        result = await Runner.run(
+            orchestrator_agent,
+            orchestrator_prompt,
+            run_config=run_config,
+            context=tutor_context # Pass the parsed TutorContext object
         )
-        print(f"[Debug /plan] Planner prompt:\n{plan_prompt[:500]}...") # Log start of prompt
-
+        print(f"[Debug /plan] Orchestrator run completed. Result final_output type: {type(result.final_output)}") # Add log
+
+        # The orchestrator's output *should* be the FocusObjective returned by the call_planner_agent tool
+        # We need to check this and update the context MANUALLY here before returning
+        # (Alternatively, the tool itself could update context, but that's less clean)
+        # For now, let's assume the context was updated *inside* the call_planner_agent tool run.
         result = await Runner.run(
             planner_agent,
             plan_prompt,
-            run_config=run_config,
+            run_config=run_config, # Pass RunConfig
             context=tutor_context # Pass the parsed TutorContext object
         )
         print(f"[Debug /plan] Runner.run completed. Result final_output type: {type(result.final_output)}") # Add log
@@ -169,19 +179,25 @@
 
         # --- Check the result and update session ---
         # Use final_output_as for potential validation
+
+        # Check the context *after* the run to see if the focus objective was set
+        focus_objective = tutor_context.current_focus_objective
         try:
             lesson_plan_obj = result.final_output_as(LessonPlan, raise_if_incorrect_type=True)
             lesson_plan_to_store = lesson_plan_obj
             logger.log_planner_output(lesson_plan_obj)
 
             # Update context object and save it
-            tutor_context.lesson_plan = lesson_plan_to_store
+            # tutor_context.lesson_plan = lesson_plan_to_store # Store focus objective instead
+            if not focus_objective:
+                 # This means the orchestrator/planner tool failed to set the focus
+                 raise HTTPException(status_code=500, detail="Failed to determine initial focus objective.")
+
             # Need supabase client - get it via Depends implicitly or pass it
             # Easiest is often to add Depends(get_supabase_client) to the signature or get it again
             supabase: Client = await get_supabase_client() # Or add to Depends
             success = await session_manager.update_session_context(supabase, session_id, user.id, tutor_context)
             if not success:
                  logger.log_error("SessionUpdate", f"Failed to update session {session_id} with lesson plan.")
-            print(f"[Debug /plan] LessonPlan stored in session.") # Add log
-            return lesson_plan_obj # Return the generated plan
+            print(f"[Debug /plan] FocusObjective stored in session.") # Add log
+            return focus_objective # Return the focus objective
         except Exception as parse_error: # Catch if final_output isn't a LessonPlan
             error_msg = f"Planner agent returned unexpected output or parsing failed: {parse_error}. Raw output: {result.final_output}"
             logger.log_error("PlannerAgentOutputParse", error_msg)
@@ -308,7 +324,7 @@
     print(f"[Interact] Context BEFORE Orchestrator: pending={tutor_context.user_model_state.pending_interaction_type}, topic='{tutor_context.current_teaching_topic}', segment={tutor_context.user_model_state.current_topic_segment_index}")
 
     user: User = request.state.user
-    # --- Agent Execution Logic ---
+
     final_response_data: TutorInteractionResponse
 
     print(f"[Interact] Fetching context for session {session_id}...") # Log context fetch
@@ -318,7 +334,7 @@
     )
 
     # Always run the Orchestrator first to decide the next step or handle pending interactions.
-    orchestrator_agent = create_orchestrator_agent(tutor_context.vector_store_id)
+    orchestrator_agent = create_orchestrator_agent() # Doesn't need vs_id directly
     print(f"[Interact] Orchestrator agent created.") # Log agent creation
 
     # Prepare input for the Orchestrator
@@ -328,9 +344,16 @@
         orchestrator_input = f"User Response to Pending Interaction '{tutor_context.user_model_state.pending_interaction_type}' | Type: {interaction_input.type} | Data: {json.dumps(interaction_input.data)}"
         logger.log_user_input(f"User Response (Pending): {interaction_input.type} - {interaction_input.data}") # Log user input
     else:
-        # No pending interaction, Orchestrator decides next general step
-        print("[Interact] No pending interaction. Running Orchestrator to decide next step.")
-        orchestrator_input = f"User Action | Type: {interaction_input.type} | Data: {json.dumps(interaction_input.data)}"
+        # No pending interaction. Check if focus objective exists.
+        if not tutor_context.current_focus_objective:
+            print("[Interact] No current focus. Instructing Orchestrator to call Planner.")
+            # If focus is missing, tell orchestrator to get it first.
+            orchestrator_input = "No current focus objective set. Call the `call_planner_agent` tool to determine the initial focus objective for the user."
+            # NOTE: This requires the orchestrator to handle this specific instruction.
+        else:
+            # Focus exists, proceed normally based on user input
+            print("[Interact] Focus exists. Running Orchestrator to decide next step based on user input.")
+            orchestrator_input = f"Current Focus: {tutor_context.current_focus_objective.topic} ({tutor_context.current_focus_objective.learning_goal}). User Action | Type: {interaction_input.type} | Data: {json.dumps(interaction_input.data)}"
         logger.log_user_input(f"User Action: {interaction_input.type} - {interaction_input.data}") # Log user input
 
     print(f"[Interact] Running Agent: {orchestrator_agent.name}")
@@ -347,46 +370,15 @@
     print(f"[Interact] Orchestrator Output Type: {type(orchestrator_output)}")
 
     # --- Handle Orchestrator Output ---
-    if isinstance(orchestrator_output, (MessageResponse, FeedbackResponse, ErrorResponse)):
-        # Direct responses from orchestrator (no teacher needed)
+    # The orchestrator's output *is* the final response for this turn,
+    # as it comes from the specialist agent tool call or a direct response.
+    if isinstance(orchestrator_output, TutorInteractionResponse):
         final_response_data = orchestrator_output
-        print(f"[Interact] Using direct Orchestrator response of type: {type(final_response_data)}")
-
-    elif isinstance(orchestrator_output, TutorInteractionResponse) and orchestrator_output.message_type == 'initiate_teaching':
-        # Orchestrator wants the teacher to explain/teach
-        print(f"[Interact] Orchestrator signaled teaching. Creating Teacher Agent...")
-
-        # Create Teacher Agent (lazy loading)
-        if not teacher_agent:
-            teacher_agent = create_interactive_teacher_agent(tutor_context.vector_store_id)
-            print(f"[Interact] Created new Teacher Agent: {teacher_agent.name}")
-
-        # Teacher input can be generic; its instructions guide it based on context
-        teacher_input = f"Explain segment {tutor_context.user_model_state.current_topic_segment_index} of topic '{tutor_context.current_teaching_topic}'."
-
-        print(f"[Interact] Context BEFORE Teacher: topic='{tutor_context.current_teaching_topic}', segment={tutor_context.user_model_state.current_topic_segment_index}")
-        print(f"[Interact] Running Teacher Agent: {teacher_agent.name} for topic '{tutor_context.current_teaching_topic}' segment {tutor_context.user_model_state.current_topic_segment_index}")
-        teacher_result = await Runner.run(
-            teacher_agent,
-            teacher_input,
-            context=tutor_context,
-            run_config=run_config
-        )
-
-        # The teacher's output (ExplanationResponse or QuestionResponse) is the final response
-        final_response_data = teacher_result.final_output # Type is TeacherInteractionOutput
-        logger.log_teacher_output(final_response_data) # Log teacher's specific output
-        print(f"[Interact] Context AFTER Teacher: pending={tutor_context.user_model_state.pending_interaction_type}, topic='{tutor_context.current_teaching_topic}', segment={tutor_context.user_model_state.current_topic_segment_index}")
-        print(f"[Interact] Teacher Output Type: {type(final_response_data)}, Content: {final_response_data}")
-
-        # Update context based on teacher's action (Runner updates context in-place)
-        if isinstance(final_response_data, QuestionResponse):
-            # Teacher asked a checking question - set pending state
+        print(f"[Interact] Orchestrator returned response of type: {final_response_data.response_type}")
+        # If orchestrator called quiz creator which returned a question, update pending state
+        if isinstance(final_response_data, QuestionResponse):
             tutor_context.user_model_state.pending_interaction_type = 'checking_question'
             tutor_context.user_model_state.pending_interaction_details = {
-                'question': final_response_data.question,
-                'options': final_response_data.options,
-                'correct_index': final_response_data.correct_index,
-                'topic': tutor_context.current_teaching_topic
+                'question': final_response_data.question.model_dump() # Store the question details
             }
             print(f"[Interact] Teacher asked checking question. Set pending_interaction_type='checking_question'")
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Diff
IGNORE_WHEN_COPYING_END

Final Notes:

These diffs represent a significant architectural shift. Thorough testing is crucial.

The prompts for the refactored Planner and Orchestrator agents are critical and may require iteration based on testing results.

Error handling within the new agent tool calls (call_planner_agent, etc.) should be robust.

The /interact endpoint logic, especially handling the initial state where no focus objective exists, might need refinement depending on how the Orchestrator behaves.

Remember to update any dependencies or imports as needed after applying the diffs.