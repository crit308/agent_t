Okay, let's analyze the current codebase and address the runtime issues.

Codebase Analysis & Viability Check:

Based on the provided code (repomix-output-tree-adk (1).xml) and the ADK documentation report:

Architecture Alignment: The codebase structure generally aligns with the target architecture:

Agents (PlannerAgent, TeacherAgent, OrchestratorAgent) are defined using ADK's LLMAgent (or the Agent alias).

The Orchestrator (create_orchestrator_agent) is correctly set up to use AgentTool to delegate to the Planner and Teacher agents.

The Teacher (create_interactive_teacher_agent) includes the custom AskUserQuestionTool (a LongRunningFunctionTool) and its instructions describe the intended autonomous loop.

State management uses a custom SupabaseSessionService inheriting from BaseSessionService, which is the correct approach for integrating ADK's session management with your Supabase backend.

API endpoints (/interact, /answer) are present to handle the main interaction flow and the pause/resume cycle.

File handling relies on Supabase Storage and a get_document_content tool, removing the OpenAI RAG dependency.

Viability: The core concept of using ADK with autonomous specialist agents (Teacher handling its own loop and pausing via a long-running tool) and Supabase for persistence is viable with the current structure. The pieces are mostly in place, but there are implementation details, error handling gaps, and potential inefficiencies causing the runtime errors.

Analysis of Runtime Errors & Warnings:

Let's break down the terminal output:

Default value is not supported... Warning (Repeated):

Cause: Google's AI API (which ADK uses for Gemini) does not support function declarations where parameters have default values. ADK tries to generate these declarations from your Python function signatures or Pydantic models used as tool schemas. If any function decorated with @FunctionTool or any Pydantic model used in a tool's input_schema or output_schema has fields/parameters with default values (e.g., param: str = "default_value" or field: Optional[str] = None), this warning will appear. The manual FunctionDeclaration definitions in orchestrator_tools.py for the utility functions are prime suspects.

Fix:

Remove Manual Wrappers: Remove the FunctionDeclaration and BaseTool wrapper classes for update_user_model, get_user_model_status, and reflect_on_interaction in tools/orchestrator_tools.py. Rely solely on the @FunctionTool decorator for these.

Check Tool Signatures/Schemas: Review the Python function signatures for all functions decorated with @FunctionTool. Remove any default argument values. If a default is needed, handle it inside the function logic.

Check Pydantic Schemas: Review Pydantic models used as input_schema or output_schema for any tools or agents. Ensure fields that become part of the function declaration don't have default values that might confuse the schema generation. Optional[str] is usually fine, but Optional[str] = "default" is not.

Warning: there are non-text parts in the response: ['function_call']... (Repeated):

Cause: This is an informational warning from ADK. It simply means the LLM's response included a request to call one of your defined tools (a function_call). This is expected agent behavior.

Fix: No fix needed; this is benign informational output during normal tool use.

_process_adk_event_for_api returning default UserModelState - requires refactor! (Repeated):

Cause: The _process_adk_event_for_api helper function in routers/tutor.py correctly identifies that it needs to return the latest UserModelState within the InteractionResponseData, but it doesn't have access to the final session state after the run_async loop finishes or pauses.

Fix: Modify the /interact and /answer endpoints to fetch the final session state after the run_async loop completes/breaks and pass the UserModelState from that final state to _process_adk_event_for_api or directly into the InteractionResponseData being returned.

google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED:

Cause: You are hitting the rate limits for the Google Gemini API, likely the free tier limit (e.g., 15 requests per minute for gemini-flash). The nested run_async calls shown in the traceback (Runner -> Agent -> Tool -> AgentTool -> Runner -> Agent -> ...) strongly suggest that the AgentTool mechanism might be triggering more LLM calls than anticipated, possibly by running the wrapped agent's full logic including LLM calls even when just being used as a tool by the Orchestrator. This needs investigation. Alternatively, the Teacher agent's internal loop might be too rapid.

Fix:

Immediate: Implement retry logic with exponential backoff specifically for google.genai.errors.ClientError where the error details indicate a 429 status. The error payload usually contains a suggested retryDelay.

Investigate Calls: Use logging within agents (Planner, Teacher, Orchestrator) and potentially ADK callbacks (before_model_callback, after_model_callback) to precisely track when and why LLM calls are being made. Is the Teacher's internal loop too aggressive? Is the Orchestrator calling the Teacher tool unnecessarily?

Optimize Prompts: Ensure agent prompts are clear and minimize unnecessary LLM turns.

Review AgentTool: While less likely, double-check if AgentTool has configuration options that might influence whether the wrapped agent makes LLM calls when invoked. Usually, it should just execute the required function/logic.

Check Billing/Quotas: Ensure you haven't genuinely exhausted your quota. Consider upgrading if necessary for your usage level.

AttributeError: 'ClientError' object has no attribute 'status_code':

Cause: The exception handling block in routers/tutor.py tries to access e.status_code directly on the caught ClientError object, which doesn't exist. The HTTP status code (429) is nested within the error details.

Fix: Modify the exception handler to inspect the ClientError details more robustly.

Code Diffs for Fixes:

1. Fix AttributeError in routers/tutor.py

--- a/ai_tutor/routers/tutor.py
+++ b/ai_tutor/routers/tutor.py
@@ -526,15 +526,17 @@
             )
 
     except google.genai.errors.ResourceExhausted as e: # Catch specific 429 error if possible (check exact type)
-        error_message = f"API Rate Limit Exceeded for session {session_id}: {e}"
-        print(f"[Interact] {error_message}")
-        # Try to extract retry delay
-        retry_after = "unknown"
-        if hasattr(e, 'details') and e.details:
+        # Extract details more safely
+        error_message = f"API Rate Limit Exceeded (ResourceExhausted) for session {session_id}: {e.message}"
+        logger.error(error_message) # Use standard logger
+        print(f"[Interact] Rate limit hit for session {session_id}")
+        retry_after = "Check Logs" # Placeholder
+        # Attempt to parse retryDelay from the details (error-prone, use carefully)
+        # Example: Look for retryInfo details if available in the exception object or its args
+        # For now, just log and return a generic retry message
+        # if hasattr(e, 'details') and e.details:
+        #     try:
+        #         details = getattr(e, 'details', [])
+        #         for detail in details:
+        #             if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
+        #                  retry_after = detail.get('retryDelay', 'unknown')
+        #                  break
+        #     except Exception:
+        #         pass # Ignore errors parsing details
         raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Please try again later. (Retry after: {retry_after})")
     except google.genai.errors.ClientError as e:
-        # Handle other client errors (like permission denied, invalid API key etc.)
-        if e.status_code == 429:
+        # Handle other ClientErrors, including 429 if not caught above
+        # Check the message or specific attributes for 429 if needed
+        if "RESOURCE_EXHAUSTED" in str(e): # Check message string for 429 indication
             error_message = f"API Rate Limit Exceeded (ClientError) for session {session_id}: {e.message}"
             logger.error(error_message)
             print(f"[Interact] Rate limit hit (ClientError) for session {session_id}")
@@ -546,7 +548,7 @@
                     pass # Ignore errors parsing details
             raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Please try again later. (Retry after: {retry_after})")
         else:
-            # Handle other ClientErrors as 500
+            # Handle other ClientErrors as 500 Internal Server Error
             tb_str = traceback.format_exc()
             error_message = f"Google API ClientError for session {session_id}: {e.message}\nTraceback:\n{tb_str}"
             logger.error(error_message)


2. Remove Manual Tool Wrappers in tools/orchestrator_tools.py

Delete the class UpdateUserModelTool(BaseTool): ... definition and the update_user_model_tool = UpdateUserModelTool() instantiation.

Delete the class GetUserModelStatusTool(BaseTool): ... definition and the get_user_model_status_tool = GetUserModelStatusTool() instantiation.

Delete the class ReflectOnInteractionTool(BaseTool): ... definition and the reflect_on_interaction_tool = ReflectOnInteractionTool() instantiation.

Ensure the original functions (update_user_model, get_user_model_status, reflect_on_interaction) are decorated only with @FunctionTool.

Update the __all__ list to export the decorated functions directly, not the removed tool instances:

--- a/ai_tutor/tools/orchestrator_tools.py
+++ b/ai_tutor/tools/orchestrator_tools.py
@@ -376,8 +376,8 @@
     'call_quiz_teacher_evaluate',
     'call_quiz_creator_agent', # Keep this one too
     'reflect_on_interaction', # Export the decorated function
-    'update_user_model_tool', # Export the tool instance, not the function
-    'get_user_model_status_tool', # Export the custom tool instance
+    'update_user_model',      # Export the decorated function
+    'get_user_model_status',  # Export the decorated function
     # --- Removed ---
     # 'call_quiz_creator_mini',
     # 'determine_next_learning_step',
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Diff
IGNORE_WHEN_COPYING_END

3. Address Default Value Warnings

Review the signatures of update_user_model, get_user_model_status, reflect_on_interaction and any other functions decorated with @FunctionTool.

Example Fix: If get_user_model_status had topic: Optional[str] = None, it's likely okay, but if it was topic: str = "default_topic", change it to topic: Optional[str] and handle the None case inside the function.

Do the same for any Pydantic models used as schemas if they have default values problematic for Google's schema generation.

4. Fix _process_adk_event_for_api State Handling in routers/tutor.py

Modify the function and its callers (/interact, /answer) to include the final state.

--- a/ai_tutor/routers/tutor.py
+++ b/ai_tutor/routers/tutor.py
@@ -498,7 +498,10 @@
     last_agent_event: Optional[Event] = None
     question_for_user: Optional[QuizQuestion] = None
     paused_tool_call_id: Optional[str] = None # Store the ID needed for resume
-
+    
+    # Define final_context variable in the outer scope
+    final_context: TutorContext = tutor_context # Initialize with initial context
+    
     try:
         # Use run_async and process events
         async for event in adk_runner.run_async(
@@ -509,7 +512,7 @@
         ):
             print(f"[Interact] Received Event: ID={event.id}, Author={event.author}, Actions={event.actions}")
             logger.log_orchestrator_output(event.content) # Log content
-            last_agent_event = event # Keep track of the last agent event
+            last_agent_event = event # Keep track of the last event processed
 
             # --- Check for Pause/Input Request ---
             # Check the specific action yielded by AskUserQuestionTool
@@ -562,14 +565,7 @@
             raise HTTPException(status_code=500, detail="Internal error during agent execution.")
 
     # --- Load final context state AFTER the run OR PAUSE ---
-    # The Runner should have updated the state via SessionService.append_event up to the pause/completion
-    final_session = session_service.get_session("ai_tutor", str(user.id), str(session_id))
-    if final_session and final_session.state:
-         final_context = TutorContext.model_validate(final_session.state)
-         # Store the paused tool call ID in the context if we paused
-         if paused_tool_call_id:
-             final_context.user_model_state.pending_interaction_type = 'checking_question' # Mark as pending
-             final_context.user_model_state.pending_interaction_details = {"paused_tool_call_id": paused_tool_call_id}
-             # Persist this update immediately so /answer can retrieve it
-             await _update_context_in_db(session_id, user.id, final_context, supabase)
-             logger.info(f"Stored paused_tool_call_id {paused_tool_call_id} in context for session {session_id}")
+    # Fetch the latest state after the run/pause
+    final_context = await get_tutor_context(session_id, request, session_service) # Use dependency logic
+    # Update context with pause details if needed (moved from loop for clarity)
+    if paused_tool_call_id and not question_for_user: # Update only if we actually paused and broke loop
+        if final_context.user_model_state.pending_interaction_type != 'checking_question':
+            final_context.user_model_state.pending_interaction_type = 'checking_question' # Mark as pending
+            final_context.user_model_state.pending_interaction_details = {"paused_tool_call_id": paused_tool_call_id}
+            # Persist this update immediately so /answer can retrieve it
+            await _update_context_in_db(session_id, user.id, final_context, supabase)
+            logger.info(f"Stored paused_tool_call_id {paused_tool_call_id} in context for session {session_id}")
     else:
         # Should not happen if session exists, but handle gracefully
         logger.error(f"ContextFetchAfterRun: Failed to fetch session state after run/pause for {session_id}")
@@ -585,14 +581,17 @@
         print(f"[Interact] Responding with question. Session {session_id} is now paused waiting for answer.")
         # IMPORTANT: The session state (context) should have been implicitly saved by the Runner/SessionService *before* the pause signal.
         # The latest state is already loaded into final_context above.
-
+        
     # --- Logic to handle the COMPLETED state (no pause detected) ---
     elif last_agent_event and last_agent_event.content:
         # Agent run completed normally. Assume the last event's content is the response.
         # This should ideally be the final output from the Orchestrator agent based on its output_schema.
-        # TODO: Adapt this based on Orchestrator's final output schema.
-        # For now, assume it's a text message.
-        response_text = last_agent_event.content.parts[0].text if last_agent_event.content.parts else "Interaction complete."
+        # Attempt to process the last event content into one of the known API response types
+        processed_data = _process_adk_event_content(last_agent_event.content) # NEW helper
+        if isinstance(processed_data, TutorInteractionResponse):
+             final_response_data = processed_data
+        else: # Fallback to simple message
+             response_text = processed_data.get("text", "Interaction complete.")
         final_response_data = MessageResponse(response_type="message", text=response_text)
         print(f"[Interact] Orchestrator returned response of type: {final_response_data.response_type}")
         # If orchestrator called quiz creator which returned a question, update pending state
@@ -611,11 +610,7 @@
             message="There was an internal error processing your request."
         )
 
-    # --- Load final context state AFTER the run OR PAUSE ---
-    # The Runner should have updated the state via SessionService.append_event up to the pause/completion
-    final_session = session_service.get_session("ai_tutor", str(user.id), str(session_id))
-    if final_session and final_session.state:
-         final_context = TutorContext.model_validate(final_session.state)
+    # --- Final Context State is already loaded into final_context ---
 
     # --- Final Context State is already loaded into final_context ---
     print(f"[Interact] Saving final context state to Supabase for session {session_id}")
@@ -646,7 +641,7 @@
     request: Request,
     interaction_input: InteractionRequestData = Body(...), # Should have type 'answer'
     supabase: Client = Depends(get_supabase_client), # Get supabase for context update helper
-    session_service: SupabaseSessionService = Depends(get_session_service),
+    session_service: SupabaseSessionService = Depends(get_session_service), # Inject ADK Service
     tutor_context: TutorContext = Depends(get_tutor_context) # Get current context to find pause details
 ):
     logger = get_session_logger(session_id)
@@ -655,14 +650,14 @@
     print(f"Input Type: {interaction_input.type}, Data: {interaction_input.data}")
 
     if interaction_input.type != 'answer' or 'answer_index' not in interaction_input.data:
-        raise HTTPException(status_code=400, detail="Invalid input type or data for /answer endpoint. Expected type='answer' and data={'answer_index': number}.")
+        raise HTTPException(status_code=400, detail="Invalid input type or data for /answer endpoint. Expected type='answer' and data={'answer_index': number}. Received: type={}, data={}".format(interaction_input.type, interaction_input.data))
 
     # --- Retrieve details about the paused interaction ---
     if tutor_context.user_model_state.pending_interaction_type != 'checking_question' or \
        not tutor_context.user_model_state.pending_interaction_details or \
        'paused_tool_call_id' not in tutor_context.user_model_state.pending_interaction_details:
         logger.warning(f"Received answer for session {session_id}, but no valid pending interaction found in context.")
-        raise HTTPException(status_code=400, detail="No pending question found for this session.")
+        raise HTTPException(status_code=400, detail=f"No pending question found for this session {session_id}. Current pending type: {tutor_context.user_model_state.pending_interaction_type}")
 
     paused_tool_call_id = tutor_context.user_model_state.pending_interaction_details['paused_tool_call_id']
     user_answer_index = interaction_input.data['answer_index']
@@ -675,11 +670,11 @@
             parts=[
                 adk_types.Part.from_function_response(
                     name="ask_user_question_and_get_answer", # Tool name
-                    id=paused_tool_call_id, # CRUCIAL: Match the original tool call ID
+                    # id=paused_tool_call_id, # CRUCIAL: Match the original tool call ID - ADK uses 'name' for matching response
                     response={"answer_index": user_answer_index} # The data the tool's caller expects
                 )
-            ]
-        ),
+            ],
+            tool_code_parts=[adk_types.FunctionResponse(name='ask_user_question_and_get_answer', id=paused_tool_call_id, response={'answer_index': user_answer_index})], # Pass ID here for ADK matching
+        ),
         invocation_id=tutor_context.last_interaction_summary or f"resume_{session_id}", # Find appropriate invocation ID? Use last one?
     )
 
@@ -691,7 +686,12 @@
     # --- Resume the ADK Runner ---
     # Re-initialize runner and run_async, providing the answer event.
     # ADK's Runner should handle routing this event correctly based on the function_call_id.
-    orchestrator_agent = create_orchestrator_agent() # Recreate agent instance
+    # Recreate agent instance - ideally Runner holds/caches agent state
+    try:
+        orchestrator_agent = create_orchestrator_agent()
+    except Exception as agent_create_e:
+        logger.error(f"AgentCreation (Resume): Failed to create orchestrator: {agent_create_e}")
+        raise HTTPException(status_code=500, detail="Internal server error: Could not initialize agent for resume.")
     adk_runner = Runner("ai_tutor", orchestrator_agent, session_service)
     run_config = RunConfig(workflow_name="Tutor_Interaction_Resume", group_id=str(session_id))
 
@@ -700,7 +690,7 @@
         last_agent_event_after_resume: Optional[Event] = None
         question_after_resume: Optional[QuizQuestion] = None
         paused_id_after_resume: Optional[str] = None
-
+        
         async for event in adk_runner.run_async(
             user_id=str(user.id),
             session_id=str(session_id),
@@ -709,7 +699,7 @@
         ):
             print(f"[Answer] Received Event after resume: ID={event.id}, Author={event.author}")
             logger.log_orchestrator_output(event.content)
-            last_agent_event_after_resume = event
+            last_agent_event_after_resume = event # Track last event
 
             # Check for ANOTHER pause signal immediately after resume
             if event.actions and event.actions.custom_action:
@@ -725,20 +715,17 @@
                              break # Exit loop to send the new question
                          except Exception as parse_err:
                               logger.error(f"PauseSignalParse (Resume): Failed to parse question: {parse_err}")
-
+        # Handle case where loop finishes without yielding events
+        else: 
+             if not last_agent_event_after_resume and not question_after_resume:
+                 print("[Answer] Warning: ADK run after resume completed without yielding agent events or pause signals.")
+                 
     except Exception as resume_err:
          logger.log_error("ADKRunnerResume", f"Error during agent run after resume: {resume_err}")
          print(f"[Answer] Error during ADK Runner execution after resume: {resume_err}\n{traceback.format_exc()}")
          raise HTTPException(status_code=500, detail="Internal error resuming agent execution.")
 
     # --- Load final context state AFTER the resume run ---
-    final_session_after_resume = session_service.get_session("ai_tutor", str(user.id), str(session_id))
-    if final_session_after_resume and final_session_after_resume.state:
-         final_context_after_resume = TutorContext.model_validate(final_session_after_resume.state)
-         # Store the new paused tool call ID if another pause occurred
-         if paused_id_after_resume:
-             final_context_after_resume.user_model_state.pending_interaction_type = 'checking_question'
-             final_context_after_resume.user_model_state.pending_interaction_details = {"paused_tool_call_id": paused_id_after_resume}
-             await _update_context_in_db(session_id, user.id, final_context_after_resume, supabase)
-             logger.info(f"Stored new paused_tool_call_id {paused_id_after_resume} after resume.")
+    final_context_after_resume = await get_tutor_context(session_id, request, session_service) # Reload latest context
+    # Update context with pause details if needed (moved for clarity)
+    if paused_id_after_resume and not question_after_resume:
+        if final_context_after_resume.user_model_state.pending_interaction_type != 'checking_question':
+            final_context_after_resume.user_model_state.pending_interaction_type = 'checking_question'
+            final_context_after_resume.user_model_state.pending_interaction_details = {"paused_tool_call_id": paused_id_after_resume}
+            await _update_context_in_db(session_id, user.id, final_context_after_resume, supabase)
+            logger.info(f"Stored new paused_tool_call_id {paused_id_after_resume} after resume.")
     else:
         logger.error(f"ContextFetchAfterResume: Failed to fetch session state after resume for {session_id}")
         final_context_after_resume = tutor_context # Fallback (stale)
@@ -753,7 +740,11 @@
         print(f"[Answer] Responding with NEW question. Session {session_id} paused again.")
     elif last_agent_event_after_resume and last_agent_event_after_resume.content:
         # Completed after resume
-        response_text = last_agent_event_after_resume.content.parts[0].text if last_agent_event_after_resume.content.parts else "Processing complete."
+        processed_data = _process_adk_event_content(last_agent_event_after_resume.content) # Use helper
+        if isinstance(processed_data, TutorInteractionResponse):
+            final_response_data_after_resume = processed_data
+        else: # Fallback
+            response_text = processed_data.get("text", "Processing complete.")
         final_response_data_after_resume = MessageResponse(response_type="message", text=response_text)
         print(f"[Answer] Interaction completed after resume. Final response type: {final_response_data_after_resume.response_type}")
     else:
@@ -765,10 +756,10 @@
     return InteractionResponseData(
         content_type=final_response_data_after_resume.response_type,
         data=final_response_data_after_resume,
-        user_model_state=final_context_after_resume.user_model_state
+        user_model_state=final_context_after_resume.user_model_state # Return the LATEST state
     )
 
-# Helper function to process ADK events into API response data
+# Helper function to process ADK event content into API response data
 def _process_adk_event_for_api(event: Event, logger: TutorOutputLogger) -> InteractionResponseData:
     """Processes an ADK event and formats it for the API response."""
     # Initialize response variables
@@ -785,11 +776,6 @@
         response_data = ErrorResponse(response_type="error", message=error_text)
         response_type = "error"
 
-        # Return the error response directly. Need to wrap in InteractionResponseData
-        # with appropriate state. Since we don't have context here, use default state.
-        error_payload = ErrorResponse(
-            error=error_text, message="An error occurred during agent processing."
-        )
         # Log and return immediately for error
         logger.log_error("AgentEventError", f"Error in event {event.id}: {error_text}")
         # Need to get the current UserModelState somehow to return here.
@@ -797,13 @@
         # Placeholder - returning default. THIS NEEDS REFACTORING.
         # ** Refactor Recommendation: This function should NOT return InteractionResponseData **
         # ** Instead, it should return the processed Pydantic model (MessageResponse, QuestionResponse, etc.) **
-        # ** or an error string/dict. The calling endpoint (/interact, /answer) should assemble **
+        # ** or an error string/dict. The calling endpoint (/interact, /answer) should assemble the **
         # ** the final InteractionResponseData using the latest UserModelState. **
-        # For now, returning default state:
+        # Returning default state for now:
         return InteractionResponseData(
             content_type="error", # Use 'error' as the content type
             data=error_payload, # Embed the ErrorResponse
             user_model_state=UserModelState() # Provide a default UserModelState
         )
-
     # Process event content if available
     if event.content:
         for part in event.content.parts:
@@ -812,8 @@
                 text = part.text.strip()
                 if text: # Ensure there's text content
                     logger.info(f"Processing text part: {text[:100]}...")
-                    # Assume text is a general message unless specific parsing is added
                     response_type = "message"
+                    # Assume text is a general message unless specific parsing is added
                     response_data = MessageResponse(response_type="message", text=text)
                     break  # Text content takes precedence
 
@@ -823,7 @@
                 tool_args = part.function_call.args or {}
                 logger.info(f"Processing function call part: {tool_name}")
 
-                # Example: Map tool names to response types (NEEDS EXPANSION)
+                # Example: Map tool names to response types (NEEDS EXPANSION based on tools)
                 if tool_name == "present_explanation_tool": # Hypothetical tool name
                     try:
                         # Assuming args match ExplanationResponse directly
@@ -832,7 @@
                         logger.error(f"Validation failed for explanation tool args: {e}")
                         response_data = ErrorResponse(response_type="error", message=f"Invalid data from explanation tool: {e}")
                         response_type = "error"
-                elif tool_name == "ask_checking_question_tool": # Hypothetical tool name
+                elif tool_name == "ask_user_question_and_get_answer": # Match the actual tool name
                     try:
                         # Args should match QuizQuestion schema
                         question = QuizQuestion.model_validate(tool_args)
@@ -841,15 +827,12 @@
                             question=question,
                             topic=tool_args.get("topic", "Unknown Topic") # Assuming topic is passed
                         )
-                        response_type = "question"
-                    except ValidationError as e:
+                    except (ValidationError, KeyError) as e:
                         logger.error(f"Validation failed for question tool args: {e}")
                         response_data = ErrorResponse(response_type="error", message=f"Invalid data from question tool: {e}")
                         response_type = "error"
-                # Handle other tool calls as intermediate steps
-                else:
-                    response_type = "intermediate"
-                    response_data = {"tool_name": tool_name, "args": tool_args}
+                # Add other tool call mappings if needed
+                # If no mapping, treat as intermediate? Or error? For now, maybe message.
+                else: 
+                     response_type = "message"
+                     response_data = MessageResponse(response_type="message", text=f"Agent requested tool: {tool_name}")
 
                 break  # Process one function call at a time
 
@@ -857,35 +840,40 @@
             elif part.function_response:
                 # Process tool results if needed (often just updates state, maybe no API response)
                 tool_name = part.function_response.name
-                tool_response = part.function_response.response or {}
+                # Safely access response, default to empty dict if None
+                tool_response = part.function_response.response if part.function_response.response is not None else {}
                 logger.info(f"Processing function response for tool: {tool_name}")
                 # Example: If feedback tool returns data, format it
-                if tool_name == "call_quiz_teacher_evaluate_tool": # Hypothetical name
+                if tool_name == "call_quiz_teacher_evaluate": # Match actual tool name
                     try:
                         feedback_item = QuizFeedbackItem.model_validate(tool_response)
-                        response_data = FeedbackResponse(response_type="feedback", feedback=feedback_item)
+                        response_data = FeedbackResponse(
+                            response_type="feedback", 
+                            feedback=feedback_item,
+                            topic=tool_response.get("topic", "Feedback Topic") # Assuming topic might be in response
+                        )
                         response_type = "feedback"
                     except ValidationError as e:
                         logger.error(f"Validation failed for feedback tool response: {e}")
                         response_data = ErrorResponse(response_type="error", message=f"Invalid data from feedback tool: {e}")
                         response_type = "error"
                 else:
-                    # Other tool responses might just be intermediate updates
+                    # Other tool responses often just update state, maybe no specific API response needed
                     response_type = "intermediate"
                     response_data = {
                         "tool_name": tool_name,
                         "response": tool_response,
                     }
-
+                break # Process one function response
+                
     # Handle case where no meaningful response was generated
     if response_type == "intermediate" and response_data is None:
         logger.warning(f"Event {event.id} did not yield a direct API response.")
-        # Fallback to a generic message or processing status
-        response_data = {"status": "processing"}
         response_data = MessageResponse(response_type="message", text="Processing your request...")
         response_type = "message" # Change type to message
 
-    # Cannot return a dict, must be InteractionResponseData.
-    # Return default state for intermediate for now.
-    # ** THIS IS THE REFACTOR POINT - Return the Pydantic model, not InteractionResponseData **
+    # ** Refactor Point: Return the processed Pydantic model (or error dict) **
+    # The caller (/interact or /answer) will construct InteractionResponseData
+    if isinstance(response_data, (ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse)):
+        return response_data
+    else: # Handle intermediate or unexpected cases
+        # Return a generic message or a dict indicating intermediate status
+        # For safety, return a MessageResponse
+        return MessageResponse(response_type="message", text="Processing...")
+
+
 # --- Helper to update context in DB (replace direct calls) ---
-async def _update_context_in_db(session_id: UUID, user_id: UUID, context: TutorContext, supabase: Client):
+async def _update_context_in_db(session_id: UUID, user_id: UUID, context: TutorContext, supabase: Client) -> bool: # Added return type hint
     """Helper to persist context via SupabaseSessionService interface."""
     # This mimics how the SessionService's append_event would work
     try:
         context_dict = context.model_dump(mode='json')
         response: PostgrestAPIResponse = supabase.table("sessions").update(
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Diff
IGNORE_WHEN_COPYING_END

5. Refactor _process_adk_event_for_api Callers in routers/tutor.py

Modify the /interact and /answer endpoints to:

Call _process_adk_event_for_api to get the core response payload (e.g., MessageResponse, QuestionResponse).

Load the final UserModelState after the run_async loop.

Construct the InteractionResponseData using the payload from the helper and the loaded final state.

--- a/ai_tutor/routers/tutor.py
+++ b/ai_tutor/routers/tutor.py
@@ -591,11 +591,15 @@
     # --- Logic to handle the COMPLETED state (no pause detected) ---
     elif last_agent_event and last_agent_event.content:
         # Agent run completed normally. Assume the last event's content is the response.
-        # This should ideally be the final output from the Orchestrator agent based on its output_schema.
-        # TODO: Adapt this based on Orchestrator's final output schema.
-        # For now, assume it's a text message.
-        response_text = last_agent_event.content.parts[0].text if last_agent_event.content.parts else "Interaction complete."
+        # Process the event content to get the core payload
+        processed_payload = _process_adk_event_for_api(last_agent_event, logger) # Use logger instance
+        if isinstance(processed_payload, ErrorResponse):
+             # Handle error payload specifically
+             final_response_data = processed_payload
+        elif isinstance(processed_payload, TutorInteractionResponse):
+             # Got a valid response payload
+             final_response_data = processed_payload
+        else: # Fallback if helper returned unexpected type
+             response_text = str(processed_payload) # Convert to string as fallback
         final_response_data = MessageResponse(response_type="message", text=response_text)
         print(f"[Interact] Orchestrator returned response of type: {final_response_data.response_type}")
         # If orchestrator called quiz creator which returned a question, update pending state
@@ -608,7 +612,7 @@
     else:
         # No event received or last event had no content
         error_msg = f"Agent interaction finished without a final response event."
-        print(f"[Interact] Error: {error_msg}")
+        logger.error(f"[Interact] Error for session {session_id}: {error_msg}") # Use logger
         final_response_data = ErrorResponse(
             error=error_msg,
             message="There was an internal error processing your request."
@@ -626,7 @@
 
     # Return the structured response
     return InteractionResponseData(
-        content_type=final_response_data.response_type,
+        content_type=getattr(final_response_data, 'response_type', 'error'), # Safely get type
         data=final_response_data, # Send the response from the final agent run
         user_model_state=final_context.user_model_state # Send final updated state
     )
@@ -712,7 +716,11 @@
         print(f"[Answer] Responding with NEW question. Session {session_id} paused again.")
     elif last_agent_event_after_resume and last_agent_event_after_resume.content:
         # Completed after resume
-        response_text = last_agent_event_after_resume.content.parts[0].text if last_agent_event_after_resume.content.parts else "Processing complete."
+        processed_payload = _process_adk_event_for_api(last_agent_event_after_resume, logger) # Use logger
+        if isinstance(processed_payload, TutorInteractionResponse):
+            final_response_data_after_resume = processed_payload
+        else: # Fallback
+            response_text = str(processed_payload)
         final_response_data_after_resume = MessageResponse(response_type="message", text=response_text)
         print(f"[Answer] Interaction completed after resume. Final response type: {final_response_data_after_resume.response_type}")
     else:
@@ -722,11 +730,11 @@
         final_response_data_after_resume = ErrorResponse(error=error_msg, message="Internal processing error after submitting answer.")
 
     return InteractionResponseData(
-        content_type=final_response_data_after_resume.response_type,
+        content_type=getattr(final_response_data_after_resume, 'response_type', 'error'), # Safely get type
         data=final_response_data_after_resume,
         user_model_state=final_context_after_resume.user_model_state # Return the LATEST state
     )
 
-# Helper function to process ADK events into API response data
+# Helper function to process ADK event content into API response payload (not InteractionResponseData)
 def _process_adk_event_for_api(event: Event, logger: TutorOutputLogger) -> InteractionResponseData:
     """Processes an ADK event and formats it for the API response."""
     # Initialize response variables
@@ -736,25 +744,16 @@
 
     # Handle error events first
     if event.error_code or event.error_message:
-        error_text = event.error_message or f"An error occurred (Code: {event.error_code})"
-        response_data = ErrorResponse(response_type="error", message=error_text)
+        error_text = event.error_message or f"Agent error occurred (Code: {event.error_code})"
+        response_payload = ErrorResponse(response_type="error", message=error_text)
         response_type = "error"
-
-        # Need to get the current UserModelState somehow to return here.
-        # This function signature does not have access to the full TutorContext.
-        # Placeholder - returning default. THIS NEEDS REFACTORING.
-        # ** Refactor Point: Return the processed Pydantic model (or error dict) **
-        # The caller (/interact or /answer) will construct InteractionResponseData
-        # Returning default state for now:
-        error_payload = ErrorResponse(
-            error=error_text, message="An error occurred during agent processing."
-        )
         logger.log_error("AgentEventError", f"Error in event {event.id}: {error_text}")
         # ** Return the payload, not the full InteractionResponseData **
         return error_payload
+        
 
     # Process event content if available
     if event.content:
         for part in event.content.parts:
@@ -763,7 @@
                 text = part.text.strip()
                 if text: # Ensure there's text content
                     logger.info(f"Processing text part: {text[:100]}...")
-                    response_data = MessageResponse(response_type="message", text=text)
+                    response_payload = MessageResponse(response_type="message", text=text)
                     response_type = "message"
                     break  # Text content takes precedence
 
@@ -773,28 +772,28 @@
                 tool_args = part.function_call.args or {}
                 logger.info(f"Processing function call part: {tool_name}")
 
-                # Example: Map tool names to response types (NEEDS EXPANSION based on tools)
+                # Example: Map tool names to response types
                 if tool_name == "present_explanation_tool": # Hypothetical tool name
                     try:
-                        response_data = ExplanationResponse.model_validate(tool_args)
+                        response_payload = ExplanationResponse.model_validate(valid_args)
                         response_type = "explanation"
                     except ValidationError as e:
                         logger.error(f"Validation failed for explanation tool args: {e}")
-                        response_data = ErrorResponse(response_type="error", message=f"Invalid data from explanation tool: {e}")
+                        response_payload = ErrorResponse(response_type="error", message=f"Invalid data from explanation tool: {e}")
                         response_type = "error"
-                elif tool_name == "ask_checking_question_tool": # Hypothetical tool name
+                elif tool_name == "ask_user_question_and_get_answer": # Match the actual tool name
                     try:
                         # Args should match QuizQuestion schema
                         question = QuizQuestion.model_validate(tool_args)
-                        response_data = QuestionResponse(
+                        response_payload = QuestionResponse(
                             response_type="question",
                             question=question,
                             topic=tool_args.get("topic", "Unknown Topic") # Assuming topic is passed
                         )
                         response_type = "question"
                     except (ValidationError, KeyError) as e:
                         logger.error(f"Validation failed for question tool args: {e}")
-                        response_data = ErrorResponse(response_type="error", message=f"Invalid data from question tool: {e}")
+                        response_payload = ErrorResponse(response_type="error", message=f"Invalid data from question tool: {e}")
                         response_type = "error"
                 # Add other tool call mappings if needed
                 # If no mapping, treat as intermediate? Or error? For now, maybe message.
@@ -809,18 +808,18 @@
                 # Process tool results if needed (often just updates state, maybe no API response)
                 tool_name = part.function_response.name
                 tool_response = part.function_response.response if part.function_response.response is not None else {}
-                logger.info(f"Processing function response for tool: {tool_name}")
+                logger.info(f"Processing function response for tool: {tool_name}") # Process function response
                 # Example: If feedback tool returns data, format it
                 if tool_name == "call_quiz_teacher_evaluate": # Match actual tool name
                     try:
                         feedback_item = QuizFeedbackItem.model_validate(tool_response)
-                        response_data = FeedbackResponse(
+                        response_payload = FeedbackResponse(
                             response_type="feedback",
                             feedback=feedback_item,
                             topic=tool_response.get("topic", "Feedback Topic") # Assuming topic might be in response
                         )
                         response_type = "feedback"
-                    except ValidationError as e:
+                    except (ValidationError, KeyError) as e:
                         logger.error(f"Validation failed for feedback tool response: {e}")
                         response_data = ErrorResponse(response_type="error", message=f"Invalid data from feedback tool: {e}")
                         response_type = "error"
@@ -831,18 +830,16 @@
                         "tool_name": tool_name,
                         "response": tool_response,
                     }
+                # If a specific payload isn't generated, maybe return the response dict
+                response_payload = response_data if response_type != "intermediate" else {"status": "tool_executed", "tool_name": tool_name}
 
                 break # Process one function response
-                
+
     # Handle case where no meaningful response was generated
     if response_type == "intermediate" and response_data is None:
         logger.warning(f"Event {event.id} did not yield a direct API response.")
-        # Fallback to a generic message or processing status
         response_payload = MessageResponse(response_type="message", text="Processing your request...")
-        response_type = "message" # Change type to message
-
-    # Return the processed Pydantic model (or error dict)
-    if isinstance(response_data, (ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse)):
+        response_type = "message"
+
+    if isinstance(response_payload, (ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse)):
         return response_payload
     else: # Handle intermediate or unexpected cases
         # Return a generic message or a dict indicating intermediate status
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Diff
IGNORE_WHEN_COPYING_END

