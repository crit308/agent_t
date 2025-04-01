from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import json

from agents import Agent, FileSearchTool, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from agents.run_context import RunContextWrapper

from ai_tutor.agents.models import LearningObjective, LessonSection, LessonPlan
from ai_tutor.agents.utils import RoundingModelWrapper
from ai_tutor.context import TutorContext


def create_planner_agent(vector_store_id: str) -> Agent[TutorContext]:
    """Create a planner agent that uses context for document analysis."""

    # Create a FileSearchTool that can search the vector store containing the uploaded documents
    file_search_tool = FileSearchTool(
        vector_store_ids=[vector_store_id],
        max_num_results=5,
        include_search_results=True,
    )

    print(f"Created FileSearchTool for vector store: {vector_store_id}")

    # Instantiate the base model provider and get the base model
    provider: ModelProvider = OpenAIProvider()
    base_model = provider.get_model("gpt-4o-2024-08-06")

    # Create the planner agent with access to the file search tool
    planner_agent = Agent(
        name="Lesson Planner",
        instructions="""
        You are an expert curriculum designer. Your task is to create a 
        well-structured lesson plan based on the documents that have been uploaded.
        
        AVAILABLE INFORMATION:
        - The results of a prior document analysis (key concepts, terms, etc.) are available in the context (`analysis_result`). You DO NOT need a tool to read this.
        - You have a `file_search` tool to look up specific details within the documents if needed.
        
        STEP 1: KNOWLEDGE BASE ANALYSIS
        - Review the key concepts and terms provided in the context (`analysis_result`).
        
        STEP 2: DETAILED DOCUMENT SEARCH
        - Use the `file_search` tool ONLY IF necessary to clarify details about the concepts or structure identified in the analysis.
        - Perform multiple searches to gather comprehensive information.
        
        STEP 3: LESSON PLAN CREATION
        - Based on the analysis results and document searches, create a comprehensive lesson plan.
        - Your lesson plan must include:
          * Clear learning objectives for each section
          * Logical sequence of sections
          * Key concepts to cover in each section (`concepts_to_cover` field is crucial)
          * Estimated duration for each section
          * Target audience considerations
          * Any prerequisites
        - Output the lesson plan as a complete structured LessonPlan object.
        
        CRITICAL:
        - DO NOT use the `read_knowledge_base` tool. Use the analysis results already available in your context.
        - DO NOT call any handoff tools. Your only output should be the LessonPlan object.
        """,
        tools=[file_search_tool],
        output_type=LessonPlan,
        model=RoundingModelWrapper(base_model),
    )
    
    return planner_agent 