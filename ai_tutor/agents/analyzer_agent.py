import os
import logging
from typing import Optional, List, Dict
from pydantic import BaseModel
from uuid import UUID
from supabase import Client

from google.adk.agents import LlmAgent
from google.adk.runners import Runner, RunConfig
from google.adk.tools import FunctionTool, ToolContext
from google.adk.models import Gemini

logger = logging.getLogger(__name__)

class FileMetadata(BaseModel):
    """Metadata for a single file."""
    filename: str
    file_type: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    version: Optional[str] = None

class KeyConcept(BaseModel):
    """Represents a key concept found in the documents."""
    name: str
    description: str
    examples: List[str]
    related_concepts: List[str] = []

class KeyTerm(BaseModel):
    """Represents an important term and its definition."""
    term: str
    definition: str
    context: Optional[str] = None
    examples: List[str] = []

class AnalysisResult(BaseModel):
    """Complete analysis results for a set of documents."""
    file_metadata: List[FileMetadata]
    key_concepts: List[KeyConcept]
    key_terms: List[KeyTerm]
    file_names: List[str]

def create_analyzer_agent() -> LlmAgent:
    """Create an analyzer agent that reads full document content."""
    
    # Import the document content tool
    from ai_tutor.tools.orchestrator_tools import get_document_content
    
    analyzer_tools = [get_document_content]
    
    # Use Gemini model via ADK
    model_name = "gemini-1.5-pro"  # Using Pro for better analysis capabilities
    
    # Create the analyzer agent
    analyzer_agent = LlmAgent(
        name="Document Analyzer",
        instructions="""
        You are an expert document analyzer. Your task is to analyze the documents whose content will be provided by tools.
        
        ANALYSIS GOALS:
        1. Document metadata and structure
        2. Key concepts and their relationships
        3. Important terminology
        4. Key terms and their definitions
        
        ANALYSIS PROCESS:
        1. You will receive the full text content of one or more documents via the `get_document_content` tool results.
        2. If not all document content is provided initially, use the `get_document_content` tool, providing the necessary file path.
        3. Analyze the FULL TEXT content provided for each document.
        4. Extract file names (from tool arguments or context), metadata (if found within text), key concepts, and key terms/definitions.
        5. Organize all findings into a comprehensive analysis.
        
        ANALYSIS STRATEGY:
        - Read through the provided text content for each document
        - Look for document headers, introductions, conclusions, summaries, and definition sections
        - Identify recurring themes, topics, and specialized vocabulary
        - Look for key section headers and topics
        - Search for defined terms, glossary sections, or key terminology with explanations
        
        FORMAT INSTRUCTIONS:
        - Present your analysis in a clear, structured format
        - Include the following sections:
          * OVERVIEW: Brief summary of analyzed documents
          * FILE METADATA: Any metadata you find for each file
          * KEY CONCEPTS: List of main topics/concepts found across all documents
          * CONCEPT DETAILS: Examples or details for each key concept
          * KEY TERMS GLOSSARY: List of important terminology with their definitions
        
        DO NOT:
        - Do not make assumptions about content you haven't seen
        - Do not stop until you have analyzed all relevant documents
        - Do not include placeholder or generic content
        """,
        tools=analyzer_tools,
        model=model_name,
    )
    
    return analyzer_agent

async def analyze_documents(context=None, supabase: Client = None) -> Optional[AnalysisResult]:
    """
    Analyze documents based on file paths stored in the context.
    
    Args:
        context: Context object with session_id and file paths
        supabase: Optional Supabase client instance for saving KB
        
    Returns:
        An AnalysisResult object containing the analysis text and extracted metadata, or None on failure.
    """
    if not context or not hasattr(context, 'uploaded_file_paths') or not context.uploaded_file_paths:
        logger.error("analyze_documents: No uploaded file paths found in context.")
        return None

    file_paths = context.uploaded_file_paths
    if not file_paths:
        logger.warning("analyze_documents: File paths list in context is empty.")
        return None

    # Create the analyzer agent
    analyzer_agent = create_analyzer_agent()
    
    # Setup RunConfig for tracing if we have context
    run_config = None
    if context and hasattr(context, 'session_id'):
        run_config = RunConfig(
            workflow_name="AI Tutor - Document Analysis",
            group_id=str(context.session_id)
        )
    
    # Create prompt for the agent
    file_list_str = "\n - ".join(file_paths)
    prompt = f"""
    Please analyze the content of the following documents. Use the `get_document_content` tool for each file path listed below to retrieve its full text content first:
    - {file_list_str}
    
    After retrieving the content for all files, perform a comprehensive analysis based on your instructions, extracting:
    - File names (use the paths provided) and any metadata found within the text
    - Key concepts/topics across all documents
    - Key terms and their definitions found within the text
    
    Present your combined findings in the specified structured format.
    """
    
    # Run the analyzer agent
    try:
        result = await Runner.run(
            analyzer_agent,
            prompt,
            run_config=run_config
        )
        
        if not result or not result.content:
            logger.error("analyze_documents: No content in analysis result")
            return None
            
        # Parse the analysis text into structured data
        analysis_text = result.content.parts[0].text if result.content.parts else ""
        
        # Extract components from the analysis text
        file_metadata: List[FileMetadata] = []
        key_concepts: List[KeyConcept] = []
        key_terms: List[KeyTerm] = []
        
        # TODO: Implement proper parsing of the analysis text into structured data
        # For now, create basic metadata from file paths
        for file_path in file_paths:
            file_metadata.append(FileMetadata(
                filename=os.path.basename(file_path),
                file_type=os.path.splitext(file_path)[1]
            ))
        
        # Create the analysis result
        analysis_result = AnalysisResult(
            file_metadata=file_metadata,
            key_concepts=key_concepts,
            key_terms=key_terms,
            file_names=[os.path.basename(path) for path in file_paths]
        )
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"analyze_documents: Error during analysis: {str(e)}")
        return None 