import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, model_validator
from uuid import UUID
from supabase import Client

from agents import Agent, FileSearchTool, Runner, trace, gen_trace_id, set_tracing_export_api_key, RunConfig, ModelProvider
from agents.models.openai_provider import OpenAIProvider
from ai_tutor.agents.utils import RoundingModelWrapper


class FileMetadata(BaseModel):
    """Metadata for a single file."""
    title: Optional[str] = Field(None, description="Title of the file if available")
    author: Optional[str] = Field(None, description="Author of the file if available")
    date: Optional[str] = Field(None, description="Date of the file if available")
    type: Optional[str] = Field(None, description="Type or format of the file")
    size: Optional[str] = Field(None, description="Size of the file if available")
    
    # Allow additional properties for any other metadata found
    model_config = {
        "extra": "allow"
    }


class ConceptInfo(BaseModel):
    """Information about a single concept."""
    examples: List[str] = Field(default_factory=list, description="Examples of the concept from documents")
    description: Optional[str] = Field(None, description="Brief description of the concept")
    
    model_config = {
        "extra": "allow"
    }


class DocumentAnalysis(BaseModel):
    """Analysis results of documents in a vector store."""
    file_names: List[str] = Field(default_factory=list, description="Names of the files analyzed")
    file_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Metadata of the files by file name")
    key_concepts: List[str] = Field(default_factory=list, description="Key concepts or topics extracted from the documents")
    concept_details: Dict[str, List[str]] = Field(default_factory=dict, description="Details about each key concept")
    key_terms: Dict[str, str] = Field(default_factory=dict, description="Important terminology with their definitions")
    vector_store_id: str = Field("", description="ID of the vector store containing the documents")
    file_ids: List[str] = Field(default_factory=list, description="Vector store reference IDs for the files")
    
    @model_validator(mode='after')
    def ensure_defaults(self):
        """Ensure all fields have at least empty default values."""
        if not self.file_names:
            self.file_names = []
        if not self.file_metadata:
            self.file_metadata = {}
        if not self.key_concepts:
            self.key_concepts = []
        if not self.concept_details:
            self.concept_details = {}
        if not self.key_terms:
            self.key_terms = {}
        if not self.file_ids:
            self.file_ids = []
        return self


class AnalysisResult(BaseModel):
    """Structured result from document analysis."""
    analysis_text: str
    key_concepts: List[str] = Field(default_factory=list)
    key_terms: Dict[str, str] = Field(default_factory=dict)
    file_names: List[str] = Field(default_factory=list)
    vector_store_id: str


def create_analyzer_agent(vector_store_id: str, api_key: str = None):
    """Create an analyzer agent with access to the provided vector store."""
    
    # If API key is provided, ensure it's set in environment
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Ensure OPENAI_API_KEY is set in the environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable is not set for analyzer agent!")
    else:
        print(f"Using OPENAI_API_KEY from environment for analyzer agent")
    
    # Create a FileSearchTool that can search the vector store containing the uploaded documents
    file_search_tool = FileSearchTool(
        vector_store_ids=[vector_store_id],
        max_num_results=10,  # Increase max results to get more comprehensive analysis
        include_search_results=True,
    )
    
    print(f"Created FileSearchTool for analyzer agent using vector store: {vector_store_id}")
    
    # Instantiate the base model provider and get the base model
    provider = OpenAIProvider()
    base_model = provider.get_model("o3-mini")
    
    # Create the analyzer agent with access to the file search tool
    analyzer_agent = Agent(
        name="Document Analyzer",
        instructions="""
        You are an expert document analyzer. Your task is to analyze the documents in the vector store
        and extract the following information:
        
        1. File names and metadata
        2. Key concepts or topics from the documents
        3. Vector store reference IDs
        4. Key terms and their definitions
        
        ANALYSIS PROCESS:
        1. Use the file_search tool with broad queries to understand what documents are available
        2. Conduct systematic searches for common document metadata fields
        3. Extract key concepts by analyzing document content and structure
        4. Identify and record vector store reference IDs
        5. Extract important terminology and their definitions
        6. Organize all findings into a comprehensive analysis
        
        SEARCH STRATEGY:
        - Start with general searches like "document", "overview", "introduction"
        - Search for specific metadata terms like "author", "date", "title", "version" 
        - Look for key section headers and topics
        - Extract unique identifiers and reference numbers
        - Search for defined terms, glossary sections, or key terminology with explanations
        
        FORMAT INSTRUCTIONS:
        - Present your analysis in a clear, structured text format
        - Include the following sections:
          * VECTOR STORE ID: The ID of the vector store
          * FILES: List of all document names you discover
          * FILE METADATA: Any metadata you find for each file
          * KEY CONCEPTS: List of main topics/concepts found across all documents
          * CONCEPT DETAILS: Examples or details for each key concept
          * KEY TERMS GLOSSARY: List of important terminology with their definitions
          * FILE IDS: Any reference IDs you discover
        
        DO NOT:
        - Do not reference any tools or future steps in your output
        - Do not return incomplete analysis
        """,
        tools=[file_search_tool],
        # No specific output type - will return plain text
        model=RoundingModelWrapper(base_model),
    )
    
    return analyzer_agent


async def analyze_documents(vector_store_id: str, api_key: str = None, context=None, supabase: Client = None) -> Optional[AnalysisResult]:
    """Analyze documents in the provided vector store.
    
    Args:
        vector_store_id: ID of the vector store containing documents to analyze
        api_key: Optional OpenAI API key
        context: Optional context object with session_id for tracing
        supabase: Optional Supabase client instance for saving KB.
        
    Returns:
        An AnalysisResult object containing the text and extracted metadata, or None on failure.
    """
    # Create the analyzer agent
    agent = create_analyzer_agent(vector_store_id, api_key)
    
    # Setup RunConfig for tracing
    run_config = None
    if context and hasattr(context, 'session_id'):
        run_config = RunConfig(
            workflow_name="AI Tutor - Document Analysis",
            group_id=str(context.session_id) # Convert UUID to string
        )
    elif api_key:
        # If no context provided but we have API key, create a basic RunConfig
        run_config = RunConfig(
            workflow_name="AI Tutor - Document Analysis"
        )
    
    # Create a prompt that instructs the agent to perform comprehensive analysis
    prompt = """
    Analyze all documents in the vector store thoroughly.
    
    Search across the entire content of all documents to:
    1. Identify all file names and their metadata
    2. Extract key concepts, topics, and themes
    3. Find and record any vector store reference IDs
    4. Extract important terminology and provide clear definitions
    
    Be methodical and comprehensive in your analysis. Start with broad searches 
    and then focus on specific areas. Present your findings in a clear, structured format.
    
    The vector store ID you are analyzing is: {0}
    """.format(vector_store_id)
    
    # Run the analyzer agent to perform document analysis
    result = await Runner.run(
        agent, 
        prompt,
        run_config=run_config,
        context=context
    )
    
    # Get the text output directly
    analysis_text = result.final_output if isinstance(result.final_output, str) else str(result.final_output)
    if not analysis_text:
         print("Error: Document analysis agent returned empty output.")
         return None

    print("Successfully completed document analysis")
    
    # --- Save analysis text (Knowledge Base) to Supabase folders table ---
    if supabase and context and context.folder_id:
        folder_id_to_update = str(context.folder_id) # Ensure UUID is string
        try:
            update_response = supabase.table("folders").update(
                {"knowledge_base": analysis_text}
            ).eq("id", folder_id_to_update).execute()

            if update_response.data:
                print(f"Knowledge Base saved to Supabase for folder {context.folder_id}")
            elif update_response.error: # Check for explicit error
                print(f"Error saving Knowledge Base to Supabase folder {context.folder_id}: {update_response.error}")
        except Exception as e:
            print(f"Exception saving Knowledge Base to Supabase folder {context.folder_id}: {e}")
    # --- End Supabase KB saving ---
    
    # Extract key concepts for use in other parts of the application
    key_concepts = []
    key_terms = {}
    file_names = []
    try:
        # Parse key concepts from the text output for easier access
        if "KEY CONCEPTS:" in analysis_text:
            concepts_section = analysis_text.split("KEY CONCEPTS:")[1].split("CONCEPT DETAILS:")[0]
            key_concepts = [c.strip() for c in concepts_section.strip().split("\n") if c.strip()]

        # Extract key terms if available
        if "KEY TERMS GLOSSARY:" in analysis_text:
            terms_section = analysis_text.split("KEY TERMS GLOSSARY:")[1]
            # Check if there's a section after KEY TERMS GLOSSARY
            next_sections = ["FILE IDS:", "VECTOR STORE ID:"]
            for section in next_sections:
                if section in terms_section:
                    terms_section = terms_section.split(section)[0]
                    break
            
            # Process the terms section to extract terms and definitions
            terms_lines = [line.strip() for line in terms_section.strip().split("\n") if line.strip()]
            for line in terms_lines:
                if ":" in line:
                    term, definition = line.split(":", 1)
                    key_terms[term.strip()] = definition.strip()
                elif "–" in line or "-" in line:
                    # Handle terms with dash separator
                    parts = line.split("–", 1) if "–" in line else line.split("-", 1)
                    if len(parts) == 2:
                        term, definition = parts
                        key_terms[term.strip()] = definition.strip()
        
        # Extract file names if possible
        if "FILES:" in analysis_text:
            files_section = analysis_text.split("FILES:")[1].split("FILE METADATA:")[0]
            file_names = [f.strip() for f in files_section.strip().split("\n") if f.strip()]
        
    except Exception as e:
        print(f"Warning: Could not parse key concepts from analysis: {e}")
    
    # Return the structured result
    return AnalysisResult(
        analysis_text=analysis_text,
        key_concepts=key_concepts,
        key_terms=key_terms,
        file_names=file_names,
        vector_store_id=vector_store_id
    ) 