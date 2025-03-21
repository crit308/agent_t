import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, model_validator

from agents import Agent, FileSearchTool, Runner, trace, gen_trace_id, set_tracing_export_api_key


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
        if not self.file_ids:
            self.file_ids = []
        return self


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
    
    # Create the analyzer agent with access to the file search tool
    analyzer_agent = Agent(
        name="Document Analyzer",
        instructions="""
        You are an expert document analyzer. Your task is to analyze the documents in the vector store
        and extract the following information:
        
        1. File names and metadata
        2. Key concepts or topics from the documents
        3. Vector store reference IDs
        
        ANALYSIS PROCESS:
        1. Use the file_search tool with broad queries to understand what documents are available
        2. Conduct systematic searches for common document metadata fields
        3. Extract key concepts by analyzing document content and structure
        4. Identify and record vector store reference IDs
        5. Organize all findings into a comprehensive analysis
        
        SEARCH STRATEGY:
        - Start with general searches like "document", "overview", "introduction"
        - Search for specific metadata terms like "author", "date", "title", "version" 
        - Look for key section headers and topics
        - Extract unique identifiers and reference numbers
        
        FORMAT INSTRUCTIONS:
        - Present your analysis in a clear, structured text format
        - Include the following sections:
          * VECTOR STORE ID: The ID of the vector store
          * FILES: List of all document names you discover
          * FILE METADATA: Any metadata you find for each file
          * KEY CONCEPTS: List of main topics/concepts found across all documents
          * CONCEPT DETAILS: Examples or details for each key concept
          * FILE IDS: Any reference IDs you discover
        
        DO NOT:
        - Do not reference any tools or future steps in your output
        - Do not return incomplete analysis
        """,
        tools=[file_search_tool],
        # No specific output type - will return plain text
        model="o3-mini",  # Using a model that's good at analysis
    )
    
    return analyzer_agent


async def analyze_documents(vector_store_id: str, api_key: str = None) -> str:
    """Analyze documents in the provided vector store.
    
    Returns:
        A string containing the analysis results in text format.
    """
    # Create the analyzer agent
    agent = create_analyzer_agent(vector_store_id, api_key)
    
    # Create trace ID for monitoring the analysis process
    trace_id = gen_trace_id()
    print(f"Analyzing documents with trace ID: {trace_id}")
    print(f"View trace at: https://platform.openai.com/traces/{trace_id}")
    
    # Ensure API key is set for tracing
    if api_key:
        set_tracing_export_api_key(api_key)
    
    with trace("Document analysis", trace_id=trace_id):
        # Create a prompt that instructs the agent to perform comprehensive analysis
        prompt = """
        Analyze all documents in the vector store thoroughly.
        
        Search across the entire content of all documents to:
        1. Identify all file names and their metadata
        2. Extract key concepts, topics, and themes
        3. Find and record any vector store reference IDs
        
        Be methodical and comprehensive in your analysis. Start with broad searches 
        and then focus on specific areas. Present your findings in a clear, structured format.
        
        The vector store ID you are analyzing is: {0}
        """.format(vector_store_id)
        
        # Run the analyzer agent to perform document analysis
        result = await Runner.run(agent, prompt)
        
        # Get the text output directly
        analysis_text = result.final_output
        print("Successfully completed document analysis")
        
        # Extract key concepts for use in other parts of the application
        try:
            # Parse key concepts from the text output for easier access
            key_concepts = []
            if "KEY CONCEPTS:" in analysis_text:
                concepts_section = analysis_text.split("KEY CONCEPTS:")[1].split("CONCEPT DETAILS:")[0]
                key_concepts = [c.strip() for c in concepts_section.strip().split("\n") if c.strip()]
            
            # Attach the concepts as an attribute to the text for easy access
            setattr(analysis_text, "key_concepts", key_concepts)
            
            # Attach the vector store ID as an attribute
            setattr(analysis_text, "vector_store_id", vector_store_id)
            
            # Extract file names if possible
            file_names = []
            if "FILES:" in analysis_text:
                files_section = analysis_text.split("FILES:")[1].split("FILE METADATA:")[0]
                file_names = [f.strip() for f in files_section.strip().split("\n") if f.strip()]
            setattr(analysis_text, "file_names", file_names)
            
        except Exception as e:
            print(f"Warning: Could not parse key concepts from analysis: {e}")
        
        return analysis_text 