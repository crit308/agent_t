#!/usr/bin/env python3
"""
Standalone script to run the document analyzer agent when a vector store is created.
Usage: python run_analyzer.py <vector_store_id> [api_key]
"""

import os
import sys
import asyncio
import json
from typing import Optional, Any

from ai_tutor.agents.analyzer_agent import analyze_documents
from ai_tutor.utils.file_upload import FileUploadManager
from ai_tutor.dependencies import SUPABASE_CLIENT


def save_analysis_to_file(analysis: Any, output_file: str) -> None:
    """Save analysis results to a file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            if isinstance(analysis, str):
                f.write(analysis)
            else:
                # Fallback to string representation
                f.write(str(analysis))
        
        print(f"Analysis saved to {output_file}")
    except Exception as e:
        print(f"Error saving analysis to file: {e}")
        # Try with a different encoding as fallback
        try:
            with open(output_file, 'w', encoding='ascii', errors='ignore') as f:
                if isinstance(analysis, str):
                    f.write(analysis)
                else:
                    f.write(str(analysis))
            print(f"Analysis saved to {output_file} (with encoding fallback)")
        except Exception as e2:
            print(f"Could not save analysis to file: {e2}")


class VectorStoreWatcher:
    """Watches for vector store creation and triggers analysis."""
    
    def __init__(self):
        """Initialize the watcher."""
        # API key is expected to be set globally via main.py or environment
        
        # Initialize file upload manager to check for vector stores
        self.file_upload_manager = FileUploadManager()
    
    async def analyze_on_vector_store_creation(self, vector_store_id: Optional[str] = None):
        """
        Watch for vector store creation and run analysis when detected.
        If vector_store_id is provided, analyze that specific vector store.
        """
        # Use provided vector store ID or check for existing one
        target_vector_store_id = vector_store_id or self.file_upload_manager.get_vector_store_id()
        
        if not target_vector_store_id:
            print("No vector store ID provided or found in the file upload manager.")
            print("Waiting for vector store creation...")
            
            # Poll until a vector store is created
            while not target_vector_store_id:
                await asyncio.sleep(5)  # Check every 5 seconds
                target_vector_store_id = self.file_upload_manager.get_vector_store_id()
                if target_vector_store_id:
                    print(f"Vector store detected: {target_vector_store_id}")
                    break
        
        # Run the analyzer on the detected vector store
        print(f"Running document analysis on vector store: {target_vector_store_id}")
        analysis = await analyze_documents(target_vector_store_id, supabase=SUPABASE_CLIENT)
        
        if analysis:
            # Generate output filename with vector store ID
            output_file = f"document_analysis_{target_vector_store_id}.txt"
            save_analysis_to_file(analysis, output_file)
            
            # Extract metadata for summary if possible
            file_count = len(getattr(analysis, "file_names", []))
            concept_count = len(getattr(analysis, "key_concepts", []))
            key_concepts = getattr(analysis, "key_concepts", [])
            
            # Print summary of the analysis
            print("\nAnalysis Summary:")
            print(f"Files analyzed: {file_count}")
            print(f"Key concepts identified: {concept_count}")
            if key_concepts and len(key_concepts) > 0:
                print(f"Top concepts: {', '.join(key_concepts[:5])}")
            print(f"Vector store ID: {target_vector_store_id}")
            print(f"Full analysis saved to: {output_file}")
        else:
            print("Analysis failed or returned no results.")


async def main():
    # Get vector store ID from command-line arguments if provided
    vector_store_id = None
    
    if len(sys.argv) > 1:
        vector_store_id = sys.argv[1]
    
    # Create and run the watcher
    watcher = VectorStoreWatcher()
    await watcher.analyze_on_vector_store_creation(vector_store_id)


if __name__ == "__main__":
    asyncio.run(main()) 