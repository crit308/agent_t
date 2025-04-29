#!/usr/bin/env python3
"""
Offline DAG Extractor
Extracts prerequisite edges between concepts from document knowledge bases using OpenAI function-calling and saves them to the Supabase 'concept_graph' table.
"""
import os
import json
from dotenv import load_dotenv
from supabase import create_client
import openai


def extract_edges(kb_text: str) -> list[dict[str, str]]:
    """Use OpenAI function-calling to extract prerequisite relationships between concepts."""
    functions = [
        {
            "name": "extract_edges",
            "description": "Extract prerequisite relationships between concepts. Return a list of edges with 'prereq' and 'concept' fields.",
            "parameters": {
                "type": "object",
                "properties": {
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prereq": {"type": "string"},
                                "concept": {"type": "string"}
                            },
                            "required": ["prereq", "concept"]
                        }
                    }
                },
                "required": ["edges"]
            }
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4.1-2025-04-14",
        messages=[{"role": "user", "content": kb_text}],
        functions=functions,
        function_call={"name": "extract_edges"}
    )
    message = response.choices[0].message
    if message.get("function_call") and message["function_call"].get("arguments"):
        try:
            args = json.loads(message["function_call"]["arguments"])
            return args.get("edges", [])
        except json.JSONDecodeError:
            print("Failed to parse function arguments:", message["function_call"]["arguments"])
    else:
        print("No function call returned. Response content:", message.content)
    return []


def main():
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL or SUPABASE_SERVICE_KEY not set in environment.")
        return
    supabase = create_client(supabase_url, supabase_key)

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not set in environment.")
        return

    # Optional: Clear existing entries to rebuild from scratch
    print("Clearing existing entries in 'concept_graph' table...")
    try:
        supabase.table("concept_graph").delete().neq("prereq", "").execute()
    except Exception as e:
        print(f"Warning: Could not clear existing table: {e}")

    # Fetch all folders and their knowledge base texts
    print("Fetching folders from Supabase...")
    try:
        result = supabase.table("folders").select("id, knowledge_base").execute()
        folders = result.data or []
    except Exception as e:
        print(f"Error fetching folders: {e}")
        return

    for folder in folders:
        folder_id = folder.get("id")
        kb_text = folder.get("knowledge_base")
        if not kb_text:
            print(f"Skipping folder {folder_id}: no knowledge base.")
            continue
        print(f"Processing folder {folder_id}...")
        edges = extract_edges(kb_text)
        print(f"Extracted {len(edges)} edges for folder {folder_id}.")
        records = [{"prereq": e["prereq"], "concept": e["concept"]} for e in edges]
        if not records:
            continue
        try:
            insert_resp = supabase.table("concept_graph").insert(records).execute()
            if insert_resp.error:
                print(f"Error inserting edges for folder {folder_id}: {insert_resp.error}")
            else:
                print(f"Inserted {len(records)} edges for folder {folder_id}.")
        except Exception as e:
            print(f"Exception inserting edges for folder {folder_id}: {e}")

    print("Concept graph extraction complete.")

if __name__ == "__main__":
    main() 