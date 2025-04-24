import os
from agents import Agent
from ai_tutor.skills import list_tools

# Placeholder for prompt content until the file is created
DEFAULT_TUTOR_INSTRUCTIONS = """
You are an expert AI Tutor. Your goal is to guide the student through a lesson plan, explaining concepts clearly, asking checking questions, and providing quizzes.
Use the available tools to interact with the lesson material, assess student understanding, and adjust your approach based on their responses.
Be patient, encouraging, and adapt to the student's pace and needs.
"""

def build_tutor_agent():
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts/tutor_agent.md")
    instructions = DEFAULT_TUTOR_INSTRUCTIONS # Default fallback
    # try:
    #     # Ensure the prompts directory exists before trying to read
    #     prompts_dir = os.path.dirname(prompt_path)
    #     if not os.path.exists(prompts_dir):
    #         os.makedirs(prompts_dir)
    #         print(f"Created missing directory: {prompts_dir}") # Log creation
    #         # Optionally create an empty file if it doesn't exist
    #         if not os.path.exists(prompt_path):
    #             with open(prompt_path, 'w') as f:
    #                 f.write(DEFAULT_TUTOR_INSTRUCTIONS) # Write default content
    #             print(f"Created missing prompt file with default content: {prompt_path}")

    #     with open(prompt_path, 'r') as f:
    #         instructions = f.read()
    # except FileNotFoundError:
    #     print(f"Warning: Prompt file not found at {prompt_path}. Using default instructions.")
    # except Exception as e:
    #      print(f"Warning: Error reading prompt file {prompt_path}: {e}. Using default instructions.")


    return Agent(
        name="Tutor",
        instructions=instructions,
        tools=list_tools(),     # every FunctionTool
        model="gpt-4o-mini",
    ) 