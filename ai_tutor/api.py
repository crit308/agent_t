from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ai_tutor.routers import sessions, tutor
from agents import set_default_openai_key, set_default_openai_api

# --- SDK Configuration ---
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    # Decide if the app should exit or continue with limited functionality
    # exit(1) # Or raise an exception
else:
    set_default_openai_key(api_key)
    set_default_openai_api("responses") # Ensure using API needed for models like o3-mini
    print("OpenAI API key configured for agents SDK.")

app = FastAPI(
    title="AI Tutor API",
    description="API for generating lessons and quizzes using AI agents.",
    version="1.0.0",
)

# --- CORS Configuration ---
# Adjust origins based on your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for now, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mount Routers ---
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(tutor.router, prefix="/api/v1")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the AI Tutor API!"}

# To run the API: uvicorn ai_tutor.api:app --reload 