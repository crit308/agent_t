import streamlit as st
import os
import asyncio
import json
from pathlib import Path
import sys

# Add the parent directory to the Python path so we can import the AI tutor package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_tutor.manager import AITutorManager

# Set page config
st.set_page_config(
    page_title="AI Tutor",
    page_icon="🧠",
    layout="wide"
)

# Session state initialization
if 'api_key' not in st.session_state:
    st.session_state.api_key = os.environ.get("OPENAI_API_KEY", "")
if 'manager' not in st.session_state:
    st.session_state.manager = None
if 'lesson_plan' not in st.session_state:
    st.session_state.lesson_plan = None
if 'lesson_content' not in st.session_state:
    st.session_state.lesson_content = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'step' not in st.session_state:
    st.session_state.step = 1

# Helper functions
def create_manager():
    if st.session_state.api_key:
        st.session_state.manager = AITutorManager(st.session_state.api_key)
        return True
    else:
        st.error("Please enter your OpenAI API key")
        return False

async def upload_documents():
    if not st.session_state.manager:
        if not create_manager():
            return
    
    with st.spinner("Uploading documents..."):
        try:
            result = await st.session_state.manager.upload_documents(st.session_state.uploaded_files)
            st.success("Documents uploaded successfully")
            st.session_state.step = 2
            return result
        except Exception as e:
            st.error(f"Error uploading documents: {str(e)}")
            return None

async def generate_lesson_plan():
    if not st.session_state.manager:
        st.error("Please upload documents first")
        return
    
    with st.spinner("Generating lesson plan..."):
        try:
            lesson_plan = await st.session_state.manager.generate_lesson_plan()
            st.session_state.lesson_plan = lesson_plan
            st.session_state.step = 3
            st.success("Lesson plan generated successfully")
            return lesson_plan
        except Exception as e:
            st.error(f"Error generating lesson plan: {str(e)}")
            return None

async def generate_lesson():
    if not st.session_state.lesson_plan:
        st.error("Please generate a lesson plan first")
        return
    
    with st.spinner("Generating lesson content..."):
        try:
            lesson_content = await st.session_state.manager.generate_lesson()
            st.session_state.lesson_content = lesson_content
            st.session_state.step = 4
            st.success("Lesson content generated successfully")
            return lesson_content
        except Exception as e:
            st.error(f"Error generating lesson content: {str(e)}")
            return None

# Main UI
st.title("AI Tutor")
st.markdown("Upload documents and generate personalized lessons")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        if api_key:
            create_manager()
    
    st.markdown("---")
    st.markdown("### Progress")
    st.progress(st.session_state.step / 4)
    step_names = {
        1: "Upload Documents", 
        2: "Generate Lesson Plan", 
        3: "Generate Lesson Content",
        4: "View Lesson"
    }
    st.info(f"Current step: {step_names[st.session_state.step]}")

# Step 1: Upload Documents
if st.session_state.step == 1:
    st.header("Step 1: Upload Documents")
    uploaded_files = st.file_uploader("Upload documents for the lesson", accept_multiple_files=True)
    
    if uploaded_files:
        # Save the uploaded files to a temporary directory
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(str(file_path))
        
        st.session_state.uploaded_files = file_paths
        
        if st.button("Upload Documents"):
            asyncio.run(upload_documents())

# Step 2: Generate Lesson Plan
elif st.session_state.step == 2:
    st.header("Step 2: Generate Lesson Plan")
    st.write("Documents uploaded successfully. Now you can generate a lesson plan.")
    
    if st.button("Generate Lesson Plan"):
        lesson_plan = asyncio.run(generate_lesson_plan())

# Step 3: Generate Lesson Content
elif st.session_state.step == 3:
    st.header("Step 3: Generate Lesson Content")
    
    if st.session_state.lesson_plan:
        st.subheader("Lesson Plan")
        st.write(f"**Title:** {st.session_state.lesson_plan.title}")
        st.write(f"**Description:** {st.session_state.lesson_plan.description}")
        st.write(f"**Target Audience:** {st.session_state.lesson_plan.target_audience}")
        st.write(f"**Total Duration:** {st.session_state.lesson_plan.total_estimated_duration_minutes} minutes")
        
        st.subheader("Sections")
        for i, section in enumerate(st.session_state.lesson_plan.sections):
            st.write(f"**{i+1}. {section.title}** ({section.estimated_duration_minutes} min)")
            st.write(section.learning_objectives)
    
    if st.button("Generate Lesson Content"):
        lesson_content = asyncio.run(generate_lesson())

# Step 4: View Lesson
elif st.session_state.step == 4:
    st.header("Step 4: View Lesson")
    
    if st.session_state.lesson_content:
        st.title(st.session_state.lesson_content.title)
        st.markdown(st.session_state.lesson_content.introduction)
        
        for i, section in enumerate(st.session_state.lesson_content.sections):
            with st.expander(f"Section {i+1}: {section.title}"):
                st.markdown(section.content)
                
                if section.examples:
                    st.subheader("Examples")
                    st.markdown(section.examples)
                
                if section.exercises:
                    st.subheader("Exercises")
                    st.markdown(section.exercises)
        
        # Save button
        if st.button("Save Lesson"):
            output_file = f"lesson_{st.session_state.lesson_content.title.replace(' ', '_')}.json"
            with open(output_file, "w") as f:
                f.write(json.dumps(st.session_state.lesson_content.dict(), indent=2))
            st.success(f"Lesson saved to {output_file}")

# Reset button
with st.sidebar:
    if st.button("Start Over"):
        st.session_state.step = 1
        st.session_state.lesson_plan = None
        st.session_state.lesson_content = None
        st.experimental_rerun()

if __name__ == "__main__":
    print("Running Streamlit app for AI Tutor") 