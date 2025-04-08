import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

class TutorOutputLogger:
    """Logger for capturing outputs from all AI Tutor agents."""
    
    def __init__(self, output_file: Optional[str] = None):
        """Initialize the logger with an optional output file path.
        
        Args:
            output_file: Path to output file. If None, generates a default name.
        """
        if output_file is None:
            # Create a timestamped file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"ai_tutor_output_{timestamp}.txt"
        
        self.output_file = output_file
        self.logs = {
            "timestamp": datetime.now().isoformat(),
            "planner_agent_output": "",
            "teacher_agent_output": "",
            "orchestrator_agent_output": "",
            "orchestrator_reasoning": [],
            "user_input": "",
            "quiz_creator_agent_output": "",
            "user_summaries": [],
            "mini_quiz_attempts": [],
            "quiz_user_answers": [],
            "user_answers": "",
            "quiz_teacher_agent_output": "",
            "session_analysis_output": "",
            "full_session": [],
            "errors": [],
            "session_log": []
        }
    
    def log_planner_output(self, output: Any) -> None:
        """Log output from the planner agent."""
        self.logs["planner_agent_output"] = self._format_output(output)
        self._append_to_session("Planner Agent", output)
    
    def log_orchestrator_output(self, output: Any) -> None:
        """Log output from the orchestrator agent."""
        self.logs["orchestrator_agent_output"] = self._format_output(output)
        self._append_to_session("Orchestrator Agent", output)
    
    def log_orchestrator_decision(self, decision: str, reasoning: Optional[str] = None, data: Optional[Dict] = None) -> None:
        """Log a specific decision or reasoning step from the orchestrator."""
        log_entry = {
            "decision": decision,
            "reasoning": reasoning,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        self.logs["orchestrator_reasoning"].append(log_entry)
        self._append_to_session("Orchestrator Decision", log_entry)
    
    def log_user_input(self, user_input: str) -> None:
        """Log the user's input."""
        self.logs["user_input"] = self._format_output(user_input)
        self._append_to_session("User Input", user_input)
    
    def log_teacher_output(self, output: Any) -> None:
        """Log output from the teacher agent."""
        self.logs["teacher_agent_output"] = self._format_output(output)
        self._append_to_session("Teacher Agent", output)
    
    def log_quiz_creator_output(self, output: Any) -> None:
        """Log output from the quiz creator agent."""
        self.logs["quiz_creator_agent_output"] = self._format_output(output)
        self._append_to_session("Quiz Creator Agent", output)
    
    def log_quiz_user_answer(self, question: str, options: List[str], 
                           selected_idx: int, correct_idx: int) -> None:
        """Log a user answer to a quiz question."""
        answer_log = {
            "question": question,
            "options": options,
            "selected_option_index": selected_idx,
            "correct_option_index": correct_idx,
            "is_correct": selected_idx == correct_idx
        }
        self.logs["quiz_user_answers"].append(answer_log)
        
        # Format for session log
        answer_text = (
            f"Question: {question}\n"
            f"Options: {', '.join(options)}\n"
            f"Your Answer: {options[selected_idx]}\n"
            f"Correct Answer: {options[correct_idx]}\n"
            f"Result: {'✓ Correct' if selected_idx == correct_idx else '✗ Incorrect'}"
        )
        self._append_to_session("Quiz User Answer", answer_text)
    
    def log_raw_user_answers(self, user_answers: Any) -> None:
        """Log the complete raw user answers object."""
        self.logs["user_answers"] = self._format_output(user_answers)
        self._append_to_session("Raw User Answers", user_answers)
    
    def log_quiz_teacher_output(self, output: Any) -> None:
        """Log output from the quiz teacher agent."""
        self.logs["quiz_teacher_agent_output"] = self._format_output(output)
        self._append_to_session("Quiz Teacher Agent", output)
    
    def log_session_analysis_output(self, output: Any) -> None:
        """Log output from the session analyzer agent."""
        self.logs["session_analysis_output"] = self._format_output(output)
        self._append_to_session("Session Analyzer Agent", output)
    
    def log_mini_quiz_attempt(self, question: str, selected_option: str, correct_option: str, is_correct: bool) -> None:
        """Log a user attempt on an in-lesson mini-quiz question."""
        attempt_log = {
            "question": question,
            "selected_option": selected_option,
            "correct_option": correct_option,
            "is_correct": is_correct,
            "timestamp": datetime.now().isoformat()
        }
        self.logs["mini_quiz_attempts"].append(attempt_log)
        # Also add to the chronological session log
        self._append_to_session("Mini-Quiz Attempt", attempt_log)
    
    def log_user_summary(self, section_title: str, topic: str, summary_text: str) -> None:
        """Log a user's summary attempt."""
        summary_log = {
            "section": section_title,
            "topic": topic,
            "summary": summary_text,
            "timestamp": datetime.now().isoformat()
        }
        self.logs["user_summaries"].append(summary_log)
        self._append_to_session("User Summary Attempt", summary_log)
    
    def get_agent_outputs(self) -> Dict[str, str]:
        """Get all agent outputs as a dictionary."""
        return {
            "planner_agent": self.logs["planner_agent_output"],
            "teacher_agent": self.logs["teacher_agent_output"],
            "orchestrator_agent": self.logs["orchestrator_agent_output"],
            "quiz_creator_agent": self.logs["quiz_creator_agent_output"],
            "quiz_teacher_agent": self.logs["quiz_teacher_agent_output"],
            "session_analysis_agent": self.logs["session_analysis_output"],
        }
    
    def log_error(self, agent_name: str, error: Exception) -> None:
        """Log an error that occurred during an agent's execution."""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "error_type": type(error).__name__,
            "error_message": str(error)
        }
        self.logs["errors"].append(error_entry)
        self._append_to_session(f"{agent_name} Error", str(error))
    
    def _format_output(self, output: Any) -> str:
        """Format an output object to string representation."""
        if output is None:
            return "None"
        
        if hasattr(output, "model_dump"):
            # Handle Pydantic models
            return json.dumps(output.model_dump(), indent=2)
        elif hasattr(output, "__dict__"):
            # Handle regular objects
            return json.dumps(output.__dict__, indent=2, default=str)
        else:
            # Handle primitive types
            return str(output)
    
    def _append_to_session(self, agent_name: str, output: Any) -> None:
        """Append formatted output to the full session log."""
        formatted_output = self._format_output(output)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "output": formatted_output
        }
        self.logs["full_session"].append(entry)
    
    def save_to_file(self) -> None:
        """Save all logs to the output file if one was specified."""
        if not self.output_file:
            return

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            f.write("SESSION LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Started: {self.logs['timestamp']}\n\n")

            f.write("CHRONOLOGICAL EVENT LOG\n")
            f.write("-" * 80 + "\n")
            for entry in self.logs["session_log"]:
                f.write(f"{entry}\n")
            f.write("\n")

            f.write("PLANNER AGENT OUTPUT (LAST)\n")
            f.write("-" * 80 + "\n")
            f.write(self.logs["planner_agent_output"])
            f.write("\n\n")
            
            f.write("TEACHER AGENT OUTPUT (LAST)\n")
            f.write("-" * 80 + "\n")
            f.write(self.logs["teacher_agent_output"])
            f.write("\n\n")
            
            f.write("ORCHESTRATOR AGENT OUTPUT (LAST)\n")
            f.write("-" * 80 + "\n")
            f.write(self.logs["orchestrator_agent_output"])
            f.write("\n\n")

            f.write("ORCHESTRATOR DECISIONS/REASONING\n")
            f.write("-" * 80 + "\n")
            for reasoning_entry in self.logs["orchestrator_reasoning"]:
                f.write(f"{self._format_output(reasoning_entry)}\n")
            f.write("\n\n")

            f.write("QUIZ CREATOR AGENT OUTPUT\n")
            f.write("-" * 80 + "\n")
            f.write(self.logs["quiz_creator_agent_output"])
            f.write("\n\n")

# Global instance for easy access
_logger = None

def get_logger(output_file: Optional[str] = None) -> TutorOutputLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = TutorOutputLogger(output_file)
    return _logger 