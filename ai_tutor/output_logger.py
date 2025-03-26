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
            "planner_output": "",
            "teacher_output": "",
            "quiz_creator_output": "",
            "quiz_user_answers": [],
            "quiz_teacher_output": "",
            "full_session": []
        }
    
    def log_planner_output(self, output: Any) -> None:
        """Log output from the planner agent."""
        self.logs["planner_output"] = self._format_output(output)
        self._append_to_session("Planner Agent", output)
    
    def log_teacher_output(self, output: Any) -> None:
        """Log output from the teacher agent."""
        self.logs["teacher_output"] = self._format_output(output)
        self._append_to_session("Teacher Agent", output)
    
    def log_quiz_creator_output(self, output: Any) -> None:
        """Log output from the quiz creator agent."""
        self.logs["quiz_creator_output"] = self._format_output(output)
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
    
    def log_quiz_teacher_output(self, output: Any) -> None:
        """Log output from the quiz teacher agent."""
        self.logs["quiz_teacher_output"] = self._format_output(output)
        self._append_to_session("Quiz Teacher Agent", output)
    
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
    
    def save(self) -> str:
        """Save the logs to the output file.
        
        Returns:
            Path to the saved file.
        """
        with open(self.output_file, "w", encoding="utf-8") as f:
            # Write a readable format with headers and sections
            f.write("=" * 80 + "\n")
            f.write(f"AI TUTOR SESSION LOG - {self.logs['timestamp']}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write each section
            f.write("PLANNER AGENT OUTPUT\n")
            f.write("-" * 80 + "\n")
            f.write(self.logs["planner_output"])
            f.write("\n\n")
            
            f.write("TEACHER AGENT OUTPUT\n")
            f.write("-" * 80 + "\n")
            f.write(self.logs["teacher_output"])
            f.write("\n\n")
            
            f.write("QUIZ CREATOR AGENT OUTPUT\n")
            f.write("-" * 80 + "\n")
            f.write(self.logs["quiz_creator_output"])
            f.write("\n\n")
            
            f.write("QUIZ USER ANSWERS\n")
            f.write("-" * 80 + "\n")
            for i, answer in enumerate(self.logs["quiz_user_answers"]):
                f.write(f"Question {i+1}: {answer['question']}\n")
                f.write(f"Options: {', '.join(answer['options'])}\n")
                f.write(f"Your Answer: {answer['options'][answer['selected_option_index']]}\n")
                f.write(f"Correct Answer: {answer['options'][answer['correct_option_index']]}\n")
                f.write(f"Result: {'✓ Correct' if answer['is_correct'] else '✗ Incorrect'}\n\n")
            
            f.write("QUIZ TEACHER AGENT OUTPUT\n")
            f.write("-" * 80 + "\n")
            f.write(self.logs["quiz_teacher_output"])
            f.write("\n\n")
            
            f.write("FULL SESSION LOG (CHRONOLOGICAL)\n")
            f.write("=" * 80 + "\n\n")
            for entry in self.logs["full_session"]:
                f.write(f"[{entry['timestamp']}] {entry['agent']}\n")
                f.write("-" * 80 + "\n")
                f.write(entry['output'])
                f.write("\n\n")
                
        print(f"AI Tutor session log saved to: {self.output_file}")
        return self.output_file

# Global instance for easy access
_logger = None

def get_logger(output_file: Optional[str] = None) -> TutorOutputLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = TutorOutputLogger(output_file)
    return _logger 