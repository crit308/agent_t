from __future__ import annotations
from agents import function_tool, Runner, RunConfig
from agents.run_context import RunContextWrapper
from typing import Any, Optional, Literal, Union, cast, Dict, List
import os
from datetime import datetime
import traceback # Import traceback for error logging
from ai_tutor.dependencies import SUPABASE_CLIENT  # Supabase client for logging events
from ai_tutor.utils.tool_helpers import invoke
from ai_tutor.context import TutorContext, UserConceptMastery, UserModelState
from ai_tutor.context import is_mastered
from ai_tutor.agents.models import FocusObjective, QuizQuestion, QuizFeedbackItem, ExplanationResult, QuizCreationResult
from ai_tutor.api_models import (
    ExplanationResponse, QuestionResponse, FeedbackResponse, MessageResponse, ErrorResponse
)

# All orchestrator tools have been moved to ai_tutor.tools.__init__.py as the single barrel file.
# This file now only contains imports and helpers if needed.

# Re-export tool functions from ai_tutor.tools barrel
import ai_tutor.tools as _barrel
for _name in _barrel.__all__:
    globals()[_name] = getattr(_barrel, _name)
__all__ = _barrel.__all__ 