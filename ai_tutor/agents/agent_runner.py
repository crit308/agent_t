import re
from typing import Any, Dict, Optional, List
from agents import Runner
from agents.run_context import RunContextWrapper
from agents.tool import FileSearchTool
from ai_tutor.agents.models import ExplanationResult, QuizCreationResult
from ai_tutor.utils.embedding_utils import cosine_similarity
from ai_tutor.dependencies import get_openai_client  # shared singleton

# Utility: Token count (simple whitespace split, replace with tiktoken if needed)
def count_tokens(text: str) -> int:
    return len(text.split())

async def semantic_similarity(a: str, b: str) -> float:
    """Cosine over OpenAI embeddings (cached). Uses the shared AsyncOpenAI client from dependencies to avoid per‑call instantiation overhead."""
    client = get_openai_client()  # singleton – created once per worker
    async def embed(text: str):
        resp = await client.embeddings.create(model="text-embedding-3-small", input=text)
        return resp.data[0].embedding
    return await cosine_similarity(a, b, embed)

class AgentRunner:
    def __init__(self, agent, context, vector_store_id: str, file_search_threshold: int = 50):
        self.agent = agent
        self.context = context
        self.vector_store_id = vector_store_id
        self.file_search_threshold = file_search_threshold

    async def preflight_file_search(self, query: str) -> Dict[str, Any]:
        from agents import Agent, Runner
        from agents.models.openai_provider import OpenAIProvider

        # Create the file_search tool
        file_search_tool = FileSearchTool(
            vector_store_ids=[self.vector_store_id],
            max_num_results=10,
            include_search_results=True,
        )
        # Spin up a temporary agent that only has the FileSearchTool
        provider = OpenAIProvider()
        temp_agent = Agent(
            name="TempFileSearcher",
            instructions="Use the file_search tool to find relevant document snippets.",
            tools=[file_search_tool],
            model=provider.get_model("o4-mini"),
        )
        # Run the temp agent with the raw query
        result = await Runner.run(temp_agent, query, context=self.context)
        # Extract the final_output as a single chunk
        chunk = result.final_output if isinstance(result.final_output, str) else str(result.final_output)
        return {"results": [chunk], "chunks": [chunk]}

    async def run_teacher(self, prompt: str, section: Optional[str] = None) -> ExplanationResult:
        # 1. Pre-flight file_search for relevant section/concepts
        search_query = section or "relevant concepts"
        search_result = await self.preflight_file_search(search_query)
        all_chunks = " ".join(search_result.get("chunks", []))
        token_count = count_tokens(all_chunks)
        if token_count <= self.file_search_threshold:
            return ExplanationResult(status="need_more_context", details="Not enough content found in file_search. Please provide more context or upload more documents.")
        # 2. Append system message
        guardrail_msg = "You must only teach using chunks you found. If you didn't find enough, respond with status 'need_more_context' and a brief note."
        full_prompt = f"{prompt}\n\n{guardrail_msg}\n\n[FILE_SEARCH RESULTS]\n{all_chunks}"
        # 3. Call agent
        result = await Runner.run(self.agent, full_prompt, context=self.context)
        # 4. Post-process: ensure explanation only uses found chunks
        if hasattr(result, 'final_output_as'):
            explanation = result.final_output_as(ExplanationResult)
            # If the agent returned a raw string, wrap it into an ExplanationResult
            if isinstance(explanation, str):
                return ExplanationResult(status="delivered", details=explanation)
            # If parsing failed or we didn't get an ExplanationResult, fail gracefully
            if not isinstance(explanation, ExplanationResult):
                return ExplanationResult(status="failed", details=f"Unexpected output from agent: {result.final_output}")
            if explanation.status == "delivered":
                # Check if explanation is grounded in file_search
                sim = await semantic_similarity(explanation.details, all_chunks)
                if sim < 0.3:
                    explanation.status = "need_more_context"
                    explanation.details = "Explanation not sufficiently grounded in file_search results."
            return explanation
        return ExplanationResult(status="failed", details="Unexpected output from agent.")

    async def run_quiz_creator(self, prompt: str, mastered_sections: List[str]) -> QuizCreationResult:
        # 1. Pre-flight file_search scoped to mastered sections
        search_results = []
        for section in mastered_sections:
            res = await self.preflight_file_search(section)
            search_results.extend(res.get("chunks", []))
        all_chunks = " ".join(search_results)
        # 2. Call agent
        guardrail_msg = "You must only create questions whose answers are exactly found in the file_search results. If not, return status: 'failed' with reason: 'answer not grounded in content'."
        full_prompt = f"{prompt}\n\n{guardrail_msg}\n\n[FILE_SEARCH RESULTS]\n{all_chunks}"
        result = await Runner.run(self.agent, full_prompt, context=self.context)
        # 3. Post-process: check each question's answer is grounded
        if hasattr(result, 'final_output_as'):
            quiz = result.final_output_as(QuizCreationResult)
            if quiz and hasattr(quiz, 'quiz') and quiz.quiz and hasattr(quiz.quiz, 'questions'):
                for q in quiz.quiz.questions:
                    if not any(q.explanation and q.explanation in chunk for chunk in search_results):
                        quiz.status = "failed"
                        quiz.details = "answer not grounded in content"
                        break
            return quiz
        return QuizCreationResult(status="failed", details="Unexpected output from agent.")

    async def run_explanation_checker(self, agent_answer: str, section: str) -> Dict[str, Any]:
        # 1. Pre-flight file_search for the same section
        search_result = await self.preflight_file_search(section)
        all_chunks = " ".join(search_result.get("chunks", []))
        # 2. Cosine similarity
        sim = await semantic_similarity(agent_answer, all_chunks)
        if sim < 0.3:
            return {"status": "failed", "reason": "low_similarity", "similarity": sim}
        return {"status": "completed", "similarity": sim} 