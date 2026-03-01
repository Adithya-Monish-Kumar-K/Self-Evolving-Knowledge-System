"""
Ollama LLM wrapper – provides a unified interface for all LLM calls
in the system (epistemic analysis, reasoning, query generation, etc.).
"""

import json
import logging
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

import config

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper around Ollama via LangChain for structured + free-form generation."""

    def __init__(
        self,
        model: str = config.OLLAMA_MODEL,
        temperature: float = config.OLLAMA_TEMPERATURE,
        base_url: str = config.OLLAMA_BASE_URL,
        timeout: int = config.OLLAMA_REQUEST_TIMEOUT,
    ):
        self.model = model
        self.llm = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            request_timeout=timeout,
        )
        logger.info("LLMClient initialised with model=%s", model)

    # ── Free-form generation ─────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a free-form text response."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            raise

    # ── JSON-structured generation ───────────────────────────────────
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Generate a response and parse it as JSON.

        The prompt should instruct the model to respond exclusively in JSON.
        Includes retry logic with a stricter nudge on failure.
        """
        base_system = (system_prompt or "") + (
            "\n\nIMPORTANT: You MUST respond with valid JSON only. "
            "No markdown fences, no explanation outside the JSON object."
        )

        for attempt in range(2):
            raw = self.generate(prompt, system_prompt=base_system)
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(
                    "JSON parse failed (attempt %d): %s…",
                    attempt + 1,
                    cleaned[:200],
                )
                if attempt == 0:
                    base_system += (
                        "\nYour previous response was not valid JSON. "
                        "Reply ONLY with a JSON object, nothing else."
                    )

        # Last-resort: return a fallback dict so the pipeline doesn't crash
        logger.error("Could not parse JSON after retries. Raw: %s", raw[:500])
        return {"error": "json_parse_failed", "raw": raw[:1000]}

    # ── Convenience: yes / no question ───────────────────────────────
    def ask_yes_no(self, question: str, context: str = "") -> bool:
        """Ask the LLM a yes/no question. Returns True for yes."""
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer with ONLY 'yes' or 'no'."
        )
        answer = self.generate(prompt).strip().lower()
        return answer.startswith("yes")

    # ── Health check ─────────────────────────────────────────────────
    def ping(self) -> bool:
        """Return True if Ollama is reachable and the model responds."""
        try:
            resp = self.generate("Say 'ok'.")
            return len(resp) > 0
        except Exception:
            return False


# Module-level singleton (lazy)
_client: Optional[LLMClient] = None


def get_llm(model: Optional[str] = None) -> LLMClient:
    """Return (or create) the module-level LLMClient singleton."""
    global _client
    if _client is None or (model and model != _client.model):
        _client = LLMClient(model=model or config.OLLAMA_MODEL)
    return _client
