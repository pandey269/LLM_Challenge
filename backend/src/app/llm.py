"""LLM client helpers (Ollama Mistral + Gemini helper)."""
from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .config import settings


class LLMService:
    """Wraps the chat model and prompt templates used in the graph."""

    def __init__(self) -> None:
        if settings.model.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set but llm_provider=openai")
            openai_kwargs = {
                "model": settings.model.llm_model,
                "temperature": 0.2,
                "max_retries": 2,
                "max_tokens": settings.model.max_output_tokens,
                "api_key": api_key,
            }
            if settings.model.openai_api_base:
                openai_kwargs["base_url"] = settings.model.openai_api_base

            self.llm = ChatOpenAI(**openai_kwargs)
        else:
            self.llm = ChatOllama(
                model=settings.model.llm_model,
                base_url=settings.model.llm_base_url,
                temperature=0.2,
                max_retries=2,
                num_ctx=settings.model.max_input_tokens,
            )
        self.draft_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a retrieval-grounded assistant. Use only provided context. Cite sources by doc id and page. If evidence missing, say you don't know.",
                ),
                (
                    "human",
                    "Question: {question}\n\nContext:\n{context}\n\nRespond with JSON containing answer, citations, evidence, confidence, and not_found_reason.",
                ),
            ]
        )
        self.reflection_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You critique QA outputs for groundedness. Flag if any claim lacks citation or context coverage is low.",
                ),
                (
                    "human",
                    "Question: {question}\nAnswer: {answer}\nContext:\n{context}\nProvide score between 0 and 1 and reasoning.",
                ),
            ]
        )

    def draft(self, question: str, context: str) -> dict[str, Any]:
        response = self.llm.invoke(self.draft_prompt.format_messages(question=question, context=context))
        return self._safe_json(response)

    def reflect(self, question: str, answer: str, context: str) -> dict[str, Any]:
        response = self.llm.invoke(
            self.reflection_prompt.format_messages(question=question, answer=answer, context=context)
        )
        return self._safe_json(response)

    @staticmethod
    def _safe_json(message: BaseMessage) -> dict[str, Any]:
        content: str
        if isinstance(message.content, str):
            content = message.content
        else:
            # LangChain >=0.2 may return a list of parts
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in message.content)

        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw": content}


llm_service = LLMService()
