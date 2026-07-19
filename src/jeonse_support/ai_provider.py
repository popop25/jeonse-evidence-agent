"""Bounded Azure OpenAI presentation adapter.

The adapter is intentionally presentation-only: deterministic policy remains the
sole authority for grades and numeric conclusions.
"""

from __future__ import annotations

import os
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


PROMPT_VERSION = "advisory-presentation-v1"
_AZURE_API_VERSION = "2024-10-21"
_ENV_NAMES = (
    "AOAI_API_KEY",
    "AOAI_ENDPOINT",
    "AOAI_DEPLOY_GPT41_MINI",
    "AOAI_MODEL_GPT41_MINI",
    "AOAI_DEPLOY_EMBED_3_SMALL",
    "AOAI_MODEL_EMBED_3_SMALL",
)
_URL = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_NUMBER = re.compile(r"\d")
_RISK_WORD = re.compile(r"\b(?:low|medium|high|unavailable|risk)\b|위험", re.IGNORECASE)
_PROHIBITED_ASSERTION = re.compile(
    r"법률\s*자문|법적으로\s*안전|안전(?:하|을\s*보장)|사기(?:가\s*(?:아니|아닙)|로\s*판정)|"
    r"legal\s+advice|legally\s+safe|safety\s+guarantee|fraud(?:ulent|\s+determination)",
    re.IGNORECASE,
)


class AzureOpenAIConfigurationError(ValueError):
    """Raised before a provider can use incomplete Azure OpenAI configuration."""


class ProviderPresentationError(RuntimeError):
    """A bounded provider/presentation failure safe for explicit fallback status."""

    def __init__(self, cause: Exception) -> None:
        super().__init__("Azure OpenAI presentation failed")
        self.trace_code = f"presentation_failed_{type(cause).__name__}"


class AzureOpenAISettings(BaseModel):
    """The only environment-backed Azure OpenAI configuration accepted here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_key: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    chat_deployment: str = Field(min_length=1)
    chat_model: str = Field(min_length=1)
    embedding_deployment: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "AzureOpenAISettings":
        values = os.environ if environ is None else environ
        missing = tuple(name for name in _ENV_NAMES if not values.get(name))
        if missing:
            raise AzureOpenAIConfigurationError(
                "Missing required Azure OpenAI environment variables: " + ", ".join(missing)
            )
        return cls(
            api_key=values["AOAI_API_KEY"],
            endpoint=values["AOAI_ENDPOINT"],
            chat_deployment=values["AOAI_DEPLOY_GPT41_MINI"],
            chat_model=values["AOAI_MODEL_GPT41_MINI"],
            embedding_deployment=values["AOAI_DEPLOY_EMBED_3_SMALL"],
            embedding_model=values["AOAI_MODEL_EMBED_3_SMALL"],
        )


class AdvisoryPresentation(BaseModel):
    """Strict, non-grade presentation returned to the application."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt_version: Literal["advisory-presentation-v1"] = PROMPT_VERSION
    summary: str = Field(min_length=1, max_length=1_200)
    evidence_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("summary")
    @classmethod
    def no_grades_numbers_or_urls(cls, value: str) -> str:
        if (
            _URL.search(value)
            or _NUMBER.search(value)
            or _RISK_WORD.search(value)
            or _PROHIBITED_ASSERTION.search(value)
        ):
            raise ValueError(
                "presentation summary may not contain grades, numbers, URLs, "
                "legal advice, safety guarantees, or fraud determinations"
            )
        return value


class AdvisoryResponse(BaseModel):
    """Presentation plus non-sensitive observability metadata."""

    model_config = ConfigDict(extra="forbid", strict=True)

    presentation: AdvisoryPresentation
    trace_codes: tuple[str, ...]



class ChatBackend(Protocol):
    def invoke(self, input: Any, **kwargs: Any) -> Any: ...


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


OfficialDocumentSearch = Callable[[str], Sequence[Any]]


class AzureOpenAIProvider:
    """A two-turn, one-tool-call Azure OpenAI adapter with injectable backends."""

    def __init__(
        self,
        *,
        settings: AzureOpenAISettings | None = None,
        chat_backend: ChatBackend | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        construct_azure: bool = False,
    ) -> None:
        self._settings = settings or AzureOpenAISettings.from_environment()
        if construct_azure:
            if chat_backend is not None or embedding_backend is not None:
                raise ValueError("construct_azure cannot be combined with injected backends")
            chat_backend, embedding_backend = self._construct_azure_backends(self._settings)
        self._chat = chat_backend
        self._embeddings = embedding_backend

    @staticmethod
    def _construct_azure_backends(settings: AzureOpenAISettings) -> tuple[ChatBackend, EmbeddingBackend]:
        # Lazy import prevents import-time client construction and keeps tests offline.
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

        common = {
            "azure_endpoint": settings.endpoint,
            "api_key": settings.api_key,
            "api_version": _AZURE_API_VERSION,
            "timeout": 15,
            "max_retries": 1,
        }
        return (
            AzureChatOpenAI(
                azure_deployment=settings.chat_deployment,
                model=settings.chat_model,
                temperature=0,
                **common,
            ),
            AzureOpenAIEmbeddings(
                azure_deployment=settings.embedding_deployment,
                model=settings.embedding_model,
                **common,
            ),
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed Chroma documents through the explicitly configured backend."""
        if self._embeddings is None:
            raise RuntimeError("No embedding backend configured")
        return self._embeddings.embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a Chroma query through the explicitly configured backend."""
        if self._embeddings is None:
            raise RuntimeError("No embedding backend configured")
        return self._embeddings.embed_query(text)

    def present(
        self,
        question: str,
        *,
        evidence: Mapping[str, str],
        official_document_search: OfficialDocumentSearch,
    ) -> AdvisoryResponse:
        """Run the provider boundary and normalize expected external failures."""
        try:
            return self._present(
                question,
                evidence=evidence,
                official_document_search=official_document_search,
            )
        except ProviderPresentationError:
            raise
        except Exception as error:
            raise ProviderPresentationError(error) from error

    def _present(
        self,
        question: str,
        *,
        evidence: Mapping[str, str],
        official_document_search: OfficialDocumentSearch,
    ) -> AdvisoryResponse:
        """Return a bounded presentation grounded exclusively in supplied evidence IDs.

        Tool results are never returned or traced.  The model receives only excerpts
        selected from the caller's allowlist, not arbitrary search payload fields.
        """
        if self._chat is None:
            raise RuntimeError("No chat backend configured")
        if not evidence:
            raise ValueError("At least one allowlisted evidence item is required")

        trace = ["presentation_started"]
        first_messages = [
            {"role": "system", "content": self._tool_prompt()},
            {"role": "user", "content": question},
        ]
        first = self._invoke_with_tool(first_messages)
        tool_calls = self._tool_calls(first)
        selected_ids: set[str] = set()
        if tool_calls:
            call = tool_calls[0]
            query = self._tool_query(call, question)
            results = official_document_search(query)
            selected_ids = self._selected_ids(results, evidence)
            trace.append("official_document_tool_called")
        else:
            trace.append("official_document_tool_not_called")

        # An empty tool result still permits the supplied evidence allowlist; no tool
        # payload is propagated into the final model turn.
        permitted_ids = selected_ids if tool_calls else set(evidence)
        if not permitted_ids:
            raise ValueError("official-document tool returned no allowlisted evidence")
        safe_evidence = [
            {"evidence_id": evidence_id, "excerpt": evidence[evidence_id]}
            for evidence_id in sorted(permitted_ids)
        ]
        second_messages = [
            {"role": "system", "content": self._presentation_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "evidence": safe_evidence},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        structured = self._structured_chat().invoke(second_messages)
        presentation = self._presentation_from(structured)
        if not set(presentation.evidence_ids).issubset(permitted_ids):
            raise ValueError("presentation references evidence outside the allowlist")
        trace.append("presentation_completed")
        return AdvisoryResponse(presentation=presentation, trace_codes=tuple(trace))

    def _invoke_with_tool(self, messages: list[dict[str, Any]]) -> Any:
        tool = {
            "name": "search_official_documents",
            "description": "Search approved official-document evidence only.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        }
        binder = getattr(self._chat, "bind_tools", None)
        if callable(binder):
            return binder([tool]).invoke(messages)
        return self._chat.invoke(messages, tools=[tool])

    def _structured_chat(self) -> ChatBackend:
        structured = getattr(self._chat, "with_structured_output", None)
        if not callable(structured):
            raise TypeError("Chat backend must support with_structured_output")
        return structured(AdvisoryPresentation)

    @staticmethod
    def _tool_calls(response: Any) -> list[Mapping[str, Any]]:
        calls = response.get("tool_calls", ()) if isinstance(response, Mapping) else getattr(response, "tool_calls", ())
        return [call for call in calls if isinstance(call, Mapping)][:1]

    @staticmethod
    def _tool_query(call: Mapping[str, Any], fallback: str) -> str:
        arguments = call.get("args", call.get("arguments", {}))
        return arguments.get("query", fallback) if isinstance(arguments, Mapping) else fallback

    @staticmethod
    def _selected_ids(results: Sequence[Any], evidence: Mapping[str, str]) -> set[str]:
        selected: set[str] = set()
        for result in results:
            if isinstance(result, Mapping):
                evidence_id = result.get("evidence_id")
            else:
                evidence_id = getattr(result, "evidence_id", None)
                nested_evidence = getattr(result, "evidence", None)
                if evidence_id is None:
                    evidence_id = getattr(nested_evidence, "evidence_id", None)
            if isinstance(evidence_id, str) and evidence_id in evidence:
                selected.add(evidence_id)
        return selected

    @staticmethod
    def _presentation_from(value: Any) -> AdvisoryPresentation:
        if isinstance(value, AdvisoryPresentation):
            return value
        if isinstance(value, Mapping):
            return AdvisoryPresentation.model_validate(value)
        return AdvisoryPresentation.model_validate(value.model_dump())

    @staticmethod
    def _tool_prompt() -> str:
        return (
            "Use search_official_documents at most once when official evidence is needed. "
            "Do not grade, calculate, invent facts, reveal URLs, or provide chain-of-thought."
        )

    @staticmethod
    def _presentation_prompt() -> str:
        return (
            "Return only the requested structured presentation. Summarize only supplied excerpts. "
            "Do not create claims, grades, numbers, URLs, or chain-of-thought; reference only supplied evidence IDs."
        )
