from __future__ import annotations

import pytest

from jeonse_support.ai_provider import (
    PROMPT_VERSION,
    AdvisoryPresentation,
    AzureOpenAIConfigurationError,
    AzureOpenAIProvider,
    AzureOpenAISettings,
    ProviderPresentationError,
)


ENV = {
    "AOAI_API_KEY": "not-returned-secret",
    "AOAI_ENDPOINT": "https://example.openai.azure.com",
    "AOAI_DEPLOY_GPT41_MINI": "chat-deployment",
    "AOAI_MODEL_GPT41_MINI": "gpt-4.1-mini",
    "AOAI_DEPLOY_EMBED_3_SMALL": "embedding-deployment",
    "AOAI_MODEL_EMBED_3_SMALL": "text-embedding-3-small",
}


class FakeStructuredChat:
    def __init__(self, result: AdvisoryPresentation) -> None:
        self.result = result
        self.messages: list[object] = []

    def invoke(self, messages: object) -> AdvisoryPresentation:
        self.messages.append(messages)
        return self.result


class FakeChat:
    def __init__(self, tool_calls: list[dict[str, object]], result: AdvisoryPresentation) -> None:
        self.tool_calls = tool_calls
        self.bound_tools: list[object] = []
        self.structured = FakeStructuredChat(result)

    def bind_tools(self, tools: list[object]) -> "FakeChat":
        self.bound_tools.extend(tools)
        return self

    def invoke(self, messages: object) -> dict[str, object]:
        return {"tool_calls": self.tool_calls}

    def with_structured_output(self, schema: type[AdvisoryPresentation]) -> FakeStructuredChat:
        assert schema is AdvisoryPresentation
        return self.structured


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


def provider(chat: FakeChat) -> AzureOpenAIProvider:
    return AzureOpenAIProvider(
        settings=AzureOpenAISettings.from_environment(ENV),
        chat_backend=chat,
        embedding_backend=FakeEmbeddings(),
    )


def test_presentation_calls_official_tool_once_and_only_returns_allowlisted_ids() -> None:
    chat = FakeChat(
        [{"name": "search_official_documents", "args": {"query": "official guidance"}}, {"name": "search_official_documents", "args": {"query": "again"}}],
        AdvisoryPresentation(summary="공식 안내 발췌문을 확인하세요.", evidence_ids=("official-a",)),
    )
    calls: list[str] = []

    response = provider(chat).present(
        "secret user question",
        evidence={"official-a": "공식 안내 발췌문", "official-b": "다른 공식 안내"},
        official_document_search=lambda query: calls.append(query) or [{"evidence_id": "official-a", "private": "not propagated"}],
    )

    assert calls == ["official guidance"]
    assert response.presentation.evidence_ids == ("official-a",)
    assert response.presentation.prompt_version == PROMPT_VERSION
    assert response.trace_codes == (
        "presentation_started",
        "official_document_tool_called",
        "presentation_completed",
    )
    assert "secret" not in repr(response)
    assert "private" not in repr(chat.structured.messages)


def test_presentation_rejects_model_evidence_outside_allowlist() -> None:
    chat = FakeChat(
        [],
        AdvisoryPresentation(summary="공식 안내를 확인하세요.", evidence_ids=("invented",)),
    )

    with pytest.raises(ProviderPresentationError) as caught:
        provider(chat).present(
            "question",
            evidence={"official-a": "공식 안내"},
            official_document_search=lambda query: [],
        )
    assert caught.value.trace_code == "presentation_failed_ValueError"


def test_summary_rejects_numbers_urls_and_grades() -> None:
    with pytest.raises(ValueError):
        AdvisoryPresentation(summary="위험 등급은 high 입니다", evidence_ids=("official-a",))
    with pytest.raises(ValueError):
        AdvisoryPresentation(summary="https://example.test", evidence_ids=("official-a",))
    with pytest.raises(ValueError):
        AdvisoryPresentation(summary="숫자 1", evidence_ids=("official-a",))
    for prohibited in (
        "이 계약은 법적으로 안전합니다",
        "이 계약은 사기가 아닙니다",
        "This is legal advice",
        "This is a safety guarantee",
        "This listing is fraudulent",
    ):
        with pytest.raises(ValueError):
            AdvisoryPresentation(summary=prohibited, evidence_ids=("official-a",))


def test_missing_environment_is_explicit_and_does_not_expose_values() -> None:
    incomplete = dict(ENV)
    incomplete.pop("AOAI_API_KEY")

    with pytest.raises(AzureOpenAIConfigurationError) as error:
        AzureOpenAISettings.from_environment(incomplete)

    assert "AOAI_API_KEY" in str(error.value)
    assert "not-returned-secret" not in str(error.value)


def test_injected_embeddings_support_chroma_ingestion() -> None:
    chat = FakeChat([], AdvisoryPresentation(summary="공식 안내를 확인하세요.", evidence_ids=("official-a",)))
    ai_provider = provider(chat)

    assert ai_provider.embed_documents(["a", "bb"]) == [[1.0], [2.0]]
    assert ai_provider.embed_query("abc") == [3.0]
