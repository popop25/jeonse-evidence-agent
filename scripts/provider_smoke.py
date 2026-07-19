"""Run a secret-safe Azure OpenAI and Chroma provider smoke test."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import chromadb

from jeonse_support.ai_provider import AzureOpenAIProvider, AzureOpenAISettings
EXPECTED_EMBEDDING_DIMENSION = 1536
PROFILE = "azure-openai-chroma-smoke-v1"


def _write_receipt(receipt: dict[str, object]) -> None:
    output = Path("artifacts/provider-smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


def main() -> None:
    stage = "configuration"
    try:
        settings = AzureOpenAISettings.from_environment()
        provider = AzureOpenAIProvider(settings=settings, construct_azure=True)

        stage = "chat"
        evidence_id = "evidence-provider-smoke"
        presentation = provider.present(
            "공식 근거를 확인해야 하는 이유를 간단히 정리하세요.",
            evidence={
                evidence_id: "계약 전에 최신 공식 원문과 적용 범위를 직접 확인해야 합니다."
            },
            official_document_search=lambda _query: ({"evidence_id": evidence_id},),
        )
        presentation_valid = bool(presentation.presentation.summary.strip())
        evidence_bound = presentation.presentation.evidence_ids == (evidence_id,)
        if not presentation_valid or not evidence_bound:
            raise RuntimeError("Azure presentation smoke returned an invalid result")

        stage = "embedding"
        vector = provider.embed_query("전세 계약 전 공식 근거 확인")
        if (
            len(vector) != EXPECTED_EMBEDDING_DIMENSION
            or not all(isinstance(value, float) for value in vector)
        ):
            raise RuntimeError("Azure embedding smoke returned an invalid vector")

        stage = "chroma"
        with tempfile.TemporaryDirectory(prefix="provider-chroma-smoke-") as directory:
            client = chromadb.PersistentClient(path=directory)
            collection = client.get_or_create_collection(
                "provider-smoke",
                metadata={"hnsw:space": "cosine"},
            )
            collection.add(
                ids=["smoke-document"],
                documents=["전세 계약 전 공식 근거 확인"],
                embeddings=[vector],
                metadatas=[{"profile": PROFILE}],
            )
            reopened = chromadb.PersistentClient(path=directory).get_collection(
                "provider-smoke"
            )
            result = reopened.query(query_embeddings=[vector], n_results=1)
            if result["ids"] != [["smoke-document"]]:
                raise RuntimeError("Chroma persistence/query smoke failed")
    except Exception:
        _write_receipt(
            {
                "status": "FAIL",
                "profile": PROFILE,
                "failed_stage": stage,
                "secrets_recorded": False,
            }
        )
        raise SystemExit(1) from None

    _write_receipt(
        {
            "status": "PASS",
            "profile": PROFILE,
            "chat": {
                "presentation_valid": presentation_valid,
                "evidence_bound": evidence_bound,
            },
            "embedding": {
                "dimension": len(vector),
                "expected_dimension": EXPECTED_EMBEDDING_DIMENSION,
            },
            "chroma": {"persist_reopen_query": "PASS", "count": 1},
            "secrets_recorded": False,
        }
    )


if __name__ == "__main__":
    main()
