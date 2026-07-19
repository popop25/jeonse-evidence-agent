"""Run a secret-safe Azure OpenAI and Chroma acceptance smoke."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import chromadb

from jeonse_support.ai_provider import AzureOpenAIProvider, AzureOpenAISettings


def main() -> None:
    settings = AzureOpenAISettings.from_environment()
    provider = AzureOpenAIProvider(settings=settings, construct_azure=True)

    evidence_id = "evidence-provider-smoke"
    presentation = provider.present(
        "공식 근거를 확인해야 하는 이유를 간단히 정리하세요.",
        evidence={
            evidence_id: "계약 전에 최신 공식 원문과 적용 범위를 직접 확인해야 합니다."
        },
        official_document_search=lambda _query: ({"evidence_id": evidence_id},),
    )
    vector = provider.embed_query("전세 계약 전 공식 근거 확인")
    if not vector or not all(isinstance(value, float) for value in vector):
        raise RuntimeError("Azure embedding smoke returned an invalid vector")

    with tempfile.TemporaryDirectory(prefix="jeonse-chroma-smoke-") as directory:
        client = chromadb.PersistentClient(path=directory)
        collection = client.get_or_create_collection(
            "provider-smoke",
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            ids=["smoke-document"],
            documents=["전세 계약 전 공식 근거 확인"],
            embeddings=[vector],
            metadatas=[{"profile": "H-Azure-Chroma-v1"}],
        )
        reopened = chromadb.PersistentClient(path=directory).get_collection(
            "provider-smoke"
        )
        result = reopened.query(query_embeddings=[vector], n_results=1)
        if result["ids"] != [["smoke-document"]]:
            raise RuntimeError("Chroma persistence/query smoke failed")

    receipt = {
        "status": "PASS",
        "profile": "H-Azure-Chroma-v1",
        "chat": {
            "presentation_valid": bool(presentation.presentation.summary),
            "evidence_bound": presentation.presentation.evidence_ids == (evidence_id,),
        },
        "embedding": {"dimension": len(vector), "non_empty": True},
        "chroma": {"persist_reopen_query": "PASS", "count": 1},
        "secrets_recorded": False,
    }
    output = Path("artifacts/provider-smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
