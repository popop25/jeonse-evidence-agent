"""Tests for offline JSON adapters and fail-closed official-evidence retrieval."""

from __future__ import annotations

import hashlib
import asyncio
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from jeonse_support.adapters import (
    JsonHugIncidentRepository,
    JsonListingCatalog,
    JsonTransactionRepository,
    SnapshotArtifactError,
    SnapshotArtifactValidationError,
)
from jeonse_support.models import OfficialCorpusAvailability
from jeonse_support.rag import (
    AllowlistedOfficialDocumentRepository,
    ChromaAzureVectorStoreAdapter,
    ImmutableEvidenceValidationError,
    PersistentChromaAzureBackend,
    SemanticOfficialDocumentRepository,
    official_guidance_chunks,
)
from jeonse_support.repositories import (
    ComparableTransactionQuery,
    HugIncidentQuery,
    OfficialDocumentQuery,
    VectorDocument,
    VectorSearchMatch,
)


ROOT = Path(__file__).resolve().parents[2]
LISTINGS = ROOT / "data/catalog/listings.json"
TRANSACTIONS = ROOT / "data/public/transactions.json"
HUG_INCIDENTS = ROOT / "data/public/hug_incidents.json"
AS_OF = date(2026, 6, 30)
PERIOD_START = date(2026, 1, 1)
PERIOD_END = date(2026, 6, 30)
def _refresh_content_sha256(payload: dict[str, object]) -> None:
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "content_sha256"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()




def test_listing_snapshots_load_read_only_and_missing_listing_abstains() -> None:
    async def exercise() -> None:
        catalog = JsonListingCatalog(LISTINGS)
        before = LISTINGS.read_bytes()
        listings = tuple(
            [
                await catalog.get_listing(listing_id)
                for listing_id in (
                    "listing-mapogu-low",
                    "listing-songpagu-medium",
                    "listing-gangseogu-high",
                )
            ]
        )

        assert all(listing is not None for listing in listings)
        assert [listing.listing_id for listing in listings if listing is not None] == [
            "listing-mapogu-low",
            "listing-songpagu-medium",
            "listing-gangseogu-high",
        ]
        assert await catalog.get_listing("unknown-synthetic-listing") is None
        assert LISTINGS.read_bytes() == before

    asyncio.run(exercise())


def test_transaction_repository_returns_complete_multiset_deterministically() -> None:
    async def exercise() -> None:
        catalog = JsonListingCatalog(LISTINGS)
        listing = await catalog.get_listing("listing-mapogu-low")
        assert listing is not None
        repository = JsonTransactionRepository(TRANSACTIONS)
        before = TRANSACTIONS.read_bytes()
        query = ComparableTransactionQuery(listing=listing, as_of=AS_OF)

        first = await repository.list_comparables(query)
        second = await repository.list_comparables(query)

        assert first == second
        assert len(first) == 14
        identifiers = {sample.transaction_id for sample in first}
        assert {"tx-map-01", "tx-map-02", "tx-map-03"} <= identifiers
        assert {
            "tx-map-stale-date",
            "tx-map-future-date",
            "tx-map-small-area",
            "tx-map-large-area",
            "tx-map-wrong-type",
            "tx-song-01",
            "tx-gang-01",
        } <= identifiers
        no_dong_listing = listing.model_copy(update={"legal_dong": "공덕동"})
        assert await repository.list_comparables(
            ComparableTransactionQuery(listing=no_dong_listing, as_of=AS_OF)
        ) == first
        assert TRANSACTIONS.read_bytes() == before

    asyncio.run(exercise())


def test_hug_subject_and_fixed_period_references_return_twenty_distinct_districts() -> None:
    async def exercise() -> None:
        repository = JsonHugIncidentRepository(HUG_INCIDENTS)
        before = HUG_INCIDENTS.read_bytes()
        query = HugIncidentQuery(as_of=AS_OF, period_start=PERIOD_START, period_end=PERIOD_END)

        subject = await repository.get_subject_statistic(query)
        references = await repository.list_reference_statistics(query)

        assert subject is not None
        assert subject.statistic_id == "hug-subject-2026h1"
        assert subject.granularity == "district"
        assert subject.geography == "마포구"
        assert len(references) == 20
        assert len({statistic.geography for statistic in references}) == 20
        assert {statistic.granularity for statistic in references} == {subject.granularity}
        assert {(statistic.period_start, statistic.period_end) for statistic in references} == {
            (PERIOD_START, PERIOD_END)
        }
        assert {statistic.source_name for statistic in references} == {subject.source_name}
        assert {statistic.snapshot_as_of for statistic in references} == {
            subject.snapshot_as_of
        }
        assert {statistic.metric_definition for statistic in references} == {
            subject.metric_definition
        }
        assert all(statistic.eligible_contract_count > 0 for statistic in references)
        assert subject.geography in {statistic.geography for statistic in references}
        assert [statistic.statistic_id for statistic in references] == [
            f"hug-ref-{index:02d}" for index in range(1, 21)
        ]
        no_match = HugIncidentQuery(
            as_of=AS_OF,
            period_start=date(2025, 1, 1),
            period_end=date(2025, 6, 30),
        )
        assert await repository.get_subject_statistic(no_match) is None
        assert await repository.list_reference_statistics(no_match) == ()
        assert HUG_INCIDENTS.read_bytes() == before

    asyncio.run(exercise())


def test_semantically_invalid_snapshot_artifacts_fail_explicitly(tmp_path: Path) -> None:
    async def exercise() -> None:
        missing = JsonListingCatalog(tmp_path / "missing.json")
        with pytest.raises(SnapshotArtifactError, match="missing"):
            await missing.get_listing("listing-mapogu-low")

        malformed_path = tmp_path / "malformed.json"
        malformed_path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(SnapshotArtifactValidationError, match="invalid"):
            await JsonListingCatalog(malformed_path).get_listing("listing-mapogu-low")

        missing_dataset_as_of = json.loads(LISTINGS.read_text(encoding="utf-8"))
        missing_dataset_as_of.pop("snapshot_as_of")
        _refresh_content_sha256(missing_dataset_as_of)
        missing_dataset_as_of_path = tmp_path / "missing-dataset-as-of.json"
        missing_dataset_as_of_path.write_text(
            json.dumps(missing_dataset_as_of), encoding="utf-8"
        )
        with pytest.raises(SnapshotArtifactValidationError, match="snapshot_as_of"):
            await JsonListingCatalog(missing_dataset_as_of_path).get_listing(
                "listing-mapogu-low"
            )

        inconsistent_provenance = json.loads(LISTINGS.read_text(encoding="utf-8"))
        inconsistent_provenance["listings"][0]["provenance_id"] = "different-snapshot"
        _refresh_content_sha256(inconsistent_provenance)
        inconsistent_provenance_path = tmp_path / "inconsistent-provenance.json"
        inconsistent_provenance_path.write_text(
            json.dumps(inconsistent_provenance), encoding="utf-8"
        )
        with pytest.raises(SnapshotArtifactValidationError, match="provenance"):
            await JsonListingCatalog(inconsistent_provenance_path).get_listing(
                "listing-mapogu-low"
            )

        duplicate_listing = json.loads(LISTINGS.read_text(encoding="utf-8"))
        duplicate_listing["listings"].append(duplicate_listing["listings"][0].copy())
        _refresh_content_sha256(duplicate_listing)
        duplicate_path = tmp_path / "duplicate-listing.json"
        duplicate_path.write_text(json.dumps(duplicate_listing), encoding="utf-8")
        with pytest.raises(SnapshotArtifactValidationError, match="duplicate"):
            await JsonListingCatalog(duplicate_path).get_listing("listing-mapogu-low")

        malformed_transaction = json.loads(TRANSACTIONS.read_text(encoding="utf-8"))
        malformed_transaction["transactions"][0].pop("provenance_id")
        _refresh_content_sha256(malformed_transaction)
        malformed_transaction_path = tmp_path / "malformed-transaction.json"
        malformed_transaction_path.write_text(json.dumps(malformed_transaction), encoding="utf-8")
        listing = await JsonListingCatalog(LISTINGS).get_listing("listing-mapogu-low")
        assert listing is not None
        with pytest.raises(SnapshotArtifactValidationError, match="provenance"):
            await JsonTransactionRepository(malformed_transaction_path).list_comparables(
                ComparableTransactionQuery(listing=listing, as_of=AS_OF)
            )

        duplicate_hug = json.loads(HUG_INCIDENTS.read_text(encoding="utf-8"))
        duplicate_hug["statistics"].append(duplicate_hug["statistics"][1].copy())
        _refresh_content_sha256(duplicate_hug)
        duplicate_hug_path = tmp_path / "duplicate-hug.json"
        duplicate_hug_path.write_text(json.dumps(duplicate_hug), encoding="utf-8")
        with pytest.raises(SnapshotArtifactValidationError, match="duplicate"):
            await JsonHugIncidentRepository(duplicate_hug_path).list_reference_statistics(
                HugIncidentQuery(as_of=AS_OF, period_start=PERIOD_START, period_end=PERIOD_END)
            )

    asyncio.run(exercise())


def test_snapshot_envelope_requires_untampered_checksum_and_qualified_identity(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        missing_checksum = json.loads(LISTINGS.read_text(encoding="utf-8"))
        missing_checksum.pop("content_sha256")
        missing_checksum_path = tmp_path / "missing-checksum.json"
        missing_checksum_path.write_text(json.dumps(missing_checksum), encoding="utf-8")
        with pytest.raises(SnapshotArtifactValidationError, match="content_sha256"):
            await JsonListingCatalog(missing_checksum_path).get_listing("listing-mapogu-low")

        tampered_checksum = json.loads(LISTINGS.read_text(encoding="utf-8"))
        tampered_checksum["snapshot_notice"] = "tampered snapshot notice"
        tampered_checksum_path = tmp_path / "tampered-checksum.json"
        tampered_checksum_path.write_text(json.dumps(tampered_checksum), encoding="utf-8")
        with pytest.raises(SnapshotArtifactValidationError, match="content_sha256"):
            await JsonListingCatalog(tampered_checksum_path).get_listing("listing-mapogu-low")

        mismatched_identity = json.loads(LISTINGS.read_text(encoding="utf-8"))
        mismatched_identity["dataset_id"] = "qualified-listings"
        mismatched_identity["dataset_version"] = "v1"
        mismatched_identity["listings"][0]["dataset_id"] = "qualified-listings"
        mismatched_identity["listings"][0]["dataset_version"] = "v2"
        _refresh_content_sha256(mismatched_identity)
        mismatched_identity_path = tmp_path / "mismatched-identity.json"
        mismatched_identity_path.write_text(json.dumps(mismatched_identity), encoding="utf-8")
        with pytest.raises(SnapshotArtifactValidationError, match="dataset identity"):
            await JsonListingCatalog(mismatched_identity_path).get_listing("listing-mapogu-low")

    asyncio.run(exercise())


def test_official_retrieval_is_allowlisted_provenance_bearing_and_can_abstain() -> None:
    async def exercise() -> None:
        repository = AllowlistedOfficialDocumentRepository()
        documents = await repository.retrieve(OfficialDocumentQuery(query_text="전세 계약"))

        assert documents
        assert {document.document_id for document in documents} <= {
            "molit-jeonse-source-pointer",
            "hug-guarantee-source-pointer",
            "korea-legal-aid-housing-source-pointer",
        }
        for document in documents:
            evidence = document.evidence
            assert evidence.source_record_id == document.document_id
            assert evidence.source_name == document.publisher
            assert evidence.url is not None
            assert evidence.content_hash is not None
        assert await repository.retrieve(OfficialDocumentQuery(query_text="unmatchedtoken")) == ()

    asyncio.run(exercise())


def test_adapter_and_rag_import_and_snapshot_load_do_not_access_network() -> None:
    code = f'''\
import socket
import sys
from datetime import date
from pathlib import Path

class NoNetworkSocket:
    def __init__(self, *args, **kwargs):
        raise AssertionError("network access is forbidden")

socket.socket = NoNetworkSocket
sys.path.insert(0, {str(ROOT / "src")!r})

from jeonse_support.adapters import JsonHugIncidentRepository, JsonListingCatalog, JsonTransactionRepository
from jeonse_support.rag import AllowlistedOfficialDocumentRepository
from jeonse_support.repositories import ComparableTransactionQuery, HugIncidentQuery, OfficialDocumentQuery

root = Path({str(ROOT)!r})
def complete(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("offline adapter unexpectedly suspended")

listing = complete(JsonListingCatalog(root / "data/catalog/listings.json").get_listing("listing-mapogu-low"))
assert listing is not None
complete(JsonTransactionRepository(root / "data/public/transactions.json").list_comparables(ComparableTransactionQuery(listing=listing, as_of=date(2026, 6, 30))))
hug_query = HugIncidentQuery(as_of=date(2026, 6, 30), period_start=date(2026, 1, 1), period_end=date(2026, 6, 30))
complete(JsonHugIncidentRepository(root / "data/public/hug_incidents.json").get_subject_statistic(hug_query))
complete(AllowlistedOfficialDocumentRepository().retrieve(OfficialDocumentQuery(query_text="전세")))
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("offline adapter import and snapshot load exceeded the 10-second timeout")

    assert result.returncode == 0, result.stderr
class _DeterministicEmbeddings:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.lower()
        return [
            float("주택도시보증공사" in normalized or "hug" in normalized),
            float("대한법률구조공단" in normalized or "legal aid" in normalized),
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)


def _lawful_corpus(tmp_path: Path, content: str = "HUG 주택도시보증공사 전세보증 안내") -> Path:
    artifact = tmp_path / "official.txt"
    artifact.write_text(content, encoding="utf-8")
    checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest = {
        "manifest_version": "official-evidence-corpus-v2",
        "profile_id": "H-Azure-Chroma-v2",
        "index_version": "immutable-official-evidence-v1",
        "sources": [
            {
                "source_id": "hug-guidance",
                "publisher": "주택도시보증공사",
                "canonical_url": "https://www.khug.or.kr/",
                "license": "reviewed sandbox license",
                "license_status": "approved_for_indexing",
                "terms_review_status": "approved_for_indexing",
            }
        ],
        "records": [
            {
                "document_id": "hug-immutable-guidance",
                "evidence_id": "evidence-hug-immutable-guidance",
                "source_id": "hug-guidance",
                "title": "전세보증 안내",
                "document_type": "guarantee-guidance",
                "content_path": "official.txt",
                "content_sha256": checksum,
                "span_sha256": checksum,
                "as_of": "2026-06-30T00:00:00+00:00",
                "page_start": 1,
                "page_end": 1,
                "span_start": 0,
                "span_end": len(content),
            }
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


class _FakeChromaCollection:
    def __init__(self, metadata: dict[str, str]) -> None:
        self.metadata = metadata
        self.rows: dict[str, tuple[str, dict[str, str], list[float]]] = {}
        self.last_n_results: int | None = None

    def get(self, *, ids: list[str], include: list[str]) -> dict[str, list[str]]:
        return {"ids": [item for item in ids if item in self.rows]}

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        self.rows.update(
            {
                document_id: (text, metadata, embedding)
                for document_id, text, embedding, metadata in zip(
                    ids, documents, embeddings, metadatas, strict=True
                )
            }
        )

    def query(
        self, *, query_embeddings: list[list[float]], n_results: int, include: list[str]
    ) -> dict[str, list[list[object]]]:
        assert include == ["documents", "metadatas", "distances"]
        self.last_n_results = n_results
        query = query_embeddings[0]
        rows = sorted(
            self.rows.items(),
            key=lambda item: sum(left * right for left, right in zip(query, item[1][2], strict=True)),
            reverse=True,
        )[:n_results]
        scores = [sum(left * right for left, right in zip(query, row[2], strict=True)) for _, row in rows]
        return {
            "ids": [[document_id for document_id, _ in rows]],
            "documents": [[text for _, (text, _, _) in rows]],
            "metadatas": [[metadata for _, (_, metadata, _) in rows]],
            "distances": [[1 - score for score in scores]],
        }


class _FakeChromaClient:
    def __init__(self) -> None:
        self.collection: _FakeChromaCollection | None = None

    def factory(self, path: str) -> "_FakeChromaClient":
        return self

    def get_or_create_collection(
        self, *, name: str, metadata: dict[str, str]
    ) -> _FakeChromaCollection:
        assert name == "jeonse-support-h-azure-chroma-v2-immutable-official-evidence-v1"
        if self.collection is None:
            self.collection = _FakeChromaCollection(metadata)
        return self.collection


def test_valid_immutable_corpus_enables_source_relevant_semantic_retrieval(tmp_path: Path) -> None:
    async def exercise() -> None:
        embeddings = _DeterministicEmbeddings()
        client = _FakeChromaClient()
        repository = SemanticOfficialDocumentRepository(
            ChromaAzureVectorStoreAdapter(
                PersistentChromaAzureBackend(
                    embeddings, tmp_path / "chroma", chroma_client_factory=client.factory
                )
            ),
            manifest_path=_lawful_corpus(tmp_path),
        )

        documents = await repository.retrieve(
            OfficialDocumentQuery(query_text="HUG 주택도시보증공사 전세보증", max_results=1)
        )

        assert [document.document_id for document in documents] == ["hug-immutable-guidance"]
        assert documents[0].evidence.kind.value == "official_document"
        assert documents[0].source_id == "hug-guidance"
        assert documents[0].page_start == documents[0].page_end == 1
        assert documents[0].span_hash == documents[0].evidence.content_hash
        assert documents[0].evidence.content_hash is not None
        assert client.collection is not None
        assert client.collection.metadata["index_version"] == "immutable-official-evidence-v1"
        assert embeddings.document_calls
        assert await repository.retrieve(OfficialDocumentQuery(query_text="unrelated out of corpus")) == ()

    asyncio.run(exercise())


def test_manifest_or_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest = _lawful_corpus(tmp_path)
    content_path = tmp_path / "official.txt"
    content_path.write_text("modified after manifest approval", encoding="utf-8")

    with pytest.raises(ImmutableEvidenceValidationError, match="checksum mismatch"):
        SemanticOfficialDocumentRepository(
            ChromaAzureVectorStoreAdapter(_FailingBackend()), manifest_path=manifest
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["index_version"] = "unapproved-index"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ImmutableEvidenceValidationError, match="manifest profile"):
        SemanticOfficialDocumentRepository(
            ChromaAzureVectorStoreAdapter(_FailingBackend()), manifest_path=manifest
        )

def test_product_semantic_retrieval_excludes_homepage_pointers_and_empty_manifest_abstains(
    tmp_path: Path,
) -> None:
    assert official_guidance_chunks() == ()

    async def exercise() -> None:
        manifest = _lawful_corpus(tmp_path)
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["records"] = []
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        repository = SemanticOfficialDocumentRepository(
            ChromaAzureVectorStoreAdapter(_FailingBackend()),  # type: ignore[arg-type]
            manifest_path=manifest,
        )
        assert await repository.retrieve(OfficialDocumentQuery(query_text="전세 계약")) == ()

    asyncio.run(exercise())


def test_semantic_corpus_availability_distinguishes_empty_corpus_from_abstention(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        empty_manifest = _lawful_corpus(empty_dir)
        payload = json.loads(empty_manifest.read_text(encoding="utf-8"))
        payload["records"] = []
        empty_manifest.write_text(json.dumps(payload), encoding="utf-8")
        empty_repository = SemanticOfficialDocumentRepository(
            ChromaAzureVectorStoreAdapter(_FailingBackend()),  # type: ignore[arg-type]
            manifest_path=empty_manifest,
        )
        assert empty_repository.corpus_availability is OfficialCorpusAvailability.UNAVAILABLE
        assert await empty_repository.retrieve(OfficialDocumentQuery(query_text="전세 계약")) == ()

        present_dir = tmp_path / "present"
        present_dir.mkdir()
        present_repository = SemanticOfficialDocumentRepository(
            ChromaAzureVectorStoreAdapter(
                PersistentChromaAzureBackend(
                    _DeterministicEmbeddings(),
                    tmp_path / "present-chroma",
                    chroma_client_factory=_FakeChromaClient().factory,
                )
            ),
            manifest_path=_lawful_corpus(present_dir),
        )
        assert present_repository.corpus_availability is OfficialCorpusAvailability.AVAILABLE
        assert await present_repository.retrieve(
            OfficialDocumentQuery(query_text="unrelated out of corpus")
        ) == ()

    asyncio.run(exercise())


def test_semantic_retrieval_rejects_stale_chunk_with_current_evidence_id(tmp_path: Path) -> None:
    class StaleResultBackend:
        async def upsert_documents(self, documents: tuple[VectorDocument, ...]) -> None:
            self.documents = documents

        async def search_documents(self, query: object) -> tuple[VectorSearchMatch, ...]:
            expected = self.documents[0]
            return (
                VectorSearchMatch(
                    document=VectorDocument(
                        document_id=expected.document_id,
                        evidence_id=expected.evidence_id,
                        text=f"{expected.text} stale",
                    ),
                    cosine_relevance=1.0,
                ),
            )

    async def exercise() -> None:
        repository = SemanticOfficialDocumentRepository(
            ChromaAzureVectorStoreAdapter(StaleResultBackend()),
            manifest_path=_lawful_corpus(tmp_path),
        )
        assert await repository.retrieve(
            OfficialDocumentQuery(query_text="HUG 주택도시보증공사 전세보증")
        ) == ()

    asyncio.run(exercise())


def test_semantic_retrieval_abstains_below_frozen_cosine_threshold(tmp_path: Path) -> None:
    async def exercise() -> None:
        repository = SemanticOfficialDocumentRepository(
            ChromaAzureVectorStoreAdapter(
                PersistentChromaAzureBackend(
                    _DeterministicEmbeddings(),
                    tmp_path / "chroma",
                    chroma_client_factory=_FakeChromaClient().factory,
                )
            ),
            manifest_path=_lawful_corpus(tmp_path),
        )
        assert await repository.retrieve(OfficialDocumentQuery(query_text="unrelated out of corpus")) == ()

    asyncio.run(exercise())


class _FailingBackend:
    async def upsert_documents(self, documents: tuple[object, ...]) -> None:
        raise RuntimeError("vector backend unavailable")

    async def search_documents(self, query: object) -> tuple[object, ...]:
        raise RuntimeError("vector backend unavailable")


def test_semantic_retrieval_propagates_backend_failure_without_lexical_downgrade(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        repository = SemanticOfficialDocumentRepository(
            ChromaAzureVectorStoreAdapter(_FailingBackend()),  # type: ignore[arg-type]
            manifest_path=_lawful_corpus(tmp_path),
        )
        with pytest.raises(RuntimeError, match="vector backend unavailable"):
            await repository.retrieve(OfficialDocumentQuery(query_text="HUG 주택도시보증공사"))

    asyncio.run(exercise())