"""Fail-closed retrieval of lawfully acquired immutable official evidence."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import yaml

from .ai_provider import AzureOpenAIProvider
from .models import Evidence, EvidenceKind, OfficialCorpusAvailability
from .repositories import (
    OfficialDocument,
    OfficialDocumentQuery,
    VectorDocument,
    VectorSearchMatch,
    VectorSearchQuery,
    VectorStore,
)

_BUNDLE_DOCUMENT_TYPE = "official-source-pointer"
_PROFILE_ID = "H-Azure-Chroma-v2"
_INDEX_VERSION = "immutable-official-evidence-v1"
_COLLECTION_NAME = "jeonse-support-h-azure-chroma-v2-immutable-official-evidence-v1"
_MAX_SEARCH_RESULTS = 20
_MIN_COSINE_RELEVANCE = 0.80
_TOKEN = re.compile(r"[가-힣A-Za-z0-9]+")
_DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data" / "rag" / "manifest.yaml"


class ImmutableEvidenceValidationError(ValueError):
    """A lawful official corpus is absent or failed immutable-artifact validation."""


class VectorBackend(Protocol):
    """Injection boundary for a configured vector backend."""

    async def upsert_documents(self, documents: tuple[VectorDocument, ...]) -> None: ...

    async def search_documents(self, query: VectorSearchQuery) -> tuple[VectorSearchMatch, ...]: ...


class ChromaAzureVectorStoreAdapter(VectorStore):
    """Vector-store seam with no lexical or in-memory fallback."""

    def __init__(self, backend: VectorBackend) -> None:
        self._backend = backend

    async def upsert(self, documents: tuple[VectorDocument, ...]) -> None:
        await self._backend.upsert_documents(documents)

    async def search(self, query: VectorSearchQuery) -> tuple[VectorSearchMatch, ...]:
        return await self._backend.search_documents(query)


class PersistentChromaAzureBackend:
    """Persistent cosine Chroma backend using only Azure embeddings."""

    def __init__(
        self,
        provider: AzureOpenAIProvider,
        persist_directory: Path,
        *,
        profile_id: str = _PROFILE_ID,
        index_version: str = _INDEX_VERSION,
        collection_name: str = _COLLECTION_NAME,
        chroma_client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._provider = provider
        self._persist_directory = persist_directory
        self._profile_id = profile_id
        self._index_version = index_version
        self._collection_name = collection_name
        self._chroma_client_factory = chroma_client_factory
        self._collection: Any | None = None
        self._operation_lock = asyncio.Lock()

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        factory = self._chroma_client_factory
        if factory is None:
            from chromadb import PersistentClient

            factory = lambda path: PersistentClient(path=path)
        client = factory(str(self._persist_directory))
        metadata = {
            "hnsw:space": "cosine",
            "profile_id": self._profile_id,
            "index_version": self._index_version,
        }
        collection = client.get_or_create_collection(name=self._collection_name, metadata=metadata)
        actual_metadata = getattr(collection, "metadata", None)
        if not isinstance(actual_metadata, dict) or any(
            actual_metadata.get(key) != value for key, value in metadata.items()
        ):
            raise RuntimeError("Chroma collection profile or distance configuration mismatch")
        self._collection = collection
        return collection

    async def upsert_documents(self, documents: tuple[VectorDocument, ...]) -> None:
        if not documents:
            return
        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Vector document IDs must be unique")
        async with self._operation_lock:
            await asyncio.to_thread(self._upsert_documents_sync, documents)

    def _upsert_documents_sync(self, documents: tuple[VectorDocument, ...]) -> None:
        collection = self._get_collection()
        document_ids = [document.document_id for document in documents]
        existing = collection.get(ids=document_ids, include=["metadatas"])
        existing_ids = set(existing.get("ids", ()))
        missing = [document for document in documents if document.document_id not in existing_ids]
        if not missing:
            return
        embeddings = self._provider.embed_documents([document.text for document in missing])
        if len(embeddings) != len(missing):
            raise RuntimeError("Azure embedding response count did not match Chroma documents")
        collection.upsert(
            ids=[document.document_id for document in missing],
            documents=[document.text for document in missing],
            embeddings=embeddings,
            metadatas=[
                {
                    "evidence_id": document.evidence_id,
                    "profile_id": self._profile_id,
                    "index_version": self._index_version,
                }
                for document in missing
            ],
        )

    async def search_documents(self, query: VectorSearchQuery) -> tuple[VectorSearchMatch, ...]:
        async with self._operation_lock:
            return await asyncio.to_thread(self._search_documents_sync, query)

    def _search_documents_sync(
        self, query: VectorSearchQuery
    ) -> tuple[VectorSearchMatch, ...]:
        collection = self._get_collection()
        query_embedding = self._provider.embed_query(query.query_text)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(query.limit, _MAX_SEARCH_RESULTS),
            include=["documents", "metadatas", "distances"],
        )
        ids = self._first_query_result(result, "ids")
        texts = self._first_query_result(result, "documents")
        metadata = self._first_query_result(result, "metadatas")
        distances = self._first_query_result(result, "distances")
        if not (len(ids) == len(texts) == len(metadata) == len(distances)):
            raise RuntimeError("Chroma search result fields have inconsistent lengths")
        matches: list[VectorSearchMatch] = []
        for document_id, text, item_metadata, distance in zip(
            ids, texts, metadata, distances, strict=True
        ):
            if (
                not isinstance(document_id, str)
                or not isinstance(text, str)
                or not isinstance(item_metadata, dict)
                or item_metadata.get("profile_id") != self._profile_id
                or item_metadata.get("index_version") != self._index_version
                or not isinstance(item_metadata.get("evidence_id"), str)
                or not isinstance(distance, (int, float))
                or isinstance(distance, bool)
                or not math.isfinite(distance)
                or not 0 <= float(distance) <= 2
            ):
                raise RuntimeError("Chroma search returned invalid isolated result metadata or distance")
            matches.append(
                VectorSearchMatch(
                    document=VectorDocument(
                        document_id=document_id,
                        text=text,
                        evidence_id=item_metadata["evidence_id"],
                    ),
                    cosine_relevance=1 - float(distance),
                )
            )
        return tuple(matches)

    @staticmethod
    def _first_query_result(result: Any, field: str) -> Sequence[Any]:
        value = result.get(field) if isinstance(result, dict) else None
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
            raise RuntimeError(f"Chroma search returned invalid {field}")
        return value[0]


class ImmutableOfficialEvidenceLoader:
    """Load only manifest-allowlisted, checksum-verified local official artifacts."""

    def __init__(self, manifest_path: Path = _DEFAULT_MANIFEST) -> None:
        self._manifest_path = manifest_path
        self._corpus_availability = OfficialCorpusAvailability.UNAVAILABLE
        self._manifest_identity: str | None = None

    @property
    def corpus_availability(self) -> OfficialCorpusAvailability:
        return self._corpus_availability

    @property
    def manifest_identity(self) -> str:
        if self._manifest_identity is None:
            raise RuntimeError("official evidence manifest has not been validated")
        return self._manifest_identity

    def load(self) -> tuple[OfficialDocument, ...]:
        if not self._manifest_path.is_file():
            raise ImmutableEvidenceValidationError("official evidence manifest is missing")
        try:
            manifest_bytes = self._manifest_path.read_bytes()
            manifest = yaml.safe_load(manifest_bytes)
        except (OSError, yaml.YAMLError) as error:
            raise ImmutableEvidenceValidationError("official evidence manifest is unreadable") from error
        if not isinstance(manifest, dict) or manifest.get("manifest_version") != "official-evidence-corpus-v2":
            raise ImmutableEvidenceValidationError("official evidence manifest version is invalid")
        if manifest.get("profile_id") != _PROFILE_ID or manifest.get("index_version") != _INDEX_VERSION:
            raise ImmutableEvidenceValidationError("official evidence manifest profile is invalid")
        sources = manifest.get("sources")
        records = manifest.get("records")
        if not isinstance(sources, list) or not isinstance(records, list):
            raise ImmutableEvidenceValidationError("official evidence manifest requires sources and records")
        self._manifest_identity = hashlib.sha256(manifest_bytes).hexdigest()
        self._corpus_availability = (
            OfficialCorpusAvailability.AVAILABLE
            if records
            else OfficialCorpusAvailability.UNAVAILABLE
        )
        source_by_id: dict[str, dict[str, Any]] = {}
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("source_id"), str):
                raise ImmutableEvidenceValidationError("official evidence source metadata is invalid")
            source_id = source["source_id"]
            if source_id in source_by_id or not all(
                isinstance(source.get(field), str) and source[field].strip()
                for field in ("publisher", "canonical_url", "license")
            ):
                raise ImmutableEvidenceValidationError("official evidence source allowlist is invalid")
            source_by_id[source_id] = source
        documents: list[OfficialDocument] = []
        evidence_ids: set[str] = set()
        for record in records:
            document = self._load_record(record, source_by_id)
            if document.document_id in {item.document_id for item in documents}:
                raise ImmutableEvidenceValidationError("official evidence document IDs must be unique")
            evidence_id = document.evidence.evidence_id
            if evidence_id in evidence_ids:
                raise ImmutableEvidenceValidationError("official evidence IDs must be unique")
            evidence_ids.add(evidence_id)
            documents.append(document)
        return tuple(documents)

    def _load_record(
        self, record: Any, source_by_id: dict[str, dict[str, Any]]
    ) -> OfficialDocument:
        if not isinstance(record, dict):
            raise ImmutableEvidenceValidationError("official evidence record is invalid")
        required_text = (
            "document_id",
            "evidence_id",
            "source_id",
            "title",
            "document_type",
            "content_path",
            "content_sha256",
            "span_sha256",
            "as_of",
        )
        if not all(isinstance(record.get(field), str) and record[field].strip() for field in required_text):
            raise ImmutableEvidenceValidationError("official evidence record metadata is incomplete")
        source = source_by_id.get(record["source_id"])
        if source is None or any(
            source.get(field) != "approved_for_indexing"
            for field in ("license_status", "terms_review_status")
        ):
            raise ImmutableEvidenceValidationError("official evidence source is not approved for indexing")
        if not all(
            isinstance(record.get(field), int) and not isinstance(record[field], bool)
            for field in ("page_start", "page_end", "span_start", "span_end")
        ) or record["page_start"] < 1 or record["page_end"] < record["page_start"]:
            raise ImmutableEvidenceValidationError("official evidence page metadata is invalid")
        content_path = (self._manifest_path.parent / record["content_path"]).resolve()
        if self._manifest_path.parent.resolve() not in content_path.parents or not content_path.is_file():
            raise ImmutableEvidenceValidationError("official evidence content artifact is unavailable")
        try:
            content_bytes = content_path.read_bytes()
            content = content_bytes.decode("utf-8")
        except (OSError, UnicodeError) as error:
            raise ImmutableEvidenceValidationError("official evidence content artifact is unreadable") from error
        if hashlib.sha256(content_bytes).hexdigest() != record["content_sha256"]:
            raise ImmutableEvidenceValidationError("official evidence content checksum mismatch")
        span_start, span_end = record["span_start"], record["span_end"]
        if span_start < 0 or span_end <= span_start or span_end > len(content):
            raise ImmutableEvidenceValidationError("official evidence span metadata is invalid")
        excerpt = content[span_start:span_end]
        if hashlib.sha256(excerpt.encode("utf-8")).hexdigest() != record["span_sha256"]:
            raise ImmutableEvidenceValidationError("official evidence span checksum mismatch")
        try:
            as_of = datetime.fromisoformat(record["as_of"].replace("Z", "+00:00"))
        except ValueError as error:
            raise ImmutableEvidenceValidationError("official evidence as-of metadata is invalid") from error
        if as_of.tzinfo is None:
            raise ImmutableEvidenceValidationError("official evidence as-of metadata requires a timezone")
        return OfficialDocument(
            document_id=record["document_id"],
            title=record["title"],
            publisher=source["publisher"],
            document_type=record["document_type"],
            retrieved_at=as_of,
            source_id=record["source_id"],
            as_of=as_of,
            page_start=record["page_start"],
            page_end=record["page_end"],
            span_start=span_start,
            span_end=span_end,
            span_hash=record["span_sha256"],
            evidence=Evidence(
                evidence_id=record["evidence_id"],
                kind=EvidenceKind.OFFICIAL_DOCUMENT,
                source_name=source["publisher"],
                source_record_id=record["document_id"],
                retrieved_at=as_of,
                observed_at=as_of,
                url=source["canonical_url"],
                excerpt=excerpt,
                content_hash=record["span_sha256"],
            ),
        )


class SemanticOfficialDocumentRepository:
    """Search validated immutable official evidence with frozen cosine abstention."""

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        manifest_path: Path = _DEFAULT_MANIFEST,
    ) -> None:
        self._vector_store = vector_store
        loader = ImmutableOfficialEvidenceLoader(manifest_path)
        self._documents = loader.load()
        self._corpus_availability = loader.corpus_availability
        self._chunks = _chunks_for_documents(self._documents, loader.manifest_identity)
        self._chunks_by_document_id = {chunk.document_id: chunk for chunk in self._chunks}
        self._documents_by_evidence_id = {
            document.evidence.evidence_id: document for document in self._documents
        }

    @property
    def corpus_availability(self) -> OfficialCorpusAvailability:
        """Expose corpus unavailability without conflating it with an empty query result."""
        return self._corpus_availability

    async def retrieve(self, query: OfficialDocumentQuery) -> tuple[OfficialDocument, ...]:
        if self._corpus_availability is OfficialCorpusAvailability.UNAVAILABLE:
            return ()
        allowed_types = set(query.document_types)
        await self._vector_store.upsert(self._chunks)
        matches = await self._vector_store.search(
            VectorSearchQuery(query_text=query.query_text, limit=min(query.max_results, _MAX_SEARCH_RESULTS))
        )
        selected: list[OfficialDocument] = []
        seen: set[str] = set()
        for match in matches:
            if match.cosine_relevance < _MIN_COSINE_RELEVANCE:
                continue
            expected_chunk = self._chunks_by_document_id.get(match.document.document_id)
            if (
                expected_chunk is None
                or match.document.evidence_id != expected_chunk.evidence_id
                or match.document.text != expected_chunk.text
            ):
                continue
            document = self._documents_by_evidence_id.get(match.document.evidence_id)
            if document is not None and (
                not allowed_types or document.document_type in allowed_types
            ) and document.document_id not in seen:
                selected.append(document)
                seen.add(document.document_id)
            if len(selected) == query.max_results:
                break
        return tuple(selected)


class AllowlistedOfficialDocumentRepository:
    """Offline pointer helper; its application-authored records are never product evidence."""

    def __init__(self, documents: tuple[OfficialDocument, ...] | None = None) -> None:
        self._documents = official_guidance_documents() if documents is None else documents

    async def retrieve(self, query: OfficialDocumentQuery) -> tuple[OfficialDocument, ...]:
        allowed_types = set(query.document_types)
        tokens = set(_tokens(query.query_text))
        scored: list[tuple[int, OfficialDocument]] = []
        for document in self._documents:
            if allowed_types and document.document_type not in allowed_types:
                continue
            excerpt = document.evidence.excerpt or ""
            score = len(tokens.intersection(_tokens(f"{document.title} {excerpt}")))
            if score:
                scored.append((score, document))
        return tuple(
            document
            for _, document in sorted(scored, key=lambda item: (-item[0], item[1].document_id))[
                : query.max_results
            ]
        )


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN.findall(text))


def _chunk_id(
    document: OfficialDocument, index: int, text: str, manifest_identity: str
) -> str:
    immutable_identity = ":".join(
        (
            manifest_identity,
            document.document_id,
            document.evidence.evidence_id,
            document.evidence.content_hash or "",
            document.span_hash or "",
            text,
        )
    )
    digest = hashlib.sha256(
        f"{immutable_identity}:{index}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{document.document_id}-chunk-{index}-{digest}"


def official_guidance_documents() -> tuple[OfficialDocument, ...]:
    """Return offline-only application-authored homepage pointers, not official evidence."""
    rows = (
        ("molit-jeonse-source-pointer", "국토교통부 전세 계약 정보 공식 출처", "https://www.molit.go.kr/"),
        ("hug-guarantee-source-pointer", "주택도시보증공사 전세보증 정보 공식 출처", "https://www.khug.or.kr/"),
        ("korea-legal-aid-housing-source-pointer", "대한법률구조공단 주택임대차 정보 공식 출처", "https://www.klac.or.kr/"),
    )
    bundled_at = datetime.now(timezone.utc)
    documents: list[OfficialDocument] = []
    for document_id, title, url in rows:
        excerpt = "애플리케이션 작성 출처 포인터입니다. 기관 원문과 시점을 직접 확인해야 합니다."
        documents.append(
            OfficialDocument(
                document_id=document_id,
                title=title,
                publisher="Jeonse Support application",
                document_type=_BUNDLE_DOCUMENT_TYPE,
                retrieved_at=bundled_at,
                evidence=Evidence(
                    evidence_id=f"evidence-{document_id}",
                    kind=EvidenceKind.BUNDLED_GUIDANCE,
                    source_name="Jeonse Support application",
                    source_record_id=document_id,
                    retrieved_at=bundled_at,
                    url=url,
                    excerpt=excerpt,
                    content_hash=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
                ),
            )
        )
    return tuple(documents)


def _chunks_for_documents(
    documents: tuple[OfficialDocument, ...], manifest_identity: str
) -> tuple[VectorDocument, ...]:
    chunks: list[VectorDocument] = []
    for document in documents:
        excerpt = document.evidence.excerpt
        if excerpt is None:
            raise ImmutableEvidenceValidationError("official evidence requires an immutable span")
        chunks.append(
            VectorDocument(
                document_id=_chunk_id(document, 0, excerpt, manifest_identity),
                text=excerpt,
                evidence_id=document.evidence.evidence_id,
            )
        )
    return tuple(chunks)


def official_guidance_chunks() -> tuple[VectorDocument, ...]:
    """Offline pointers have no product semantic chunks."""
    return ()
