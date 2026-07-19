"""Small, typed adapter boundaries for the advisory-only MVP.

These interfaces intentionally exclude live listing crawling, rights or landlord
verification, safety guarantees, legal advice, accounts, payments, and production ops.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import Field

from .models import (
    ContractModel,
    Evidence,
    HugIncidentStatistic,
    Identifier,
    ListingConditions,
    NonEmptyText,
    SampleListing,
    SessionMemorySummary,
)


class ComparableTransactionQuery(ContractModel):
    listing: ListingConditions
    as_of: date


class HugIncidentQuery(ContractModel):
    as_of: date
    period_start: date
    period_end: date


class OfficialDocumentQuery(ContractModel):
    query_text: NonEmptyText
    document_types: tuple[NonEmptyText, ...] = ()
    max_results: int = Field(default=5, ge=1, le=20)


class OfficialDocument(ContractModel):
    document_id: Identifier
    title: NonEmptyText
    publisher: NonEmptyText
    document_type: NonEmptyText
    published_on: date | None = None
    retrieved_at: datetime
    evidence: Evidence
    source_id: Identifier | None = None
    as_of: datetime | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=1)
    span_hash: str | None = Field(default=None, pattern=r"^[A-Fa-f0-9]{64}$")


class VectorDocument(ContractModel):
    document_id: Identifier
    text: NonEmptyText
    evidence_id: Identifier


class VectorSearchMatch(ContractModel):
    document: VectorDocument
    cosine_relevance: float = Field(ge=-1, le=1)

class VectorSearchQuery(ContractModel):
    query_text: NonEmptyText
    limit: int = Field(default=5, ge=1, le=20)


class StructuredInvocation(ContractModel):
    invocation_id: Identifier
    system_instruction: NonEmptyText
    user_input: NonEmptyText


@runtime_checkable
class ListingCatalog(Protocol):
    """Read-only catalog boundary; it does not crawl live listings."""

    async def get_listing(self, listing_id: str) -> ListingConditions | None: ...


@runtime_checkable
class TransactionRepository(Protocol):
    async def list_comparables(self, query: ComparableTransactionQuery) -> tuple[SampleListing, ...]: ...


@runtime_checkable
class HugIncidentRepository(Protocol):
    async def get_subject_statistic(self, query: HugIncidentQuery) -> HugIncidentStatistic | None: ...

    async def list_reference_statistics(self, query: HugIncidentQuery) -> tuple[HugIncidentStatistic, ...]: ...


@runtime_checkable
class OfficialDocumentRepository(Protocol):
    """Read-only official-document retrieval, not rights or landlord verification."""

    async def retrieve(self, query: OfficialDocumentQuery) -> tuple[OfficialDocument, ...]: ...


@runtime_checkable
class SessionMemoryRepository(Protocol):
    async def load(self, session_id: str) -> SessionMemorySummary | None: ...

    async def save(self, summary: SessionMemorySummary) -> None: ...


@runtime_checkable
class VectorStore(Protocol):
    async def upsert(self, documents: tuple[VectorDocument, ...]) -> None: ...

    async def search(self, query: VectorSearchQuery) -> tuple[VectorSearchMatch, ...]: ...


StructuredOutput = TypeVar("StructuredOutput", bound=ContractModel)


@runtime_checkable
class StructuredLlm(Protocol):
    """Structured invocation boundary; callers choose an explicit Pydantic response model."""

    async def invoke(
        self, invocation: StructuredInvocation, response_model: type[StructuredOutput]
    ) -> StructuredOutput: ...
