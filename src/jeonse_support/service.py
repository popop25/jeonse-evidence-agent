"""Analysis application service with bounded in-process reports and session summaries."""
from __future__ import annotations
import asyncio
import calendar
import re
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Iterable, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from .agents import ADVISORY, supervisor_agent
from .ai_provider import AzureOpenAIProvider, ProviderPresentationError
from .config import Settings
from .graph import build_workflow
from .models import (
    AgentResult, AnalysisStatus, ApiErrorCode, ChecklistStatus, ClaimKind, ContractModel,
    Evidence, EvidenceKind, FinalReport, FitLevel, FollowUpOperation, HugFixedPeriod,
    HugIncidentStatistic, ListingConditions, NormalizedUserConditions, RetainedListing,
    SessionMemorySummary, TemporalProvenance, TransactionRetrievalStatus, UnavailableReason,
    TransactionWindow, UserConditions,
)
from .policy import HugPolicyInput, RiskPolicyAssessment, RiskPolicyInput
from .repositories import ComparableTransactionQuery, HugIncidentQuery, OfficialDocumentRepository, TransactionRepository, HugIncidentRepository

PII_PATTERN = re.compile(r"(?:\b\d{3}[- ]?\d{3,4}[- ]?\d{4}\b|\b\d{6}[- ]?\d{7}\b|@)")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
MAX_RETAINED_RECORDS = 100
MAX_RETAINED_SESSIONS = 100
RETENTION_TTL = timedelta(hours=1)
SESSION_MEMORY_LIMIT = 5
SESSION_MEMORY_BYTES = 4_096


class AnalysisRequest(ContractModel):
    session_id: str = Field(min_length=1, max_length=128)
    listing_id: str = Field(min_length=1, max_length=128)
    conditions: UserConditions | None = None

    @model_validator(mode="after")
    def reject_personal_identifiers(self) -> "AnalysisRequest":
        if PII_PATTERN.search(self.session_id):
            raise ValueError("PII_NOT_ALLOWED")
        if not SESSION_ID_PATTERN.fullmatch(self.session_id):
            raise ValueError("INVALID_SESSION_ID")
        if not SESSION_ID_PATTERN.fullmatch(self.listing_id):
            raise ValueError("INVALID_LISTING_ID")
        if self.conditions and self.conditions.region and PII_PATTERN.search(self.conditions.region):
            raise ValueError("PII_NOT_ALLOWED")
        if self.conditions and any(
            PII_PATTERN.search(property_type)
            for property_type in self.conditions.property_types
        ):
            raise ValueError("PII_NOT_ALLOWED")
        return self



class AnalysisRecord(ContractModel):
    analysis_id: str
    session_id: str
    status: AnalysisStatus
    created_at: datetime
    agent_results: tuple[AgentResult, ...] = ()
    error_code: ApiErrorCode | None = None
    report_id: str | None = None
    ai_presentation: str | None = None
    ai_evidence_ids: tuple[str, ...] = ()
    ai_trace_codes: tuple[str, ...] = ()
    @model_validator(mode="after")
    def matches_lifecycle_contract(self) -> "AnalysisRecord":
        completed_statuses = {
            AnalysisStatus.COMPLETED,
            AnalysisStatus.COMPLETED_WITH_EXTERNAL_FALLBACK,
        }
        if self.status in completed_statuses:
            if not self.report_id:
                raise ValueError("completed analysis requires report_id")
            if self.error_code is not None:
                raise ValueError("completed analysis cannot carry error_code")
        elif self.status in {AnalysisStatus.NEEDS_REVIEW, AnalysisStatus.FAILED}:
            if self.report_id is not None:
                raise ValueError("non-completed terminal analysis cannot carry report_id")
        elif self.report_id is not None or self.error_code is not None:
            raise ValueError("active analysis cannot carry report_id or error_code")
        return self
class TerminalError(ContractModel):
    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=256)
    retryable: bool
    details: tuple[str, ...] = Field(default_factory=tuple, max_length=8)


class TerminalJobFailure(AnalysisRecord):
    status: Literal[AnalysisStatus.NEEDS_REVIEW, AnalysisStatus.FAILED]
    error: TerminalError

    @model_validator(mode="after")
    def matches_terminal_error_contract(self) -> "TerminalJobFailure":
        if self.status == AnalysisStatus.NEEDS_REVIEW:
            if self.error.code != "EVIDENCE_GATE_FAILED" or self.error.retryable:
                raise ValueError("needs_review requires non-retryable EVIDENCE_GATE_FAILED")
        elif self.error.code == "EVIDENCE_GATE_FAILED":
            raise ValueError("EVIDENCE_GATE_FAILED requires needs_review")
        elif self.error.code in {"ANALYSIS_TIMEOUT", "AI_PROVIDER_FAILED"} and not self.error.retryable:
            raise ValueError("external terminal failures must be retryable")
        return self


class TerminalJobCompleted(AnalysisRecord):
    status: Literal[
        AnalysisStatus.COMPLETED,
        AnalysisStatus.COMPLETED_WITH_EXTERNAL_FALLBACK,
    ]
    report_id: str
    error_code: None = None



@dataclass(slots=True)
class DataAdapters:
    """DI seam: snapshot adapters are read-only; no live listing crawling is supported."""
    listings: dict[str, ListingConditions] = field(default_factory=dict)
    transactions: TransactionRepository | None = None
    hug: HugIncidentRepository | None = None
    documents: OfficialDocumentRepository | None = None


class AnalysisService:
    def __init__(
        self,
        settings: Settings | None = None,
        adapters: DataAdapters | None = None,
        ai_provider: AzureOpenAIProvider | None = None,
    ) -> None:
        self.settings = settings or Settings.from_environment()
        self.adapters = adapters or DataAdapters()
        self.ai_provider = ai_provider
        self._records: OrderedDict[str, AnalysisRecord] = OrderedDict()
        self._reports: OrderedDict[str, FinalReport] = OrderedDict()
        self._sessions: OrderedDict[str, list[SessionMemorySummary]] = OrderedDict()
        self._session_accessed_at: dict[str, datetime] = {}
        self._session_versions: dict[str, int] = {}
        self._active_sessions: set[str] = set()
        self._session_lease_owners: dict[str, str] = {}
        retrieve = self.adapters.documents.retrieve if self.adapters.documents else None
        self._workflow = build_workflow(retrieve)

    def list_listings(self) -> tuple[ListingConditions, ...]:
        return tuple(self.adapters.listings.values())

    async def analyze(
        self,
        request: AnalysisRequest,
        analysis_id: str | None = None,
    ) -> AnalysisRecord:
        self._prune()
        if analysis_id and PII_PATTERN.search(analysis_id):
            raise ValueError("PII_NOT_ALLOWED")
        listing = self.adapters.listings.get(request.listing_id)
        if listing is None:
            raise KeyError(request.listing_id)
        if (
            listing.snapshot_as_of is None
            or listing.source_name is None
            or listing.source_record_id is None
            or listing.provenance_id is None
        ):
            raise ValueError("INVALID_CANONICAL_LISTING")
        acquired_lease = False
        if self._session_lease_owners.get(request.session_id) != analysis_id:
            acquired_lease = self.acquire_session_lease(request.session_id)
            if not acquired_lease:
                raise ValueError("SESSION_ANALYSIS_IN_PROGRESS")
        expected_session_version = self._session_versions.get(request.session_id, 0)
        as_of = listing.snapshot_as_of
        analysis_id = analysis_id or uuid4().hex
        created = datetime.now(UTC)
        self._records[analysis_id] = AnalysisRecord(
            analysis_id=analysis_id, session_id=request.session_id,
            status=AnalysisStatus.RUNNING, created_at=created,
        )
        try:
            samples, transaction_retrieval_status, subject, reference = await self._policy_data(listing, as_of)
            risk_input = RiskPolicyInput(
                listing=listing, transaction_samples=samples,
                transaction_retrieval_status=transaction_retrieval_status,
                hug=HugPolicyInput(subject=subject, reference_rows=reference)
                if subject and reference else None,
                as_of=as_of,
            )
            state = await self._workflow.ainvoke(
                {
                    "listing": listing,
                    "conditions": request.conditions,
                    "risk_input": risk_input,
                    "results": (),
                },
                config={"configurable": {"thread_id": analysis_id}},
            )
            results = state["results"]
            try:
                report = self._report(
                    analysis_id, request.session_id, listing, as_of, results,
                    samples, subject, reference, state["assessment"], state["official_evidence"],
                )
                gate_error = self._evidence_gate(report, results)
            except ValueError:
                gate_error = "INVALID_REPORT_EVIDENCE"
                report = None
            if gate_error is not None:
                results = (*results[:3], supervisor_agent(results[:3]))
                try:
                    report = self._repair_report_evidence(
                        analysis_id, request.session_id, listing, as_of, results, samples,
                        subject, reference, state["assessment"], state["official_evidence"],
                    )
                    gate_error = self._evidence_gate(report, results)
                except ValueError:
                    gate_error = "INVALID_REPORT_EVIDENCE"
            if gate_error is not None or report is None:
                record = AnalysisRecord(
                    analysis_id=analysis_id,
                    session_id=request.session_id,
                    status=AnalysisStatus.NEEDS_REVIEW,
                    created_at=created,
                    agent_results=results,
                    error_code=ApiErrorCode.ANALYSIS_FAILED,
                    ai_trace_codes=(gate_error or "INVALID_REPORT_EVIDENCE",),
                )
                self._records[analysis_id] = record
                self._reports.pop(analysis_id, None)
                self._prune()
                return record
            try:
                ai_presentation, ai_evidence_ids, ai_trace_codes = await self._presentation(
                    report
                )
            except ProviderPresentationError as error:
                record = AnalysisRecord(
                    analysis_id=analysis_id,
                    session_id=request.session_id,
                    status=AnalysisStatus.FAILED,
                    created_at=created,
                    agent_results=results,
                    ai_trace_codes=(error.trace_code,),
                    error_code=ApiErrorCode.AI_PROVIDER_FAILED,
                )
                self._records[analysis_id] = record
                self._reports.pop(analysis_id, None)
                self._prune()
                return record
            record = AnalysisRecord(
                analysis_id=analysis_id, session_id=request.session_id,
                status=AnalysisStatus.COMPLETED, created_at=created, agent_results=results,
                report_id=report.report_id, ai_presentation=ai_presentation,
                ai_evidence_ids=ai_evidence_ids, ai_trace_codes=ai_trace_codes,
            )
            if not self._remember(
                request.session_id,
                listing,
                report,
                state["assessment"],
                request.conditions,
                subject,
                reference,
                expected_session_version,
            ):
                record = AnalysisRecord(
                    analysis_id=analysis_id,
                    session_id=request.session_id,
                    status=AnalysisStatus.FAILED,
                    created_at=created,
                    agent_results=results,
                    error_code=ApiErrorCode.ANALYSIS_FAILED,
                    ai_trace_codes=("SESSION_MEMORY_CONFLICT",),
                )
                self._records[analysis_id] = record
                self._reports.pop(analysis_id, None)
                self._prune()
                return record
            self._records[analysis_id] = record
            self._reports[analysis_id] = report
            self._prune()
            return record
        except asyncio.CancelledError:
            self._reports.pop(analysis_id, None)
            self._records[analysis_id] = AnalysisRecord(
                analysis_id=analysis_id, session_id=request.session_id,
                status=AnalysisStatus.FAILED, created_at=created,
                error_code=ApiErrorCode.ANALYSIS_FAILED,
            )
            self._prune()
            raise
        except Exception:
            self._reports.pop(analysis_id, None)
            self._records[analysis_id] = AnalysisRecord(
                analysis_id=analysis_id, session_id=request.session_id,
                status=AnalysisStatus.FAILED, created_at=created,
            )
            self._prune()
            raise
        finally:
            if acquired_lease:
                self.release_session_lease(request.session_id)

    async def _policy_data(
        self, listing: ListingConditions, as_of: date,
    ) -> tuple[
        tuple[SampleListing, ...],
        TransactionRetrievalStatus,
        HugIncidentStatistic | None,
        tuple[HugIncidentStatistic, ...],
    ]:
        samples: tuple[SampleListing, ...] = ()
        transaction_retrieval_status = TransactionRetrievalStatus.MISSING
        subject: HugIncidentStatistic | None = None
        reference: tuple[HugIncidentStatistic, ...] = ()
        if self.adapters.transactions:
            samples = await self.adapters.transactions.list_comparables(
                ComparableTransactionQuery(listing=listing, as_of=as_of)
            )
            transaction_retrieval_status = TransactionRetrievalStatus.SUCCESS
        if self.adapters.hug:
            period_end = min(as_of, date(as_of.year, 6, 30))
            query = HugIncidentQuery(
                as_of=as_of, period_start=date(as_of.year, 1, 1),
                period_end=period_end,
            )
            subject = await self.adapters.hug.get_subject_statistic(query)
            reference = await self.adapters.hug.list_reference_statistics(query)
        return samples, transaction_retrieval_status, subject, reference

    async def _presentation(self, report: FinalReport) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
        evidence = {
            item.evidence_id: item.excerpt
            for item in report.evidence
            if item.kind is EvidenceKind.OFFICIAL_DOCUMENT and item.excerpt
        }
        if self.ai_provider is None or not evidence:
            return None, (), ()

        def search(query: str) -> list[dict[str, str]]:
            tokens = {token.casefold() for token in query.split() if len(token) > 1}
            return [
                {"evidence_id": evidence_id}
                for evidence_id, excerpt in evidence.items()
                if not tokens or any(token in excerpt.casefold() for token in tokens)
            ]

        response = await asyncio.to_thread(
            self.ai_provider.present,
            "공식 근거에 연결된 문의 준비 요약을 작성하세요.",
            evidence=evidence,
            official_document_search=search,
        )
        return response.presentation.summary, response.presentation.evidence_ids, response.trace_codes

    def get_analysis(self, analysis_id: str) -> AnalysisRecord | None:
        self._prune()
        return self._records.get(analysis_id)

    def get_report(self, analysis_id: str) -> FinalReport | None:
        self._prune()
        return self._reports.get(analysis_id)

    def follow_up(
        self,
        session_id: str,
        operation: FollowUpOperation | str,
        target_report_id: str | None = None,
    ) -> dict[str, object]:
        self._prune()
        if not SESSION_ID_PATTERN.fullmatch(session_id) or PII_PATTERN.search(session_id):
            raise ValueError("INVALID_SESSION_ID")
        try:
            resolved_operation = FollowUpOperation(operation)
        except ValueError as exc:
            raise ValueError("INVALID_FOLLOW_UP_OPERATION") from exc
        summaries = self._sessions.get(session_id)
        if not summaries:
            raise KeyError(session_id)
        self._sessions.move_to_end(session_id)
        self._session_accessed_at[session_id] = datetime.now(UTC)
        by_report_id = {summary.report_id: summary for summary in summaries}
        latest = summaries[-1]
        if target_report_id is None:
            if resolved_operation is FollowUpOperation.COMPARE:
                if len(summaries) < 2:
                    raise ValueError("NO_PRIOR_REPORT")
                target = summaries[-2]
            else:
                target = latest
        else:
            target = by_report_id.get(target_report_id)
            if target is None:
                raise KeyError(target_report_id)
        if (
            resolved_operation is FollowUpOperation.COMPARE
            and target.report_id == latest.report_id
        ):
            raise ValueError("COMPARE_REQUIRES_DISTINCT_REPORTS")
        target_payload = self._temporal_payload(target)
        response: dict[str, object] = {
            "session_id": session_id,
            "status": "advisory_only",
            "operation": resolved_operation.value,
            "target": target_payload,
            "advisory_notice": ADVISORY,
        }
        if resolved_operation is FollowUpOperation.COMPARE:
            response["current"] = self._temporal_payload(latest)
            response["message"] = "Compare the selected retained report with the current retained report using their distinct dates and provenance versions."
        elif resolved_operation is FollowUpOperation.RECHECK:
            response["message"] = "Recheck the listed source records at their retained dates and provenance versions; this service does not perform live verification."
        else:
            response["message"] = "Clarify this retained report through its listed evidence, dates, and provenance versions."
        return response
    def acquire_session_lease(self, session_id: str, analysis_id: str | None = None) -> bool:
        """Reserve a session from API admission through its terminal outcome."""
        if session_id in self._active_sessions:
            return False
        entries = self._sessions.get(session_id)
        cutoff = datetime.now(UTC) - RETENTION_TTL
        if not entries or (
            self._session_accessed_at.get(session_id, entries[-1].completed_at) < cutoff
        ):
            self.clear_session_memory(session_id)
        self._active_sessions.add(session_id)
        if analysis_id is not None:
            self._session_lease_owners[session_id] = analysis_id
        return True

    def release_session_lease(self, session_id: str, analysis_id: str | None = None) -> None:
        """Touch existing retained memory before allowing its idle TTL to resume."""
        owner = self._session_lease_owners.get(session_id)
        if owner is not None and owner != analysis_id:
            return
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            self._session_accessed_at[session_id] = datetime.now(UTC)
        self._session_lease_owners.pop(session_id, None)
        self._active_sessions.discard(session_id)
        if session_id not in self._sessions:
            self._session_versions.pop(session_id, None)

    def clear_session_memory(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._session_accessed_at.pop(session_id, None)
        if session_id in self._active_sessions:
            self._session_versions[session_id] = self._session_versions.get(session_id, 0) + 1
        else:
            self._session_versions.pop(session_id, None)
            self._session_lease_owners.pop(session_id, None)

    def is_session_active(self, session_id: str) -> bool:
        return session_id in self._active_sessions


    @staticmethod
    def _temporal_payload(summary: SessionMemorySummary) -> dict[str, object]:
        return {
            "report_id": summary.report_id,
            "generated_at": summary.completed_at,
            "completed_at": summary.completed_at,
            "as_of": summary.report_as_of,
            "overall_risk": summary.risk_level,
            "fit_level": summary.fit_level,
            "listing": summary.listing.model_dump(),
            "conditions": summary.conditions.model_dump(),
            "claim_ids": summary.claim_ids,
            "checklist_ids": summary.checklist_ids,
            "temporal_provenance": tuple(
                item.model_dump(exclude_none=True)
                for item in summary.temporal_provenance
            ),
            "unavailable_ids": summary.unavailable_ids,
            "next_action_ids": summary.next_action_ids,
        }

    @staticmethod
    def _window_start(as_of: date, months: int) -> date:
        month_index = as_of.year * 12 + as_of.month - 1 - months
        year, month_zero = divmod(month_index, 12)
        month = month_zero + 1
        return date(year, month, min(as_of.day, calendar.monthrange(year, month)[1]))

    @staticmethod
    def _fit_level(
        listing: ListingConditions, conditions: UserConditions | None,
    ) -> FitLevel:
        if conditions is None:
            return FitLevel.UNAVAILABLE
        active = any((
            conditions.region is not None,
            conditions.max_deposit_won is not None,
            bool(conditions.property_types),
            conditions.min_area_sqm is not None,
            conditions.max_area_sqm is not None,
        ))
        if not active:
            return FitLevel.UNAVAILABLE
        mismatches = (
            conditions.region is not None and conditions.region != listing.legal_dong,
            conditions.max_deposit_won is not None
            and listing.deposit_won > conditions.max_deposit_won,
            bool(conditions.property_types)
            and listing.property_type not in conditions.property_types,
            conditions.min_area_sqm is not None
            and listing.area_sqm < conditions.min_area_sqm,
            conditions.max_area_sqm is not None
            and listing.area_sqm > conditions.max_area_sqm,
        )
        return FitLevel.MISMATCH if any(mismatches) else FitLevel.MATCH

    def _remember(
        self,
        session_id: str,
        listing: ListingConditions,
        report: FinalReport,
        assessment: RiskPolicyAssessment,
        conditions: UserConditions | None,
        subject: HugIncidentStatistic | None,
        reference: Iterable[HugIncidentStatistic],
        expected_session_version: int,
    ) -> bool:
        evidence_by_id = {item.evidence_id: item for item in report.evidence}
        hug_periods = {
            item.evidence_id: HugFixedPeriod(
                start=item.period_start, end=item.period_end,
            )
            for item in ((subject,) if subject else ()) + tuple(reference)
        }
        transaction_window = TransactionWindow(
            start=self._window_start(
                listing.snapshot_as_of or report.generated_at.date(), 24,
            ),
            end=listing.snapshot_as_of or report.generated_at.date(),
        )

        def projection(
            evidence_id: str,
            *,
            signal_id: str | None = None,
            checklist_id: str | None = None,
            unavailable_reason: UnavailableReason | None = None,
            unavailable_id: str | None = None,
            next_action_id: str | None = None,
        ) -> TemporalProvenance:
            evidence = evidence_by_id[evidence_id]
            evidence_as_of = (
                evidence.snapshot_as_of
                or (evidence.observed_at.date() if evidence.observed_at else None)
                or evidence.retrieved_at.date()
            )
            snapshot_version = (
                evidence.provenance_id if evidence.snapshot_as_of is not None else None
            )
            corpus_version = (
                evidence.provenance_id
                if evidence.kind is EvidenceKind.OFFICIAL_DOCUMENT
                else None
            )
            return TemporalProvenance(
                signal_id=signal_id,
                checklist_id=checklist_id,
                source_id=evidence.source_record_id,
                evidence_as_of=evidence_as_of,
                transaction_window=(
                    transaction_window
                    if evidence.kind is EvidenceKind.TRANSACTION_RECORD
                    else None
                ),
                hug_fixed_period=hug_periods.get(evidence_id),
                snapshot_version=snapshot_version,
                snapshot_as_of=evidence.snapshot_as_of,
                corpus_version=corpus_version,
                corpus_as_of=(
                    evidence.snapshot_as_of or evidence_as_of
                    if corpus_version is not None
                    else None
                ),
                unavailable_reason=unavailable_reason,
                unavailable_id=unavailable_id,
                next_action_id=next_action_id,
            )

        def preferred_signal_evidence_id(signal_id: str, evidence_ids: tuple[str, ...]) -> str:
            candidates = tuple(
                evidence_id
                for evidence_id in evidence_ids
                if evidence_id in evidence_by_id
            )
            if signal_id == assessment.transaction_signal.signal_id:
                comparable = tuple(
                    evidence_id
                    for evidence_id in candidates
                    if (
                        evidence_by_id[evidence_id].kind
                        is EvidenceKind.TRANSACTION_RECORD
                        and evidence_id != listing.evidence_id
                    )
                )
                transactions = tuple(
                    evidence_id
                    for evidence_id in candidates
                    if evidence_by_id[evidence_id].kind
                    is EvidenceKind.TRANSACTION_RECORD
                )
                return (comparable or transactions or candidates)[0]
            if signal_id == assessment.hug_signal.signal_id:
                subject_evidence_id = subject.evidence_id if subject else None
                hug = tuple(
                    evidence_id for evidence_id in candidates if evidence_id in hug_periods
                )
                return (
                    (subject_evidence_id,)
                    if subject_evidence_id in hug
                    else hug or candidates
                )[0]
            return candidates[0]

        def preferred_checklist_evidence_id(
            checklist_id: str, evidence_ids: tuple[str, ...]
        ) -> str:
            if checklist_id == "transaction-comparables":
                return preferred_signal_evidence_id(
                    assessment.transaction_signal.signal_id, evidence_ids
                )
            if checklist_id == "hug-percentiles":
                return preferred_signal_evidence_id(
                    assessment.hug_signal.signal_id, evidence_ids
                )
            candidates = tuple(
                evidence_id
                for evidence_id in evidence_ids
                if evidence_id in evidence_by_id
            )
            authoritative = tuple(
                evidence_id
                for evidence_id in candidates
                if evidence_by_id[evidence_id].kind is not EvidenceKind.AGENT_OBSERVATION
            )
            return (authoritative or candidates)[0]

        unavailable_ids: list[str] = []
        next_action_ids: list[str] = []
        temporal_provenance: list[TemporalProvenance] = []
        projection_keys: set[str] = set()

        def append_projection(value: TemporalProvenance) -> None:
            key = value.model_dump_json(exclude_none=True)
            if key not in projection_keys:
                projection_keys.add(key)
                temporal_provenance.append(value)

        def append_or_merge_checklist(value: TemporalProvenance) -> None:
            for index, current in enumerate(temporal_provenance):
                if (
                    current.signal_id is not None
                    and current.checklist_id is None
                    and current.source_id == value.source_id
                    and current.evidence_as_of == value.evidence_as_of
                    and current.transaction_window == value.transaction_window
                    and current.hug_fixed_period == value.hug_fixed_period
                    and current.snapshot_version == value.snapshot_version
                    and current.corpus_version == value.corpus_version
                ):
                    temporal_provenance[index] = current.model_copy(
                        update={"checklist_id": value.checklist_id}
                    )
                    return
            append_projection(value)

        for signal in (assessment.transaction_signal, assessment.hug_signal):
            unavailable_id = None
            next_action_id = None
            if signal.unavailable_reason is not None:
                unavailable_id = (
                    f"{signal.signal_id}-unavailable-{signal.unavailable_reason.value}"
                )
                next_action_id = f"recheck-{signal.signal_id}"
                unavailable_ids.append(unavailable_id)
                next_action_ids.append(next_action_id)
            append_projection(projection(
                preferred_signal_evidence_id(signal.signal_id, signal.evidence_ids),
                signal_id=signal.signal_id,
                unavailable_reason=signal.unavailable_reason,
                unavailable_id=unavailable_id,
                next_action_id=next_action_id,
            ))

        claim_ids = tuple(claim.claim_id for claim in report.claims[:12])
        remaining = 12 - len(claim_ids)
        checklist_ids = tuple(
            item.checklist_id for item in report.checklist_items[:remaining]
        )
        retained_checklists = {
            item.checklist_id
            for item in report.checklist_items
            if item.checklist_id in checklist_ids
        }
        for checklist in report.checklist_items:
            if checklist.checklist_id not in retained_checklists:
                continue
            unavailable_id = None
            next_action_id = None
            if checklist.status is ChecklistStatus.UNAVAILABLE:
                unavailable_id = f"{checklist.checklist_id}-unavailable"
                next_action_id = f"recheck-{checklist.checklist_id}"
                unavailable_ids.append(unavailable_id)
                next_action_ids.append(next_action_id)
            append_or_merge_checklist(projection(
                preferred_checklist_evidence_id(
                    checklist.checklist_id, checklist.evidence_ids
                ),
                checklist_id=checklist.checklist_id,
                unavailable_id=unavailable_id,
                next_action_id=next_action_id,
            ))

        summary = SessionMemorySummary(
            session_id=session_id,
            report_id=report.report_id,
            completed_at=report.generated_at,
            report_as_of=listing.snapshot_as_of or report.generated_at.date(),
            listing=RetainedListing(
                listing_id=listing.listing_id,
                legal_dong=listing.legal_dong,
                deposit_won=listing.deposit_won,
                area_sqm=listing.area_sqm,
                contract_date=listing.contract_date,
                property_type=listing.property_type,
                floor=listing.floor,
                built_year=listing.built_year,
            ),
            conditions=NormalizedUserConditions(
                max_deposit_won=conditions.max_deposit_won if conditions else None,
                property_types=conditions.property_types if conditions else (),
                min_area_sqm=conditions.min_area_sqm if conditions else None,
                max_area_sqm=conditions.max_area_sqm if conditions else None,
            ),
            fit_level=self._fit_level(listing, conditions),
            risk_level=report.overall_risk,
            claim_ids=claim_ids,
            checklist_ids=checklist_ids,
            temporal_provenance=tuple(temporal_provenance),
            unavailable_ids=tuple(unavailable_ids),
            next_action_ids=tuple(next_action_ids),
        )
        return self._store_session_summary(
            session_id, summary, expected_session_version
        )

    def _store_session_summary(
        self,
        session_id: str,
        summary: SessionMemorySummary,
        expected_session_version: int,
    ) -> bool:
        if self._session_versions.get(session_id, 0) != expected_session_version:
            return False
        if self._session_container_bytes((summary,)) > SESSION_MEMORY_BYTES:
            return False
        entries = [*self._sessions.get(session_id, ()), summary]
        while len(entries) > SESSION_MEMORY_LIMIT or self._session_container_bytes(entries) > SESSION_MEMORY_BYTES:
            entries.pop(0)
        self._sessions[session_id] = entries
        self._sessions.move_to_end(session_id)
        self._session_accessed_at[session_id] = datetime.now(UTC)
        self._session_versions[session_id] = expected_session_version + 1
        return True


    @staticmethod
    def _session_container_bytes(entries: Iterable[SessionMemorySummary]) -> int:
        container = [
            item.model_dump(mode="json", exclude_none=True)
            for item in entries
        ]
        return len(json.dumps(
            container, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8"))

    def _prune(self) -> None:
        cutoff = datetime.now(UTC) - RETENTION_TTL
        for analysis_id, record in tuple(self._records.items()):
            if (
                record.created_at < cutoff
                and record.status not in {AnalysisStatus.PENDING, AnalysisStatus.RUNNING}
            ):
                self._records.pop(analysis_id, None)
                self._reports.pop(analysis_id, None)
        for session_id, entries in tuple(self._sessions.items()):
            if (
                not entries
                or (
                    session_id not in self._active_sessions
                    and self._session_accessed_at.get(session_id, entries[-1].completed_at) < cutoff
                )
            ):
                self.clear_session_memory(session_id)
        while len(self._records) > MAX_RETAINED_RECORDS:
            analysis_id = next(
                (
                    record_id
                    for record_id, record in self._records.items()
                    if record.status not in {AnalysisStatus.PENDING, AnalysisStatus.RUNNING}
                ),
                None,
            )
            if analysis_id is None:
                break
            self._records.pop(analysis_id, None)
            self._reports.pop(analysis_id, None)
        while len(self._reports) > MAX_RETAINED_RECORDS:
            self._reports.popitem(last=False)
        while len(self._sessions) > MAX_RETAINED_SESSIONS:
            session_id = next(
                (
                    candidate
                    for candidate in self._sessions
                    if candidate not in self._active_sessions
                ),
                None,
            )
            if session_id is None:
                break
            self.clear_session_memory(session_id)

    @staticmethod
    def _evidence(
        identifier: str,
        kind: EvidenceKind,
        source: str,
        observed: date,
        *,
        source_record_id: str | None = None,
        snapshot_as_of: date | None = None,
        provenance_id: str | None = None,
    ) -> Evidence:
        now = datetime.now(UTC)
        return Evidence(
            evidence_id=identifier,
            kind=kind,
            source_name=source,
            source_record_id=source_record_id or identifier,
            retrieved_at=now,
            observed_at=datetime(observed.year, observed.month, observed.day, tzinfo=UTC),
            snapshot_as_of=snapshot_as_of,
            provenance_id=provenance_id,
        )

    @staticmethod
    def _evidence_gate(
        report: FinalReport, results: tuple[AgentResult, ...],
    ) -> str | None:
        """Reject publication unless every role's output has resolved, scoped evidence."""
        expected_roles = ("fit", "risk", "contract-prep", "supervisor")
        if tuple(result.agent_name for result in results) != expected_roles:
            return "ROLE_ORDER_OR_COVERAGE"
        evidence_by_id = {item.evidence_id: item for item in report.evidence}
        if len(evidence_by_id) != len(report.evidence):
            return "DUPLICATE_EVIDENCE"
        if any(item.kind is EvidenceKind.BUNDLED_GUIDANCE for item in report.evidence):
            return "NON_OFFICIAL_GUIDANCE_EVIDENCE"
        for result in results:
            if result.status is not AnalysisStatus.COMPLETED or not result.evidence_ids:
                return "INCOMPLETE_ROLE_RESULT"
            if not set(result.evidence_ids).issubset(evidence_by_id):
                return "UNRESOLVED_ROLE_EVIDENCE"
        for item in report.evidence:
            if item.kind in {EvidenceKind.TRANSACTION_RECORD, EvidenceKind.HUG_STATISTIC}:
                if item.snapshot_as_of is None or item.provenance_id is None:
                    return "UNSCOPED_SNAPSHOT_EVIDENCE"
            if item.kind is EvidenceKind.OFFICIAL_DOCUMENT and (
                not item.excerpt or not item.content_hash
            ):
                return "UNVERIFIABLE_OFFICIAL_EVIDENCE"
        contract_prep = results[2]
        if not contract_prep.checklist_items:
            return "MISSING_CONTRACT_PREP_CHECKLIST"
        available_items = tuple(
            item
            for item in contract_prep.checklist_items
            if item.status is not ChecklistStatus.UNAVAILABLE
        )
        for item in available_items:
            if not item.evidence_ids:
                return "MISSING_CONTRACT_PREP_ITEM_EVIDENCE"
            if not set(item.evidence_ids).issubset(evidence_by_id):
                return "UNRESOLVED_CONTRACT_PREP_ITEM_EVIDENCE"
            if any(
                evidence_by_id[evidence_id].kind is not EvidenceKind.OFFICIAL_DOCUMENT
                for evidence_id in item.evidence_ids
            ):
                return "NON_OFFICIAL_CONTRACT_PREP_ITEM_EVIDENCE"
        if available_items and (
            not contract_prep.evidence_ids
            or any(
                evidence_by_id[evidence_id].kind is not EvidenceKind.OFFICIAL_DOCUMENT
                for evidence_id in contract_prep.evidence_ids
            )
        ):
            return "NON_OFFICIAL_CONTRACT_PREP_EVIDENCE"
        for claim in report.claims:
            if not set(claim.evidence_ids).issubset(evidence_by_id):
                return "UNRESOLVED_CLAIM_EVIDENCE"
            if claim.kind is ClaimKind.FACT and any(
                evidence_by_id[evidence_id].kind is EvidenceKind.AGENT_OBSERVATION
                for evidence_id in claim.evidence_ids
            ):
                return "WEAK_FACT_EVIDENCE"
        if not report.conclusions or not report.policy_snapshot_evidence:
            return "MISSING_TOP_LEVEL_POLICY_COVERAGE"
        return None

    def _repair_report_evidence(
        self,
        analysis_id: str,
        session_id: str,
        listing: ListingConditions,
        as_of: date,
        results: tuple[AgentResult, ...],
        samples: Iterable[SampleListing],
        subject: HugIncidentStatistic | None,
        reference: Iterable[HugIncidentStatistic],
        assessment: RiskPolicyAssessment,
        official_evidence: tuple[Evidence, ...],
    ) -> FinalReport:
        """Perform the single bounded deterministic repair by rematerializing report evidence."""
        return self._report(
            analysis_id,
            session_id,
            listing,
            as_of,
            results,
            samples,
            subject,
            reference,
            assessment,
            official_evidence,
        )
    def _report(self, analysis_id: str, session_id: str, listing: ListingConditions, as_of: date,
                results: tuple[AgentResult, ...], samples: Iterable[SampleListing], subject: HugIncidentStatistic | None,
                reference: Iterable[HugIncidentStatistic], assessment: RiskPolicyAssessment,
                official_evidence: tuple[Evidence, ...]) -> FinalReport:
        evidence = {
            listing.evidence_id: self._evidence(
                listing.evidence_id,
                EvidenceKind.TRANSACTION_RECORD,
                listing.source_name or "snapshot listing",
                as_of,
                source_record_id=listing.source_record_id,
                snapshot_as_of=listing.snapshot_as_of,
                provenance_id=listing.provenance_id,
            )
        }
        policy_evidence_ids = {
            reference_item.evidence_id
            for reference_item in assessment.snapshot_evidence
        }
        for sample in samples:
            if sample.evidence_id not in policy_evidence_ids:
                continue
            evidence[sample.evidence_id] = self._evidence(
                sample.evidence_id,
                EvidenceKind.TRANSACTION_RECORD,
                sample.source_name or "snapshot transaction",
                sample.occurred_on,
                source_record_id=sample.source_record_id,
                snapshot_as_of=sample.snapshot_as_of,
                provenance_id=sample.provenance_id,
            )
        for statistic in ([subject] if subject else []) + list(reference):
            if statistic.evidence_id not in policy_evidence_ids:
                continue
            evidence[statistic.evidence_id] = self._evidence(
                statistic.evidence_id,
                EvidenceKind.HUG_STATISTIC,
                statistic.source_name or "snapshot HUG statistic",
                statistic.period_end,
                source_record_id=statistic.source_record_id,
                snapshot_as_of=statistic.snapshot_as_of,
                provenance_id=statistic.provenance_id,
            )
        evidence.update({item.evidence_id: item for item in official_evidence})
        claims = tuple(claim for result in results for claim in result.claims)
        checks = tuple(item for result in results for item in result.checklist_items)
        conclusion = assessment.conclusion
        return FinalReport(report_id=f"report-{analysis_id}", session_id=session_id, generated_at=datetime.now(UTC),
            status=AnalysisStatus.COMPLETED, overall_risk=assessment.overall_risk,
            conclusions=(conclusion,), claims=claims, checklist_items=checks, evidence=tuple(evidence.values()),
            policy_snapshot_evidence=assessment.snapshot_evidence,
            limitations=(
                "Snapshot data is non-live and may be incomplete or changed after its as-of date.",
                "Bundled listing, transaction, and HUG fixtures are synthetic demo data, not real listings or official records.",
                "No live rights, building-register, or landlord verification is performed.",
                "No official contract checklist evidence is available."
                if not any(item.kind is EvidenceKind.OFFICIAL_DOCUMENT for item in official_evidence)
                else "Official contract checklist evidence is limited to the cited retrieved documents.",
            ), advisory_notice=ADVISORY)
