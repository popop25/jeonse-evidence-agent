"""Strict, advisory-only domain contracts for jeonse decision support.

These models deliberately do not represent legal conclusions, safety guarantees, fraud
findings, rights verification, landlord verification, accounts, or payments.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class ContractModel(BaseModel):
    """Base for public contracts: unknown fields and implicit coercion are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


Identifier = Annotated[str, Field(min_length=1, max_length=128)]
NonEmptyText = Annotated[str, Field(min_length=1, max_length=4_000)]
WonAmount = Annotated[Decimal, Field(ge=0, max_digits=18, decimal_places=0)]
SquareMeters = Annotated[Decimal, Field(gt=0, le=10_000, max_digits=10, decimal_places=2)]


class RiskLevel(StrEnum):
    """Advisory risk categories; they are not safety or fraud determinations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNAVAILABLE = "unavailable"


class AnalysisStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_EXTERNAL_FALLBACK = "completed_with_external_fallback"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"

class PropertyTypeV1(StrEnum):
    """Versioned allowlist for property categories accepted as user constraints."""

    APARTMENT = "아파트"
    MULTI_FAMILY = "연립·다세대"
    OFFICETEL = "오피스텔"
    APARTMENT_EN = "apartment"

class TransactionRetrievalStatus(StrEnum):
    SUCCESS = "success"
    MISSING = "missing"

class TransactionReconciliationState(StrEnum):
    """Outcome of reconciling declared transaction snapshot rows."""

    EXACT_DUPLICATE = "exact_duplicate"
    SOURCE_CONFLICT = "source_conflict"
    CARDINALITY_BREACH = "cardinality_breach"
    CORRECTION = "correction"
    NOT_COMPARABLE = "not_comparable"

class EvidenceKind(StrEnum):
    USER_INPUT = "user_input"
    TRANSACTION_RECORD = "transaction_record"
    HUG_STATISTIC = "hug_statistic"
    OFFICIAL_DOCUMENT = "official_document"
    AGENT_OBSERVATION = "agent_observation"
    BUNDLED_GUIDANCE = "bundled_guidance"

class OfficialCorpusAvailability(StrEnum):
    """Whether a validated lawful official-document corpus is available for retrieval."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ClaimKind(StrEnum):
    FACT = "fact"
    INFERENCE = "inference"
    LIMITATION = "limitation"


class ChecklistStatus(StrEnum):
    PASS = "pass"
    REVIEW = "review"
    UNAVAILABLE = "unavailable"


class AgentEventKind(StrEnum):
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ApiErrorCode(StrEnum):
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    ANALYSIS_FAILED = "analysis_failed"
    INTERNAL_ERROR = "internal_error"
    AI_PROVIDER_FAILED = "ai_provider_failed"
class FollowUpOperation(StrEnum):
    """Bounded, evidence-linked operations supported for a retained report."""

    CLARIFY = "clarify"
    COMPARE = "compare"
    RECHECK = "recheck"


class UserConditions(ContractModel):
    """Canonical user-submitted constraints evaluated against the selected snapshot."""

    region: NonEmptyText | None = Field(default=None, max_length=100)
    max_deposit_won: WonAmount | None = None
    property_types: tuple[PropertyTypeV1, ...] = Field(default=(), max_length=20)
    min_area_sqm: SquareMeters | None = None
    max_area_sqm: SquareMeters | None = None

    @field_validator("property_types", mode="before")
    @classmethod
    def accept_json_property_type_array(cls, value: object) -> object:
        if isinstance(value, list):
            value = tuple(value)
        if isinstance(value, tuple):
            return tuple(
                (
                    PropertyTypeV1(item)
                    if isinstance(item, str)
                    and item in PropertyTypeV1._value2member_map_
                    else item
                )
                for item in value
            )
        return value

    @field_validator("max_deposit_won", "min_area_sqm", "max_area_sqm", mode="before")
    @classmethod
    def accept_json_decimal_number(cls, value: object) -> object:
        if value is None or isinstance(value, Decimal):
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        return Decimal(str(value))

    @model_validator(mode="after")
    def canonical_bounds_and_types(self) -> UserConditions:
        if (
            self.min_area_sqm is not None
            and self.max_area_sqm is not None
            and self.min_area_sqm > self.max_area_sqm
        ):
            raise ValueError("min_area_sqm cannot exceed max_area_sqm")
        if len(set(self.property_types)) != len(self.property_types):
            raise ValueError("property_types must not contain duplicates")
        return self


class ListingConditions(ContractModel):
    """User-supplied listing facts; unverified inputs are not official verification."""

    listing_id: Identifier
    address_text: NonEmptyText
    legal_dong: NonEmptyText
    deposit_won: WonAmount
    area_sqm: SquareMeters
    contract_date: date
    property_type: NonEmptyText
    snapshot_as_of: date | None = None
    source_name: NonEmptyText | None = None
    source_record_id: Identifier | None = None
    provenance_id: Identifier | None = None
    floor: int | None = Field(default=None, ge=-10, le=300)
    built_year: int | None = Field(default=None, ge=1800, le=2200)
    evidence_id: Identifier


class SampleListing(ContractModel):
    """Comparable transaction listing used only for the deterministic policy."""

    transaction_id: Identifier
    occurred_on: date
    area_sqm: SquareMeters
    deposit_won: WonAmount
    address_text: NonEmptyText
    source_record_id: Identifier | None = None
    evidence_id: Identifier
    legal_dong: NonEmptyText
    property_type: NonEmptyText
    monthly_rent_won: WonAmount
    cancelled: bool
    renewal: bool
    floor: int | None = Field(default=None, ge=-10, le=300)
    source_name: NonEmptyText
    snapshot_as_of: date
    provenance_id: Identifier
    dataset_id: Identifier | None = None
    dataset_version: Identifier | None = None
    stable_row_id: Identifier | None = None

    @model_validator(mode="after")
    def qualified_identity_is_complete(self) -> SampleListing:
        identity = (self.dataset_id, self.dataset_version, self.stable_row_id)
        if any(value is not None for value in identity) and any(value is None for value in identity):
            raise ValueError("transaction dataset-qualified identity must be complete")
        return self



class Evidence(ContractModel):
    """Provenance-bearing evidence, never an assertion of legal validity."""

    evidence_id: Identifier
    kind: EvidenceKind
    source_name: NonEmptyText
    source_record_id: Identifier
    retrieved_at: datetime
    observed_at: datetime | None = None
    url: HttpUrl | None = None
    excerpt: str | None = Field(default=None, min_length=1, max_length=4_000)
    content_hash: str | None = Field(default=None, pattern=r"^[A-Fa-f0-9]{64}$")
    snapshot_as_of: date | None = None
    provenance_id: Identifier | None = None

    @model_validator(mode="after")
    def snapshot_evidence_has_provenance(self) -> Evidence:
        if self.kind in {EvidenceKind.TRANSACTION_RECORD, EvidenceKind.HUG_STATISTIC}:
            if self.snapshot_as_of is None or self.provenance_id is None:
                raise ValueError("snapshot-backed evidence requires snapshot_as_of and provenance_id")
        elif (self.snapshot_as_of is None) != (self.provenance_id is None):
            raise ValueError("snapshot evidence date and provenance must be supplied together")
        return self


class TypedClaim(ContractModel):
    claim_id: Identifier
    kind: ClaimKind
    statement: NonEmptyText
    rule_ids: tuple[Identifier, ...] = Field(min_length=1)
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)


class ChecklistItem(ContractModel):
    checklist_id: Identifier
    status: ChecklistStatus
    label: NonEmptyText
    rationale: NonEmptyText
    rule_ids: tuple[Identifier, ...] = Field(min_length=1)
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)


class UnavailableReason(StrEnum):
    MISSING = "missing"
    STALE = "stale"
    INSUFFICIENT = "insufficient"
    NO_MATCH = "no_match"
    UNSUPPORTED = "unsupported"
    SOURCE_CONFLICT = "source_conflict"
    POPULATION_MISMATCH = "population_mismatch"
    TIED_THRESHOLDS = "tied_thresholds"
    NOT_COMPARABLE = "not_comparable"
    INVALID_ARTIFACT = "invalid_artifact"
class GradeSignal(ContractModel):
    """A grade-bearing policy signal with explicit freshness and sufficiency."""

    signal_id: Identifier
    level: RiskLevel
    is_fresh: bool
    is_sufficient: bool
    has_signal: bool
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)
    rationale: NonEmptyText
    unavailable_reason: UnavailableReason | None = None

    @model_validator(mode="after")
    def valid_level(self) -> GradeSignal:
        if self.level is RiskLevel.UNAVAILABLE:
            if self.has_signal:
                raise ValueError("unavailable signals cannot carry a risk signal")
            if self.unavailable_reason is None:
                raise ValueError("unavailable signals require a data-quality reason")
        elif not self.is_fresh or not self.is_sufficient:
            raise ValueError("available signals must be fresh and sufficient")
        elif self.unavailable_reason is not None:
            raise ValueError("available signals cannot carry an unavailable reason")
        return self


class SnapshotEvidenceReference(ContractModel):
    evidence_id: Identifier
    kind: EvidenceKind
    source_record_id: Identifier
    snapshot_as_of: date
    provenance_id: Identifier


class AgentEvent(ContractModel):
    event_id: Identifier
    agent_name: NonEmptyText
    kind: AgentEventKind
    occurred_at: datetime
    message: NonEmptyText
    evidence_ids: tuple[Identifier, ...] = ()


class AgentResult(ContractModel):
    agent_name: NonEmptyText
    status: AnalysisStatus
    claims: tuple[TypedClaim, ...] = ()
    checklist_items: tuple[ChecklistItem, ...] = ()
    events: tuple[AgentEvent, ...] = ()
    evidence_ids: tuple[Identifier, ...] = ()
    error_code: ApiErrorCode | None = None

    @model_validator(mode="after")
    def status_matches_error(self) -> AgentResult:
        if self.status is AnalysisStatus.FAILED and self.error_code is None:
            raise ValueError("failed agent results require an error_code")
        if self.status is not AnalysisStatus.FAILED and self.error_code is not None:
            raise ValueError("only failed agent results may carry an error_code")
        return self


class ReportConclusion(ContractModel):
    conclusion_id: Identifier
    risk_level: RiskLevel
    statement: NonEmptyText
    rule_ids: tuple[Identifier, ...] = Field(min_length=1)
    checklist_ids: tuple[Identifier, ...] = Field(min_length=1)
    evidence_ids: tuple[Identifier, ...] = Field(min_length=1)


class FinalReport(ContractModel):
    """Advisory report; it cannot claim safety, fraud certainty, or legal advice."""

    report_id: Identifier
    session_id: Identifier
    generated_at: datetime
    status: AnalysisStatus
    overall_risk: RiskLevel
    conclusions: tuple[ReportConclusion, ...] = Field(min_length=1)
    claims: tuple[TypedClaim, ...] = ()
    checklist_items: tuple[ChecklistItem, ...] = ()
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    policy_snapshot_evidence: tuple[SnapshotEvidenceReference, ...] = ()
    limitations: tuple[NonEmptyText, ...] = Field(min_length=1)
    advisory_notice: Literal[
        "Advisory decision support only; not legal advice, a safety guarantee, or a fraud determination."
    ]

    @model_validator(mode="after")
    def references_are_resolved(self) -> FinalReport:
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        checklist_ids = tuple(item.checklist_id for item in self.checklist_items)
        claim_ids = tuple(item.claim_id for item in self.claims)
        conclusion_ids = tuple(item.conclusion_id for item in self.conclusions)
        for label, identifiers in (
            ("evidence", evidence_ids),
            ("checklist", checklist_ids),
            ("claim", claim_ids),
            ("conclusion", conclusion_ids),
        ):
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"report contains duplicate {label} identifiers")
        resolved_evidence_ids = set(evidence_ids)
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        for reference in self.policy_snapshot_evidence:
            evidence = evidence_by_id.get(reference.evidence_id)
            if evidence is None or (
                evidence.kind,
                evidence.source_record_id,
                evidence.snapshot_as_of,
                evidence.provenance_id,
            ) != (
                reference.kind,
                reference.source_record_id,
                reference.snapshot_as_of,
                reference.provenance_id,
            ):
                raise ValueError("report snapshot evidence must match policy assessment provenance")
        snapshot_evidence_ids = {
            evidence.evidence_id
            for evidence in self.evidence
            if evidence.kind in {EvidenceKind.TRANSACTION_RECORD, EvidenceKind.HUG_STATISTIC}
        }
        reference_ids = {
            reference.evidence_id for reference in self.policy_snapshot_evidence
        }
        if snapshot_evidence_ids != reference_ids:
            raise ValueError("report snapshot evidence must exactly match policy assessment provenance")
        for item in (*self.claims, *self.checklist_items, *self.conclusions):
            if not set(item.evidence_ids).issubset(resolved_evidence_ids):
                raise ValueError("report item references unknown evidence")
        resolved_checklist_ids = set(checklist_ids)
        for conclusion in self.conclusions:
            if not set(conclusion.checklist_ids).issubset(resolved_checklist_ids):
                raise ValueError("conclusion references unknown checklist items")
        policy_conclusions = tuple(
            conclusion
            for conclusion in self.conclusions
            if "risk-policy-v1.overall-grade" in conclusion.rule_ids
        )
        if len(policy_conclusions) != 1:
            raise ValueError("report requires exactly one deterministic policy conclusion")
        if policy_conclusions[0].risk_level is not self.overall_risk:
            raise ValueError("overall_risk must match the deterministic policy conclusion")
        if any(conclusion.risk_level is not self.overall_risk for conclusion in self.conclusions):
            raise ValueError("all report conclusions must match overall_risk")
        return self


class FitLevel(StrEnum):
    """Deterministic fit result retained without explanatory text."""

    MATCH = "match"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"


class TransactionWindow(ContractModel):
    start: date
    end: date
    inclusive: Literal[True] = True


class HugFixedPeriod(ContractModel):
    start: date
    end: date


class TemporalProvenance(ContractModel):
    """One retained, evidence-linked grade or checklist source projection."""

    signal_id: Identifier | None = None
    checklist_id: Identifier | None = None
    source_id: Identifier
    evidence_as_of: date
    transaction_window: TransactionWindow | None = None
    hug_fixed_period: HugFixedPeriod | None = None
    snapshot_version: Identifier | None = None
    snapshot_as_of: date | None = None
    corpus_version: Identifier | None = None
    corpus_as_of: date | None = None
    unavailable_reason: UnavailableReason | None = None
    unavailable_id: Identifier | None = None
    next_action_id: Identifier | None = None

    @model_validator(mode="after")
    def identifies_retained_source(self) -> TemporalProvenance:
        if self.signal_id is None and self.checklist_id is None:
            raise ValueError("temporal provenance requires a signal or checklist ID")
        return self


class RetainedListing(ContractModel):
    """Allowlisted canonical listing fields only; address text is intentionally excluded."""

    listing_id: Identifier
    legal_dong: NonEmptyText
    deposit_won: WonAmount
    area_sqm: SquareMeters
    contract_date: date
    property_type: NonEmptyText
    floor: int | None = Field(default=None, ge=-10, le=300)
    built_year: int | None = Field(default=None, ge=1800, le=2200)


class NormalizedUserConditions(ContractModel):
    """Structured user constraints only; free-form region input is not retained."""

    max_deposit_won: WonAmount | None = None
    property_types: tuple[PropertyTypeV1, ...] = Field(default=(), max_length=20)
    min_area_sqm: SquareMeters | None = None
    max_area_sqm: SquareMeters | None = None


class SessionMemorySummary(ContractModel):
    """Bounded completed-report projection with no evidence content or free text."""

    session_id: Identifier
    report_id: Identifier
    completed_at: datetime
    report_as_of: date
    listing: RetainedListing
    conditions: NormalizedUserConditions
    fit_level: FitLevel
    risk_level: RiskLevel
    claim_ids: tuple[Identifier, ...] = Field(default=(), max_length=12)
    checklist_ids: tuple[Identifier, ...] = Field(default=(), max_length=12)
    temporal_provenance: tuple[TemporalProvenance, ...] = ()
    unavailable_ids: tuple[Identifier, ...] = ()
    next_action_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def retained_identifiers_are_bounded(self) -> SessionMemorySummary:
        if len(self.claim_ids) + len(self.checklist_ids) > 12:
            raise ValueError("retained claim and checklist IDs cannot exceed 12")
        if len(set((*self.unavailable_ids, *self.next_action_ids))) != (
            len(self.unavailable_ids) + len(self.next_action_ids)
        ):
            raise ValueError("retained unavailable and next-action IDs must be distinct")
        return self


class ApiError(ContractModel):
    code: ApiErrorCode
    message: NonEmptyText
    request_id: Identifier | None = None
    details: tuple[NonEmptyText, ...] = ()


class HugIncidentStatistic(ContractModel):
    """HUG incident statistic for one region at a fixed period and granularity."""

    statistic_id: Identifier
    period_start: date
    period_end: date
    geography: NonEmptyText
    granularity: NonEmptyText
    incident_count: int = Field(ge=0)
    eligible_contract_count: int = Field(gt=0)
    metric_definition: Literal["incident_rate"] = "incident_rate"
    source_record_id: Identifier
    evidence_id: Identifier
    source_name: NonEmptyText
    snapshot_as_of: date
    provenance_id: Identifier

    @field_validator("period_end")
    @classmethod
    def period_is_ordered(cls, value: date, info: object) -> date:
        start = getattr(info, "data", {}).get("period_start")
        if start is not None and value < start:
            raise ValueError("period_end must not precede period_start")
        return value
