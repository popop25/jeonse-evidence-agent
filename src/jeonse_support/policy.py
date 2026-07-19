"""Deterministic advisory implementation of the approved ``risk-policy-v1``.

The policy is deliberately conservative. It reports unavailable rather than inferring a
safety, fraud, legal, rights, or landlord-verification conclusion from weak data.
"""

from __future__ import annotations

import calendar
import hashlib
import json
from pathlib import Path

import yaml
from datetime import date
from decimal import Decimal
from math import ceil

import re
import unicodedata
from pydantic import Field, model_validator

from .models import (
    ChecklistItem,
    ChecklistStatus,
    ClaimKind,
    ContractModel,
    EvidenceKind,
    GradeSignal,
    HugIncidentStatistic,
    Identifier,
    ListingConditions,
    ReportConclusion,
    RiskLevel,
    SampleListing,
    SnapshotEvidenceReference,
    TypedClaim,
    TransactionRetrievalStatus,
    UnavailableReason,
    TransactionReconciliationState,
)

POLICY_ID = "risk-policy-v1"
TRANSACTION_RULE_ID = "risk-policy-v1.transaction-comparables"
HUG_RULE_ID = "risk-policy-v1.hug-percentiles"
OVERALL_RULE_ID = "risk-policy-v1.overall-grade"
TRANSACTION_CHECKLIST_ID = "transaction-comparables"
HUG_CHECKLIST_ID = "hug-percentiles"

TRANSACTION_WINDOW_MONTHS = 24
HUG_WINDOW_MONTHS = 18
MINIMUM_COMPARABLE_COUNT = 3
MINIMUM_HUG_ROW_COUNT = 20
AREA_TOLERANCE = Decimal("0.10")
DEPOSIT_MEDIAN_MEDIUM_THRESHOLD = Decimal("0.90")
DEPOSIT_MEDIAN_HIGH_THRESHOLD = Decimal("1.00")
SUPPORTED_PROPERTY_TYPES = frozenset({"아파트", "연립·다세대", "오피스텔", "apartment"})
HUG_MEDIUM_PERCENTILE = Decimal("0.50")
HUG_HIGH_PERCENTILE = Decimal("0.75")
POLICY_PATH = Path(__file__).resolve().parents[2] / "policies" / "risk-policy-v1.yaml"
POLICY_CHECKSUM_SIGNAL_IDS = (
    "R-TX-COVERAGE",
    "R-DEPOSIT-RATIO",
    "R-HUG-REGION",
    "R-SOURCE-CONFLICT",
)


class HugPolicyInput(ContractModel):
    """Subject HUG value and its fixed-period regional reference population."""

    subject: HugIncidentStatistic
    reference_rows: tuple[HugIncidentStatistic, ...] = Field(min_length=1)
    @model_validator(mode="after")
    def reference_population_is_valid(self) -> HugPolicyInput:
        population_keys = tuple(
            (row.period_start, row.period_end, row.granularity, row.geography)
            for row in self.reference_rows
        )
        if len(set(population_keys)) != len(population_keys):
            raise ValueError("HUG reference rows must have distinct population keys")
        if len({row.statistic_id for row in self.reference_rows}) != len(self.reference_rows):
            raise ValueError("HUG reference rows must have distinct statistic_ids")
        target_rows = tuple(
            row for row in self.reference_rows if row.geography == self.subject.geography
        )
        if len(target_rows) != 1:
            raise ValueError("HUG reference population requires exactly one subject geography")
        target = target_rows[0]
        subject_measurement = (
            self.subject.period_start,
            self.subject.period_end,
            self.subject.granularity,
            self.subject.metric_definition,
            self.subject.source_name,
            self.subject.snapshot_as_of,
            self.subject.incident_count,
            self.subject.eligible_contract_count,
        )
        target_measurement = (
            target.period_start,
            target.period_end,
            target.granularity,
            target.metric_definition,
            target.source_name,
            target.snapshot_as_of,
            target.incident_count,
            target.eligible_contract_count,
        )
        if target_measurement != subject_measurement:
            raise ValueError("HUG subject measurement must equal its population row")
        return self



class RiskPolicyInput(ContractModel):
    listing: ListingConditions
    transaction_samples: tuple[SampleListing, ...] = ()
    transaction_retrieval_status: TransactionRetrievalStatus = TransactionRetrievalStatus.SUCCESS
    hug: HugPolicyInput | None = None
    as_of: date


def _runtime_parameters() -> dict[str, object]:
    return {
        "freshness": {
            "transactions_months": TRANSACTION_WINDOW_MONTHS,
            "hug_months": HUG_WINDOW_MONTHS,
        },
        "R-TX-COVERAGE": {
            "eligibility": {
                "window_calendar_months": TRANSACTION_WINDOW_MONTHS,
                "same_legal_dong": True,
                "supported_property_type": True,
                "supported_property_types": sorted(SUPPORTED_PROPERTY_TYPES),
                "exclusive_area_tolerance_percent": int(AREA_TOLERANCE * 100),
                "jeonse_only": {
                    "monthly_rent_krw": 0,
                    "deposit_won_positive": True,
                    "cancelled": False,
                    "renewal": False,
                },
                "minimum_comparables": MINIMUM_COMPARABLE_COUNT,
            },
        },
        "R-DEPOSIT-RATIO": {
            "calculation": "listing_deposit_won / median(eligible_transaction_deposit_won)",
            "thresholds": {
                "none": f"< {DEPOSIT_MEDIAN_MEDIUM_THRESHOLD:.2f}",
                "medium": (
                    f">= {DEPOSIT_MEDIAN_MEDIUM_THRESHOLD:.2f} and "
                    f"< {DEPOSIT_MEDIAN_HIGH_THRESHOLD:.2f}"
                ),
                "high": f">= {DEPOSIT_MEDIAN_HIGH_THRESHOLD:.2f}",
            },
        },
        "R-HUG-REGION": {
            "population": {
                "same_source": True,
                "same_as_of": True,
                "same_fixed_period": True,
                "same_metric_definition": True,
                "same_granularity": True,
                "nationwide_rows": True,
                "target_included": True,
                "numeric_rate_required": True,
                "positive_denominator_required": True,
                "minimum_rows": MINIMUM_HUG_ROW_COUNT,
            },
            "percentile": {
                "method": "nearest_rank",
                "formula": "Pq = x[ceil(q*n)-1]",
                "p50": float(HUG_MEDIUM_PERCENTILE),
                "p75": float(HUG_HIGH_PERCENTILE),
                "unavailable_when_p50_equals_p75": True,
                "inclusive_boundaries": True,
            },
            "thresholds": {
                "none": f"target < P{int(HUG_MEDIUM_PERCENTILE * 100)}",
                "medium": (
                    f"target >= P{int(HUG_MEDIUM_PERCENTILE * 100)} and "
                    f"target < P{int(HUG_HIGH_PERCENTILE * 100)}"
                ),
                "high": f"target >= P{int(HUG_HIGH_PERCENTILE * 100)}",
            },
        },
        "R-SOURCE-CONFLICT": {
            "identity": {
                "fields": ["dataset_id", "dataset_version", "stable_row_id"],
                "legacy_single_snapshot": True,
                "measured_fields_never_identity": ["area_sqm", "monthly_rent_won", "deposit_won"],
            },
            "reconciliation": {
                "unit": "locator_to_measurement_multiset",
                "exact_duplicate": "retain one deterministic row",
                "later_as_of_correction": "retain the latest declared snapshot row",
                "same_as_of_measurement_difference": "source_conflict",
                "mixed_identity_capability": "not_comparable",
                "cardinality_breach": "invalid_artifact",
            },
            "normalization": {
                "strings": "Unicode NFC, trim, collapse internal ASCII whitespace, casefold",
                "dates": "ISO-8601 calendar date",
                "area_sqm": "decimal square metres rounded to 2 decimal places",
                "deposit_won": "integer won with no scaling",
                "property_type": "allowlisted canonical Korean label",
            },
            "measurement_inequality": {
                "fields": [
                    "occurred_on",
                    "legal_dong",
                    "property_type",
                    "area_sqm",
                    "floor",
                    "deposit_won",
                    "monthly_rent_won",
                ],
                "operator": "any normalized field value differs",
            },
        },
        "overall_grade": {
            "exact_low_grade_bearing_set": ["R-DEPOSIT-RATIO", "R-HUG-REGION"],
            "precedence": [
                {"schema_checksum_config_mapping_profile_or_corpus_breach": "failed_no_report"},
                {"any_valid_high": "high"},
                {"no_high_and_any_valid_medium": "medium"},
                {"both_exact_low_signals_fresh_sufficient_none": "low"},
                {"otherwise_required_grade_signal_unavailable": "unavailable"},
            ],
            "coverage_incomplete_when_any_signal_missing": True,
        },
    }


def _artifact_runtime_parameters(policy: dict[str, object]) -> dict[str, object]:
    signals = policy["signals"]
    return {
        "freshness": policy["provenance"]["freshness"],
        "R-TX-COVERAGE": {"eligibility": signals["R-TX-COVERAGE"]["eligibility"]},
        "R-DEPOSIT-RATIO": {key: signals["R-DEPOSIT-RATIO"][key] for key in ("calculation", "thresholds")},
        "R-HUG-REGION": {
            "population": signals["R-HUG-REGION"]["population"],
            "percentile": signals["R-HUG-REGION"]["percentile"],
            "thresholds": signals["R-HUG-REGION"]["thresholds"],
        },
        "R-SOURCE-CONFLICT": {
            key: signals["R-SOURCE-CONFLICT"][key]
            for key in ("identity", "reconciliation", "normalization", "measurement_inequality")
        },
        "overall_grade": {
            key: policy["overall_grade"][key]
            for key in ("exact_low_grade_bearing_set", "precedence", "coverage_incomplete_when_any_signal_missing")
        },
    }


def validate_policy_artifact(path: Path = POLICY_PATH) -> str:
    """Fail closed unless the pinned artifact and executable parameters agree."""
    try:
        policy = yaml.safe_load(path.read_text(encoding="utf-8"))
        checksum = policy["provenance"]["parameter_checksum"]
        scoped_mapping = {
            "signals": {
                signal_id: policy["signals"][signal_id]
                for signal_id in POLICY_CHECKSUM_SIGNAL_IDS
            },
            "overall_grade": policy["overall_grade"],
        }
        canonical = json.dumps(
            scoped_mapping,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        artifact_parameters = _artifact_runtime_parameters(policy)
    except (OSError, KeyError, TypeError, yaml.YAMLError) as error:
        raise ValueError("INVALID_POLICY_ARTIFACT") from error
    if (
        policy.get("policy_id") != POLICY_ID
        or policy.get("status") != "v1-active"
        or checksum.get("canonical_input") != canonical
        or checksum.get("value") != digest
        or artifact_parameters != _runtime_parameters()
    ):
        raise ValueError("INVALID_POLICY_ARTIFACT")
    return digest


class RiskPolicyAssessment(ContractModel):
    policy_id: Identifier = POLICY_ID
    transaction_signal: GradeSignal
    hug_signal: GradeSignal
    overall_risk: RiskLevel
    claims: tuple[TypedClaim, ...]
    checklist_items: tuple[ChecklistItem, ...]
    conclusion: ReportConclusion
    snapshot_evidence: tuple[SnapshotEvidenceReference, ...] = ()

    @model_validator(mode="after")
    def conclusion_matches_grade(self) -> RiskPolicyAssessment:
        if self.conclusion.risk_level is not self.overall_risk:
            raise ValueError("conclusion risk_level must match overall_risk")
        return self


def _months_before(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 - months
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    return date(year, month, min(value.day, calendar.monthrange(year, month)[1]))


def _nearest_rank(values: tuple[Decimal, ...], percentile: Decimal) -> Decimal:
    """Return the nearest-rank percentile; callers supply a non-empty population."""
    ordered = sorted(values)
    rank = max(1, ceil(len(ordered) * float(percentile)))
    return ordered[rank - 1]


def _unavailable_signal(
    signal_id: str,
    evidence_ids: tuple[str, ...],
    rationale: str,
    reason: UnavailableReason,
    *,
    is_fresh: bool = False,
    is_sufficient: bool = False,
) -> GradeSignal:
    return GradeSignal(
        signal_id=signal_id,
        level=RiskLevel.UNAVAILABLE,
        is_fresh=is_fresh,
        is_sufficient=is_sufficient,
        has_signal=False,
        evidence_ids=evidence_ids,
        rationale=rationale,
        unavailable_reason=reason,
    )
def _snapshot_reference(
    evidence_id: str,
    kind: EvidenceKind,
    source_record_id: str | None,
    snapshot_as_of: date | None,
    provenance_id: str | None,
) -> SnapshotEvidenceReference | None:
    if source_record_id is None or snapshot_as_of is None or provenance_id is None:
        return None
    return SnapshotEvidenceReference(
        evidence_id=evidence_id,
        kind=kind,
        source_record_id=source_record_id,
        snapshot_as_of=snapshot_as_of,
        provenance_id=provenance_id,
    )






def _normalized_string(value: str) -> str:
    return re.sub(r"[ \t\n\r\f\v]+", " ", unicodedata.normalize("NFC", value).strip()).casefold()


def _normalized_measurement(sample: SampleListing) -> tuple[object, ...]:
    return (
        sample.occurred_on.isoformat(),
        _normalized_string(sample.legal_dong),
        _normalized_string(sample.property_type),
        sample.area_sqm.quantize(Decimal("0.01")),
        sample.floor,
        sample.deposit_won,
        sample.monthly_rent_won,
    )


def _normalized_locator(sample: SampleListing) -> tuple[object, ...]:
    if sample.source_record_id is not None:
        return ("source_record_id", _normalized_string(sample.source_record_id))
    return (
        "fallback",
        sample.occurred_on.isoformat(),
        _normalized_string(sample.legal_dong),
        _normalized_string(sample.property_type),
        sample.floor,
    )


def _reconcile_transaction_rows(
    samples: tuple[SampleListing, ...],
) -> tuple[TransactionReconciliationState | None, tuple[SampleListing, ...]]:
    """Reconcile immutable, qualified snapshot rows without using measurements as identity."""

    qualified = tuple(
        sample.dataset_id is not None and sample.dataset_version is not None and sample.stable_row_id is not None
        for sample in samples
    )
    if any(qualified) and not all(qualified):
        return TransactionReconciliationState.NOT_COMPARABLE, ()
    if not any(qualified):
        measurements_by_locator: dict[tuple[object, ...], tuple[object, ...]] = {}
        rows_by_locator: dict[tuple[object, ...], list[SampleListing]] = {}
        for sample in samples:
            locator = _normalized_locator(sample)
            measurement = _normalized_measurement(sample)
            previous = measurements_by_locator.setdefault(locator, measurement)
            if previous != measurement:
                return TransactionReconciliationState.SOURCE_CONFLICT, ()
            rows_by_locator.setdefault(locator, []).append(sample)
        reconciled = []
        outcome: TransactionReconciliationState | None = None
        for rows in rows_by_locator.values():
            if len(rows) > 1:
                outcome = TransactionReconciliationState.EXACT_DUPLICATE
            reconciled.append(min(rows, key=lambda row: (row.provenance_id, row.evidence_id, row.transaction_id)))
        return outcome, tuple(sorted(reconciled, key=lambda row: (row.occurred_on, row.transaction_id)))

    by_locator: dict[tuple[str, str], list[SampleListing]] = {}
    for sample in samples:
        by_locator.setdefault((sample.dataset_id, sample.stable_row_id), []).append(sample)

    reconciled: list[SampleListing] = []
    outcome: TransactionReconciliationState | None = None
    for rows in by_locator.values():
        by_snapshot: dict[tuple[str, date], list[SampleListing]] = {}
        for row in rows:
            by_snapshot.setdefault((row.dataset_version, row.snapshot_as_of), []).append(row)
        candidates: list[SampleListing] = []
        versions_to_as_of: dict[str, set[date]] = {}
        for row in rows:
            versions_to_as_of.setdefault(row.dataset_version, set()).add(row.snapshot_as_of)
        if any(len(as_of_values) > 1 for as_of_values in versions_to_as_of.values()):
            return TransactionReconciliationState.CARDINALITY_BREACH, ()
        for snapshot_rows in by_snapshot.values():
            measurements = {_normalized_measurement(row) for row in snapshot_rows}
            if len(measurements) != 1:
                return TransactionReconciliationState.SOURCE_CONFLICT, ()
            candidates.append(
                min(snapshot_rows, key=lambda row: (row.provenance_id, row.evidence_id, row.transaction_id))
            )
            if len(snapshot_rows) > 1:
                outcome = TransactionReconciliationState.EXACT_DUPLICATE
        newest_as_of = max(row.snapshot_as_of for row in candidates)
        newest = [row for row in candidates if row.snapshot_as_of == newest_as_of]
        if len({_normalized_measurement(row) for row in newest}) != 1:
            return TransactionReconciliationState.SOURCE_CONFLICT, ()
        selected = min(
            newest,
            key=lambda row: (row.dataset_version, row.provenance_id, row.evidence_id, row.transaction_id),
        )
        if any(_normalized_measurement(row) != _normalized_measurement(selected) for row in candidates):
            outcome = TransactionReconciliationState.CORRECTION
        reconciled.append(selected)
    return outcome, tuple(sorted(reconciled, key=lambda row: (row.occurred_on, row.transaction_id)))




def _median(values: tuple[Decimal, ...]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)




def assess_transaction_signal(input: RiskPolicyInput) -> GradeSignal:
    """Assess deposit versus median comparable deposit in the approved 24-month window."""
    validate_policy_artifact()
    listing = input.listing
    base_evidence = (listing.evidence_id,)
    used_rows = input.transaction_samples
    evidence_ids = base_evidence + tuple(sample.evidence_id for sample in used_rows)
    if input.transaction_retrieval_status is TransactionRetrievalStatus.MISSING:
        return _unavailable_signal(
            "transaction-deposit-ratio",
            base_evidence,
            "Comparable transaction retrieval is unavailable.",
            UnavailableReason.MISSING,
        )
    reconciliation_state, reconciled_rows = _reconcile_transaction_rows(used_rows)
    if reconciliation_state in (
        TransactionReconciliationState.SOURCE_CONFLICT,
        TransactionReconciliationState.NOT_COMPARABLE,
        TransactionReconciliationState.CARDINALITY_BREACH,
    ):
        reason = (
            UnavailableReason.NOT_COMPARABLE
            if reconciliation_state is TransactionReconciliationState.NOT_COMPARABLE
            else UnavailableReason.INVALID_ARTIFACT
            if reconciliation_state is TransactionReconciliationState.CARDINALITY_BREACH
            else UnavailableReason.SOURCE_CONFLICT
        )
        return _unavailable_signal(
            "transaction-deposit-ratio",
            evidence_ids,
            "Comparable transaction snapshots cannot be reconciled under the declared identity contract.",
            reason,
            is_fresh=True,
        )
    non_temporal_samples = tuple(
        sample
        for sample in reconciled_rows
        if sample.legal_dong == listing.legal_dong
        and sample.property_type == listing.property_type
        and sample.property_type in SUPPORTED_PROPERTY_TYPES
        and sample.monthly_rent_won == 0
        and sample.deposit_won > 0
        and not sample.cancelled
        and not sample.renewal
        and abs(sample.area_sqm - listing.area_sqm) <= listing.area_sqm * AREA_TOLERANCE
    )
    fresh_samples = tuple(
        sample
        for sample in non_temporal_samples
        if sample.snapshot_as_of <= input.as_of
        and _months_before(input.as_of, TRANSACTION_WINDOW_MONTHS)
        <= sample.occurred_on
        <= input.as_of
    )
    is_fresh = (
        len(fresh_samples) == len(non_temporal_samples)
        or len(fresh_samples) >= MINIMUM_COMPARABLE_COUNT
    )
    is_sufficient = len(non_temporal_samples) >= MINIMUM_COMPARABLE_COUNT
    if not is_sufficient or not is_fresh:
        if not used_rows:
            reason = UnavailableReason.NO_MATCH
        elif listing.property_type not in SUPPORTED_PROPERTY_TYPES or not any(
            sample.property_type in SUPPORTED_PROPERTY_TYPES for sample in used_rows
        ):
            reason = UnavailableReason.UNSUPPORTED
        elif not is_fresh:
            reason = UnavailableReason.STALE
        elif not non_temporal_samples:
            reason = UnavailableReason.NO_MATCH
        else:
            reason = UnavailableReason.INSUFFICIENT
        return _unavailable_signal(
            "transaction-deposit-ratio",
            evidence_ids,
            "Comparable transaction data cannot support an advisory grade.",
            reason,
            is_fresh=is_fresh,
            is_sufficient=is_sufficient,
        )
    samples = fresh_samples
    evidence_ids = base_evidence + tuple(sample.evidence_id for sample in samples)
    deposits = tuple(sample.deposit_won for sample in samples)
    if any(value == 0 for value in deposits) or listing.deposit_won == 0:
        return _unavailable_signal(
            "transaction-deposit-ratio",
            evidence_ids,
            "Zero deposit data cannot support an advisory grade.",
            UnavailableReason.INSUFFICIENT,
            is_fresh=True,
        )
    median = _median(deposits)
    ratio = listing.deposit_won / median
    if ratio >= DEPOSIT_MEDIAN_HIGH_THRESHOLD:
        level, has_signal = RiskLevel.HIGH, True
    elif ratio >= DEPOSIT_MEDIAN_MEDIUM_THRESHOLD:
        level, has_signal = RiskLevel.MEDIUM, True
    else:
        level, has_signal = RiskLevel.LOW, False
    return GradeSignal(
        signal_id="transaction-deposit-ratio",
        level=level,
        is_fresh=True,
        is_sufficient=True,
        has_signal=has_signal,
        evidence_ids=evidence_ids,
        rationale=(
            f"{len(samples)} comparable transactions produced a median deposit of {median} won "
            f"and a listing-to-median ratio of {ratio:.4f}."
        ),
    )


def assess_hug_signal(input: RiskPolicyInput) -> GradeSignal:
    """Assess a regional HUG subject value against P50/P75 nearest-rank references."""
    validate_policy_artifact()
    if input.hug is None:
        return _unavailable_signal(
            "hug-incident-percentile",
            (input.listing.evidence_id,),
            "No HUG statistic was supplied for the advisory grade.",
            UnavailableReason.MISSING,
        )
    subject = input.hug.subject
    rows = input.hug.reference_rows
    evidence_ids = (input.listing.evidence_id, subject.evidence_id) + tuple(
        row.evidence_id for row in rows
    )
    cutoff = _months_before(input.as_of, HUG_WINDOW_MONTHS)
    fresh = all(
        row.period_end >= cutoff
        and row.period_end <= input.as_of
        and row.snapshot_as_of <= input.as_of
        for row in (subject, *rows)
    )
    same_population = all(
        row.period_start == subject.period_start
        and row.period_end == subject.period_end
        and row.granularity == subject.granularity
        and row.metric_definition == subject.metric_definition
        and row.source_name == subject.source_name
        and row.snapshot_as_of == subject.snapshot_as_of
        for row in rows
    )
    has_conflicting_ids = len({row.statistic_id for row in rows}) != len(rows)
    has_duplicate_regions = len({row.geography for row in rows}) != len(rows)
    sufficient = (
        same_population
        and not has_conflicting_ids
        and not has_duplicate_regions
        and len(rows) >= MINIMUM_HUG_ROW_COUNT
    )
    if not fresh or not sufficient:
        return _unavailable_signal(
            "hug-incident-percentile",
            evidence_ids,
            "HUG data is stale, mismatched, conflicting, or has fewer than 20 valid regional rows.",
            UnavailableReason.STALE if not fresh else (
                UnavailableReason.INSUFFICIENT
                if len(rows) < MINIMUM_HUG_ROW_COUNT
                else UnavailableReason.POPULATION_MISMATCH
            ),
            is_fresh=fresh,
            is_sufficient=sufficient,
        )
    subject_rate = Decimal(subject.incident_count) / Decimal(subject.eligible_contract_count)
    rates = tuple(
        Decimal(row.incident_count) / Decimal(row.eligible_contract_count) for row in rows
    )
    p50 = _nearest_rank(rates, HUG_MEDIUM_PERCENTILE)
    p75 = _nearest_rank(rates, HUG_HIGH_PERCENTILE)
    if p50 == p75:
        return _unavailable_signal(
            "hug-incident-percentile",
            evidence_ids,
            "Tied HUG P50 and P75 rates cannot support a distinct advisory grade.",
            UnavailableReason.TIED_THRESHOLDS,
            is_fresh=True,
            is_sufficient=True,
        )
    if subject_rate >= p75:
        level, has_signal = RiskLevel.HIGH, True
    elif subject_rate >= p50:
        level, has_signal = RiskLevel.MEDIUM, True
    else:
        level, has_signal = RiskLevel.LOW, False
    return GradeSignal(
        signal_id="hug-incident-percentile",
        level=level,
        is_fresh=True,
        is_sufficient=True,
        has_signal=has_signal,
        evidence_ids=evidence_ids,
        rationale=(
            f"Regional HUG nearest-rank P50/P75 rates are {p50:.4f}/{p75:.4f}; "
            f"the subject rate is {subject_rate:.4f}."
        ),
    )


def assess_risk_policy(input: RiskPolicyInput) -> RiskPolicyAssessment:
    """Apply ``risk-policy-v1`` and return provenance-linked advisory contracts."""
    validate_policy_artifact()
    transaction = assess_transaction_signal(input)
    hug = assess_hug_signal(input)
    signals = (transaction, hug)
    evidence_ids = tuple(dict.fromkeys(eid for signal in signals for eid in signal.evidence_ids))
    if any(signal.level is RiskLevel.HIGH for signal in signals):
        overall = RiskLevel.HIGH
    elif any(signal.level is RiskLevel.MEDIUM for signal in signals):
        overall = RiskLevel.MEDIUM
    elif all(
        signal.level is RiskLevel.LOW
        and signal.is_fresh
        and signal.is_sufficient
        and not signal.has_signal
        for signal in signals
    ):
        overall = RiskLevel.LOW
    else:
        overall = RiskLevel.UNAVAILABLE

    checklist_items = (
        ChecklistItem(
            checklist_id=TRANSACTION_CHECKLIST_ID,
            status=(
                ChecklistStatus.UNAVAILABLE
                if transaction.level is RiskLevel.UNAVAILABLE
                else ChecklistStatus.PASS
                if transaction.level is RiskLevel.LOW and not transaction.has_signal
                else ChecklistStatus.REVIEW
            ),
            label="Comparable transaction signal",
            rationale=transaction.rationale,
            rule_ids=(TRANSACTION_RULE_ID,),
            evidence_ids=transaction.evidence_ids,
        ),
        ChecklistItem(
            checklist_id=HUG_CHECKLIST_ID,
            status=(
                ChecklistStatus.UNAVAILABLE
                if hug.level is RiskLevel.UNAVAILABLE
                else ChecklistStatus.PASS
                if hug.level is RiskLevel.LOW and not hug.has_signal
                else ChecklistStatus.REVIEW
            ),
            label="Regional HUG percentile signal",
            rationale=hug.rationale,
            rule_ids=(HUG_RULE_ID,),
            evidence_ids=hug.evidence_ids,
        ),
    )
    claim = TypedClaim(
        claim_id="risk-policy-v1-grade",
        kind=ClaimKind.INFERENCE,
        statement=(
            f"The deterministic advisory policy produced a {overall.value} risk grade; "
            "this is not legal advice, a safety guarantee, or a fraud determination."
        ),
        rule_ids=(TRANSACTION_RULE_ID, HUG_RULE_ID, OVERALL_RULE_ID),
        evidence_ids=evidence_ids,
    )
    conclusion = ReportConclusion(
        conclusion_id="risk-policy-v1-conclusion",
        risk_level=overall,
        statement=claim.statement,
        rule_ids=claim.rule_ids,
        checklist_ids=(TRANSACTION_CHECKLIST_ID, HUG_CHECKLIST_ID),
        evidence_ids=evidence_ids,
    )
    references = tuple(
        reference
        for reference in (
            _snapshot_reference(
                input.listing.evidence_id,
                EvidenceKind.TRANSACTION_RECORD,
                input.listing.source_record_id,
                input.listing.snapshot_as_of,
                input.listing.provenance_id,
            ),
            *(
                _snapshot_reference(
                    sample.evidence_id,
                    EvidenceKind.TRANSACTION_RECORD,
                    sample.source_record_id,
                    sample.snapshot_as_of,
                    sample.provenance_id,
                )
                for sample in input.transaction_samples
                if sample.evidence_id in evidence_ids
            ),
            *(
                _snapshot_reference(
                    statistic.evidence_id,
                    EvidenceKind.HUG_STATISTIC,
                    statistic.source_record_id,
                    statistic.snapshot_as_of,
                    statistic.provenance_id,
                )
                for statistic in (
                    (input.hug.subject, *input.hug.reference_rows) if input.hug else ()
                )
                if statistic.evidence_id in evidence_ids
            ),
        )
        if reference is not None
    )
    return RiskPolicyAssessment(
        transaction_signal=transaction,
        hug_signal=hug,
        overall_risk=overall,
        claims=(claim,),
        checklist_items=checklist_items,
        conclusion=conclusion,
        snapshot_evidence=references,
    )
