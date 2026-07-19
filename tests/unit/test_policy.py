import hashlib
import json
import re
from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from jeonse_support import policy
from jeonse_support.models import (
    AnalysisStatus,
    Evidence,
    EvidenceKind,
    FinalReport,
    GradeSignal,
    HugIncidentStatistic,
    ListingConditions,
    RiskLevel,
    SampleListing,
    TransactionRetrievalStatus,
    UnavailableReason,
)
from jeonse_support.policy import (
    HugPolicyInput,
    RiskPolicyInput,
    assess_hug_signal,
    assess_risk_policy,
    assess_transaction_signal,
    validate_policy_artifact,
)


AS_OF = date(2026, 7, 18)
PERIOD_START = date(2025, 7, 1)
PERIOD_END = date(2025, 7, 31)


def listing(*, deposit: Decimal = Decimal("89")) -> ListingConditions:
    return ListingConditions(
        listing_id="listing-1",
        address_text="Synthetic test listing",
        legal_dong="Synthetic-dong",
        deposit_won=deposit,
        area_sqm=Decimal("100.00"),
        contract_date=AS_OF,
        property_type="apartment",
        evidence_id="listing-evidence",
    )


def sample(
    number: int,
    *,
    occurred_on: date = AS_OF,
    area: Decimal = Decimal("100.00"),
    deposit: Decimal = Decimal("100"),
) -> SampleListing:
    return SampleListing(
        transaction_id=f"transaction-{number}",
        occurred_on=occurred_on,
        area_sqm=area,
        deposit_won=deposit,
        address_text=f"Synthetic comparable {number}",
        source_record_id=f"record-{number}",
        evidence_id=f"transaction-evidence-{number}",
        legal_dong="Synthetic-dong",
        property_type="apartment",
        monthly_rent_won=Decimal("0"),
        cancelled=False,
        renewal=False,
        floor=1,
        source_name="Synthetic transaction source",
        snapshot_as_of=AS_OF,
        provenance_id=f"transaction-provenance-{number}",
    )


def hug_statistic(
    number: int,
    count: int,
    *,
    geography: str | None = None,
    eligible_contract_count: int = 100,
) -> HugIncidentStatistic:
    return HugIncidentStatistic(
        statistic_id=f"hug-{number}",
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        geography=geography or f"Synthetic district {number}",
        granularity="district",
        incident_count=count,
        eligible_contract_count=eligible_contract_count,
        metric_definition="incident_rate",
        source_record_id=f"hug-record-{number}",
        evidence_id=f"hug-evidence-{number}",
        source_name="Synthetic HUG source",
        snapshot_as_of=PERIOD_END,
        provenance_id=f"hug-provenance-{number}",
    )


def policy_input(
    *,
    deposit: Decimal = Decimal("89"),
    samples: tuple[SampleListing, ...] = (),
    hug: HugPolicyInput | None = None,
) -> RiskPolicyInput:
    return RiskPolicyInput(
        listing=listing(deposit=deposit),
        transaction_samples=samples,
        hug=hug,
        as_of=AS_OF,
    )


def hug_input(subject_count: int, counts: tuple[int, ...]) -> HugPolicyInput:
    subject = hug_statistic(0, subject_count)
    rows = [
        hug_statistic(number, count) for number, count in enumerate(counts, start=1)
    ]
    rows[-1] = hug_statistic(
        len(rows),
        subject_count,
        geography=subject.geography,
    )
    return HugPolicyInput(subject=subject, reference_rows=tuple(rows))


def test_transaction_window_and_area_boundaries_require_three_comparables() -> None:
    valid = (
        sample(1, occurred_on=date(2024, 7, 18), area=Decimal("90.00")),
        sample(2, area=Decimal("110.00")),
        sample(3),
    )
    assert assess_transaction_signal(policy_input(samples=valid)).level is RiskLevel.LOW

    invalid_third_samples = (
        sample(3, occurred_on=date(2024, 7, 17)),
        sample(3, occurred_on=AS_OF + timedelta(days=1)),
        sample(3, area=Decimal("89.99")),
        sample(3, area=Decimal("110.01")),
    )
    for invalid_third in invalid_third_samples:
        assert (
            assess_transaction_signal(policy_input(samples=valid[:2] + (invalid_third,))).level
            is RiskLevel.UNAVAILABLE
        )


@pytest.mark.parametrize(
    ("deposit", "expected"),
    [
        (Decimal("89"), RiskLevel.LOW),
        (Decimal("90"), RiskLevel.MEDIUM),
        (Decimal("91"), RiskLevel.MEDIUM),
        (Decimal("99"), RiskLevel.MEDIUM),
        (Decimal("100"), RiskLevel.HIGH),
        (Decimal("101"), RiskLevel.HIGH),
    ],
)
def test_transaction_deposit_ratio_thresholds(
    deposit: Decimal, expected: RiskLevel
) -> None:
    samples = (sample(1), sample(2), sample(3))
    assert assess_transaction_signal(policy_input(deposit=deposit, samples=samples)).level is expected


def test_transaction_two_comparables_are_unavailable_and_three_are_sufficient() -> None:
    assert assess_transaction_signal(
        policy_input(samples=(sample(1), sample(2)))
    ).level is RiskLevel.UNAVAILABLE
    assert assess_transaction_signal(
        policy_input(samples=(sample(1), sample(2), sample(3)))
    ).level is RiskLevel.LOW


def test_transaction_source_conflict_makes_signal_unavailable() -> None:
    first = sample(1)
    conflicting = sample(2, deposit=Decimal("120")).model_copy(
        update={"source_record_id": first.source_record_id}
    )
    signal = assess_transaction_signal(
        policy_input(samples=(first, conflicting, sample(3)))
    )
    assert signal.level is RiskLevel.UNAVAILABLE

def test_transaction_normalized_duplicate_measurement_is_reconciled_once() -> None:
    first = sample(1).model_copy(update={"source_record_id": "Record   1"})
    equivalent = sample(2).model_copy(update={"source_record_id": "record 1"})
    signal = assess_transaction_signal(
        policy_input(samples=(first, equivalent, sample(3), sample(4)))
    )
    assert signal.level is RiskLevel.LOW
    assert signal.rationale.startswith("3 comparable transactions")


def test_transaction_normalized_duplicate_measurement_conflict_is_unavailable() -> None:
    first = sample(1).model_copy(update={"source_record_id": "Record   1"})
    conflicting = sample(2, deposit=Decimal("120")).model_copy(
        update={"source_record_id": "record 1"}
    )
    signal = assess_transaction_signal(policy_input(samples=(first, conflicting, sample(3))))
    assert signal.unavailable_reason is UnavailableReason.SOURCE_CONFLICT
def test_transaction_ascii_whitespace_locator_conflict_is_unavailable() -> None:
    first = sample(1).model_copy(update={"source_record_id": "Record\t1"})
    conflicting = sample(2, deposit=Decimal("120")).model_copy(
        update={"source_record_id": "record\f1"}
    )
    signal = assess_transaction_signal(policy_input(samples=(first, conflicting, sample(3))))
    assert signal.unavailable_reason is UnavailableReason.SOURCE_CONFLICT
def test_qualified_transaction_identity_reconciles_later_correction_deterministically() -> None:
    current = sample(1).model_copy(
        update={"dataset_id": "official-transactions", "dataset_version": "2026-07", "stable_row_id": "row-1"}
    )
    correction_source = sample(4, deposit=Decimal("120")).model_copy(
        update={
            "dataset_id": "official-transactions",
            "dataset_version": "2026-06",
            "stable_row_id": "row-1",
            "snapshot_as_of": date(2026, 6, 18),
        }
    )
    other_rows = tuple(
        sample(number).model_copy(
            update={
                "dataset_id": "official-transactions",
                "dataset_version": "2026-07",
                "stable_row_id": f"row-{number}",
            }
        )
        for number in (2, 3)
    )
    signal = assess_transaction_signal(policy_input(samples=(current, correction_source) + other_rows))
    assert signal.level is RiskLevel.LOW
    assert signal.rationale.startswith("3 comparable transactions")
def test_qualified_exact_duplicates_are_reconciled_as_one_row() -> None:
    first = sample(1).model_copy(
        update={"dataset_id": "official-transactions", "dataset_version": "2026-07", "stable_row_id": "row-1"}
    )
    duplicate = sample(2).model_copy(
        update={"dataset_id": "official-transactions", "dataset_version": "2026-07", "stable_row_id": "row-1"}
    )
    third = sample(3).model_copy(
        update={"dataset_id": "official-transactions", "dataset_version": "2026-07", "stable_row_id": "row-3"}
    )
    signal = assess_transaction_signal(policy_input(samples=(first, duplicate, third)))
    assert signal.unavailable_reason is UnavailableReason.INSUFFICIENT
def test_qualified_multiset_reconciliation_is_deterministic_for_duplicates_and_corrections() -> None:
    current = sample(1).model_copy(
        update={
            "dataset_id": "official-transactions",
            "dataset_version": "2026-07",
            "stable_row_id": "row-1",
        }
    )
    exact_duplicate = sample(4).model_copy(
        update={
            "dataset_id": "official-transactions",
            "dataset_version": "2026-07",
            "stable_row_id": "row-1",
        }
    )
    correction = sample(5, deposit=Decimal("120")).model_copy(
        update={
            "dataset_id": "official-transactions",
            "dataset_version": "2026-06",
            "stable_row_id": "row-1",
            "snapshot_as_of": date(2026, 6, 18),
        }
    )
    other_rows = tuple(
        sample(number).model_copy(
            update={
                "dataset_id": "official-transactions",
                "dataset_version": "2026-07",
                "stable_row_id": f"row-{number}",
            }
        )
        for number in (2, 3)
    )
    rows = (current, exact_duplicate, correction, *other_rows)
    forward = assess_transaction_signal(policy_input(samples=rows))
    reverse = assess_transaction_signal(policy_input(samples=tuple(reversed(rows))))

    assert forward.model_dump() == reverse.model_dump()
    assert forward.level is RiskLevel.LOW
    assert forward.rationale.startswith("3 comparable transactions")


def test_qualified_identity_reuse_with_unequal_snapshot_cardinality_fails_closed() -> None:
    current = sample(1).model_copy(
        update={
            "dataset_id": "official-transactions",
            "dataset_version": "2026-07",
            "stable_row_id": "row-1",
        }
    )
    exact_duplicate = sample(4).model_copy(
        update={
            "dataset_id": "official-transactions",
            "dataset_version": "2026-07",
            "stable_row_id": "row-1",
        }
    )
    reused_identity = sample(5, deposit=Decimal("120")).model_copy(
        update={
            "dataset_id": "official-transactions",
            "dataset_version": "2026-07",
            "stable_row_id": "row-1",
            "snapshot_as_of": date(2026, 7, 17),
        }
    )
    third = sample(3).model_copy(
        update={
            "dataset_id": "official-transactions",
            "dataset_version": "2026-07",
            "stable_row_id": "row-3",
        }
    )
    signal = assess_transaction_signal(
        policy_input(samples=(current, exact_duplicate, reused_identity, third))
    )

    assert signal.level is RiskLevel.UNAVAILABLE
    assert signal.unavailable_reason is UnavailableReason.INVALID_ARTIFACT


def test_qualified_identity_measurement_mutation_at_same_as_of_is_source_conflict() -> None:
    first = sample(1).model_copy(
        update={"dataset_id": "official-transactions", "dataset_version": "2026-07", "stable_row_id": "row-1"}
    )
    conflicting = sample(2, deposit=Decimal("120")).model_copy(
        update={"dataset_id": "official-transactions", "dataset_version": "2026-07", "stable_row_id": "row-1"}
    )
    third = sample(3).model_copy(
        update={"dataset_id": "official-transactions", "dataset_version": "2026-07", "stable_row_id": "row-3"}
    )
    signal = assess_transaction_signal(policy_input(samples=(first, conflicting, third)))
    assert signal.unavailable_reason is UnavailableReason.SOURCE_CONFLICT


def test_mixed_qualified_and_legacy_transaction_rows_are_not_comparable() -> None:
    qualified = sample(1).model_copy(
        update={"dataset_id": "official-transactions", "dataset_version": "2026-07", "stable_row_id": "row-1"}
    )
    signal = assess_transaction_signal(policy_input(samples=(qualified, sample(2), sample(3))))
    assert signal.unavailable_reason is UnavailableReason.NOT_COMPARABLE
def test_qualified_identity_reused_in_one_dataset_version_is_invalid_artifact() -> None:
    first = sample(1).model_copy(
        update={"dataset_id": "official-transactions", "dataset_version": "2026-07", "stable_row_id": "row-1"}
    )
    repeated = sample(2).model_copy(
        update={
            "dataset_id": "official-transactions",
            "dataset_version": "2026-07",
            "stable_row_id": "row-1",
            "snapshot_as_of": date(2026, 7, 17),
        }
    )
    third = sample(3).model_copy(
        update={"dataset_id": "official-transactions", "dataset_version": "2026-07", "stable_row_id": "row-3"}
    )
    signal = assess_transaction_signal(policy_input(samples=(first, repeated, third)))
    assert signal.unavailable_reason is UnavailableReason.INVALID_ARTIFACT


def test_transaction_unavailable_reasons_keep_freshness_and_sufficiency_independent() -> None:
    unsupported = tuple(
        sample(number).model_copy(update={"property_type": "unsupported"})
        for number in range(1, 4)
    )
    unsupported_signal = assess_transaction_signal(policy_input(samples=unsupported))
    assert unsupported_signal.unavailable_reason is UnavailableReason.UNSUPPORTED
    assert unsupported_signal.is_fresh
    assert not unsupported_signal.is_sufficient

    stale_signal = assess_transaction_signal(
        policy_input(samples=(sample(1, occurred_on=date(2020, 1, 1)),))
    )
    assert stale_signal.unavailable_reason is UnavailableReason.STALE
    assert not stale_signal.is_fresh
    assert not stale_signal.is_sufficient


def test_assessment_carries_immutable_snapshot_references() -> None:
    assessment = assess_risk_policy(
        policy_input(samples=(sample(1), sample(2), sample(3)))
    )
    sample_reference = next(
        item
        for item in assessment.snapshot_evidence
        if item.evidence_id == "transaction-evidence-1"
    )
    assert sample_reference.provenance_id == "transaction-provenance-1"
def test_final_report_rejects_snapshot_provenance_substitution() -> None:
    assessment = assess_risk_policy(
        policy_input(samples=(sample(1), sample(2), sample(3)))
    )
    evidence = tuple(
        Evidence(
            evidence_id=reference.evidence_id,
            kind=reference.kind,
            source_name="Synthetic snapshot source",
            source_record_id=reference.source_record_id,
            retrieved_at=datetime(2026, 7, 18),
            snapshot_as_of=reference.snapshot_as_of,
            provenance_id=reference.provenance_id,
        )
        for reference in assessment.snapshot_evidence
    )
    corrupted = list(evidence)
    corrupted[0] = corrupted[0].model_copy(update={"provenance_id": "substituted"})
    with pytest.raises(ValidationError, match="must match policy assessment provenance"):
        FinalReport(
            report_id="report",
            session_id="session",
            generated_at=datetime(2026, 7, 18),
            status=AnalysisStatus.COMPLETED,
            overall_risk=assessment.overall_risk,
            conclusions=(assessment.conclusion,),
            claims=assessment.claims,
            checklist_items=assessment.checklist_items,
            evidence=tuple(corrupted),
            policy_snapshot_evidence=assessment.snapshot_evidence,
            limitations=("Synthetic limitation.",),
            advisory_notice=(
                "Advisory decision support only; not legal advice, a safety guarantee, "
                "or a fraud determination."
            ),
        )



def test_direct_signal_assessors_enforce_policy_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_gate() -> str:
        raise ValueError("INVALID_POLICY_ARTIFACT")

    monkeypatch.setattr(policy, "validate_policy_artifact", fail_gate)
    with pytest.raises(ValueError, match="INVALID_POLICY_ARTIFACT"):
        assess_transaction_signal(policy_input())
    with pytest.raises(ValueError, match="INVALID_POLICY_ARTIFACT"):
        assess_hug_signal(policy_input())


def test_runtime_parameter_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy, "MINIMUM_COMPARABLE_COUNT", 4)
    with pytest.raises(ValueError, match="INVALID_POLICY_ARTIFACT"):
        validate_policy_artifact()
@pytest.mark.parametrize(
    ("constant", "value"),
    [
        ("DEPOSIT_MEDIAN_HIGH_THRESHOLD", Decimal("1.10")),
        ("HUG_WINDOW_MONTHS", 19),
    ],
)
def test_all_runtime_threshold_and_window_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch, constant: str, value: object
) -> None:
    monkeypatch.setattr(policy, constant, value)
    with pytest.raises(ValueError, match="INVALID_POLICY_ARTIFACT"):
        validate_policy_artifact()
def test_synchronized_hug_threshold_artifact_drift_fails_closed(tmp_path: object) -> None:
    path = tmp_path / "risk-policy-v1.yaml"
    text = policy.POLICY_PATH.read_text(encoding="utf-8")
    match = re.search(r"    canonical_input: '(.*)'\n", text)
    assert match is not None
    mapping = json.loads(match.group(1))
    mapping["signals"]["R-HUG-REGION"]["thresholds"]["high"] = "target >= P80"
    canonical = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text = text.replace('high: "target >= P75"', 'high: "target >= P80"')
    text = text[: match.start(1)] + canonical + text[match.end(1) :]
    text = re.sub(
        r"(    value: )[a-f0-9]{64}",
        rf"\g<1>{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}",
        text,
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="INVALID_POLICY_ARTIFACT"):
        validate_policy_artifact(path)


def test_empty_transaction_response_is_fresh_no_match() -> None:
    signal = assess_transaction_signal(policy_input())
    assert signal.unavailable_reason is UnavailableReason.NO_MATCH
    assert signal.is_fresh
    assert not signal.is_sufficient


def test_stale_comparables_are_sufficient_but_not_fresh() -> None:
    stale = tuple(
        sample(number, occurred_on=date(2020, 1, 1)) for number in range(1, 4)
    )
    signal = assess_transaction_signal(policy_input(samples=stale))
    assert signal.unavailable_reason is UnavailableReason.STALE
    assert not signal.is_fresh
    assert signal.is_sufficient
def test_missing_transaction_retrieval_is_not_a_successful_no_match() -> None:
    signal = assess_transaction_signal(
        RiskPolicyInput(
            listing=listing(),
            transaction_retrieval_status=TransactionRetrievalStatus.MISSING,
            as_of=AS_OF,
        )
    )
    assert signal.unavailable_reason is UnavailableReason.MISSING
    assert not signal.is_fresh
    assert not signal.is_sufficient





def test_synchronized_artifact_parameter_drift_fails_closed(tmp_path: object) -> None:
    path = tmp_path / "risk-policy-v1.yaml"
    text = policy.POLICY_PATH.read_text(encoding="utf-8")
    match = re.search(r"    canonical_input: '(.*)'\n", text)
    assert match is not None
    mapping = json.loads(match.group(1))
    mapping["signals"]["R-DEPOSIT-RATIO"]["thresholds"]["high"] = ">= 1.10"
    canonical = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    text = text[: match.start(1)] + canonical + text[match.end(1) :]
    text = re.sub(
        r"(    value: )[a-f0-9]{64}",
        rf"\g<1>{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}",
        text,
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="INVALID_POLICY_ARTIFACT"):
        validate_policy_artifact(path)


def test_grade_signal_state_invariants() -> None:
    kwargs = {
        "signal_id": "signal",
        "has_signal": False,
        "evidence_ids": ("evidence",),
        "rationale": "Synthetic rationale.",
    }
    with pytest.raises(ValidationError, match="fresh and sufficient"):
        GradeSignal(level=RiskLevel.LOW, is_fresh=False, is_sufficient=True, **kwargs)
    with pytest.raises(ValidationError, match="require a data-quality reason"):
        GradeSignal(level=RiskLevel.UNAVAILABLE, is_fresh=False, is_sufficient=False, **kwargs)


def test_snapshot_evidence_requires_and_preserves_provenance() -> None:
    kwargs = {
        "evidence_id": "evidence",
        "kind": EvidenceKind.TRANSACTION_RECORD,
        "source_name": "Synthetic source",
        "source_record_id": "record",
        "retrieved_at": datetime(2026, 7, 18),
    }
    with pytest.raises(ValidationError, match="requires snapshot_as_of"):
        Evidence(**kwargs)
    evidence = Evidence(
        **kwargs,
        snapshot_as_of=AS_OF,
        provenance_id="snapshot-provenance",
    )
    assert evidence.provenance_id == "snapshot-provenance"

@pytest.mark.parametrize(
    "excluded_update",
    [
        {"monthly_rent_won": Decimal("1")},
        {"cancelled": True},
        {"renewal": True},
        {"snapshot_as_of": AS_OF + timedelta(days=1)},
        {"property_type": "unsupported"},
    ],
)
def test_transaction_jeonse_only_and_snapshot_dimensions_are_enforced(
    excluded_update: dict[str, object],
) -> None:
    excluded = sample(3).model_copy(update=excluded_update)
    signal = assess_transaction_signal(
        policy_input(samples=(sample(1), sample(2), excluded))
    )
    assert signal.level is RiskLevel.UNAVAILABLE


def test_hug_requires_twenty_valid_reference_rows() -> None:
    assert assess_hug_signal(
        policy_input(hug=hug_input(10, tuple(range(1, 20))))
    ).level is RiskLevel.UNAVAILABLE
    assert assess_hug_signal(
        policy_input(hug=hug_input(0, tuple(range(1, 21))))
    ).level is RiskLevel.LOW


@pytest.mark.parametrize(
    ("subject_count", "expected"),
    [
        (0, RiskLevel.LOW),
        (10, RiskLevel.MEDIUM),
        (15, RiskLevel.HIGH),
    ],
)
def test_hug_nearest_rank_boundaries(subject_count: int, expected: RiskLevel) -> None:
    assert assess_hug_signal(
        policy_input(hug=hug_input(subject_count, tuple(range(1, 21))))
    ).level is expected


def test_hug_p50_and_p75_boundaries_are_inclusive() -> None:
    counts = tuple(range(1, 21))
    assert assess_hug_signal(policy_input(hug=hug_input(10, counts))).level is RiskLevel.MEDIUM
    assert assess_hug_signal(policy_input(hug=hug_input(15, counts))).level is RiskLevel.HIGH


def test_hug_tied_reference_rates_are_unavailable() -> None:
    assert assess_hug_signal(
        policy_input(hug=hug_input(5, (5,) * 20))
    ).level is RiskLevel.UNAVAILABLE


def test_hug_zero_rate_with_positive_denominator_is_valid() -> None:
    counts = (0,) + tuple(range(1, 20))
    assert assess_hug_signal(
        policy_input(hug=hug_input(0, counts))
    ).level is RiskLevel.LOW


def test_hug_percentiles_rank_rates_with_varying_denominators() -> None:
    subject = hug_statistic(0, 10, eligible_contract_count=100)
    rows = tuple(
        hug_statistic(number, 10, eligible_contract_count=100 + number * 5)
        for number in range(1, 20)
    ) + (
        hug_statistic(
            20,
            10,
            geography=subject.geography,
            eligible_contract_count=100,
        ),
    )
    signal = assess_hug_signal(
        policy_input(hug=HugPolicyInput(subject=subject, reference_rows=rows))
    )
    assert signal.level is RiskLevel.HIGH


def test_hug_zero_denominator_is_rejected() -> None:
    with pytest.raises(ValidationError):
        hug_statistic(1, 0, eligible_contract_count=0)


def test_hug_reference_population_requires_subject_geography() -> None:
    subject = hug_statistic(0, 5)
    references = tuple(hug_statistic(number, number) for number in range(1, 21))
    with pytest.raises(ValidationError, match="subject geography"):
        HugPolicyInput(subject=subject, reference_rows=references)


def test_hug_subject_population_row_must_match_exact_measurement() -> None:
    base = hug_input(10, tuple(range(1, 21)))
    target_index = next(
        index
        for index, row in enumerate(base.reference_rows)
        if row.geography == base.subject.geography
    )
    mutated = list(base.reference_rows)
    mutated[target_index] = mutated[target_index].model_copy(
        update={"incident_count": base.subject.incident_count + 1}
    )
    with pytest.raises(ValidationError, match="must equal"):
        HugPolicyInput(subject=base.subject, reference_rows=tuple(mutated))


@pytest.mark.parametrize(
    "field_update",
    [
        {"source_name": "Different source"},
        {"snapshot_as_of": date(2025, 7, 30)},
        {"metric_definition": "different_rate"},
    ],
)
def test_hug_reference_population_rejects_mismatched_source_as_of_and_metric(
    field_update: dict[str, object],
) -> None:
    base = hug_input(10, tuple(range(1, 21)))
    mismatched = base.reference_rows[0].model_copy(update=field_update)
    signal = assess_hug_signal(
        policy_input(
            hug=HugPolicyInput(
                subject=base.subject,
                reference_rows=(mismatched, *base.reference_rows[1:]),
            )
        )
    )
    assert signal.level is RiskLevel.UNAVAILABLE


def test_hug_stale_and_granularity_mismatched_populations_are_unavailable() -> None:
    base = hug_input(10, tuple(range(1, 21)))
    stale_subject = base.subject.model_copy(
        update={
            "period_start": date(2024, 1, 1),
            "period_end": date(2024, 6, 30),
            "snapshot_as_of": date(2024, 6, 30),
        }
    )
    stale_rows = tuple(
        row.model_copy(
            update={
                "period_start": date(2024, 1, 1),
                "period_end": date(2024, 6, 30),
                "snapshot_as_of": date(2024, 6, 30),
            }
        )
        for row in base.reference_rows
    )
    assert assess_hug_signal(
        policy_input(hug=HugPolicyInput(subject=stale_subject, reference_rows=stale_rows))
    ).level is RiskLevel.UNAVAILABLE

    mismatch = base.reference_rows[0].model_copy(update={"granularity": "province"})
    assert assess_hug_signal(
        policy_input(
            hug=HugPolicyInput(
                subject=base.subject,
                reference_rows=(mismatch, *base.reference_rows[1:]),
            )
        )
    ).level is RiskLevel.UNAVAILABLE


@pytest.mark.parametrize(
    ("transaction_input", "hug", "transaction_level", "hug_level", "expected"),
    [
        (policy_input(samples=(sample(1), sample(2))), None, RiskLevel.UNAVAILABLE, RiskLevel.UNAVAILABLE, RiskLevel.UNAVAILABLE),
        (policy_input(samples=(sample(1), sample(2))), hug_input(0, tuple(range(1, 21))), RiskLevel.UNAVAILABLE, RiskLevel.LOW, RiskLevel.UNAVAILABLE),
        (policy_input(samples=(sample(1), sample(2))), hug_input(11, tuple(range(1, 21))), RiskLevel.UNAVAILABLE, RiskLevel.MEDIUM, RiskLevel.MEDIUM),
        (policy_input(samples=(sample(1), sample(2))), hug_input(15, tuple(range(1, 21))), RiskLevel.UNAVAILABLE, RiskLevel.HIGH, RiskLevel.HIGH),
        (policy_input(samples=(sample(1), sample(2), sample(3))), None, RiskLevel.LOW, RiskLevel.UNAVAILABLE, RiskLevel.UNAVAILABLE),
        (policy_input(samples=(sample(1), sample(2), sample(3))), hug_input(0, tuple(range(1, 21))), RiskLevel.LOW, RiskLevel.LOW, RiskLevel.LOW),
        (policy_input(samples=(sample(1), sample(2), sample(3))), hug_input(11, tuple(range(1, 21))), RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.MEDIUM),
        (policy_input(samples=(sample(1), sample(2), sample(3))), hug_input(15, tuple(range(1, 21))), RiskLevel.LOW, RiskLevel.HIGH, RiskLevel.HIGH),
        (policy_input(deposit=Decimal("90"), samples=(sample(1), sample(2), sample(3))), None, RiskLevel.MEDIUM, RiskLevel.UNAVAILABLE, RiskLevel.MEDIUM),
        (policy_input(deposit=Decimal("90"), samples=(sample(1), sample(2), sample(3))), hug_input(0, tuple(range(1, 21))), RiskLevel.MEDIUM, RiskLevel.LOW, RiskLevel.MEDIUM),
        (policy_input(deposit=Decimal("90"), samples=(sample(1), sample(2), sample(3))), hug_input(11, tuple(range(1, 21))), RiskLevel.MEDIUM, RiskLevel.MEDIUM, RiskLevel.MEDIUM),
        (policy_input(deposit=Decimal("90"), samples=(sample(1), sample(2), sample(3))), hug_input(15, tuple(range(1, 21))), RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.HIGH),
        (policy_input(deposit=Decimal("100"), samples=(sample(1), sample(2), sample(3))), None, RiskLevel.HIGH, RiskLevel.UNAVAILABLE, RiskLevel.HIGH),
        (policy_input(deposit=Decimal("100"), samples=(sample(1), sample(2), sample(3))), hug_input(0, tuple(range(1, 21))), RiskLevel.HIGH, RiskLevel.LOW, RiskLevel.HIGH),
        (policy_input(deposit=Decimal("100"), samples=(sample(1), sample(2), sample(3))), hug_input(11, tuple(range(1, 21))), RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.HIGH),
        (policy_input(deposit=Decimal("100"), samples=(sample(1), sample(2), sample(3))), hug_input(15, tuple(range(1, 21))), RiskLevel.HIGH, RiskLevel.HIGH, RiskLevel.HIGH),
    ],
)
def test_overall_risk_aggregates_every_signal_pair_symmetrically(
    transaction_input: RiskPolicyInput,
    hug: HugPolicyInput | None,
    transaction_level: RiskLevel,
    hug_level: RiskLevel,
    expected: RiskLevel,
) -> None:
    assessment = assess_risk_policy(
        RiskPolicyInput(
            listing=transaction_input.listing,
            transaction_samples=transaction_input.transaction_samples,
            hug=hug,
            as_of=AS_OF,
        )
    )
    assert assessment.transaction_signal.level is transaction_level
    assert assessment.hug_signal.level is hug_level
    assert assessment.overall_risk is expected
