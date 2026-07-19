"""Four bounded, fact-first workflow roles.  They do not retain prompts or reasoning."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Awaitable, Callable

from .models import (
    AgentEvent, AgentEventKind, AgentResult, AnalysisStatus, ChecklistItem,
    ChecklistStatus, ClaimKind, Evidence, EvidenceKind, ListingConditions, TypedClaim,
    UserConditions,
)
from .policy import RiskPolicyAssessment, RiskPolicyInput, assess_risk_policy
from .repositories import OfficialDocument, OfficialDocumentQuery

ADVISORY = "Advisory decision support only; not legal advice, a safety guarantee, or a fraud determination."


def _event(agent: str, kind: AgentEventKind, suffix: str, evidence_ids: tuple[str, ...] = ()) -> AgentEvent:
    """Events contain stable codes only, never prompts, user instructions, or chain of thought."""
    return AgentEvent(event_id=f"{agent}-{suffix}", agent_name=agent, kind=kind,
                      occurred_at=datetime.now(UTC), message=f"{agent.upper()}_{kind.value.upper()}", evidence_ids=evidence_ids)


def fit_agent(
    listing: ListingConditions, conditions: UserConditions | None = None,
) -> AgentResult:
    """Compare only canonical submitted conditions against the selected snapshot."""
    if conditions is None:
        claim = TypedClaim(
            claim_id="FIT-conditions-unavailable",
            kind=ClaimKind.LIMITATION,
            statement="No canonical user conditions were submitted, so no fit comparison is available.",
            rule_ids=("FIT-conditions-v1",),
            evidence_ids=(listing.evidence_id,),
        )
        claims = (claim,)
    else:
        claims: list[TypedClaim] = []
        if conditions.region is not None:
            matches = conditions.region.casefold() in listing.address_text.casefold()
            claims.append(TypedClaim(
                claim_id="FIT-region",
                kind=ClaimKind.FACT,
                statement=(
                    f"Submitted region '{conditions.region}' {'matches' if matches else 'does not match'} "
                    f"the selected snapshot address."
                ),
                rule_ids=("FIT-region-v1",),
                evidence_ids=(listing.evidence_id,),
            ))
        if conditions.max_deposit_won is not None:
            difference = listing.deposit_won - conditions.max_deposit_won
            claims.append(TypedClaim(
                claim_id="FIT-max-deposit-won",
                kind=ClaimKind.FACT,
                statement=(
                    f"Snapshot deposit differs from the submitted maximum by {difference} won "
                    f"({'within' if difference <= 0 else 'above'} the maximum)."
                ),
                rule_ids=("FIT-max-deposit-won-v1",),
                evidence_ids=(listing.evidence_id,),
            ))
        if conditions.property_types:
            matches = listing.property_type in conditions.property_types
            claims.append(TypedClaim(
                claim_id="FIT-property-type",
                kind=ClaimKind.FACT,
                statement=(
                    f"Snapshot property type '{listing.property_type}' "
                    f"{'is' if matches else 'is not'} in the submitted allowed types."
                ),
                rule_ids=("FIT-property-type-v1",),
                evidence_ids=(listing.evidence_id,),
            ))
        if conditions.min_area_sqm is not None:
            difference = listing.area_sqm - conditions.min_area_sqm
            claims.append(TypedClaim(
                claim_id="FIT-min-area-sqm",
                kind=ClaimKind.FACT,
                statement=(
                    f"Snapshot area differs from the submitted minimum by {difference} square meters "
                    f"({'meets' if difference >= 0 else 'is below'} the minimum)."
                ),
                rule_ids=("FIT-min-area-sqm-v1",),
                evidence_ids=(listing.evidence_id,),
            ))
        if conditions.max_area_sqm is not None:
            difference = listing.area_sqm - conditions.max_area_sqm
            claims.append(TypedClaim(
                claim_id="FIT-max-area-sqm",
                kind=ClaimKind.FACT,
                statement=(
                    f"Snapshot area differs from the submitted maximum by {difference} square meters "
                    f"({'within' if difference <= 0 else 'is above'} the maximum)."
                ),
                rule_ids=("FIT-max-area-sqm-v1",),
                evidence_ids=(listing.evidence_id,),
            ))
        if not claims:
            claims.append(TypedClaim(
                claim_id="FIT-no-active-constraints",
                kind=ClaimKind.LIMITATION,
                statement="Canonical conditions contained no active constraints for fit comparison.",
                rule_ids=("FIT-conditions-v1",),
                evidence_ids=(listing.evidence_id,),
            ))
    return AgentResult(
        agent_name="fit",
        status=AnalysisStatus.COMPLETED,
        claims=tuple(claims),
        evidence_ids=(listing.evidence_id,),
        events=(_event("fit", AgentEventKind.COMPLETED, "done", (listing.evidence_id,)),),
    )


def risk_agent(input: RiskPolicyInput) -> tuple[AgentResult, RiskPolicyAssessment]:
    assessment = assess_risk_policy(input)
    result = AgentResult(agent_name="risk", status=AnalysisStatus.COMPLETED,
        claims=assessment.claims, checklist_items=assessment.checklist_items,
        evidence_ids=assessment.conclusion.evidence_ids,
        events=(_event("risk", AgentEventKind.COMPLETED, "done", assessment.conclusion.evidence_ids),))
    return result, assessment


async def contract_prep_agent(
    listing: ListingConditions,
    retrieve: Callable[[OfficialDocumentQuery], Awaitable[tuple[OfficialDocument, ...]]] | None,
) -> tuple[AgentResult, tuple[Evidence, ...]]:
    documents: tuple[OfficialDocument, ...] = ()
    if retrieve is not None:
        documents = await retrieve(
            OfficialDocumentQuery(query_text="jeonse contract checklist", max_results=1)
        )
    if documents:
        evidence = tuple(document.evidence for document in documents)
        evidence_ids = tuple(item.evidence_id for item in evidence)
        item = ChecklistItem(checklist_id="CHK-official-document", status=ChecklistStatus.REVIEW,
            label="Confirm official contract checklist items", rationale="Review the cited official checklist before signing.",
            rule_ids=("CHK-contract-prep-v1",), evidence_ids=evidence_ids)
    else:
        unavailable_evidence = Evidence(
            evidence_id="evidence-official-document-unavailable",
            kind=EvidenceKind.AGENT_OBSERVATION,
            source_name="official document retrieval",
            source_record_id="official-document-retrieval",
            retrieved_at=datetime.now(UTC),
            excerpt="No scoped official checklist evidence was retrieved.",
        )
        evidence = (unavailable_evidence,)
        evidence_ids = (unavailable_evidence.evidence_id,)
        rationale = (
            "Official-document retrieval is unavailable in this snapshot."
            if retrieve is None
            else "No scoped official checklist evidence was available."
        )
        item = ChecklistItem(checklist_id="CHK-official-document-unavailable", status=ChecklistStatus.UNAVAILABLE,
            label="Official contract checklist", rationale=rationale,
            rule_ids=("CHK-contract-prep-v1",), evidence_ids=evidence_ids)
    result = AgentResult(agent_name="contract-prep", status=AnalysisStatus.COMPLETED,
        checklist_items=(item,), evidence_ids=evidence_ids,
        events=(_event("contract-prep", AgentEventKind.COMPLETED, "done", evidence_ids),))
    return result, evidence


def supervisor_agent(results: tuple[AgentResult, ...]) -> AgentResult:
    evidence_ids = tuple(dict.fromkeys(e for result in results for e in result.evidence_ids))
    claim = TypedClaim(claim_id="supervisor-advisory-boundary", kind=ClaimKind.LIMITATION,
        statement=ADVISORY, rule_ids=("SUPERVISOR-advisory-boundary-v1",), evidence_ids=evidence_ids)
    return AgentResult(agent_name="supervisor", status=AnalysisStatus.COMPLETED, claims=(claim,), evidence_ids=evidence_ids,
        events=(_event("supervisor", AgentEventKind.COMPLETED, "done", evidence_ids),))
