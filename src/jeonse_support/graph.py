"""Sequential LangGraph orchestration for the four observable advisory roles."""
from __future__ import annotations

from typing import Annotated, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import contract_prep_agent, fit_agent, risk_agent, supervisor_agent
from .models import AgentResult, Evidence, ListingConditions, UserConditions
from .policy import RiskPolicyAssessment, RiskPolicyInput
from .repositories import OfficialDocument, OfficialDocumentQuery


def _append(left: tuple[AgentResult, ...], right: tuple[AgentResult, ...]) -> tuple[AgentResult, ...]:
    return left + right


class WorkflowState(TypedDict):
    listing: ListingConditions
    conditions: UserConditions | None
    risk_input: RiskPolicyInput
    results: Annotated[tuple[AgentResult, ...], _append]
    assessment: RiskPolicyAssessment
    official_evidence: tuple[Evidence, ...]


def build_workflow(
    retrieve: Callable[[OfficialDocumentQuery], Awaitable[tuple[OfficialDocument, ...]]] | None = None,
):
    """Build START → fit → risk → contract-prep → supervisor → END, without LLM control flow."""
    graph = StateGraph(WorkflowState)

    def fit(state: WorkflowState) -> dict[str, tuple[AgentResult, ...]]:
        return {"results": (fit_agent(state["listing"], state.get("conditions")),)}

    def risk(state: WorkflowState) -> dict[str, object]:
        result, assessment = risk_agent(state["risk_input"])
        return {"results": (result,), "assessment": assessment}

    async def contract(state: WorkflowState) -> dict[str, object]:
        result, evidence = await contract_prep_agent(state["listing"], retrieve)
        return {"results": (result,), "official_evidence": evidence}

    def supervisor(state: WorkflowState) -> dict[str, tuple[AgentResult, ...]]:
        return {"results": (supervisor_agent(state["results"]),)}

    graph.add_node("fit", fit)
    graph.add_node("risk", risk)
    graph.add_node("contract-prep", contract)
    graph.add_node("supervisor", supervisor)
    graph.add_edge(START, "fit")
    graph.add_edge("fit", "risk")
    graph.add_edge("risk", "contract-prep")
    graph.add_edge("contract-prep", "supervisor")
    graph.add_edge("supervisor", END)
    return graph.compile()
