"""Public HTTP integration coverage for the bounded advisory job workflow."""
from __future__ import annotations

import asyncio
import json
import time
import threading
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
import jeonse_support.api as api_module
import jeonse_support.service as service_module

from jeonse_support.api import _snapshot_adapters, create_app
from jeonse_support.models import (
    AnalysisStatus,
    FitLevel,
    NormalizedUserConditions,
    RetainedListing,
    RiskLevel,
    SessionMemorySummary,
    TemporalProvenance,
)
from jeonse_support.service import AnalysisRecord
from jeonse_support.rag import AllowlistedOfficialDocumentRepository, SemanticOfficialDocumentRepository


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("JEONSE_DATA_MODE", raising=False)
    for name in (
        "AOAI_API_KEY", "AOAI_ENDPOINT", "AOAI_DEPLOY_GPT41_MINI", "AOAI_MODEL_GPT41_MINI",
        "AOAI_DEPLOY_EMBED_3_SMALL", "AOAI_MODEL_EMBED_3_SMALL",
    ):
        monkeypatch.delenv(name, raising=False)
    with TestClient(create_app()) as test_client:
        yield test_client


def _completed(client: TestClient, analysis_id: str) -> dict[str, object]:
    for _ in range(100):
        response = client.get(f"/api/v1/analyses/{analysis_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] not in {"queued", "running"}:
            return body
        time.sleep(0.01)
    pytest.fail("analysis did not reach a terminal state")


def test_snapshot_catalog_checksum_rejects_tampering(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2] / "data" / "catalog" / "listings.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["listings"][0]["deposit_won"] += 1
    catalog_dir = tmp_path / "data" / "catalog"
    catalog_dir.mkdir(parents=True)
    (catalog_dir / "listings.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="INVALID_SNAPSHOT_CATALOG_CHECKSUM"):
        _snapshot_adapters(tmp_path)


def test_snapshot_job_admission_poll_report_idempotency_and_delete(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["data_mode"] == "snapshot"
    assert health.json()["offline_demo"] is True
    assert isinstance(client.app.state.analysis_service.adapters.documents, AllowlistedOfficialDocumentRepository)
    unknown_analysis = client.get("/api/v1/analyses/unknown-analysis")
    assert unknown_analysis.status_code == 404
    assert unknown_analysis.json()["detail"]["code"] == "ANALYSIS_NOT_FOUND"
    unknown_report = client.get("/api/v1/analyses/unknown-analysis/report")
    assert unknown_report.status_code == 404
    assert unknown_report.json()["detail"]["code"] == "ANALYSIS_NOT_FOUND"

    request = {"session_id": "snapshot-job-session", "listing_id": "listing-mapogu-low"}
    admitted = client.post("/api/v1/analyses", json=request, headers={"Idempotency-Key": "request-1"})
    assert admitted.status_code == 202
    queued = admitted.json()
    assert queued["status"] in {"queued", "running"}
    analysis_id = queued["analysis_id"]

    replay = client.post("/api/v1/analyses", json=request, headers={"Idempotency-Key": "request-1"})
    assert replay.status_code == 202
    assert replay.json()["analysis_id"] == analysis_id
    collision = client.post(
        "/api/v1/analyses",
        json={"session_id": "snapshot-job-session", "listing_id": "listing-songpagu-medium"},
        headers={"Idempotency-Key": "request-1"},
    )
    assert collision.status_code == 409
    assert collision.json()["detail"]["code"] == "IDEMPOTENCY_KEY_COLLISION"

    terminal = _completed(client, analysis_id)
    assert terminal["status"] == "completed"
    assert terminal["report_id"]
    report = client.get(f"/api/v1/analyses/{analysis_id}/report")
    assert report.status_code == 200
    assert report.json()["session_id"] == "snapshot-job-session"
    follow_up = client.post(
        "/api/v1/sessions/snapshot-job-session/follow-ups",
        json={"operation": "clarify", "target_report_id": report.json()["report_id"]},
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["operation"] == "clarify"

    reset = client.post("/api/v1/sessions/snapshot-job-session/reset")
    assert reset.status_code == 204
    missing_analysis = client.get(f"/api/v1/analyses/{analysis_id}")
    assert missing_analysis.status_code == 404
    assert missing_analysis.json()["detail"]["code"] == "ANALYSIS_NOT_FOUND"
    missing_report = client.get(f"/api/v1/analyses/{analysis_id}/report")
    assert missing_report.status_code == 404
    assert missing_report.json()["detail"]["code"] == "ANALYSIS_NOT_FOUND"
    second = client.post("/api/v1/analyses", json=request)
    assert second.status_code == 202
    _completed(client, second.json()["analysis_id"])
    deleted = client.delete("/api/v1/sessions/snapshot-job-session")
    assert deleted.status_code == 204
    assert client.delete("/api/v1/sessions/snapshot-job-session").status_code == 404
def test_pii_requests_are_rejected_without_creating_jobs_or_session_memory(
    client: TestClient,
) -> None:
    service = client.app.state.analysis_service
    for request in (
        {
            "session_id": "010-1234-5678",
            "listing_id": "listing-mapogu-low",
        },
        {
            "session_id": "pii-conditions",
            "listing_id": "listing-mapogu-low",
            "conditions": {"region": "contact@example.com"},
        },
    ):
        rejected = client.post("/api/v1/analyses", json=request)
        assert rejected.status_code == 422
        assert rejected.json()["detail"]["code"] == "PII_NOT_ALLOWED"

    assert client.app.state._jobs == {}
    assert client.app.state._requests == {}
    assert service._records == {}
    assert service._reports == {}
    assert service._sessions == {}

def test_arbitrary_property_type_is_rejected_without_retained_state(client: TestClient) -> None:
    service = client.app.state.analysis_service
    rejected = client.post(
        "/api/v1/analyses",
        json={
            "session_id": "typed-property-session",
            "listing_id": "listing-mapogu-low",
            "conditions": {"property_types": ["arbitrary-free-text"]},
        },
    )
    assert rejected.status_code == 422
    assert client.app.state._jobs == {}
    assert client.app.state._requests == {}
    assert service._records == {}
    assert service._sessions == {}


def _memory_summary(session_id: str, report_id: str) -> SessionMemorySummary:
    return SessionMemorySummary(
        session_id=session_id,
        report_id=report_id,
        completed_at=datetime.now(UTC),
        report_as_of=date(2026, 6, 15),
        listing=RetainedListing(
            listing_id="listing-mapogu-low",
            legal_dong="마포구",
            deposit_won=Decimal("100000000"),
            area_sqm=Decimal("50.0"),
            contract_date=date(2026, 6, 15),
            property_type="아파트",
        ),
        conditions=NormalizedUserConditions(),
        fit_level=FitLevel.UNAVAILABLE,
        risk_level=RiskLevel.LOW,
    )


def test_session_memory_retention_is_cas_safe_bounded_and_pauses_while_leased(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = client.app.state.analysis_service
    session_id = "memory-lifecycle"
    for number in range(6):
        assert service._store_session_summary(
            session_id,
            _memory_summary(session_id, f"report-{number}"),
            number,
        )
    summaries = service._sessions[session_id]
    assert [summary.report_id for summary in summaries] == [
        "report-1", "report-2", "report-3", "report-4", "report-5",
    ]
    assert service._session_container_bytes(summaries) <= 4_096

    assert service.acquire_session_lease(session_id, "in-flight-job")
    prior_version = service._session_versions[session_id]
    service.clear_session_memory(session_id)
    assert not service._store_session_summary(
        session_id, _memory_summary(session_id, "stale-write"), prior_version
    )
    assert service._session_versions[session_id] == prior_version + 1
    service.release_session_lease(session_id, "in-flight-job")
    assert session_id not in service._sessions
    assert session_id not in service._session_accessed_at
    assert session_id not in service._session_versions
    assert session_id not in service._active_sessions
    assert session_id not in service._session_lease_owners

    oversized = _memory_summary("oversized", "oversized")
    oversized = oversized.model_copy(update={
        "temporal_provenance": tuple(
            TemporalProvenance(
                signal_id=f"signal-{number}",
                source_id="x" * 128,
                evidence_as_of=date(2026, 6, 15),
            )
            for number in range(40)
        ),
    })
    assert service._session_container_bytes((oversized,)) > 4_096
    assert not service._store_session_summary("oversized", oversized, 0)
    assert "oversized" not in service._sessions

    assert service._store_session_summary(
        session_id, _memory_summary(session_id, "expired-before-admission"), 0
    )
    service._session_accessed_at[session_id] = (
        datetime.now(UTC) - service_module.RETENTION_TTL - timedelta(seconds=1)
    )
    assert service.acquire_session_lease(session_id, "expired-job")
    assert session_id not in service._sessions
    assert session_id not in service._session_accessed_at
    assert session_id not in service._session_versions
    service.release_session_lease(session_id, "expired-job")

    assert service._store_session_summary(
        session_id, _memory_summary(session_id, "valid-at-admission"), 0
    )
    service._session_accessed_at[session_id] = (
        datetime.now(UTC) - service_module.RETENTION_TTL + timedelta(seconds=1)
    )
    assert service.acquire_session_lease(session_id, "queued-job")
    service._session_accessed_at[session_id] = (
        datetime.now(UTC) - service_module.RETENTION_TTL - timedelta(seconds=1)
    )
    service._prune()
    assert session_id in service._sessions
    service.release_session_lease(session_id, "queued-job")
    assert service._session_accessed_at[session_id] > datetime.now(UTC) - timedelta(seconds=1)

    service._session_accessed_at[session_id] = (
        datetime.now(UTC) - service_module.RETENTION_TTL - timedelta(seconds=1)
    )
    service._prune()
    assert session_id not in service._sessions
    assert session_id not in service._session_accessed_at
    assert session_id not in service._session_versions
    assert session_id not in service._active_sessions
    assert session_id not in service._session_lease_owners

    active = "active-session"
    inactive = "inactive-session"
    assert service._store_session_summary(active, _memory_summary(active, "active"), 0)
    assert service._store_session_summary(inactive, _memory_summary(inactive, "inactive"), 0)
    assert service.acquire_session_lease(active, "active-job")
    monkeypatch.setattr(service_module, "MAX_RETAINED_SESSIONS", 1)
    service._prune()
    assert active in service._sessions
    service.release_session_lease(active, "active-job")
def test_expiry_and_session_deletion_leave_no_residual_session_metadata(
    client: TestClient,
) -> None:
    service = client.app.state.analysis_service
    for number in range(4):
        session_id = f"expired-{number}"
        assert service._store_session_summary(
            session_id, _memory_summary(session_id, f"report-{number}"), 0
        )
        service._session_accessed_at[session_id] = (
            datetime.now(UTC) - service_module.RETENTION_TTL - timedelta(seconds=1)
        )
        service._prune()
        assert session_id not in service._sessions
        assert session_id not in service._session_accessed_at
        assert session_id not in service._session_versions
        assert session_id not in service._active_sessions
        assert session_id not in service._session_lease_owners

    for number, endpoint in enumerate(("reset", "delete", "reset", "delete")):
        session_id = f"deleted-{number}"
        assert service._store_session_summary(
            session_id, _memory_summary(session_id, f"report-{number}"), 0
        )
        response = (
            client.post(f"/api/v1/sessions/{session_id}/reset")
            if endpoint == "reset"
            else client.delete(f"/api/v1/sessions/{session_id}")
        )
        assert response.status_code == 204
        assert session_id not in service._sessions
        assert session_id not in service._session_accessed_at
        assert session_id not in service._session_versions
        assert session_id not in service._active_sessions
        assert session_id not in service._session_lease_owners

def test_retained_follow_ups_are_session_scoped_and_temporally_complete(
    client: TestClient,
) -> None:
    session_id = "retained-temporal"
    first = client.post(
        "/api/v1/analyses",
        json={"session_id": session_id, "listing_id": "listing-mapogu-low"},
    )
    first_terminal = _completed(client, first.json()["analysis_id"])
    assert first_terminal["status"] == "completed"
    second = client.post(
        "/api/v1/analyses",
        json={"session_id": session_id, "listing_id": "listing-songpagu-medium"},
    )
    second_terminal = _completed(client, second.json()["analysis_id"])
    assert second_terminal["status"] == "completed"

    foreign = client.post(
        "/api/v1/analyses",
        json={"session_id": "other-retained-session", "listing_id": "listing-mapogu-low"},
    )
    foreign_terminal = _completed(client, foreign.json()["analysis_id"])
    assert foreign_terminal["status"] == "completed"
    denied = client.post(
        "/api/v1/sessions/other-retained-session/follow-ups",
        json={"operation": "clarify", "target_report_id": first_terminal["report_id"]},
    )
    assert denied.status_code == 404
    assert denied.json()["detail"]["code"] == "REPORT_NOT_FOUND"
    assert first_terminal["report_id"] not in denied.text
    assert "listing-mapogu-low" not in denied.text

    service = client.app.state.analysis_service
    service._reports.pop(first.json()["analysis_id"])
    service._reports.pop(second.json()["analysis_id"])
    clarify = client.post(
        f"/api/v1/sessions/{session_id}/follow-ups",
        json={"operation": "clarify", "target_report_id": first_terminal["report_id"]},
    )
    assert clarify.status_code == 200
    comparison = client.post(
        f"/api/v1/sessions/{session_id}/follow-ups",
        json={"operation": "compare", "target_report_id": first_terminal["report_id"]},
    )
    assert comparison.status_code == 200

    clarify_target = clarify.json()["target"]
    compare_target = comparison.json()["target"]
    compare_current = comparison.json()["current"]
    assert clarify_target == compare_target
    assert compare_target["report_id"] == first_terminal["report_id"]
    assert compare_current["report_id"] == second_terminal["report_id"]
    assert compare_target["temporal_provenance"] != compare_current["temporal_provenance"]
    assert compare_target["as_of"] != compare_target["completed_at"]
    assert compare_current["as_of"] != compare_current["completed_at"]
    for temporal in (compare_target, compare_current):
        provenance = temporal["temporal_provenance"]
        assert provenance
        assert all(item["evidence_as_of"] != temporal["completed_at"] for item in provenance)
        assert all(
            bool(item.get("signal_id")) or bool(item.get("checklist_id"))
            for item in provenance
        )
        assert any(item.get("transaction_window") for item in provenance)
        assert any(item.get("hug_fixed_period") for item in provenance)
        assert any(item.get("snapshot_version") for item in provenance)
        assert (
            any(item.get("corpus_version") for item in provenance)
            or any(item.get("unavailable_id") for item in provenance)
        )


def test_retained_summaries_are_bounded_and_contain_no_free_text(client: TestClient) -> None:
    service = client.app.state.analysis_service
    session_id = "bounded-retained-session"
    for number in range(6):
        admitted = client.post(
            "/api/v1/analyses",
            json={
                "session_id": session_id,
                "listing_id": (
                    "listing-mapogu-low" if number % 2 == 0 else "listing-songpagu-medium"
                ),
            },
        )
        assert _completed(client, admitted.json()["analysis_id"])["status"] == "completed"

    summaries = service._sessions[session_id]
    serialized = [
        summary.model_dump_json(exclude_none=True).encode("utf-8") for summary in summaries
    ]
    assert len(summaries) <= 5
    assert all(len(item) <= 4_096 for item in serialized)
    assert sum(len(item) for item in serialized) <= 4_096
    prohibited = {"excerpt", "quote", "message", "prompt", "summary", "address_text", "region"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    for item in serialized:
        payload = json.loads(item)
        assert not prohibited.intersection(keys(payload))
        assert b"@" not in item


def test_capacity_and_same_session_lock(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def blocked(*_args: object, **_kwargs: object) -> object:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    monkeypatch.setattr(client.app.state.analysis_service, "analyze", blocked)
    first = client.post("/api/v1/analyses", json={"session_id": "locked", "listing_id": "listing-mapogu-low"})
    assert first.status_code == 202
    locked = client.post("/api/v1/analyses", json={"session_id": "locked", "listing_id": "listing-mapogu-low"})
    assert locked.status_code == 409
    assert locked.json()["detail"]["code"] == "SESSION_ANALYSIS_IN_PROGRESS"
    for session_id in ("capacity-1", "capacity-2", "capacity-3"):
        assert client.post("/api/v1/analyses", json={"session_id": session_id, "listing_id": "listing-mapogu-low"}).status_code == 202
    capacity = client.post("/api/v1/analyses", json={"session_id": "capacity-4", "listing_id": "listing-mapogu-low"})
    assert capacity.status_code == 503
    assert capacity.json()["detail"]["code"] == "ANALYSIS_CAPACITY_EXCEEDED"
def test_job_preserves_policy_error_code(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def invalid_policy(*_args: object, **_kwargs: object) -> object:
        raise ValueError("INVALID_POLICY_ARTIFACT")

    monkeypatch.setattr(client.app.state.analysis_service, "analyze", invalid_policy)
    admitted = client.post(
        "/api/v1/analyses",
        json={"session_id": "policy-error", "listing_id": "listing-mapogu-low"},
    )
    assert admitted.status_code == 202
    terminal = _completed(client, admitted.json()["analysis_id"])
    assert terminal["status"] == "failed"
    assert terminal["error"]["code"] == "INVALID_POLICY_ARTIFACT"
    report = client.get(f"/api/v1/analyses/{admitted.json()['analysis_id']}/report")
    assert report.status_code == 409
    assert report.json()["detail"]["code"] == "INVALID_POLICY_ARTIFACT"


def test_timeout_and_pre_record_failures_publish_terminal_error_envelopes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def pre_record_failure(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("unexpected internal detail")

    monkeypatch.setattr(client.app.state.analysis_service, "analyze", pre_record_failure)
    failed = client.post(
        "/api/v1/analyses",
        json={"session_id": "pre-record-failure", "listing_id": "listing-mapogu-low"},
    )
    failed_id = failed.json()["analysis_id"]
    failed_terminal = _completed(client, failed_id)
    assert failed_terminal["status"] == "failed"
    assert failed_terminal["error"] == {
        "code": "ANALYSIS_FAILED",
        "message": "Analysis did not complete.",
        "retryable": False,
        "details": [],
    }
    assert client.get(f"/api/v1/analyses/{failed_id}/report").json()["detail"]["code"] == "ANALYSIS_FAILED"

    async def never_finishes(*_args: object, **_kwargs: object) -> object:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(api_module, "JOB_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(client.app.state.analysis_service, "analyze", never_finishes)
    timed_out = client.post(
        "/api/v1/analyses",
        json={"session_id": "timeout-failure", "listing_id": "listing-mapogu-low"},
    )
    timed_out_id = timed_out.json()["analysis_id"]
    timeout_terminal = _completed(client, timed_out_id)
    assert timeout_terminal["status"] == "failed"
    assert timeout_terminal["error"] == {
        "code": "ANALYSIS_TIMEOUT",
        "message": "Analysis exceeded the server time limit.",
        "retryable": True,
        "details": [],
    }
    assert client.get(f"/api/v1/analyses/{timed_out_id}/report").json()["detail"]["code"] == "ANALYSIS_TIMEOUT"


def test_evidence_gate_needs_review_has_safe_reason_and_no_report(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = client.app.state.analysis_service

    async def blocked_by_evidence_gate(request: object, analysis_id: str) -> AnalysisRecord:
        record = AnalysisRecord(
            analysis_id=analysis_id,
            session_id=getattr(request, "session_id"),
            status=AnalysisStatus.NEEDS_REVIEW,
            created_at=datetime.now(UTC),
        )
        service._records[analysis_id] = record
        return record

    monkeypatch.setattr(service, "analyze", blocked_by_evidence_gate)
    admitted = client.post(
        "/api/v1/analyses",
        json={"session_id": "evidence-gated", "listing_id": "listing-mapogu-low"},
    )
    analysis_id = admitted.json()["analysis_id"]
    terminal = _completed(client, analysis_id)
    assert terminal["status"] == "needs_review"
    assert "report_id" not in terminal
    assert terminal["error"] == {
        "code": "EVIDENCE_GATE_FAILED",
        "message": "Evidence validation prevented report publication.",
        "retryable": False,
        "details": [],
    }
    unavailable = client.get(f"/api/v1/analyses/{analysis_id}/report")
    assert unavailable.status_code == 409
    assert unavailable.json()["detail"]["code"] == "EVIDENCE_GATE_FAILED"
def test_completed_fallback_status_is_report_bearing_and_retains_memory(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = client.app.state.analysis_service
    analyze = service.analyze

    async def fallback_record(request: object, analysis_id: str) -> AnalysisRecord:
        completed = await analyze(request, analysis_id)
        record = completed.model_copy(update={
            "status": AnalysisStatus.COMPLETED_WITH_EXTERNAL_FALLBACK,
        })
        service._records[analysis_id] = record
        return record

    monkeypatch.setattr(service, "analyze", fallback_record)
    admitted = client.post(
        "/api/v1/analyses",
        json={"session_id": "fallback-terminal", "listing_id": "listing-mapogu-low"},
    )
    analysis_id = admitted.json()["analysis_id"]
    terminal = _completed(client, analysis_id)
    assert terminal["status"] == "completed_with_external_fallback"
    assert terminal["report_id"]
    report = client.get(f"/api/v1/analyses/{analysis_id}/report")
    assert report.status_code == 200
    assert report.json()["report_id"] == terminal["report_id"]
    summaries = service._sessions["fallback-terminal"]
    assert summaries[-1].report_id == terminal["report_id"]


def test_real_service_releases_active_marker_after_timeout_and_exception(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert api_module.JOB_TIMEOUT_SECONDS == 240
    service = client.app.state.analysis_service
    entered = threading.Event()

    async def blocked_policy_data(*_args: object, **_kwargs: object) -> object:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(api_module, "JOB_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(service, "_policy_data", blocked_policy_data)
    timed_out = client.post(
        "/api/v1/analyses",
        json={"session_id": "real-timeout", "listing_id": "listing-mapogu-low"},
    )
    timeout_id = timed_out.json()["analysis_id"]
    timeout_terminal = _completed(client, timeout_id)
    assert entered.is_set()
    assert timeout_terminal["error"]["code"] == "ANALYSIS_TIMEOUT"
    assert not service.is_session_active("real-timeout")
    assert "real-timeout" not in service._sessions
    assert client.post("/api/v1/sessions/real-timeout/reset").status_code == 204

    async def failing_policy_data(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("dependency failed")

    monkeypatch.setattr(service, "_policy_data", failing_policy_data)
    failed = client.post(
        "/api/v1/analyses",
        json={"session_id": "real-exception", "listing_id": "listing-mapogu-low"},
    )
    failed_terminal = _completed(client, failed.json()["analysis_id"])
    assert failed_terminal["error"]["code"] == "ANALYSIS_FAILED"
    assert not service.is_session_active("real-exception")
    assert "real-exception" not in service._sessions
    assert client.post("/api/v1/sessions/real-exception/reset").status_code == 204
    assert client.post(
        "/api/v1/analyses",
        json={"session_id": "real-exception", "listing_id": "listing-mapogu-low"},
    ).status_code == 202
def test_queued_time_consumes_admission_deadline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    async def block_first(request: object, analysis_id: str) -> AnalysisRecord:
        if getattr(request, "session_id") == "queue-blocker":
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.001)
        return AnalysisRecord(
            analysis_id=analysis_id,
            session_id=getattr(request, "session_id"),
            status=AnalysisStatus.FAILED,
            created_at=datetime.now(UTC),
        )

    monkeypatch.setattr(api_module, "JOB_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(client.app.state.analysis_service, "analyze", block_first)
    first = client.post(
        "/api/v1/analyses",
        json={"session_id": "queue-blocker", "listing_id": "listing-mapogu-low"},
    )
    assert entered.wait(timeout=1)
    queued = client.post(
        "/api/v1/analyses",
        json={"session_id": "queue-expired", "listing_id": "listing-mapogu-low"},
    )
    client.app.state._jobs[queued.json()["analysis_id"]].created_at -= timedelta(seconds=1)
    release.set()
    assert _completed(client, first.json()["analysis_id"])["status"] == "failed"
    terminal = _completed(client, queued.json()["analysis_id"])
    assert terminal["status"] == "failed"
    assert terminal["error"]["code"] == "ANALYSIS_TIMEOUT"




def test_terminal_api_registries_prune_with_bounded_retention_but_keep_active_work(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = client.app.state.analysis_service
    monkeypatch.setattr(api_module, "MAX_RETAINED_RECORDS", 1)
    monkeypatch.setattr(service_module, "MAX_RETAINED_RECORDS", 1)

    async def completes(request: object, analysis_id: str) -> AnalysisRecord:
        record = AnalysisRecord(
            analysis_id=analysis_id,
            session_id=getattr(request, "session_id"),
            status=AnalysisStatus.NEEDS_REVIEW,
            created_at=datetime.now(UTC),
        )
        service._records[analysis_id] = record
        service._prune()
        return record

    monkeypatch.setattr(service, "analyze", completes)
    first = client.post(
        "/api/v1/analyses",
        json={"session_id": "retained-first", "listing_id": "listing-mapogu-low"},
        headers={"Idempotency-Key": "first"},
    )
    assert _completed(client, first.json()["analysis_id"])["status"] == "needs_review"
    second = client.post(
        "/api/v1/analyses",
        json={"session_id": "retained-second", "listing_id": "listing-mapogu-low"},
        headers={"Idempotency-Key": "second"},
    )
    first_id, second_id = first.json()["analysis_id"], second.json()["analysis_id"]
    assert _completed(client, second_id)["status"] == "needs_review"
    assert client.get(f"/api/v1/analyses/{first_id}").status_code == 404
    assert client.get(f"/api/v1/analyses/{second_id}").status_code == 200

    async def stays_active(*_args: object, **_kwargs: object) -> object:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(service, "analyze", stays_active)
    active = client.post(
        "/api/v1/analyses",
        json={"session_id": "retained-active", "listing_id": "listing-mapogu-low"},
    )
    active_id = active.json()["analysis_id"]
    assert client.get(f"/api/v1/analyses/{active_id}").json()["status"] in {"queued", "running"}
    assert set(client.app.state._jobs) == {active_id}
    assert set(client.app.state._requests) == {active_id}
    assert client.app.state._idempotency == {}



def test_modes_and_partial_azure_configuration_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    azure_names = (
        "AOAI_API_KEY", "AOAI_ENDPOINT", "AOAI_DEPLOY_GPT41_MINI", "AOAI_MODEL_GPT41_MINI",
        "AOAI_DEPLOY_EMBED_3_SMALL", "AOAI_MODEL_EMBED_3_SMALL",
    )
    for name in azure_names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AOAI_API_KEY", "configured-for-test")
    with pytest.raises(ValueError, match="PARTIAL_AZURE_CONFIGURATION"):
        create_app()
    monkeypatch.delenv("AOAI_API_KEY")
    for mode in ("api", "auto"):
        monkeypatch.setenv("JEONSE_DATA_MODE", mode)
        with pytest.raises(ValueError, match="DATA_MODE_UNCONFIGURED"):
            create_app()
    monkeypatch.setenv("JEONSE_DATA_MODE", "snapshot")

    class FakeAzureProvider:
        def __init__(self, **_kwargs: object) -> None:
            pass

    monkeypatch.setattr("jeonse_support.api.AzureOpenAIProvider", FakeAzureProvider)
    for name in azure_names:
        monkeypatch.setenv(name, "configured-for-test")
    configured = create_app()
    assert isinstance(configured.state.analysis_service.adapters.documents, SemanticOfficialDocumentRepository)
