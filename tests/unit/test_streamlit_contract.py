from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

import streamlit_app as ui
from jeonse_support.api import AnalysisJob
from jeonse_support.models import AnalysisStatus, ApiErrorCode
from jeonse_support.service import AnalysisRecord, AnalysisRequest


def terminal(status: str, code: str, retryable: bool) -> dict[str, object]:
    return {
        "analysis_id": "analysis-1",
        "session_id": "session-1",
        "status": status,
        "created_at": datetime(2026, 7, 18, tzinfo=UTC).isoformat(),
        "error": {
            "code": code,
            "message": "bounded terminal error",
            "retryable": retryable,
            "details": [],
        },
    }


def test_terminal_failure_contract_binds_status_code_and_retryability() -> None:
    assert ui.valid_terminal_job(terminal("failed", "ANALYSIS_TIMEOUT", True))
    assert ui.valid_terminal_job(terminal("failed", "AI_PROVIDER_FAILED", True))
    assert ui.valid_terminal_job(
        terminal("needs_review", "EVIDENCE_GATE_FAILED", False)
    )

    assert not ui.valid_terminal_job(terminal("needs_review", "ANALYSIS_TIMEOUT", True))
    assert not ui.valid_terminal_job(terminal("failed", "ANALYSIS_TIMEOUT", False))
    assert not ui.valid_terminal_job(
        terminal("failed", "EVIDENCE_GATE_FAILED", False)
    )
    assert not ui.valid_terminal_job(
        terminal("needs_review", "EVIDENCE_GATE_FAILED", True)
    )
    assert not ui.valid_terminal_job(
        {**terminal("failed", "ANALYSIS_FAILED", False), "unexpected": "field"}
    )
def test_terminal_completion_contract_requires_report_and_accepts_fallback() -> None:
    completed = {
        "analysis_id": "analysis-1",
        "session_id": "session-1",
        "status": "completed",
        "created_at": datetime(2026, 7, 18, tzinfo=UTC).isoformat(),
        "report_id": "report-1",
    }
    assert ui.valid_terminal_job(completed)
    assert ui.valid_terminal_job({
        **completed,
        "status": "completed_with_external_fallback",
    })
    assert not ui.valid_terminal_job({key: value for key, value in completed.items() if key != "report_id"})
    assert not ui.valid_terminal_job({**completed, "error": terminal("failed", "ANALYSIS_FAILED", False)["error"]})
def test_api_terminal_poll_payloads_are_accepted_without_ignoring_extra_fields() -> None:
    created_at = datetime(2026, 7, 18, tzinfo=UTC)
    request = AnalysisRequest(session_id="session-1", listing_id="listing-mapogu-low")

    class Service:
        def __init__(self, record: AnalysisRecord | None) -> None:
            self.record = record

        def get_analysis(self, _analysis_id: str) -> AnalysisRecord | None:
            return self.record

    cases = (
        (
            "needs_review",
            "EVIDENCE_GATE_FAILED",
            AnalysisRecord(
                analysis_id="analysis-1",
                session_id="session-1",
                status=AnalysisStatus.NEEDS_REVIEW,
                created_at=created_at,
                error_code=ApiErrorCode.ANALYSIS_FAILED,
                ai_trace_codes=("INVALID_REPORT_EVIDENCE",),
            ),
        ),
        (
            "failed",
            "AI_PROVIDER_FAILED",
            AnalysisRecord(
                analysis_id="analysis-1",
                session_id="session-1",
                status=AnalysisStatus.FAILED,
                created_at=created_at,
                error_code=ApiErrorCode.AI_PROVIDER_FAILED,
                ai_trace_codes=("PROVIDER_UNAVAILABLE",),
            ),
        ),
        ("failed", "ANALYSIS_TIMEOUT", None),
        ("failed", "SERVER_SHUTDOWN", None),
    )
    for status, error_code, record in cases:
        job = AnalysisJob(
            analysis_id="analysis-1",
            request=request,
            fingerprint="fingerprint",
            idempotency_key=None,
            created_at=created_at,
            status=status,
            service_analysis_id="analysis-1" if record else None,
            error_code=error_code,
        )
        payload = job.public(Service(record))
        assert ui.valid_terminal_job(payload)
        assert payload["error"]["code"] == error_code
        assert "agent_results" in payload


def test_poll_renders_fallback_report_with_disclosure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SessionState(dict[str, object]):
        def __setattr__(self, name: str, value: object) -> None:
            self[name] = value
    job = {
        "analysis_id": "analysis-1",
        "session_id": "session-1",
        "status": "completed_with_external_fallback",
        "created_at": datetime(2026, 7, 18, tzinfo=UTC).isoformat(),
        "report_id": "report-1",
    }
    report = {"evidence": [{"snapshot_as_of": "2026-06-15"}]}
    requests: list[str] = []
    warnings: list[str] = []
    captions: list[str] = []
    rendered: list[dict[str, object]] = []
    state = SessionState()
    monkeypatch.setattr(ui.st, "session_state", state)
    monkeypatch.setattr(
        ui,
        "api_request",
        lambda _method, path: requests.append(path)
        or (job if path.endswith("analysis-1") else report),
    )
    monkeypatch.setattr(ui, "render_events", lambda _job: None)
    monkeypatch.setattr(ui, "valid_report", lambda _report: True)
    monkeypatch.setattr(ui, "render_report", rendered.append)
    monkeypatch.setattr(ui.st, "caption", captions.append)
    monkeypatch.setattr(ui.st, "warning", warnings.append)

    ui.poll_and_render("analysis-1")

    assert requests == ["/api/v1/analyses/analysis-1", "/api/v1/analyses/analysis-1/report"]
    assert rendered == [report]
    assert warnings
    assert any("2026-06-15" in caption for caption in captions)




@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(204), True),
        (
            httpx.Response(
                404,
                json={
                    "detail": {
                        "code": "SESSION_NOT_FOUND",
                        "message": "Session was not found.",
                        "retryable": False,
                    }
                },
            ),
            True,
        ),
        (
            httpx.Response(
                404,
                json={
                    "detail": {
                        "code": "ROUTE_NOT_FOUND",
                        "message": "Wrong endpoint.",
                        "retryable": False,
                    }
                },
            ),
            False,
        ),
    ],
)
def test_server_reset_only_accepts_confirmed_reset_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
    expected: bool,
) -> None:
    errors: list[str] = []
    monkeypatch.setattr(ui.httpx, "request", lambda *_args, **_kwargs: response)
    monkeypatch.setattr(ui.st, "error", errors.append)

    assert ui.reset_server_session("session-1") is expected
    assert bool(errors) is not expected


def test_server_reset_preserves_state_on_network_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []

    def fail(*_args: object, **_kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(ui.httpx, "request", fail)
    monkeypatch.setattr(ui.st, "error", errors.append)

    assert not ui.reset_server_session("session-1")
    assert errors
@pytest.mark.parametrize(
    ("status", "report_id", "error_code", "valid"),
    [
        ("completed", "report-1", None, True),
        ("completed_with_external_fallback", "report-1", None, True),
        ("completed", None, None, False),
        ("completed", "report-1", "analysis_failed", False),
        ("needs_review", None, "analysis_failed", True),
        ("needs_review", "report-1", "analysis_failed", False),
        ("failed", None, "analysis_failed", True),
        ("failed", "report-1", "analysis_failed", False),
    ],
)
def test_analysis_record_status_report_error_matrix(
    status: str, report_id: str | None, error_code: str | None, valid: bool,
) -> None:
    payload = {
        "analysis_id": "analysis-1",
        "session_id": "session-1",
        "status": status,
        "created_at": datetime(2026, 7, 18, tzinfo=UTC).isoformat(),
        "report_id": report_id,
        "error_code": error_code,
    }
    assert ui.valid_analysis(payload) is valid
