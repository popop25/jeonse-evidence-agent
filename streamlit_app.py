"""Korean Streamlit client for the advisory, snapshot-based jeonse API.

The client owns presentation state only. All catalog and analysis data are obtained through
FastAPI so that the displayed provenance and limitations remain server-authoritative.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import time
import uuid
from pathlib import Path
import sys
from typing import Any, Literal

PROJECT_SRC = Path(__file__).resolve().parent / "src"
if PROJECT_SRC.is_dir():
    sys.path.insert(0, str(PROJECT_SRC))

import httpx
import streamlit as st
from pydantic import BaseModel, ConfigDict, ValidationError

from jeonse_support.models import FinalReport, ListingConditions
from jeonse_support.service import AnalysisRecord, TerminalJobCompleted, TerminalJobFailure

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
TERMINAL_STATUSES = {
    "completed",
    "completed_with_external_fallback",
    "needs_review",
    "failed",
}
POLL_TIMEOUT_SECONDS = 255
POLL_INTERVAL_SECONDS = 1
ROLE_LABELS = {
    "fit": "1. 적합도",
    "risk": "2. 위험 참고 신호",
    "contract-prep": "3. 계약 준비",
    "supervisor": "4. 총괄 검토",
}


class AnalysisJobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    analysis_id: str
    session_id: str
    status: Literal["queued", "running"]
    created_at: datetime





def api_base_url() -> str:
    return os.getenv("JEONSE_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def api_request(method: str, path: str, **kwargs: Any) -> Any | None:
    """Make one visible, bounded API request without falling back to local data."""
    try:
        response = httpx.request(method, f"{api_base_url()}{path}", timeout=15.0, **kwargs)
    except httpx.RequestError as exc:
        st.error(f"API 네트워크 오류: {exc}. FastAPI 주소와 연결 상태를 확인하세요.")
        return None
    if response.is_error:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        detail = payload if isinstance(payload, dict) else {}
        nested_detail = detail.get("detail")
        if isinstance(nested_detail, dict):
            detail = nested_detail
        message = detail.get("message") or detail.get("detail")
        message = message if isinstance(message, str) else "요청을 처리하지 못했습니다."
        code = detail.get("code")
        code = code if isinstance(code, str) else None
        retryable = detail.get("retryable") is True
        suffix = " 다시 시도할 수 있습니다." if retryable else ""
        st.error(f"API 오류{f' ({code})' if code else ''}: {message}{suffix}")
        return None
    if response.status_code == 204:
        return {}
    try:
        return response.json()
    except ValueError:
        st.error("API 오류: JSON 응답을 해석할 수 없습니다.")
        return None


def value(data: Any, *keys: str, default: Any = None) -> Any:
    if not isinstance(data, dict):
        return default
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def items(data: Any, *keys: str) -> list[dict[str, Any]]:
    candidate = value(data, *keys, default=[])
    if isinstance(candidate, list):
        return [item for item in candidate if isinstance(item, dict)]
    return []

def validates_contract(model: type[Any], data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    try:
        model.model_validate_json(json.dumps(data))
    except (TypeError, ValueError, ValidationError):
        return False
    return True


def valid_listing(data: Any) -> bool:
    return validates_contract(ListingConditions, data)


def valid_analysis(data: Any) -> bool:
    return validates_contract(AnalysisRecord, data)


def valid_job_status(data: Any) -> bool:
    return validates_contract(AnalysisJobStatus, data)
def valid_terminal_job(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    status = data.get("status")
    if status in {"completed", "completed_with_external_fallback"}:
        return validates_contract(TerminalJobCompleted, data)
    if status in {"needs_review", "failed"}:
        return validates_contract(TerminalJobFailure, data)
    return False


def render_terminal_error(job: dict[str, Any]) -> None:
    error = value(job, "error", default={})
    if not isinstance(error, dict):
        return
    message = str(value(error, "message", default="분석이 완료되지 않았습니다."))
    code = str(value(error, "code", default="ANALYSIS_FAILED"))
    details = value(error, "details", default=[])
    st.error(message)
    st.caption(f"오류 코드: {code}")
    if isinstance(details, list) and details:
        st.caption("검증 세부 코드: " + ", ".join(str(detail) for detail in details))




def valid_report(data: Any) -> bool:
    return validates_contract(FinalReport, data)


def valid_catalog(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and data.get("data_mode") == "snapshot"
        and data.get("non_live") is True
        and isinstance(data.get("items"), list)
        and all(valid_listing(listing) for listing in data["items"])
    )


def reset_search_state() -> None:
    for key in ("listings", "selected_listing", "analysis_id", "report", "poll_deadline", "follow_up", "analysis_conditions"):
        st.session_state.pop(key, None)


def reset_analysis_state() -> None:
    for key in ("analysis_id", "report", "poll_deadline", "follow_up"):
        st.session_state.pop(key, None)


def reset_client() -> None:
    for key in ("session_id", "listings", "selected_listing", "analysis_id", "report", "poll_deadline", "follow_up", "analysis_conditions"):
        st.session_state.pop(key, None)

def reset_server_session(session_id: str) -> bool:
    """Clear the server session before discarding its browser-side state."""
    try:
        response = httpx.request(
            "POST",
            f"{api_base_url()}/api/v1/sessions/{session_id}/reset",
            timeout=15.0,
        )
    except httpx.RequestError as exc:
        st.error(f"API 네트워크 오류: {exc}. FastAPI 주소와 연결 상태를 확인하세요.")
        return False
    if response.status_code == 204:
        return True
    if response.is_error:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        detail = payload if isinstance(payload, dict) else {}
        nested_detail = detail.get("detail")
        if isinstance(nested_detail, dict):
            detail = nested_detail
        message = detail.get("message") or detail.get("detail")
        message = message if isinstance(message, str) else "요청을 처리하지 못했습니다."
        code = detail.get("code")
        code = code if isinstance(code, str) else None
        if response.status_code == 404 and code == "SESSION_NOT_FOUND":
            return True
        retryable = detail.get("retryable") is True
        suffix = " 다시 시도할 수 있습니다." if retryable else ""
        st.error(f"API 오류{f' ({code})' if code else ''}: {message}{suffix}")
        return False
    st.error("API 오류: 세션 재설정 응답을 처리하지 못했습니다.")
    return False

def ensure_session() -> str:
    existing = st.session_state.get("session_id")
    if isinstance(existing, str):
        return existing
    session_id = uuid.uuid4().hex
    st.session_state.session_id = session_id
    return session_id


def matches_conditions(
    listing: dict[str, Any],
    region: str,
    budget: int,
    property_types: list[str],
    min_area: int,
    max_area: int,
) -> bool:
    address = str(value(listing, "address_text", default=""))
    deposit = value(listing, "deposit_won", default=0)
    area = value(listing, "area_sqm", default=0)
    property_type = str(value(listing, "property_type", default=""))
    try:
        within_budget = not budget or int(deposit) <= budget
        within_area = min_area <= float(area) <= max_area
    except (TypeError, ValueError):
        return False
    return (
        (not region or region.casefold() in address.casefold())
        and within_budget
        and within_area
        and (not property_types or property_type in property_types)
    )


def render_listing_card(listing: dict[str, Any]) -> None:
    listing_id = str(value(listing, "listing_id", default=""))
    title = value(listing, "address_text", default="샘플 매물")
    deposit = value(listing, "deposit_won", default="확인 불가")
    area = value(listing, "area_sqm", default="확인 불가")
    property_type = value(listing, "property_type", default="유형 확인 불가")
    as_of = value(listing, "snapshot_as_of", default="기준일 확인 불가")
    st.markdown(f"**{title}**")
    st.caption(f"보증금: {deposit}원 · 전용면적: {area}㎡ · 유형: {property_type}")
    st.caption(f"샘플 스냅샷 기준일: {as_of} (실시간 매물이 아님)")
    if st.button("이 샘플 선택", key=f"select-{listing_id}", disabled=not listing_id):
        reset_analysis_state()
        st.session_state.selected_listing = listing


def render_events(job: dict[str, Any]) -> None:
    st.subheader("분석 진행")
    results = items(job, "agent_results")
    result_by_role = {str(value(result, "agent_name")): result for result in results}
    for role, label in ROLE_LABELS.items():
        result = result_by_role.get(role)
        if result is None:
            st.warning(f"{label}: 대기")
            continue
        events = items(result, "events")
        message = value(events[-1], "message", default="완료 상태 수신") if events else "완료 상태 수신"
        status = str(value(result, "status", default="running"))
        text = f"{label}: {status} — {message}"
        if status in {"completed", "completed_with_external_fallback"}:
            st.success(text)
        elif status in {"failed", "cancelled"}:
            st.error(text)
        else:
            st.warning(text)


def render_section(title: str, content: Any, fields: tuple[str, ...]) -> None:
    st.subheader(title)
    if isinstance(content, list) and content:
        rendered = False
        for item in content:
            if isinstance(item, dict):
                values: list[str] = []
                for field in fields:
                    entry = item.get(field)
                    if isinstance(entry, (str, int, float)):
                        values.append(str(entry))
                    elif isinstance(entry, list) and all(
                        isinstance(value, str) for value in entry
                    ):
                        values.append(", ".join(entry))
                if values:
                    st.write(" · ".join(values))
                    rendered = True
            elif isinstance(item, str):
                st.write(item)
                rendered = True
        if rendered:
            return
    st.info("제공된 항목이 없습니다.")


def render_report(report: dict[str, Any]) -> None:
    if not valid_report(report):
        st.error("보고서 응답 계약이 올바르지 않아 표시하지 않았습니다.")
        return
    st.header("분석 결과")
    level = str(value(report, "overall_risk", default="unavailable"))
    level_text = {"low": "낮음", "medium": "중간", "high": "높음", "unavailable": "확인 불가"}.get(level, level)
    if level == "high":
        st.error(f"위험 참고 신호: {level_text}")
    elif level == "medium":
        st.warning(f"위험 참고 신호: {level_text}")
    elif level == "low":
        st.success(f"위험 참고 신호: {level_text}")
    else:
        st.info(f"위험 참고 신호: {level_text}")
    st.caption("색상은 보조 수단입니다. 이 등급은 참고용이며 안전 보증, 사기 판정 또는 법률 자문이 아닙니다.")

    render_section("판단", value(report, "conclusions", default=[]), ("statement", "risk_level", "evidence_ids"))
    render_section("근거 있는 주장", value(report, "claims", default=[]), ("statement", "kind", "evidence_ids"))
    render_section("준비 서류·체크 항목", value(report, "checklist_items", default=[]), ("label", "status", "rationale", "evidence_ids"))
    render_section(
        "근거와 기준일",
        value(report, "evidence", default=[]),
        (
            "evidence_id",
            "source_name",
            "source_record_id",
            "snapshot_as_of",
            "observed_at",
            "retrieved_at",
            "provenance_id",
            "excerpt",
        ),
    )
    render_section("한계 및 다음 확인", value(report, "limitations", default=[]), ())


def poll_and_render(analysis_id: str) -> None:
    job = api_request("GET", f"/api/v1/analyses/{analysis_id}")
    if not isinstance(job, dict):
        st.error("분석 상태 응답 계약이 올바르지 않습니다.")
        return
    status = str(job.get("status", ""))
    st.caption(f"분석 상태: {status}")
    if status not in TERMINAL_STATUSES:
        if not valid_job_status(job):
            st.error("분석 상태 응답 계약이 올바르지 않습니다.")
            return
        deadline = st.session_state.get("poll_deadline")
        if not isinstance(deadline, (int, float)):
            deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
            st.session_state.poll_deadline = deadline
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            st.session_state.pop("analysis_id", None)
            st.session_state.pop("poll_deadline", None)
            st.error("분석 상태 확인 시간이 초과되었습니다. 서버 상태를 확인한 뒤 새 분석을 시작하세요.")
            return
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))
        st.rerun()
        return
    st.session_state.pop("poll_deadline", None)
    if not valid_terminal_job(job):
        st.error("완료된 분석 응답 계약이 올바르지 않습니다.")
        return
    render_events(job)
    if status not in {"completed", "completed_with_external_fallback"}:
        render_terminal_error(job)
        st.error("분석이 완료되지 않아 보고서를 게시할 수 없습니다. 화면의 서버 이벤트와 오류를 확인하세요.")
        return
    report = api_request("GET", f"/api/v1/analyses/{analysis_id}/report")
    if valid_report(report):
        if status == "completed_with_external_fallback":
            evidence = items(report, "evidence")
            as_of = value(evidence[0], "snapshot_as_of", "observed_at", default="기준일 확인 불가") if evidence else "기준일 확인 불가"
            st.warning("외부 정보 대체 경로로 완료된 보고서입니다. 외부 정보의 최신성·완전성을 별도로 확인하세요.")
            st.caption(f"외부 대체 근거 기준일: {as_of}")
        st.session_state.report = report
        render_report(report)
    else:
        st.error("보고서 응답 계약이 올바르지 않아 저장하지 않았습니다.")


def main() -> None:
    st.set_page_config(page_title="전세 의사결정 지원", page_icon="🏠", layout="wide")
    st.markdown(
        """
        <style>
        @font-face {
          font-family: "Noto Sans KR";
          src: url("/app/static/NotoSansKR-Regular.ttf") format("truetype");
          font-weight: 100 900;
          font-display: swap;
        }
        body, p, label, input, button, textarea, h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
          font-family: "Noto Sans KR", sans-serif !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("전세 매물 의사결정 지원")
    st.warning("비실시간 합성 데모 스냅샷 기반의 교육용 참고 정보입니다. 최신성·완전성을 보장하지 않습니다.")
    st.info("정보 제공과 문의 준비용이며 법률 자문, 사기 판정 또는 안전 보증이 아닙니다. 등기·건축물·임대인 정보는 실시간 검증하지 않습니다.")
    st.caption(f"FastAPI 연결 주소: {api_base_url()}")

    with st.sidebar:
        st.header("세션")
        if st.button("처음부터 다시", type="secondary"):
            session_id = st.session_state.get("session_id")
            if isinstance(session_id, str) and reset_server_session(session_id):
                reset_client()
                st.rerun()
        st.caption("입력란에는 이름·연락처·주민등록번호·임대인 식별정보를 적지 마세요.")

    catalog_response = api_request("GET", "/api/v1/listings")
    if valid_catalog(catalog_response):
        catalog_items = catalog_response["items"]
        property_type_options = sorted(
            {str(listing["property_type"]) for listing in catalog_items}
        )
    else:
        catalog_items = []
        property_type_options = []
        st.error("매물 목록 응답 계약이 올바르지 않습니다.")

    with st.form("conditions"):
        st.subheader("조건 필터")
        region = st.text_input("희망 지역(법정동 또는 지역명)", max_chars=100)
        budget = st.number_input("최대 보증금(원)", min_value=0, step=10_000)
        property_types = st.multiselect("주택 유형", property_type_options)
        min_area, max_area = st.slider("전용면적 범위(㎡)", 1, 200, (20, 85))
        submitted = st.form_submit_button("샘플 매물 찾기")

    if submitted:
        reset_search_state()
        ensure_session()
        listings = [
            listing
            for listing in catalog_items
            if matches_conditions(
                listing, region, int(budget), property_types, min_area, max_area
            )
        ]
        if not listings:
            st.info("조건에 맞는 샘플 매물이 없습니다. 필터를 조정하세요.")
        st.session_state.analysis_conditions = {
            "region": region.strip() or None,
            "max_deposit_won": int(budget) if budget else None,
            "property_types": property_types,
            "min_area_sqm": min_area,
            "max_area_sqm": max_area,
        }
        st.session_state.listings = listings

    listings = st.session_state.get("listings", [])
    if listings:
        st.subheader("조건에 맞는 비실시간 샘플 매물")
        for listing in listings:
            with st.container(border=True):
                render_listing_card(listing)

    selected = st.session_state.get("selected_listing")
    if isinstance(selected, dict):
        st.success("샘플 매물을 선택했습니다. 아래에서 분석을 시작하세요.")
        if st.button("4단계 분석 시작", type="primary"):
            reset_analysis_state()
            if not valid_listing(selected):
                st.error("선택한 매물의 스냅샷 계약이 올바르지 않습니다.")
            else:
                session_id = ensure_session()
                response = api_request(
                    "POST",
                    "/api/v1/analyses",
                    json={
                        "session_id": session_id,
                        "listing_id": selected["listing_id"],
                        "conditions": st.session_state.get("analysis_conditions"),
                    },
                )
                if valid_job_status(response):
                    st.session_state.analysis_id = response["analysis_id"]
                    st.session_state.poll_deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
                else:
                    st.error("분석 요청 응답 계약이 올바르지 않습니다.")

    analysis_id = st.session_state.get("analysis_id")
    if isinstance(analysis_id, str):
        poll_and_render(analysis_id)
    elif isinstance(st.session_state.get("report"), dict):
        render_report(st.session_state.report)

    report = st.session_state.get("report")
    session_id = st.session_state.get("session_id")
    if isinstance(report, dict) and isinstance(session_id, str):
        st.divider()
        st.subheader("후속 확인")
        operation_labels = {
            "clarify": "근거·기준일 설명",
            "compare": "이전 보고서와 비교",
            "recheck": "공식 출처 재확인 안내",
        }
        operation = st.selectbox(
            "후속 작업",
            options=tuple(operation_labels),
            format_func=operation_labels.get,
        )
        target_report_id = st.text_input(
            "대상 보고서 ID(비우면 기본 대상)",
            max_chars=128,
        )
        if st.button("후속 작업 실행"):
            response = api_request(
                "POST",
                f"/api/v1/sessions/{session_id}/follow-ups",
                json={
                    "operation": operation,
                    "target_report_id": target_report_id or None,
                },
            )
            if isinstance(response, dict):
                message = value(response, "message")
                notice = value(response, "advisory_notice")
                if isinstance(message, str):
                    st.success(message)
                if isinstance(notice, str):
                    st.caption(notice)


if __name__ == "__main__":
    main()
