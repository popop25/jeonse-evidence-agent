"""FastAPI boundary for the bounded, in-memory advisory job service."""
from __future__ import annotations

import asyncio
import hashlib
import json
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .adapters import JsonHugIncidentRepository, JsonTransactionRepository
from .ai_provider import AzureOpenAIProvider, AzureOpenAISettings
from .config import Settings
from .models import AnalysisStatus, ListingConditions
from .rag import (
    AllowlistedOfficialDocumentRepository,
    ChromaAzureVectorStoreAdapter,
    PersistentChromaAzureBackend,
    SemanticOfficialDocumentRepository,
    official_guidance_documents,
)
from .repositories import OfficialDocumentRepository
from .service import (
    MAX_RETAINED_RECORDS,
    RETENTION_TTL,
    AnalysisRequest,
    AnalysisService,
    DataAdapters,
    TerminalJobCompleted,
    TerminalJobFailure,
)

CATALOG_SHA256 = "cd302319884f2346d697cbd9af6a6e184b60f4c255d1607476b58869f3bacca8"
CATALOG_FIELDS = {
    "dataset_kind", "snapshot_notice", "snapshot_as_of", "provenance_id", "listings", "content_sha256",
}
JOB_CAPACITY = 4
JOB_TIMEOUT_SECONDS = 240
TERMINAL_JOB_STATUSES = frozenset(
    {"completed", "completed_with_external_fallback", "needs_review", "failed"}
)
ACTIVE_JOB_STATUSES = frozenset({"queued", "running"})
COMPLETED_JOB_STATUSES = frozenset({"completed", "completed_with_external_fallback"})


class FollowUpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    operation: str = Field(min_length=1, max_length=32)
    target_report_id: str | None = Field(default=None, min_length=1, max_length=128)


@dataclass(slots=True)
class AnalysisJob:
    analysis_id: str
    request: AnalysisRequest
    fingerprint: str
    idempotency_key: str | None
    created_at: datetime
    status: str = "queued"
    service_analysis_id: str | None = None
    report_id: str | None = None
    error_code: str | None = None
    error_details: tuple[str, ...] = ()
    has_service_record: bool = False

    def public(self, service: AnalysisService) -> dict[str, object]:
        if self.service_analysis_id and self.status in TERMINAL_JOB_STATUSES:
            record = service.get_analysis(self.service_analysis_id)
            if self.status in COMPLETED_JOB_STATUSES and (
                record is None
                or not self.report_id
                or record.report_id != self.report_id
                or service.get_report(self.service_analysis_id) is None
            ):
                unavailable: dict[str, object] = {
                    "analysis_id": self.analysis_id,
                    "session_id": self.request.session_id,
                    "status": "failed",
                    "created_at": self.created_at.isoformat(),
                    "error": _terminal_error("failed", "REPORT_NOT_AVAILABLE"),
                }
                return TerminalJobFailure.model_validate_json(json.dumps(unavailable)).model_dump(
                    mode="json", exclude_none=True
                )
            if record is not None and record.status.value == self.status:
                body = record.model_dump(mode="json", exclude_none=True)
                body["analysis_id"] = self.analysis_id
                body["status"] = self.status
                if self.status not in COMPLETED_JOB_STATUSES:
                    body["error"] = _terminal_error(
                        self.status,
                        self.error_code or getattr(record.error_code, "value", None),
                        self.error_details or tuple(record.ai_trace_codes),
                    )
                terminal_model = (
                    TerminalJobCompleted
                    if self.status in COMPLETED_JOB_STATUSES
                    else TerminalJobFailure
                )
                return terminal_model.model_validate_json(json.dumps(body)).model_dump(
                    mode="json", exclude_none=True
                )
        if self.status in COMPLETED_JOB_STATUSES:
            unavailable: dict[str, object] = {
                "analysis_id": self.analysis_id,
                "session_id": self.request.session_id,
                "status": "failed",
                "created_at": self.created_at.isoformat(),
                "error": _terminal_error("failed", "REPORT_NOT_AVAILABLE"),
            }
            return TerminalJobFailure.model_validate_json(json.dumps(unavailable)).model_dump(
                mode="json", exclude_none=True
            )
        body: dict[str, object] = {
            "analysis_id": self.analysis_id,
            "session_id": self.request.session_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }
        if self.report_id:
            body["report_id"] = self.report_id
        if self.status in TERMINAL_JOB_STATUSES and self.status not in COMPLETED_JOB_STATUSES:
            body["error"] = _terminal_error(self.status, self.error_code, self.error_details)
            return TerminalJobFailure.model_validate_json(json.dumps(body)).model_dump(
                mode="json", exclude_none=True
            )
        return body


def _error(status: int, code: str, message: str, retryable: bool = False) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message, "retryable": retryable})
def _terminal_error(
    status: str,
    code: str | None,
    details: tuple[str, ...] = (),
) -> dict[str, object]:
    if code in {"ANALYSIS_TIMEOUT", "AI_PROVIDER_FAILED"}:
        message = (
            "Analysis exceeded the server time limit."
            if code == "ANALYSIS_TIMEOUT"
            else "The configured AI provider did not complete the analysis."
        )
        retryable = True
    elif status == "needs_review":
        code = "EVIDENCE_GATE_FAILED"
        message = "Evidence validation prevented report publication."
        retryable = False
    else:
        code = code or "ANALYSIS_FAILED"
        message = "Analysis did not complete."
        retryable = False
    return {"code": code, "message": message, "retryable": retryable, "details": details}


def _snapshot_adapters(root: Path | None = None, *, documents: OfficialDocumentRepository | None = None) -> DataAdapters:
    project_root = root or Path(__file__).resolve().parents[2]
    catalog_path = project_root / "data" / "catalog" / "listings.json"
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != CATALOG_FIELDS:
        raise ValueError("INVALID_SNAPSHOT_CATALOG")
    supplied_digest = payload["content_sha256"]
    canonical_payload = {key: value for key, value in payload.items() if key != "content_sha256"}
    canonical = json.dumps(canonical_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    calculated_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if (supplied_digest != CATALOG_SHA256 or calculated_digest != CATALOG_SHA256
            or payload["dataset_kind"] != "snapshot" or not isinstance(payload["listings"], list)):
        raise ValueError("INVALID_SNAPSHOT_CATALOG_CHECKSUM")
    listing_fields = set(ListingConditions.model_fields)
    listings: dict[str, ListingConditions] = {}
    for row in payload["listings"]:
        if not isinstance(row, dict) or set(row) - listing_fields:
            raise ValueError("INVALID_SNAPSHOT_CATALOG_ROW")
        provenance_id = row.get("provenance_id")
        if (row.get("snapshot_as_of") != payload["snapshot_as_of"] or not isinstance(provenance_id, str)
                or not provenance_id.startswith(f"{payload['provenance_id']}-")):
            raise ValueError("INVALID_SNAPSHOT_CATALOG_PROVENANCE")
        item = ListingConditions.model_validate_json(json.dumps({key: row[key] for key in listing_fields if key in row}))
        if item.listing_id in listings:
            raise ValueError("DUPLICATE_SNAPSHOT_LISTING_ID")
        listings[item.listing_id] = item
    return DataAdapters(
        listings=listings,
        transactions=JsonTransactionRepository(project_root / "data" / "public" / "transactions.json"),
        hug=JsonHugIncidentRepository(project_root / "data" / "public" / "hug_incidents.json"),
        documents=documents or AllowlistedOfficialDocumentRepository(official_guidance_documents()),
    )


def _request_fingerprint(request: AnalysisRequest) -> str:
    payload = request.model_dump(mode="json")
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _job_error_code(error: Exception) -> str:
    if isinstance(error, KeyError):
        return "LISTING_NOT_FOUND"
    if isinstance(error, ValueError):
        code = str(error)
        if code and code.replace("_", "").isalnum() and code.upper() == code:
            return code
    return "ANALYSIS_FAILED"


def create_app(service: AnalysisService | None = None) -> FastAPI:
    """Create an app with explicit modes and a bounded single-worker job registry."""
    settings = Settings.from_environment()
    if settings.data_mode != "snapshot":
        raise ValueError("DATA_MODE_UNCONFIGURED")
    ai_provider = None
    documents = None
    if settings.azure_configured:
        ai_provider = AzureOpenAIProvider(settings=AzureOpenAISettings.from_environment(), construct_azure=True)
        vector_store = ChromaAzureVectorStoreAdapter(
            PersistentChromaAzureBackend(ai_provider, Path(__file__).resolve().parents[2] / "data" / "chroma")
        )
        documents = SemanticOfficialDocumentRepository(vector_store)
    analysis_service = service or AnalysisService(settings, _snapshot_adapters(documents=documents), ai_provider=ai_provider)
    jobs: OrderedDict[str, AnalysisJob] = OrderedDict()
    requests: OrderedDict[str, AnalysisRequest] = OrderedDict()
    idempotency: dict[tuple[str, str], str] = {}
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=JOB_CAPACITY)
    registry_lock = asyncio.Lock()
    worker: asyncio.Task[None] | None = None

    def remove_job(analysis_id: str) -> None:
        job = jobs.pop(analysis_id, None)
        requests.pop(analysis_id, None)
        if job and job.idempotency_key:
            idempotency.pop((job.request.session_id, job.idempotency_key), None)

    def prune_registry() -> None:
        cutoff = datetime.now(UTC) - RETENTION_TTL
        for analysis_id, job in tuple(jobs.items()):
            if job.status not in ACTIVE_JOB_STATUSES and (
                job.created_at < cutoff
                or (
                    job.has_service_record
                    and job.service_analysis_id
                    and analysis_service.get_analysis(job.service_analysis_id) is None
                )
            ):
                remove_job(analysis_id)
        while len(jobs) > MAX_RETAINED_RECORDS:
            terminal_id = next(
                (analysis_id for analysis_id, job in jobs.items() if job.status not in ACTIVE_JOB_STATUSES),
                None,
            )
            if terminal_id is None:
                return
            remove_job(terminal_id)

    async def run_job(analysis_id: str) -> None:
        cancelled = False
        async with registry_lock:
            prune_registry()
            job = jobs.get(analysis_id)
            if job is None or job.status != "queued":
                return
            job.status = "running"
            job.service_analysis_id = analysis_id
        remaining_seconds = JOB_TIMEOUT_SECONDS - (
            datetime.now(UTC) - job.created_at
        ).total_seconds()
        if remaining_seconds <= 0:
            status, error_code = "failed", "ANALYSIS_TIMEOUT"
        else:
            try:
                record = await asyncio.wait_for(
                    analysis_service.analyze(job.request, analysis_id),
                    timeout=remaining_seconds,
                )
            except asyncio.CancelledError:
                status, error_code, cancelled = "failed", "SERVER_SHUTDOWN", True
            except TimeoutError:
                status, error_code = "failed", "ANALYSIS_TIMEOUT"
            except Exception as exc:
                status, error_code = "failed", _job_error_code(exc)
            else:
                match record.status:
                    case AnalysisStatus.COMPLETED:
                        status, error_code = "completed", None
                    case AnalysisStatus.COMPLETED_WITH_EXTERNAL_FALLBACK:
                        status, error_code = "completed_with_external_fallback", None
                    case AnalysisStatus.NEEDS_REVIEW:
                        status, error_code = "needs_review", "EVIDENCE_GATE_FAILED"
                    case AnalysisStatus.FAILED:
                        status = "failed"
                        record_error_code = record.error_code
                        error_code = (
                            getattr(record_error_code, "name", record_error_code)
                            or "REPORT_NOT_AVAILABLE"
                        )
                    case _:
                        status, error_code = "failed", "REPORT_NOT_AVAILABLE"
                if status in COMPLETED_JOB_STATUSES and (
                    not record.report_id
                    or analysis_service.get_report(record.analysis_id) is None
                ):
                    status, error_code = "failed", "REPORT_NOT_AVAILABLE"
        async with registry_lock:
            current = jobs.get(analysis_id)
            if current is not None:
                current.status = status
                current.error_code = error_code
                if "record" in locals():
                    current.service_analysis_id = record.analysis_id
                    current.report_id = record.report_id if status in COMPLETED_JOB_STATUSES else None
                current.has_service_record = analysis_service.get_analysis(analysis_id) is not None
                analysis_service.release_session_lease(current.request.session_id, analysis_id)
                prune_registry()
        if cancelled:
            raise asyncio.CancelledError

    async def work() -> None:
        while True:
            analysis_id = await queue.get()
            try:
                if analysis_id is None:
                    return
                await run_job(analysis_id)
            finally:
                queue.task_done()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal worker
        worker = asyncio.create_task(work())
        try:
            yield
        finally:
            async with registry_lock:
                for job in jobs.values():
                    if job.status in ACTIVE_JOB_STATUSES:
                        job.status = "failed"
                        job.error_code = "SERVER_SHUTDOWN"
                        analysis_service.release_session_lease(
                            job.request.session_id, job.analysis_id
                        )
                prune_registry()
            if worker:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

    app = FastAPI(title="Jeonse Advisory Support", version="0.1.0", lifespan=lifespan)
    app.state.analysis_service = analysis_service
    app.state._jobs = jobs
    app.state._requests = requests
    app.state._idempotency = idempotency

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        messages = " ".join(str(error.get("msg", "")) for error in exc.errors())
        if "PII_NOT_ALLOWED" in messages:
            code, message = "PII_NOT_ALLOWED", "Personal identifiers are not accepted."
        elif "INVALID_SESSION_ID" in messages:
            code, message = "INVALID_SESSION_ID", "Session identifier is invalid."
        else:
            code, message = "VALIDATION_ERROR", "Request validation failed."
        return JSONResponse(status_code=422, content={"detail": {"code": code, "message": message, "retryable": False}})

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"ready": True, "restart_epoch": "in-memory", "versions": {"risk_policy": "risk-policy-v1"}, **analysis_service.settings.public_metadata()}

    @app.get("/api/v1/listings")
    async def listings() -> dict[str, object]:
        return {"data_mode": analysis_service.settings.data_mode, "non_live": True, "items": analysis_service.list_listings()}

    @app.post("/api/v1/analyses", status_code=202)
    async def create_analysis(request: AnalysisRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")) -> dict[str, object]:
        if idempotency_key is not None and (not idempotency_key.strip() or len(idempotency_key) > 128):
            raise _error(422, "INVALID_IDEMPOTENCY_KEY", "Idempotency-Key must be 1 to 128 characters.")
        fingerprint = _request_fingerprint(request)
        async with registry_lock:
            prune_registry()
            if idempotency_key:
                previous_id = idempotency.get((request.session_id, idempotency_key))
                if previous_id:
                    previous = jobs.get(previous_id)
                    if previous is not None:
                        if previous.fingerprint != fingerprint:
                            raise _error(409, "IDEMPOTENCY_KEY_COLLISION", "Idempotency-Key was used for a different request.")
                        return previous.public(analysis_service)
                    idempotency.pop((request.session_id, idempotency_key), None)
            active_session = next((job for job in jobs.values() if job.request.session_id == request.session_id and job.status in ACTIVE_JOB_STATUSES), None)
            if active_session:
                raise _error(409, "SESSION_ANALYSIS_IN_PROGRESS", "This session already has an active analysis.", retryable=True)
            active_jobs = sum(job.status in ACTIVE_JOB_STATUSES for job in jobs.values())
            if active_jobs >= JOB_CAPACITY:
                raise _error(503, "ANALYSIS_CAPACITY_EXCEEDED", "Analysis queue is at capacity.", retryable=True)
            analysis_id = uuid4().hex
            if not analysis_service.acquire_session_lease(request.session_id, analysis_id):
                raise _error(409, "SESSION_ANALYSIS_IN_PROGRESS", "This session already has an active analysis.", retryable=True)
            job = AnalysisJob(analysis_id, request, fingerprint, idempotency_key, datetime.now(UTC))
            jobs[analysis_id] = job
            requests[analysis_id] = request
            if idempotency_key:
                idempotency[(request.session_id, idempotency_key)] = analysis_id
            prune_registry()
            queue.put_nowait(analysis_id)
            return job.public(analysis_service)

    @app.get("/api/v1/analyses/{analysis_id}")
    async def analysis(analysis_id: str) -> dict[str, object]:
        async with registry_lock:
            prune_registry()
            job = jobs.get(analysis_id)
            if job is None:
                raise _error(404, "ANALYSIS_NOT_FOUND", "Analysis was not found.")
            return job.public(analysis_service)

    @app.get("/api/v1/analyses/{analysis_id}/report")
    async def report(analysis_id: str):
        async with registry_lock:
            prune_registry()
            job = jobs.get(analysis_id)
            if job is None:
                raise _error(404, "ANALYSIS_NOT_FOUND", "Analysis was not found.")
            if job.status not in COMPLETED_JOB_STATUSES or not job.report_id:
                raise _error(
                    409,
                    job.error_code or "REPORT_NOT_AVAILABLE",
                    "Completed report is not available.",
                    retryable=job.status in ACTIVE_JOB_STATUSES,
                )
            result = analysis_service.get_report(job.service_analysis_id) if job.service_analysis_id else None
            if result is None:
                raise _error(409, "REPORT_NOT_AVAILABLE", "Completed report is not available.")
            return result

    async def clear_session(session_id: str) -> None:
        async with registry_lock:
            prune_registry()
            active = [job for job in jobs.values() if job.request.session_id == session_id and job.status in ACTIVE_JOB_STATUSES]
            if active or analysis_service.is_session_active(session_id):
                raise _error(409, "SESSION_ANALYSIS_IN_PROGRESS", "Active analyses must finish before session reset.", retryable=True)
            session_jobs = [job for job in jobs.values() if job.request.session_id == session_id]
            if (
                not session_jobs
                and session_id not in analysis_service._sessions
                and session_id not in analysis_service._session_accessed_at
            ):
                raise _error(404, "SESSION_NOT_FOUND", "Session was not found.")
            for job in session_jobs:
                remove_job(job.analysis_id)
                if job.service_analysis_id:
                    analysis_service._records.pop(job.service_analysis_id, None)
                    analysis_service._reports.pop(job.service_analysis_id, None)
            analysis_service.clear_session_memory(session_id)

    @app.post("/api/v1/sessions/{session_id}/reset", status_code=204)
    async def reset_session(session_id: str) -> Response:
        await clear_session(session_id)
        return Response(status_code=204)

    @app.delete("/api/v1/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str) -> Response:
        await clear_session(session_id)
        return Response(status_code=204)

    @app.post("/api/v1/sessions/{session_id}/follow-ups")
    async def follow_up(session_id: str, request: FollowUpRequest) -> dict[str, object]:
        try:
            return analysis_service.follow_up(session_id, request.operation, request.target_report_id)
        except KeyError:
            raise _error(404, "REPORT_NOT_FOUND", "Session or requested retained report was not found.") from None
        except ValueError as exc:
            raise _error(422, str(exc), "Follow-up operation is not accepted for this advisory service.") from None

    return app


app = create_app()
