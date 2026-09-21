"""FastAPI application. Phase 2: health + ingestion endpoints. Phase 4 adds inference,
results, review and feedback routers.

For research use only. Not for diagnostic use.
"""

from __future__ import annotations

import hmac

import structlog
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel

from secondreader import __version__
from secondreader.core.constants import RESEARCH_USE_LABEL
from secondreader.core.ingest.service import PhiLeakError
from secondreader.core.serving.deps import Container, get_container
from secondreader.core.serving.routes import auth, infer, monitoring, review, studies

log = structlog.get_logger(__name__)

app = FastAPI(
    title="Second Reader API",
    version=__version__,
    description=f"**{RESEARCH_USE_LABEL}**\n\nAI second reader for medical imaging (prototype).",
)


app.include_router(auth.router)
app.include_router(studies.router)
app.include_router(review.router)
app.include_router(review.artifacts)
app.include_router(infer.router)
app.include_router(monitoring.router)


@app.middleware("http")
async def research_use_header(request: Request, call_next):  # type: ignore[no-untyped-def]
    resp = await call_next(request)
    resp.headers["X-Research-Use"] = RESEARCH_USE_LABEL
    return resp


def container() -> Container:
    return get_container()


def require_webhook_token(
    x_webhook_token: str | None = Header(default=None), c: Container = Depends(container)
) -> None:
    expected = c.settings.webhook_token.get_secret_value() if c.settings.webhook_token else ""
    if not expected or not x_webhook_token or not hmac.compare_digest(x_webhook_token, expected):
        raise HTTPException(status_code=401, detail="invalid webhook token")


class Health(BaseModel):
    status: str
    version: str
    research_use: str
    orthanc_a: bool
    orthanc_b: bool
    models: dict[str, str] = {}


@app.get("/health", response_model=Health)
def health(c: Container = Depends(container)) -> Health:
    plugins = c.plugins().all()
    return Health(
        status="ok",
        version=__version__,
        research_use=RESEARCH_USE_LABEL,
        orthanc_a=c.orthanc_a.reachable(),
        orthanc_b=c.orthanc_b.reachable(),
        models={p.info.modality_id: p.model_version_id for p in plugins},
    )


class OrthancStudyEvent(BaseModel):
    orthanc_study_id: str
    source: str = "orthanc-a"


class IngestSummary(BaseModel):
    research_use: str = RESEARCH_USE_LABEL
    n_instances: int
    stored: int
    quarantined: int
    duplicate: int
    rejected: int
    study_ids: list[str]
    job_ids: list[str]


@app.post(
    "/v1/ingest/orthanc",
    response_model=IngestSummary,
    dependencies=[Depends(require_webhook_token)],
)
def ingest_from_orthanc(ev: OrthancStudyEvent, c: Container = Depends(container)) -> IngestSummary:
    """Pull every instance of a study from Orthanc A and run it through ingestion."""
    try:
        instance_ids = c.orthanc_a.study_instances(ev.orthanc_study_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"orthanc A: {exc}") from exc
    return _ingest_many(c, [c.orthanc_a.instance_file(i) for i in instance_ids], ev.source)


@app.post("/v1/ingest/upload", response_model=IngestSummary)
async def ingest_upload(
    files: list[UploadFile] = File(...), c: Container = Depends(container)
) -> IngestSummary:
    """Direct upload (dev/testing). Production traffic arrives via the PACS webhook."""
    payloads = [await f.read() for f in files]
    return _ingest_many(c, payloads, "upload")


def _ingest_many(c: Container, payloads: list[bytes], source: str) -> IngestSummary:
    counts: dict[str, int] = {"stored": 0, "quarantined": 0, "duplicate": 0, "rejected": 0}
    study_ids: list[str] = []
    job_ids: list[str] = []
    with c.sessions() as session:
        for raw in payloads:
            try:
                res = c.ingest.ingest_bytes(session, raw, source=source)
            except PhiLeakError as exc:
                session.rollback()
                log.error("phi_leak_blocked", error=str(exc))
                raise HTTPException(
                    status_code=500, detail="de-identification verification failed; refused"
                ) from exc
            counts[res.status] += 1
            if res.study_id and res.study_id not in study_ids:
                study_ids.append(res.study_id)
            if res.job_id:
                job_ids.append(res.job_id)
        session.commit()
    return IngestSummary(
        n_instances=len(payloads),
        study_ids=study_ids,
        job_ids=job_ids,
        stored=counts["stored"],
        quarantined=counts["quarantined"],
        duplicate=counts["duplicate"],
        rejected=counts["rejected"],
    )
