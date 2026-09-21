"""Run one instance through its modality plugin and persist everything (principles 2,3,7,8).

Used by the job worker (async path via Orthanc) and by `POST /v1/infer` (sync path).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import numpy as np
import structlog
from sqlalchemy.orm import Session

from secondreader.core.constants import GATE_UNABLE_TO_ASSESS
from secondreader.core.db.models import Instance, ModelVersion, Prediction, Study
from secondreader.core.dicom import read_dicom
from secondreader.core.dicom.imaging import to_imaging_input
from secondreader.core.plugins.protocol import Finding, GateResult, ModalityPlugin
from secondreader.core.results import build_results
from secondreader.core.serving.deps import Container

log = structlog.get_logger(__name__)


class NoPluginError(RuntimeError):
    pass


def ensure_model_version(session: Session, plugin: ModalityPlugin) -> ModelVersion:
    mv = session.get(ModelVersion, plugin.model_version_id)
    if mv is None:
        card: dict[str, Any] = dict(getattr(getattr(plugin, "bundle", None), "card", {}) or {})
        mv = ModelVersion(
            id=plugin.model_version_id,
            modality_id=plugin.info.modality_id,
            name=str(card.get("name", plugin.info.display_name)),
            weights_sha256=str(card.get("weights_sha256", "")),
            config_sha256=str(card.get("config_sha256", "")),
            split_manifest_sha256=card.get("split_manifest_sha256"),
            config={
                k: card[k] for k in ("backbone", "init", "datasets", "finding_codes") if k in card
            },
            metrics=dict(card.get("metrics", {}).get("val", {}))
            if isinstance(card.get("metrics"), dict)
            else {},
            notes=str(card.get("notes", "")),
        )
        session.add(mv)
        session.flush()
    return mv


def run_inference_for_instance(
    c: Container,
    session: Session,
    instance: Instance,
    *,
    request_id: str | None = None,
    push_to_archive: bool = True,
) -> Prediction:
    t0 = time.perf_counter()
    raw = c.blobs.get(instance.blob_key)
    ds = read_dicom(raw)
    plugin = c.plugins().route(instance.header)
    if plugin is None:
        raise NoPluginError(
            f"no plugin accepts modality={instance.header.get('Modality')} "
            f"body_part={instance.header.get('BodyPartExamined')}"
        )
    mv = ensure_model_version(session, plugin)
    inp = to_imaging_input(ds, input_sha256=instance.sha256)

    gate: GateResult = plugin.quality_gate(inp)
    findings: list[Finding] = plugin.predict(inp) if gate.ok else []
    stats = plugin.input_stats(inp) if hasattr(plugin, "input_stats") else {}
    image01 = _image01(plugin)
    latency = round((time.perf_counter() - t0) * 1000, 1)

    pid = str(uuid.uuid4())
    rb = build_results(
        ds,
        inp,
        gate,
        findings,
        model_version_id=mv.id,
        plugin_version=plugin.info.version,
        prediction_id=pid,
        image01=image01,
        latency_ms=latency,
    )

    sr_key = f"results/{pid}/sr.dcm"
    c.blobs.put(sr_key, rb.sr_bytes)
    sc_keys: dict[str, str] = {}
    for code, _uid, data in rb.sc_items:
        sc_keys[code] = f"results/{pid}/sc_{code}.dcm"
        c.blobs.put(sc_keys[code], data)
    heat_keys: dict[str, str] = {}
    for code, png in rb.heatmap_pngs.items():
        heat_keys[code] = f"results/{pid}/heat_{code}.png"
        c.blobs.put(heat_keys[code], png)

    findings_json = rb.json["findings"]
    for f in findings_json:
        f["heatmap_blob_key"] = heat_keys.get(f["code"])
        f["sc_blob_key"] = sc_keys.get(f["code"])
    pred = Prediction(
        id=pid,
        instance_id=instance.id,
        study_id=instance.series.study_id,
        model_version_id=mv.id,
        plugin_version=plugin.info.version,
        input_sha256=instance.sha256,
        gate_status=gate.status,
        gate_reasons=list(gate.reasons),
        gate_scores={
            k: (None if isinstance(v, float) and np.isnan(v) else float(v))
            for k, v in gate.scores.items()
        },
        findings=findings_json,
        input_stats=stats,
        sr_sop_uid=rb.sr_sop_uid,
        sc_sop_uids=[uid for _, uid, _ in rb.sc_items],
        sr_blob_key=sr_key,
        sc_blob_keys=list(sc_keys.values()),
        latency_ms=latency,
        request_id=request_id,
    )
    session.add(pred)
    session.flush()

    if push_to_archive:
        _push_archive(c, session, instance, raw, rb.sr_bytes, [d for _, _, d in rb.sc_items])

    log.info(
        "prediction",
        prediction_id=pid,
        instance_id=instance.id,
        gate=gate.status,
        n_present=sum(1 for f in findings if f.present),
        latency_ms=latency,
        model=mv.id,
    )
    return pred


def _image01(plugin: ModalityPlugin) -> Any:
    """Source-resolution [0,1] image for overlays, if the plugin exposes one (optional hook)."""
    fn = getattr(plugin, "display_image", None)
    return fn() if callable(fn) else None


def _push_archive(
    c: Container, session: Session, instance: Instance, image: bytes, sr: bytes, scs: list[bytes]
) -> None:
    """Best-effort push of de-identified image + results to Orthanc B (the platform archive)."""
    try:
        if not c.orthanc_b.reachable():
            log.warning("orthanc_b_unreachable")
            return
        r = c.orthanc_b.upload(image)
        for blob in [sr, *scs]:
            c.orthanc_b.upload(blob)
        study: Study | None = session.get(Study, instance.series.study_id)
        if study is not None and not study.orthanc_b_id:
            study.orthanc_b_id = str(r.get("ParentStudy", ""))
            session.flush()
    except Exception as exc:  # noqa: BLE001
        log.warning("orthanc_b_push_failed", error=str(exc))


__all__ = [
    "NoPluginError",
    "run_inference_for_instance",
    "ensure_model_version",
    "GATE_UNABLE_TO_ASSESS",
]
