"""Orthanc A (hospital PACS) plugin: notify the platform when a study becomes stable.

Runs inside Orthanc's embedded Python. It does NOT push pixel data anywhere — it
only tells the platform *which* study is ready; the platform pulls, de-identifies,
and never writes anything back to this PACS.
"""

import json
import os
import urllib.request

import orthanc  # provided by the Orthanc Python plugin at runtime

PLATFORM_URL = os.environ.get("SR_PLATFORM_URL", "http://api:8000").rstrip("/")
TOKEN = os.environ.get("SR_WEBHOOK_TOKEN", "")


def on_change(change_type, level, resource_id):
    if change_type != orthanc.ChangeType.STABLE_STUDY:
        return
    body = json.dumps({"orthanc_study_id": resource_id, "source": "orthanc-a"}).encode()
    req = urllib.request.Request(
        f"{PLATFORM_URL}/v1/ingest/orthanc",
        data=body,
        headers={"Content-Type": "application/json", "X-Webhook-Token": TOKEN},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            orthanc.LogInfo(f"[second-reader] notified platform of study {resource_id}: {resp.status}")
    except Exception as exc:  # noqa: BLE001
        orthanc.LogWarning(f"[second-reader] platform notification failed for {resource_id}: {exc}")


orthanc.RegisterOnChangeCallback(on_change)
