# OHIF / Orthanc B compatibility of the Second Reader SR

> For research use only. Not for diagnostic use.

Status as of **2026-09-22**. Concerns the DICOM SR written by
`src/secondreader/core/results/writer.py` after D-029 (Comprehensive 3D SR, TID 1500,
IHE AI Results provenance) and how it reaches OHIF through Orthanc B (D-002).

## What is verified automatically

`tests/integration/test_ohif_sr_compat.py` (marker `integration`; skips when Orthanc B
is not reachable at `SR_ORTHANC_B_URL`, default `http://localhost:8043`). Run with
Orthanc B up (`docker compose up -d orthanc-b`, image `orthancteam/orthanc:24.10.1`):

```bash
uv run pytest -q -m integration tests/integration/test_ohif_sr_compat.py
```

Passed on 2026-09-22 against `orthancteam/orthanc:24.10.1` (Orthanc API version 24,
DICOMweb plugin enabled, `/dicom-web/` root). The test:

1. Builds a synthetic 256×256 CR film (`tests/fixtures/synthetic.py`, fake PHI only),
   de-identifies it with the project profile, and writes our results for three findings:
   pneumothorax **present** with a box, pleural effusion **absent** without a box,
   nodule/mass **absent** with a box.
2. Uploads the de-identified image and the SR to Orthanc B (`POST /instances`) and
   asserts the SR lands in the **same Orthanc study** as the image (the study-level
   lookup OHIF uses to find the referenced instance).
3. **QIDO-RS**: `GET /dicom-web/studies?StudyInstanceUID=…` finds the study;
   `/series` lists exactly `CR` + `SR`; `/series/{sr}/instances` lists one instance with
   `SOPClassUID = 1.2.840.10008.5.1.4.1.1.88.34` (Comprehensive 3D SR Storage) and our
   SOP Instance UID; `/instances` of the study contains the referenced image's SOP UID.
4. **WADO-RS metadata JSON** (the representation OHIF's `dicomweb` data source and SR
   extension parse), checking:
   * root `ConceptNameCodeSequence = (126000, DCM, Imaging Measurement Report)` with a
     `ContentSequence` — OHIF's precondition for treating the SR as a measurement report;
   * `ContributingEquipmentSequence` (0018,A001) contains our item: purpose
     `(109102, DCM, Processing Equipment)`, `ManufacturerModelName = secondreader-cxr`,
     `SoftwareVersions = <model version id>` (highdicom adds the source acquisition
     equipment and itself alongside);
   * three `(125007, DCM, Measurement Group)` containers, each with an
     `ObservationUID` (0040,A171);
   * two `SCOORD` items named `(111030, DCM, Image Region)`, `GraphicType = POLYLINE`,
     `GraphicData` of 10 values (5 closed points), each with an `IMAGE` child whose
     `ReferencedSOPSequence` points at the uploaded image's SOP Instance UID;
   * codes `112040` (Tracking Unique Identifier), `373572006` (Clinical finding absent —
     exactly for the two absent findings, valued with their finding codes), `130400`
     (Geometric purpose of region) valued `75958009` (Bounded by), `246112005`
     (Presence) and `99SRD-PROB` (calibrated probability) all present.
5. **WADO-RS instance retrieve** (`multipart/related; type="application/dicom"`): the
   bytes parse with pydicom; `ContentSequence` and `ContributingEquipmentSequence` are
   **element-for-element equal** to what we wrote; `VerificationFlag = UNVERIFIED`,
   `PreliminaryFlag = PRELIMINARY`; no fake PHI survives. highdicom re-reads it as a
   `Comprehensive3DSR` and `get_planar_roi_measurement_groups()` returns the two box
   groups as `POLYLINE` with shape `(5, 2)`.
6. Deletes the study from Orthanc B afterwards.

`tests/unit/test_results_writer.py` covers the same content offline (parse-back with
highdicom) including the gate-only "unable to assess" SR.

### OHIF hydration preconditions, checked in Python

From the survey of OHIF's SR extension (`docs/research/open_source_deployment.md` §5a,
analysed on OHIF master 3.14.0-beta.35): a measurement group hydrates to a
`RectangleROI` when its `SCOORD` is `(DCM 111030, POLYLINE, 5 points)`, the group has a
`TrackingUniqueIdentifier`, and the referenced SOP Instance is in the same study. All
three hold for both box groups after the Orthanc round trip (steps 3–5 above). The
absent-without-box group and the gate-only SR have no `SCOORD`, so they are display-only
(text) by design.

## What remains manual

| Item | Why it is manual | How to check |
|---|---|---|
| Rendering in the **pinned OHIF `v3.9.2`** UI (`docker-compose.yml`) | The SR-extension analysis was done against OHIF master; the 3.9.2 code path was not executed here, and no headless OHIF harness exists in the repo | `docker compose up -d` (full stack), upload a de-identified image + SR to Orthanc B (or run the integration test with the `finally:` cleanup commented out), open `http://localhost:3000/viewer?StudyInstanceUIDs=<uid>`, confirm the SR series appears in the study panel, the measurement panel lists the two box findings, and the boxes draw on the CR image |
| Hydration to editable `RectangleROI` annotations | Requires clicking "load measurements" in the UI | Same session; the measurement panel should offer to hydrate; boxes become selectable rectangles |
| Display path of the gate-only SR (no region, `TrackingIdentifier = "gate"`) | OHIF may route an SR without `SCOORD` to the text viewport rather than the measurement panel | Upload results for a gated-out film (e.g. `view_position="LL"`); note which viewport OHIF picks and that `UNABLE TO ASSESS` and the reasons text are readable |
| Behaviour of `ContributingEquipmentSequence` / `ObservationUID` in OHIF | OHIF ignores both (harmless); no rendering to verify | — |
| Orthanc `1.13.0` vs pinned `24.10.1` image; OHIF `v3.12.18` vs `v3.9.2` | Version bumps deferred (survey §6.5) | Re-run the integration test after bumping the image tags |
| STOW-RS (`/dicom-web/studies`) instead of `POST /instances` for AIR RAD-108 conformance | Not implemented (D-029 "still open") | — |

## Known limitations of the SR for viewers (unchanged)

* 2D `SCOORD` only (no `SCOORD3D`); one region per finding (first box).
* TID 1600 image library omitted (D-022): OHIF does not need it for hydration, but
  some SR viewers list the image library as the evidence index.
* `PreliminaryFlag = PRELIMINARY`, `VerificationFlag = UNVERIFIED` — always; a clinician
  verifies, never the system (D-022).
