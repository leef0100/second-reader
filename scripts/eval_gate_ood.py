"""Gate-only evaluation on unlabelled sets: what fraction does the active model refuse, and why?

Paediatric (Kermany) should be REJECTED (wrong population); NLM TB adult frontal sets should PASS.
For research use only. Not for diagnostic use.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from secondreader.core.plugins.protocol import FrameMeta, ImagingInput
from secondreader.modalities.cxr.data import load_image, load_manifest
from secondreader.modalities.cxr.plugin import CxrPlugin

model_id = sys.argv[1]
manifests = sys.argv[2:]
plugin = CxrPlugin.from_registry(Path("models"), model_id, device="auto")
rows = []
for m in manifests:
    df = load_manifest(m)
    for ds_name, part in df.groupby("dataset"):
        reasons: Counter[str] = Counter()
        dists, n_ok = [], 0
        for _, r in part.iterrows():
            arr, meta = load_image(r["path"])
            inp = ImagingInput(
                frames=np.asarray(arr)[None], frame_meta=(meta or FrameMeta(),), modality="CR",
                sop_class_uid="1.2.840.10008.5.1.4.1.1.1", body_part_examined="CHEST",
                view_position=str(r.get("view") or "PA"), patient_sex=None, patient_age_years=None,
                manufacturer=None, manufacturer_model=None, input_sha256=str(r["image_id"]),
            )
            g = plugin.quality_gate(inp)
            if g.ok:
                n_ok += 1
            for x in g.reasons:
                reasons["ood" if "out-of-distribution" in x else "view" if "view classifier" in x else x.split("(")[0].strip()] += 1
            if "ood_distance" in g.scores and not np.isnan(g.scores["ood_distance"]):
                dists.append(float(g.scores["ood_distance"]))
        n = len(part)
        rows.append({"dataset": ds_name, "n": n, "passed": n_ok, "pass_rate": round(n_ok / n, 4),
                     "reject_reasons": dict(reasons), "ood_distance_median": float(np.median(dists)) if dists else None,
                     "ood_threshold": plugin.bundle.ood.threshold if plugin.bundle.ood else None})
        print(rows[-1])
out = Path("docs/reports/gate_ood_eval"); out.mkdir(parents=True, exist_ok=True)
(out / "gate_ood_eval.json").write_text(json.dumps({"model_version_id": model_id, "rows": rows}, indent=2))
md = ["# Gate evaluation on out-of-scope / in-scope unlabelled sets", "", "> For research use only. Not for diagnostic use.", "",
      f"Model `{model_id}`. Expected: paediatric (Kermany) → rejected; NLM Shenzhen/Montgomery adult frontal → passed.", "",
      "| dataset | n | passed | pass rate | median OOD distance (threshold) | reject reasons |", "|---|---:|---:|---:|---|---|"]
for r in rows:
    md.append(f"| {r['dataset']} | {r['n']} | {r['passed']} | {r['pass_rate']:.1%} | {r['ood_distance_median']:.1f} ({r['ood_threshold']:.1f}) | {r['reject_reasons']} |")
(out / "gate_ood_eval.md").write_text("\n".join(md) + "\n")
print("wrote", out)
