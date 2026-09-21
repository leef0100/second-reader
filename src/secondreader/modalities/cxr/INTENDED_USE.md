# Chest radiograph plugin — intended use (module-level)

**For research use only. Not for diagnostic use.**

- **Input:** single-frame frontal (PA/AP) adult chest radiographs, DICOM CR/DX.
- **Output:** calibrated probability + Grad-CAM heatmap + coarse box for
  pneumothorax, pleural effusion, pulmonary nodule/mass; or `unable_to_assess`.
- **Not for:** lateral views, paediatric patients, non-chest images, screen
  captures, ultrasound/CT, or any patient-management decision.
- The platform-level document `docs/INTENDED_USE.md` is authoritative.
