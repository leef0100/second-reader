import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, setToken, StudySummary } from "../lib/api";

export default function StudyList() {
  const nav = useNavigate();
  const [rows, setRows] = useState<StudySummary[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api<StudySummary[]>("/v1/studies").then(setRows).catch((e) => setErr(e.message)); }, []);
  return (
    <div className="page">
      <div className="topbar">
        <h1>Worklist</h1>
        <div>
          <Link to="/monitoring" style={{ marginRight: 12 }}>Monitoring</Link>
          <button className="secondary" onClick={() => { setToken(null); nav("/login"); }}>Sign out</button>
        </div>
      </div>
      {err && <p className="err">{err}</p>}
      <div className="card">
        <table>
          <thead><tr><th>Patient key</th><th>Study date</th><th>Modality</th><th>Description</th><th>Age/Sex</th><th>Images</th><th>Priors</th><th>AI status</th><th>Source</th></tr></thead>
          <tbody>
            {rows.map((s) => (
              <tr key={s.id} className="row" onClick={() => nav(`/study/${s.id}`)}>
                <td><code>{s.pseudo_key.slice(0, 12)}…</code></td>
                <td>{s.study_date ?? "—"}</td>
                <td>{s.modality ?? "—"}</td>
                <td>{s.description ?? "—"}</td>
                <td>{s.patient_age_years ?? "?"} / {s.patient_sex ?? "?"}</td>
                <td>{s.n_instances}{s.n_quarantined ? ` (${s.n_quarantined} quarantined)` : ""}</td>
                <td>{s.n_priors}</td>
                <td><span className={`pill ${s.prediction_status}`}>{s.prediction_status.replace("_", " ")}</span></td>
                <td className="label">{s.source}</td>
              </tr>
            ))}
            {rows.length === 0 && <tr><td colSpan={9} className="label">No studies yet. Send a DICOM to the hospital PACS (Orthanc A) or POST /v1/infer.</td></tr>}
          </tbody>
        </table>
      </div>
      <p className="label">Dates are shifted per patient during de-identification. Patient keys are pseudonyms; the same key across studies means the same (de-identified) patient, which is how priors are linked.</p>
    </div>
  );
}
