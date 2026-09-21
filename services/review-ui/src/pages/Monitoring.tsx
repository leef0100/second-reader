import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";

export default function Monitoring() {
  const [drift, setDrift] = useState<any>(null);
  const [models, setModels] = useState<any[]>([]);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    api<any>("/v1/monitoring/drift?days=30").then(setDrift).catch((e) => setErr(e.message));
    api<any[]>("/v1/monitoring/models").then(setModels).catch(() => {});
  }, []);
  return (
    <div className="page">
      <div className="topbar"><h1><Link to="/">Worklist</Link> / Monitoring</h1></div>
      {err && <p className="err">{err}</p>}
      {drift && (
        <div className="card">
          <strong>Drift — model <code>{drift.model_version_id}</code> · last 30 days · {drift.n_predictions} predictions · gated out {drift.n_gated_out}</strong>
          <ul>{(drift.flags as string[]).length ? drift.flags.map((f: string) => <li key={f} className="err">⚠ {f}</li>) : <li className="label">no flags</li>}</ul>
          <div className="grid">
            <table><thead><tr><th>input statistic</th><th>PSI</th></tr></thead><tbody>
              {Object.entries(drift.input_psi as Record<string, number | null>).map(([k, v]) => <tr key={k}><td>{k}</td><td>{v == null ? "—" : v.toFixed(3)}</td></tr>)}
            </tbody></table>
            <table><thead><tr><th>finding</th><th>PSI</th><th>mean p</th><th>ref mean p</th><th>flag rate</th><th>agree / disagree / unsure</th></tr></thead><tbody>
              {Object.keys(drift.output_psi).map((c) => { const f = drift.feedback?.[c] ?? {}; return (
                <tr key={c}><td>{c}</td><td>{drift.output_psi[c]?.toFixed?.(3) ?? "—"}</td><td>{drift.output_mean[c]?.toFixed?.(3) ?? "—"}</td><td>{drift.reference_output_mean[c]?.toFixed?.(3) ?? "—"}</td><td>{drift.positive_rate[c]?.toFixed?.(2) ?? "—"}</td><td>{f.agree ?? 0} / {f.disagree ?? 0} / {f.unsure ?? 0}</td></tr>
              ); })}
            </tbody></table>
          </div>
        </div>
      )}
      <div className="card" style={{ marginTop: 12 }}>
        <strong>Model versions</strong>
        <table><thead><tr><th>id</th><th>modality</th><th>name</th><th>weights sha256</th><th>split manifest</th><th>predictions</th><th>created</th></tr></thead><tbody>
          {models.map((m) => <tr key={m.id}><td><code>{m.id}</code></td><td>{m.modality_id}</td><td>{m.name}</td><td><code>{String(m.weights_sha256).slice(0, 12)}</code></td><td><code>{String(m.split_manifest_sha256 ?? "").slice(0, 12)}</code></td><td>{m.n_predictions}</td><td>{m.created_at}</td></tr>)}
        </tbody></table>
      </div>
    </div>
  );
}
