import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, fetchBlobUrl, Finding, PredictionOut, RevealOut, ReviewSession, StudyDetail, StudySummary } from "../lib/api";

const CODES = [
  { code: "pneumothorax", name: "Pneumothorax" },
  { code: "pleural_effusion", name: "Pleural effusion" },
  { code: "nodule_mass", name: "Pulmonary nodule/mass" },
];

function useImage(path: string | null) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    let alive = true; let obj: string | null = null;
    setUrl(null);
    if (path) fetchBlobUrl(path).then((u) => { if (alive) { obj = u; setUrl(u); } else URL.revokeObjectURL(u); }).catch(() => setUrl(null));
    return () => { alive = false; if (obj) URL.revokeObjectURL(obj); };
  }, [path]);
  return url;
}

function Viewer({ title, imagePath, overlayPath, boxes, children }: { title: string; imagePath: string | null; overlayPath?: string | null; boxes?: Finding["boxes"]; children?: React.ReactNode }) {
  const img = useImage(imagePath);
  const ov = useImage(overlayPath ?? null);
  return (
    <div className="card">
      <div className="topbar"><strong>{title}</strong>{children}</div>
      <div className="viewer">
        {img ? <img src={img} alt={title} /> : <span className="label">{imagePath ? "loading…" : "no image"}</span>}
        {ov && <img className="overlay" src={ov} alt="AI heatmap" />}
        {img && boxes?.map((b, i) => (
          <div key={i} className="box" style={{ left: `${b.x_min * 100}%`, top: `${b.y_min * 100}%`, width: `${(b.x_max - b.x_min) * 100}%`, height: `${(b.y_max - b.y_min) * 100}%` }} />
        ))}
      </div>
    </div>
  );
}

export default function Review() {
  const { id } = useParams();
  const [detail, setDetail] = useState<StudyDetail | null>(null);
  const [prior, setPrior] = useState<StudySummary | null>(null);
  const [priorDetail, setPriorDetail] = useState<StudyDetail | null>(null);
  const [session, setSession] = useState<ReviewSession | null>(null);
  const [mode, setMode] = useState<"read_first" | "immediate">("read_first");
  const [seen, setSeen] = useState<Record<string, boolean>>({});
  const [notes, setNotes] = useState("");
  const [reveal, setReveal] = useState<RevealOut | null>(null);
  const [shown, setShown] = useState<{ pred: PredictionOut; f: Finding } | null>(null);
  const [fb, setFb] = useState<Record<string, string>>({});
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { if (id) api<StudyDetail>(`/v1/studies/${id}`).then((d) => { setDetail(d); setPrior(d.priors[0] ?? null); }).catch((e) => setErr(e.message)); }, [id]);
  useEffect(() => { if (prior) api<StudyDetail>(`/v1/studies/${prior.id}`).then(setPriorDetail); else setPriorDetail(null); }, [prior]);

  const instance = useMemo(() => detail?.instances.find((i) => !i.quarantined) ?? null, [detail]);
  const priorInstance = useMemo(() => priorDetail?.instances.find((i) => !i.quarantined) ?? null, [priorDetail]);
  const imagePath = detail && instance ? `/v1/studies/${detail.study.id}/instances/${instance.id}/preview.png` : null;
  const priorPath = priorDetail && priorInstance ? `/v1/studies/${priorDetail.study.id}/instances/${priorInstance.id}/preview.png` : null;

  async function start() {
    setErr(null);
    try { setSession(await api<ReviewSession>("/v1/review/sessions", { method: "POST", body: JSON.stringify({ study_id: id, mode }) })); } catch (e: any) { setErr(e.message); }
  }
  async function recordRead() {
    if (!session) return;
    try { setSession(await api<ReviewSession>(`/v1/review/sessions/${session.id}/impression`, { method: "POST", body: JSON.stringify({ findings: seen, notes }) })); } catch (e: any) { setErr(e.message); }
  }
  async function doReveal() {
    if (!session) return;
    try { const r = await api<RevealOut>(`/v1/review/sessions/${session.id}/reveal`, { method: "POST" }); setReveal(r); setSession(r.session); } catch (e: any) { setErr(e.message); }
  }
  async function sendFeedback(pred: PredictionOut, f: Finding, agreement: string) {
    if (!session) return;
    try {
      await api(`/v1/review/sessions/${session.id}/feedback`, { method: "POST", body: JSON.stringify({ prediction_id: pred.id, finding_code: f.code, agreement }) });
      setFb({ ...fb, [`${pred.id}:${f.code}`]: agreement });
    } catch (e: any) { setErr(e.message); }
  }
  async function complete() { if (session) setSession(await api<ReviewSession>(`/v1/review/sessions/${session.id}/complete`, { method: "POST" })); }

  if (!detail) return <div className="page"><p className="label">{err ?? "loading…"}</p></div>;
  const s = detail.study;
  const canReveal = !!session && (session.mode === "immediate" || !!session.pre_read_impression) && !session.revealed_at;

  return (
    <div className="page">
      <div className="topbar">
        <h1><Link to="/">Worklist</Link> / study <code>{s.study_uid.slice(-10)}</code> · {s.modality} · {s.study_date ?? "—"} · age {s.patient_age_years ?? "?"} {s.patient_sex ?? ""}</h1>
        <div>{detail.ohif_url && <a href={detail.ohif_url} target="_blank" rel="noreferrer">Open in OHIF ↗</a>}</div>
      </div>
      {err && <p className="err">{err}</p>}
      <div className="grid">
        <Viewer title="Current study" imagePath={imagePath} overlayPath={shown?.f.heatmap_url ?? null} boxes={shown?.f.boxes}>
          {shown && <span className="label">showing heatmap: {shown.f.display_name} <button className="secondary" onClick={() => setShown(null)}>hide</button></span>}
        </Viewer>
        <Viewer title={prior ? `Prior · ${prior.study_date ?? "—"}` : "Prior study"} imagePath={priorPath}>
          {detail.priors.length > 0 ? (
            <select style={{ width: "auto" }} value={prior?.id ?? ""} onChange={(e) => setPrior(detail.priors.find((p) => p.id === e.target.value) ?? null)}>
              {detail.priors.map((p) => <option key={p.id} value={p.id}>{p.study_date ?? "—"} · {p.description ?? p.modality}</option>)}
            </select>
          ) : <span className="label">no priors for this patient key</span>}
        </Viewer>
      </div>

      <div className="grid3" style={{ marginTop: 12 }}>
        <div className="card">
          <strong>1 · Your read (recorded before AI is shown)</strong>
          {!session ? (
            <div style={{ marginTop: 8 }}>
              <label className="label">Mode <select style={{ width: "auto", marginLeft: 8 }} value={mode} onChange={(e) => setMode(e.target.value as any)}>
                <option value="read_first">Read first, then reveal AI (recommended)</option>
                <option value="immediate">Reveal immediately (logged)</option>
              </select></label>
              <button style={{ marginTop: 8 }} onClick={start}>Start review</button>
            </div>
          ) : (
            <div className="checks" style={{ marginTop: 8 }}>
              {CODES.map((c) => (
                <label key={c.code}><input type="checkbox" style={{ width: "auto", marginRight: 8 }} disabled={!!session.pre_read_impression} checked={!!seen[c.code]} onChange={(e) => setSeen({ ...seen, [c.code]: e.target.checked })} /> I see {c.name.toLowerCase()}</label>
              ))}
              <textarea rows={2} placeholder="notes (optional)" value={notes} disabled={!!session.pre_read_impression} onChange={(e) => setNotes(e.target.value)} />
              <div className="row-actions" style={{ marginTop: 8 }}>
                <button className="secondary" disabled={!!session.pre_read_impression} onClick={recordRead}>{session.pre_read_impression ? "Read recorded ✓" : "Record my read"}</button>
                <button disabled={!canReveal} onClick={doReveal}>{session.revealed_at ? "AI revealed" : "Reveal AI second read"}</button>
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <strong>2 · AI second read</strong>
          {!reveal ? (
            <div className="locked" style={{ marginTop: 8 }}>AI output is hidden until you record your own read.<br /><span className="label">Anti-automation-bias: the model must not anchor your interpretation.</span></div>
          ) : (
            <div style={{ marginTop: 8 }}>
              {reveal.predictions.length === 0 && <p className="label">No AI result for this study yet (pending or no eligible image).</p>}
              {reveal.predictions.map((pred) => (
                <div key={pred.id}>
                  <p className="label">model <code>{pred.model_version_id}</code> · plugin {pred.plugin_version} · <a href={`#`} onClick={(e) => { e.preventDefault(); fetchBlobUrl(pred.sr_url).then((u) => window.open(u)); }}>DICOM SR</a></p>
                  {pred.gate_status !== "ok" ? (
                    <div className="finding" style={{ borderColor: "var(--bad)" }}><strong className="err">Unable to assess</strong><ul>{pred.gate_reasons.map((r) => <li key={r}>{r}</li>)}</ul></div>
                  ) : pred.findings.map((f) => {
                    const key = `${pred.id}:${f.code}`;
                    return (
                      <div key={f.code} className={`finding ${f.present ? "present" : ""}`}>
                        <div className="topbar" style={{ marginBottom: 2 }}>
                          <strong>{f.display_name} {f.present ? <span className="pill pending">flagged</span> : <span className="pill none">below threshold</span>}</strong>
                          <span>p = {f.probability.toFixed(2)}</span>
                        </div>
                        <div className="bar" title={`calibrated probability ${f.probability.toFixed(3)}; operating threshold ${f.threshold.toFixed(2)}`}>
                          <i style={{ width: `${f.probability * 100}%` }} /><b style={{ left: `${f.threshold * 100}%` }} />
                        </div>
                        <div className="row-actions">
                          <button className="secondary" onClick={() => setShown({ pred, f })}>Show heatmap / box</button>
                          <button className={fb[key] === "agree" ? "" : "secondary"} onClick={() => sendFeedback(pred, f, "agree")}>Agree</button>
                          <button className={fb[key] === "disagree" ? "" : "secondary"} onClick={() => sendFeedback(pred, f, "disagree")}>Disagree</button>
                          <button className={fb[key] === "unsure" ? "" : "secondary"} onClick={() => sendFeedback(pred, f, "unsure")}>Unsure</button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ))}
              <button style={{ marginTop: 8 }} disabled={!!session?.completed_at} onClick={complete}>{session?.completed_at ? "Review completed ✓" : "Complete review"}</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
