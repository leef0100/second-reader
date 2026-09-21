// Thin API client. Base is /api behind the gateway; VITE_API_BASE overrides for dev.
export const API_BASE: string = (import.meta as any).env?.VITE_API_BASE ?? "/api";
export const RESEARCH_USE = "For research use only. Not for diagnostic use.";

let token: string | null = sessionStorage.getItem("sr_token");
export function setToken(t: string | null) {
  token = t;
  if (t) sessionStorage.setItem("sr_token", t);
  else sessionStorage.removeItem("sr_token");
}
export function hasToken() { return !!token; }

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string> | undefined) };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (init.body && !(init.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const r = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (r.status === 401) { setToken(null); window.location.hash = "#/login"; }
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* ignore */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json() as Promise<T>;
}

export function authedUrl(path: string): string {
  // Images are fetched via <img>, which cannot set headers → use fetch + object URL.
  return `${API_BASE}${path}`;
}

export async function fetchBlobUrl(path: string): Promise<string> {
  const r = await fetch(`${API_BASE}${path}`, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return URL.createObjectURL(await r.blob());
}

export interface StudySummary {
  id: string; pseudo_key: string; study_uid: string; study_date: string | null; modality: string | null;
  description: string | null; source: string | null; patient_age_years: number | null; patient_sex: string | null;
  n_instances: number; n_quarantined: number; prediction_status: string; n_priors: number;
  orthanc_b_id: string | null; created_at: string;
}
export interface InstanceSummary {
  id: string; sop_uid: string; instance_number: number | null; rows: number | null; columns: number | null;
  n_frames: number; quarantined: boolean; quarantine_reasons: string[]; view_position: string | null; manufacturer: string | null;
}
export interface StudyDetail {
  research_use: string; study: StudySummary; instances: InstanceSummary[]; priors: StudySummary[];
  predictions: { id: string; instance_id: string; gate_status: string; model_version_id: string; created_at: string }[];
  ohif_url: string | null;
}
export interface Finding {
  code: string; display_name: string; probability: number; threshold: number; present: boolean;
  boxes: { x_min: number; y_min: number; x_max: number; y_max: number }[];
  heatmap_url: string | null; overlay_url: string | null; snomed_code: string | null;
}
export interface PredictionOut {
  id: string; instance_id: string; model_version_id: string; plugin_version: string; gate_status: string;
  gate_reasons: string[]; findings: Finding[]; sr_url: string; created_at: string;
}
export interface ReviewSession {
  id: string; study_id: string; mode: string; started_at: string; revealed_at: string | null;
  completed_at: string | null; pre_read_impression: any;
}
export interface RevealOut { session: ReviewSession; predictions: PredictionOut[]; note: string }
