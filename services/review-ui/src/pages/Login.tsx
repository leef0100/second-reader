import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, setToken } from "../lib/api";

export default function Login() {
  const nav = useNavigate();
  const [u, setU] = useState("");
  const [p, setP] = useState("");
  const [err, setErr] = useState<string | null>(null);
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    try {
      const r = await api<{ access_token: string }>("/v1/auth/login", { method: "POST", body: JSON.stringify({ username: u, password: p }) });
      setToken(r.access_token);
      nav("/");
    } catch (e: any) { setErr(e.message); }
  }
  return (
    <div className="page" style={{ maxWidth: 420 }}>
      <div className="card">
        <h1 style={{ marginTop: 0 }}>Second Reader — clinician sign-in</h1>
        <p className="label">Every study view, reveal and feedback action is recorded in the audit log.</p>
        <form onSubmit={submit}>
          <label className="label">Username<input value={u} onChange={(e) => setU(e.target.value)} autoComplete="username" /></label>
          <label className="label" style={{ display: "block", marginTop: 8 }}>Password<input type="password" value={p} onChange={(e) => setP(e.target.value)} autoComplete="current-password" /></label>
          {err && <p className="err">{err}</p>}
          <button style={{ marginTop: 12 }} type="submit">Sign in</button>
        </form>
      </div>
    </div>
  );
}
