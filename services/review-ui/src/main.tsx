import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter, Navigate, Route, Routes } from "react-router-dom";
import "./styles.css";
import { hasToken, RESEARCH_USE } from "./lib/api";
import Login from "./pages/Login";
import StudyList from "./pages/StudyList";
import Review from "./pages/Review";
import Monitoring from "./pages/Monitoring";

function Guard({ children }: { children: React.ReactElement }) {
  return hasToken() ? children : <Navigate to="/login" replace />;
}

function App() {
  return (
    <HashRouter>
      <div className="banner" role="note">{RESEARCH_USE}</div>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<Guard><StudyList /></Guard>} />
        <Route path="/study/:id" element={<Guard><Review /></Guard>} />
        <Route path="/monitoring" element={<Guard><Monitoring /></Guard>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <div className="banner bottom">{RESEARCH_USE}</div>
    </HashRouter>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
