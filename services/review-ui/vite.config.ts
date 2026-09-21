import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Served under /review/ behind the gateway; API at /api/ (gateway) or :8000 (dev).
export default defineConfig({
  plugins: [react()],
  base: "/review/",
  server: {
    port: 5173,
    proxy: { "/api": { target: "http://localhost:8000", changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") } },
  },
});
