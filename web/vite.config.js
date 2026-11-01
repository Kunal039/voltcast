import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Builds into ../public, which Vercel serves from its CDN next to the FastAPI function.
// In development, /api calls are proxied to the local FastAPI server (uvicorn app:app --port 8000).
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../public", emptyOutDir: true, chunkSizeWarningLimit: 5000 },
  server: { proxy: { "/api": "http://localhost:8000" } },
});
