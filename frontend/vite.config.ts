/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// In development the API runs on :8000; the proxy keeps everything on one origin (no CORS), like production.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { '/api': 'http://localhost:8000', '/docs': 'http://localhost:8000', '/openapi.json': 'http://localhost:8000' } },
  test: { environment: 'jsdom', globals: true, setupFiles: ['./tests/setup.ts'], css: false },
})
