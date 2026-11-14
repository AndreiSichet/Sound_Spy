import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// The backend (soundspy serve) listens on 127.0.0.1:8765 (api.host / api.port in
// backend/configs/default.yaml; keep both in sync). The proxy targets the IPv4 address
// explicitly: on Windows "localhost" may resolve to ::1 first, where uvicorn is not listening.
const BACKEND = 'http://127.0.0.1:8765'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/ws': { target: BACKEND, ws: true, changeOrigin: true },
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    include: ['tests/**/*.test.{ts,tsx}'],
  },
})
