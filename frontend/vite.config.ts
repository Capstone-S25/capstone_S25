import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// The dashboard API (data-pipeline: `npm run api`) serves /api/v1/*.
const api = process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8000'

// Login and staff accounts (cayla-backend: `docker compose up` in cayla-backend/).
const accounts = process.env.VITE_ACCOUNTS_PROXY ?? 'http://127.0.0.1:8001'

// Checked from top to bottom: the account paths go to cayla-backend, everything else to the API above.
const proxy = {
  '/api/v1/auth': accounts,
  '/api/v1/me': accounts,
  '/api/v1/users': accounts,
  '/api/v1/audit-log': accounts,
  '/api': api,
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy },
  preview: { proxy },
})
