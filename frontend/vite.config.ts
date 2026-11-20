import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// The dashboard API (data-pipeline: `npm run api`) serves /api/v1/*: incidents,
// reviews, and the login and staff-account routes of cayla-backend mounted on it.
const api = process.env.VITE_API_PROXY ?? 'http://127.0.0.1:8000'

const proxy = { '/api': api }

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy },
  preview: { proxy },
})
