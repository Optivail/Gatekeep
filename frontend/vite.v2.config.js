// Build for the v2 dashboard only: `npx vite build --config vite.v2.config.js`.
// Writes dist/v2/index.html + hashed assets next to the classic build
// (emptyOutDir off), so both UIs are served from /dashboard/.
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

export default defineConfig({
  plugins: [react()],
  base: '/dashboard/',
  build: {
    outDir: 'dist',
    emptyOutDir: false,
    rollupOptions: { input: { v2: resolve(import.meta.dirname, 'v2/index.html') } },
  },
})
