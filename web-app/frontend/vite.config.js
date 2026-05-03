import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    open: '/',
    proxy: {
      '/api':             'http://localhost:8001',
      '/media/annotated': 'http://localhost:8001',
    },
  },
})
