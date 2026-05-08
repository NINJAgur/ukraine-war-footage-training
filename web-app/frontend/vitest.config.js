import { defineConfig } from 'vitest/config'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'url'
import { resolve, dirname } from 'path'
const __dirname = dirname(fileURLToPath(import.meta.url))
export default defineConfig({
  plugins: [vue()],
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['tests/unit/**/*.test.js'],
    setupFiles: ['tests/setup.js'],
  },
  resolve: {
    alias: { '@': resolve(__dirname, 'src') },
  },
})
