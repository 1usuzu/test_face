import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { nodePolyfills } from 'vite-plugin-node-polyfills'
import { resolve } from 'path'
import { fileURLToPath, URL } from 'node:url'

// https://vite.dev/config/
const __dirname = fileURLToPath(new URL('.', import.meta.url))
export default defineConfig({
  plugins: [
    react(),
    nodePolyfills({
      globals: { Buffer: true, global: true, process: true },
      protocolImports: true,
    }),
    {
      name: 'rewrite-consumer-routes',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url.startsWith('/consumer') && !req.url.includes('.')) {
            req.url = '/consumer/index.html'
          }
          next()
        })
      }
    }
  ],
  define: { global: 'globalThis' },
  resolve: { alias: { buffer: 'buffer' } },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        consumer: resolve(__dirname, 'consumer/index.html'),
      },
    },
  },
  appType: 'mpa',
})

