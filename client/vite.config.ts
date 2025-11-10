import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

// Polyfill for Node.js globals
const nodePolyfills = {
  name: 'polyfill-node',
  setup(build) {
    build.onResolve({ filter: /^node:/ }, (args) => ({
      path: args.path.replace(/^node:/, ''),
      namespace: 'node',
    }));
  },
};

export default defineConfig({
  base: '/',
  plugins: [
    react({
      jsxImportSource: '@emotion/react',
      babel: {
        plugins: ['@emotion/babel-plugin'],
      },
    }),
    nodePolyfills,
  ],
  define: {
    'process.env': {},
    global: 'globalThis',
    __dirname: JSON.stringify(''),
    __filename: JSON.stringify(''),
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      'react': resolve(__dirname, 'node_modules/react'),
      'react-dom': resolve(__dirname, 'node_modules/react-dom'),
      'next/router': resolve(__dirname, 'src/utils/router'),
      'next/head': resolve(__dirname, 'src/utils/head'),
    },
    extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json'],
  },
  server: {
    port: 5173,
    strictPort: true,
    host: '0.0.0.0',
    open: true,
    proxy: {
      '/api': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        secure: false,
      },
    },
    hmr: {
      host: 'localhost',
      port: 3001
    }
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'react-router-dom'],
    esbuildOptions: {
      loader: {
        '.js': 'jsx',
      },
      // Node.js global to browser globalThis
      define: {
        global: 'globalThis',
      },
    },
  },
  build: {
    commonjsOptions: {
      include: [/node_modules/],
      transformMixedEsModules: true,
    },
    minify: 'esbuild',
    rollupOptions: {
      plugins: [
        {
          name: 'replace-node-polyfills',
          resolveId(source) {
            if (source === 'node:process') {
              return { id: 'process/browser', external: true };
            }
            if (source === 'node:buffer') {
              return { id: 'buffer', external: true };
            }
            if (source === 'node:events') {
              return { id: 'events', external: true };
            }
            return null;
          },
        },
      ],
    },
  }
});