import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import { stripConsolePlugin } from './vite-plugin-strip-console';

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
    stripConsolePlugin({
      include: ['console.log', 'console.debug'],
      exclude: ['console.warn', 'console.error', 'console.info'],
    }),
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
        target: 'http://localhost:5001',
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
    target: 'es2015',
    // Enable code splitting and tree shaking
    rollupOptions: {
      output: {
        // Manual chunks for better code splitting
        manualChunks: {
          // Vendor chunks
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          'ui-vendor': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', '@radix-ui/react-toast'],
          'query-vendor': ['@tanstack/react-query'],
          // Route-based chunks
          'dashboard': ['./src/pages/Dashboard.tsx'],
          'analysis': ['./src/pages/ImageAnalysis.tsx', './src/pages/VideoAnalysis.tsx', './src/pages/AudioAnalysis.tsx'],
          'history': ['./src/pages/History.tsx'],
        },
        // Optimize chunk size
        chunkFileNames: 'assets/[name]-[hash].js',
        entryFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash].[ext]',
      },
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
    // Warn on large chunks
    chunkSizeWarningLimit: 1000, // 1MB warning
    // Enable source maps for production debugging (optional)
    sourcemap: false,
  }
});