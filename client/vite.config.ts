import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { resolve } from 'path';
import { fileURLToPath } from 'url';
import { stripConsolePlugin } from './vite-plugin-strip-console';

const __dirname = fileURLToPath(new URL('.', import.meta.url));

// Polyfill for Node.js globals
const nodePolyfills = {
  name: 'polyfill-node',
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  setup(build: any) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    build.onResolve({ filter: /^node:/ }, (args: any) => ({
      path: args.path.replace(/^node:/, ''),
      namespace: 'node',
    }));
  },
};

export default defineConfig(({ mode }) => {
  // Load env variables based on the current mode
  // eslint-disable-next-line no-undef
  const env = loadEnv(mode, process.cwd(), '');
  
  // Validate required environment variables
  const requiredVars = ['VITE_AUTH_API_URL', 'VITE_ANALYSIS_API_URL'];
  const missingVars = requiredVars.filter(varName => !env[varName]);
  
  if (missingVars.length > 0) {
    throw new Error(`Missing required environment variables: ${missingVars.join(', ')}`);
  }
  
  // Production plugins
  const plugins = [
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
  ];
  
  return {
    base: '/',
    plugins,
    define: {
      'process.env': {},
      global: 'globalThis',
      __dirname: JSON.stringify(''),
      __filename: JSON.stringify(''),
      // eslint-disable-next-line no-undef
      __APP_VERSION__: JSON.stringify(process.env?.npm_package_version || ''),
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
        port: 5173,
        protocol: 'ws',
      },
    },
    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'react-router-dom',
        '@tanstack/react-query',
      ],
      esbuildOptions: {
        target: 'esnext',
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: mode === 'development',
      commonjsOptions: {
        include: [/node_modules/],
        transformMixedEsModules: true,
      },
      minify: mode === 'production' ? 'esbuild' : false,
      target: 'esnext',
      chunkSizeWarningLimit: 1000,
      reportCompressedSize: false,
      rollupOptions: {
        output: {
          manualChunks: {
            react: ['react', 'react-dom', 'react-router-dom'],
            vendor: ['lodash', 'axios', 'date-fns'],
            ui: ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', '@radix-ui/react-toast'],
            state: ['@tanstack/react-query', 'zustand'],
            dashboard: ['./src/pages/Dashboard.tsx'],
            analysis: ['./src/pages/ImageAnalysis.tsx', './src/pages/VideoAnalysis.tsx', './src/pages/AudioAnalysis.tsx'],
            history: ['./src/pages/History.tsx'],
          },
          chunkFileNames: 'assets/[name]-[hash].js',
          entryFileNames: 'assets/[name]-[hash].js',
          assetFileNames: 'assets/[name]-[hash].[ext]',
        },
        plugins: [
          {
            name: 'replace-node-polyfills',
            resolveId(source) {
              if (source === 'node:process') return { id: 'process/browser', external: true };
              if (source === 'node:buffer') return { id: 'buffer', external: true };
              if (source === 'node:events') return { id: 'events', external: true };
              return null;
            },
          },
        ],
      },
    },
    css: {
      modules: {
        localsConvention: 'camelCaseOnly',
      },
      preprocessorOptions: {
        scss: {
          additionalData: `@import "@/styles/variables.scss";`,
        },
      },
    },
  };
});