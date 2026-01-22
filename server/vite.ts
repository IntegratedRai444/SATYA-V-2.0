import express, { type Express } from "express";
import fs from "fs";
import path from "path";
import { type Server } from "http";

// Vite dynamic imports - using any for dynamic imports to avoid type conflicts
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let vite: any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let createViteServer: any;

// This will be populated when Vite is loaded
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let viteServer: any;

// Get directory name in CommonJS
const currentDir = process.cwd();

// Simple logger since createLogger was removed
const viteLogger = {
  info: (msg: string) => console.log(`[vite] ${msg}`),
  warn: (msg: string) => console.warn(`[vite] ${msg}`),
  error: (msg: string, options?: unknown) => {
    console.error(`[vite] ${msg}`, options || '');
    process.exit(1);
  }
};

export function log(message: string, source = "express") {
  const formattedTime = new Date().toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });

  console.log(`${formattedTime} [${source}] ${message}`);
}

export async function setupVite(app: Express, server: Server) {
  try {
    // Dynamically import Vite
    vite = await import('vite');
    createViteServer = vite.createServer;

    const serverOptions = {
      middlewareMode: true,
      hmr: { server },
      allowedHosts: true,
    };

    // Create Vite server
    viteServer = await createViteServer({
      configFile: false,
      server: serverOptions,
      appType: "custom",
      logLevel: 'info',
      customLogger: viteLogger
    });

    // Add Vite's middleware
    if (viteServer.middlewares) {
      app.use(viteServer.middlewares);
    } else {
      console.warn('Vite middleware not available');
    }
    app.use("/*", async (req, res, next) => {
      const url = req.originalUrl;

      try {
        const template = fs.readFileSync(path.resolve(currentDir, 'index.html'), 'utf-8');
        
        if (!viteServer || !viteServer.ssrLoadModule) {
          throw new Error('Vite server not properly initialized');
        }
        
        const { render } = await viteServer.ssrLoadModule('/src/entry-server.ts');
        const appHtml = await render(url);
        const html = template.replace('<!--ssr-outlet-->', appHtml);
        res.status(200).set({ 'Content-Type': 'text/html' }).end(html);
      } catch (e) {
      viteServer.ssrFixStacktrace(e as Error);
        next(e);
      }
    });
  } catch (error) {
    console.error('Failed to initialize Vite:', error);
    throw error;
  }
}

export function serveStatic(app: Express) {
  const distPath = path.resolve(__dirname, "public");

  if (!fs.existsSync(distPath)) {
    throw new Error(
      `Could not find the build directory: ${distPath}, make sure to build the client first`,
    );
  }

  app.use(express.static(distPath));

  // fall through to index.html if the file doesn't exist
  app.use("/*", (_req, res) => {
    res.sendFile(path.resolve(distPath, "index.html"));
  });
}
