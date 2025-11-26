#!/usr/bin/env node

/**
 * SatyaAI Startup Script
 * Starts all required services in the correct order
 */

import { spawn } from 'child_process';
import chalk from 'chalk';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const processes = [];
let isShuttingDown = false;

// Service configurations
const services = [
  {
    name: 'Node Server',
    command: 'npm',
    args: ['run', 'dev'],
    color: 'blue',
    port: 3000,
    healthCheck: 'http://localhost:3000/api/health'
  },
  {
    name: 'Python AI Service',
    command: 'npm',
    args: ['run', 'dev:python'],
    color: 'green',
    port: 5001,
    healthCheck: 'http://localhost:5001/health',
    delay: 2000 // Wait 2s after Node server
  },
  {
    name: 'React Client',
    command: 'npm',
    args: ['run', 'dev:client'],
    color: 'yellow',
    port: 5173,
    delay: 4000 // Wait 4s after Node server
  }
];

// Utility functions
const log = (service, message, type = 'info') => {
  const colors = {
    blue: chalk.blue,
    green: chalk.green,
    yellow: chalk.yellow,
    red: chalk.red
  };
  
  const color = colors[service.color] || chalk.white;
  const prefix = color(`[${service.name}]`);
  
  if (type === 'error') {
    console.error(`${prefix} ${chalk.red(message)}`);
  } else {
    console.log(`${prefix} ${message}`);
  }
};

const startService = (service) => {
  return new Promise((resolve) => {
    setTimeout(() => {
      log(service, `Starting on port ${service.port}...`);
      
      const proc = spawn(service.command, service.args, {
        stdio: 'pipe',
        shell: true,
        cwd: __dirname
      });

      proc.stdout.on('data', (data) => {
        const output = data.toString().trim();
        if (output) {
          log(service, output);
        }
      });

      proc.stderr.on('data', (data) => {
        const output = data.toString().trim();
        if (output && !output.includes('ExperimentalWarning')) {
          log(service, output, 'error');
        }
      });

      proc.on('error', (error) => {
        log(service, `Failed to start: ${error.message}`, 'error');
      });

      proc.on('exit', (code) => {
        if (!isShuttingDown) {
          log(service, `Exited with code ${code}`, code === 0 ? 'info' : 'error');
          if (code !== 0) {
            shutdown();
          }
        }
      });

      processes.push({ service, proc });
      log(service, chalk.green('✓ Started successfully'));
      resolve();
    }, service.delay || 0);
  });
};

const shutdown = () => {
  if (isShuttingDown) return;
  isShuttingDown = true;

  console.log(chalk.yellow('\n🛑 Shutting down all services...\n'));

  processes.forEach(({ service, proc }) => {
    try {
      log(service, 'Stopping...');
      proc.kill('SIGTERM');
      
      setTimeout(() => {
        if (!proc.killed) {
          proc.kill('SIGKILL');
        }
      }, 5000);
    } catch (error) {
      log(service, `Error stopping: ${error.message}`, 'error');
    }
  });

  setTimeout(() => {
    console.log(chalk.green('\n✓ All services stopped\n'));
    process.exit(0);
  }, 6000);
};

// Main startup sequence
const main = async () => {
  console.log(chalk.cyan.bold('\n🚀 Starting SatyaAI Platform...\n'));
  console.log(chalk.gray('Press Ctrl+C to stop all services\n'));

  try {
    // Start services sequentially
    for (const service of services) {
      await startService(service);
    }

    console.log(chalk.green.bold('\n✓ All services started successfully!\n'));
    console.log(chalk.cyan('📊 Access points:'));
    console.log(chalk.white('  • Frontend:  ') + chalk.blue('http://localhost:5173'));
    console.log(chalk.white('  • Backend:   ') + chalk.blue('http://localhost:3000'));
    console.log(chalk.white('  • AI Service:') + chalk.blue('http://localhost:5001'));
    console.log(chalk.white('  • API Docs:  ') + chalk.blue('http://localhost:3000/api-docs'));
    console.log('');

  } catch (error) {
    console.error(chalk.red(`\n❌ Startup failed: ${error.message}\n`));
    shutdown();
  }
};

// Handle shutdown signals
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
process.on('uncaughtException', (error) => {
  console.error(chalk.red(`\n❌ Uncaught exception: ${error.message}\n`));
  shutdown();
});

// Start the application
main().catch((error) => {
  console.error(chalk.red(`\n❌ Fatal error: ${error.message}\n`));
  process.exit(1);
});
