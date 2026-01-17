#!/usr/bin/env tsx

/**
 * Production Readiness Check
 * Ensures system is ready for production deployment
 */

import { existsSync, readFileSync } from 'fs';
import chalk from 'chalk';

interface Check {
  name: string;
  passed: boolean;
  message: string;
}

const checks: Check[] = [];

function check(name: string, condition: boolean, message: string = ''): void {
  checks.push({ name, passed: condition, message });
  const icon = condition ? chalk.green('✓') : chalk.red('✗');
  console.log(`${icon} ${name}`, message ? chalk.gray(`(${message})`) : '');
}

console.log(chalk.cyan.bold('\n🔍 Production Readiness Check\n'));

// Environment Variables
const envExists = existsSync('.env');
check('Environment file exists', envExists);

if (envExists) {
  const env = readFileSync('.env', 'utf-8');
  check('DATABASE_URL configured', env.includes('DATABASE_URL='));
  check('JWT_SECRET configured', env.includes('JWT_SECRET='));
}

// Required Files
check('README.md exists', existsSync('README.md'));
check('LICENSE exists', existsSync('LICENSE'));
check('package.json exists', existsSync('package.json'));
check('Dockerfile exists', existsSync('Dockerfile'));

// Build Artifacts
check('Client build exists', existsSync('client/dist'));
check('Server build exists', existsSync('dist'));

// Summary
const passed = checks.filter(c => c.passed).length;
const total = checks.length;

console.log(chalk.cyan(`\n📊 ${passed}/${total} checks passed\n`));

const criticalIssues = total - passed;

if (passed === total) {
  console.log(chalk.green.bold('✅ System is production ready!\n'));
} else {
  console.log(chalk.yellow.bold('⚠️  Some checks failed. Review before deploying.\n'));
}

export function runProductionReadinessCheck() {
  return {
    ready: passed === total,
    criticalIssues,
    totalChecks: total,
    passedChecks: passed,
    checks
  };
}
