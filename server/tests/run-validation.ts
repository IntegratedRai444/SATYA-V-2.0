#!/usr/bin/env tsx

/**
 * System Validation Script
 * Validates all system components are working correctly
 */

import axios from 'axios';
import chalk from 'chalk';

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';
const PYTHON_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:5001';

interface ValidationResult {
  name: string;
  passed: boolean;
  message: string;
}

const results: ValidationResult[] = [];

async function validate(name: string, fn: () => Promise<boolean>): Promise<void> {
  try {
    const passed = await fn();
    results.push({ name, passed, message: passed ? 'OK' : 'Failed' });
    console.log(passed ? chalk.green('✓') : chalk.red('✗'), name);
  } catch (error: any) {
    results.push({ name, passed: false, message: error.message });
    console.log(chalk.red('✗'), name, chalk.gray(`(${error.message})`));
  }
}

async function runValidation() {
  console.log(chalk.cyan.bold('\n🔍 Running System Validation...\n'));

  // Node.js Server Health
  await validate('Node.js Server Health', async () => {
    const res = await axios.get(`${BASE_URL}/api/health`);
    return res.status === 200;
  });

  // Python Service Health
  await validate('Python AI Service Health', async () => {
    const res = await axios.get(`${PYTHON_URL}/health`);
    return res.status === 200;
  });

  // Database Connection
  await validate('Database Connection', async () => {
    const res = await axios.get(`${BASE_URL}/api/health`);
    return res.data.database === 'connected';
  });

  // Summary
  const passed = results.filter(r => r.passed).length;
  const total = results.length;
  
  console.log(chalk.cyan(`\n📊 Results: ${passed}/${total} checks passed\n`));
  
  if (passed === total) {
    console.log(chalk.green.bold('✅ All validations passed!\n'));
    process.exit(0);
  } else {
    console.log(chalk.red.bold('❌ Some validations failed\n'));
    process.exit(1);
  }
}

runValidation();
