import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Get current directory
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.join(__dirname, '..');

// Configuration
const IGNORE_DIRS = [
  'node_modules',
  '.git',
  '.next',
  'dist',
  'build',
  'coverage',
  '.vscode',
  '.github',
  '__mocks__',
  'cypress',
  'public',
  'types',
  '.husky',
  '.vercel',
  '.netlify'
];

const IGNORE_FILES = [
  '.DS_Store',
  '*.log',
  '*.md',
  '*.mdx',
  '*.json',
  '*.lock',
  '*.d.ts',
  '*.test.*',
  '*.spec.*',
  '*.stories.*',
  '*.config.*',
  '*.min.*',
  'babel.config.*',
  'tsconfig.*',
  'next-env.d.ts',
  '*.d.ts'
];

// Track file counts
const fileStats = {
  total: 0,
  byExtension: {} as Record<string, number>,
  byDirectory: {} as Record<string, number>,
  largestFiles: [] as Array<{ path: string; size: number; }>,
  directorySizes: {} as Record<string, number>
};

// Function to check if a file should be ignored
function shouldIgnore(filePath: string): boolean {
  const relativePath = path.relative(projectRoot, filePath);
  
  // Check if in ignored directories
  for (const dir of IGNORE_DIRS) {
    if (relativePath.split(path.sep).includes(dir)) {
      return true;
    }
  }
  
  // Check if matches ignored file patterns
  const fileName = path.basename(filePath);
  return IGNORE_FILES.some(pattern => {
    if (pattern.startsWith('*')) {
      return fileName.endsWith(pattern.substring(1));
    }
    return fileName === pattern;
  });
}

// Function to get human-readable file size
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

// Analyze a single file
function analyzeFile(filePath: string): void {
  if (shouldIgnore(filePath)) return;
  
  const stats = fs.statSync(filePath);
  const ext = path.extname(filePath).toLowerCase() || '.none';
  const dir = path.dirname(path.relative(projectRoot, filePath));
  
  // Update counters
  fileStats.total++;
  fileStats.byExtension[ext] = (fileStats.byExtension[ext] || 0) + 1;
  fileStats.byDirectory[dir] = (fileStats.byDirectory[dir] || 0) + 1;
  
  // Track directory sizes
  let currentDir = dir;
  while (currentDir) {
    fileStats.directorySizes[currentDir] = (fileStats.directorySizes[currentDir] || 0) + stats.size;
    const parentDir = path.dirname(currentDir);
    if (parentDir === currentDir) break;
    currentDir = parentDir;
  }
  
  // Track largest files
  fileStats.largestFiles.push({ path: filePath, size: stats.size });
  fileStats.largestFiles.sort((a, b) => b.size - a.size);
  if (fileStats.largestFiles.length > 10) {
    fileStats.largestFiles.pop();
  }
}

// Recursively analyze a directory
function analyzeDirectory(dirPath: string): void {
  const files = fs.readdirSync(dirPath);
  
  for (const file of files) {
    const fullPath = path.join(dirPath, file);
    
    try {
      const stats = fs.statSync(fullPath);
      
      if (stats.isDirectory()) {
        if (!shouldIgnore(fullPath)) {
          analyzeDirectory(fullPath);
        }
      } else if (stats.isFile()) {
        analyzeFile(fullPath);
      }
    } catch (error) {
      console.warn(`Error processing ${fullPath}:`, error);
    }
  }
}

// Generate report
function generateReport(): void {
  // Sort extensions by count
  const sortedExtensions = Object.entries(fileStats.byExtension)
    .sort((a, b) => b[1] - a[1]);
  
  // Sort directories by file count
  const sortedDirs = Object.entries(fileStats.byDirectory)
    .filter(([dir]) => dir) // Filter out root directory
    .sort((a, b) => b[1] - a[1]);
  
  // Sort directories by size
  const sortedDirSizes = Object.entries(fileStats.directorySizes)
    .filter(([dir]) => dir) // Filter out root directory
    .sort((a, b) => b[1] - a[1]);
  
  console.log('\n=== Project Analysis Report ===\n');
  
  // Basic stats
  console.log(`Total Files: ${fileStats.total}`);
  console.log(`Unique Extensions: ${sortedExtensions.length}\n`);
  
  // File types
  console.log('File Types (Top 10):');
  console.log('-------------------');
  sortedExtensions.slice(0, 10).forEach(([ext, count]) => {
    const percentage = ((count / fileStats.total) * 100).toFixed(1);
    console.log(`${ext.padEnd(10)}: ${count.toString().padEnd(6)} (${percentage}%)`);
  });
  
  // Largest directories by file count
  console.log('\nLargest Directories (by file count):');
  console.log('--------------------------------');
  sortedDirs.slice(0, 10).forEach(([dir, count]) => {
    const percentage = ((count / fileStats.total) * 100).toFixed(1);
    console.log(`${dir.padEnd(50)}: ${count.toString().padEnd(4)} files (${percentage}%)`);
  });
  
  // Largest directories by size
  console.log('\nLargest Directories (by size):');
  console.log('---------------------------');
  sortedDirSizes.slice(0, 10).forEach(([dir, size]) => {
    console.log(`${dir.padEnd(50)}: ${formatFileSize(size)}`);
  });
  
  // Largest files
  console.log('\nLargest Files:');
  console.log('------------');
  fileStats.largestFiles.forEach(({ path: filePath, size }) => {
    const relativePath = path.relative(projectRoot, filePath);
    console.log(`${formatFileSize(size).padEnd(10)} ${relativePath}`);
  });
}

// Run the analysis
console.log('Analyzing project structure...');
analyzeDirectory(projectRoot);
generateReport();
console.log('\nAnalysis complete!');
