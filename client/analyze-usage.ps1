# Comprehensive Frontend File Usage Analysis
$srcPath = "src"
$allFiles = Get-ChildItem -Path $srcPath -Recurse -Include "*.tsx","*.ts" -Exclude "*.d.ts"

Write-Host "=== FRONTEND FILE USAGE ANALYSIS ===" -ForegroundColor Cyan
Write-Host ""

# Count files by type
$tsxFiles = @($allFiles | Where-Object { $_.Extension -eq ".tsx" })
$tsFiles = @($allFiles | Where-Object { $_.Extension -eq ".ts" })

Write-Host "Total Files: $($allFiles.Count)" -ForegroundColor Green
Write-Host "  - TypeScript React (.tsx): $($tsxFiles.Count)" -ForegroundColor Yellow
Write-Host "  - TypeScript (.ts): $($tsFiles.Count)" -ForegroundColor Yellow
Write-Host ""

# Build file map
$fileMap = @{}
foreach ($file in $allFiles) {
    $relativePath = $file.FullName.Replace((Get-Location).Path + "\src\", "").Replace("\", "/")
    $fileMap[$relativePath] = @{
        FullPath = $file.FullName
        Name = $file.Name
        Extension = $file.Extension
        IsUsed = $false
        ImportedBy = @()
    }
}

# Entry points (always used)
$entryPoints = @("main.tsx", "App.tsx", "setupTests.ts", "vite-env.d.ts")

# Mark entry points
foreach ($path in $fileMap.Keys) {
    $fileName = $fileMap[$path].Name
    if ($fileName -in $entryPoints) {
        $fileMap[$path].IsUsed = $true
        $fileMap[$path].ImportedBy += "ENTRY_POINT"
    }
}

# Scan all files for imports
foreach ($filePath in $fileMap.Keys) {
    $content = Get-Content $fileMap[$filePath].FullPath -Raw -ErrorAction SilentlyContinue
    if (-not $content) { continue }
    
    # Find all import statements
    $importPattern = 'import\s+(?:(?:\{[^}]+\}|\*\s+as\s+\w+|\w+)\s+from\s+)?[''"`]([^''"`]+)[''"`]'
    $matches = [regex]::Matches($content, $importPattern)
    
    foreach ($match in $matches) {
        $importPath = $match.Groups[1].Value
        
        # Skip external packages
        if ($importPath -notmatch '^(\.|@/)') { continue }
        
        # Normalize path
        $importPath = $importPath -replace '^@/', ''
        $importPath = $importPath -replace '^\./', ''
        
        # Try to find matching file
        foreach ($targetPath in $fileMap.Keys) {
            $targetPathNoExt = $targetPath -replace '\.(tsx?|jsx?)$', ''
            $importPathNormalized = $importPath -replace '\.(tsx?|jsx?)$', ''
            
            # Check if paths match
            if ($targetPathNoExt -eq $importPathNormalized -or 
                $targetPath -like "*$importPathNormalized*" -or
                $targetPathNoExt -like "*$importPathNormalized") {
                
                $fileMap[$targetPath].IsUsed = $true
                $fileMap[$targetPath].ImportedBy += $filePath
            }
        }
    }
}

# Categorize files
$usedFiles = @($fileMap.Keys | Where-Object { $fileMap[$_].IsUsed })
$unusedFiles = @($fileMap.Keys | Where-Object { -not $fileMap[$_].IsUsed })

Write-Host "=== USAGE SUMMARY ===" -ForegroundColor Cyan
Write-Host "Used/Imported Files: $($usedFiles.Count)" -ForegroundColor Green
Write-Host "Potentially Unused Files: $($unusedFiles.Count)" -ForegroundColor Red
Write-Host ""

# Show unused files by category
if ($unusedFiles.Count -gt 0) {
    Write-Host "=== POTENTIALLY UNUSED FILES ===" -ForegroundColor Red
    Write-Host ""
    
    $categories = @{
        "Components" = @($unusedFiles | Where-Object { $_ -like "components/*" })
        "Pages" = @($unusedFiles | Where-Object { $_ -like "pages/*" })
        "Hooks" = @($unusedFiles | Where-Object { $_ -like "hooks/*" })
        "Services" = @($unusedFiles | Where-Object { $_ -like "services/*" })
        "Utils" = @($unusedFiles | Where-Object { $_ -like "utils/*" })
        "Lib" = @($unusedFiles | Where-Object { $_ -like "lib/*" })
        "Contexts" = @($unusedFiles | Where-Object { $_ -like "contexts/*" })
        "Types" = @($unusedFiles | Where-Object { $_ -like "types/*" })
        "Other" = @($unusedFiles | Where-Object { $_ -notlike "components/*" -and $_ -notlike "pages/*" -and $_ -notlike "hooks/*" -and $_ -notlike "services/*" -and $_ -notlike "utils/*" -and $_ -notlike "lib/*" -and $_ -notlike "contexts/*" -and $_ -notlike "types/*" })
    }
    
    foreach ($category in $categories.Keys | Sort-Object) {
        $files = $categories[$category]
        if ($files.Count -gt 0) {
            Write-Host "$category ($($files.Count)):" -ForegroundColor Yellow
            $files | Sort-Object | ForEach-Object { Write-Host "  - $_" }
            Write-Host ""
        }
    }
}

# Export detailed report
$report = @{
    TotalFiles = $allFiles.Count
    TSXFiles = $tsxFiles.Count
    TSFiles = $tsFiles.Count
    UsedFiles = $usedFiles.Count
    UnusedFiles = $unusedFiles.Count
    UnusedFilesList = $unusedFiles | Sort-Object
}

$report | ConvertTo-Json -Depth 3 | Out-File "file-usage-report.json"
Write-Host "Detailed report saved to: file-usage-report.json" -ForegroundColor Cyan
