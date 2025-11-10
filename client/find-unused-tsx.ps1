# Find unused .tsx files in the project
Write-Host "Analyzing .tsx files for usage..." -ForegroundColor Cyan

# Get all .tsx files
$allTsxFiles = Get-ChildItem -Path .\src -Filter *.tsx -Recurse -File

# Get all source files content
$allSourceContent = Get-ChildItem -Path .\src -Include *.ts,*.tsx,*.js,*.jsx -Recurse -File | Get-Content -Raw

$unusedFiles = @()
$usedFiles = @()

foreach ($file in $allTsxFiles) {
    $fileName = $file.BaseName
    $relativePath = $file.FullName.Replace((Get-Location).Path + '\src\', '').Replace('\', '/')
    
    # Skip entry points and special files
    if ($fileName -in @('main', 'App', 'index', 'router')) {
        $usedFiles += $file
        continue
    }
    
    # Check if file is imported anywhere
    $isImported = $false
    
    # Check for various import patterns
    $patterns = @(
        "from ['""].*/$fileName['""]",
        "from ['""].*/$fileName\.tsx['""]",
        "from ['""]@/.*/$fileName['""]",
        "import.*$fileName",
        "lazy\(\(\) => import\(['""].*/$fileName"
    )
    
    foreach ($pattern in $patterns) {
        if ($allSourceContent -match $pattern) {
            $isImported = $true
            break
        }
    }
    
    if ($isImported) {
        $usedFiles += $file
    } else {
        $unusedFiles += $file
    }
}

Write-Host "`n=== ANALYSIS RESULTS ===" -ForegroundColor Yellow
Write-Host "Total .tsx files: $($allTsxFiles.Count)" -ForegroundColor White
Write-Host "Used files: $($usedFiles.Count)" -ForegroundColor Green
Write-Host "Potentially unused files: $($unusedFiles.Count)" -ForegroundColor Red

if ($unusedFiles.Count -gt 0) {
    Write-Host "`n=== POTENTIALLY UNUSED FILES ===" -ForegroundColor Red
    foreach ($file in $unusedFiles) {
        $relativePath = $file.FullName.Replace((Get-Location).Path + '\', '')
        Write-Host "  - $relativePath" -ForegroundColor Yellow
    }
}

Write-Host "`nNote: This is a basic analysis. Some files may be used dynamically or in ways not detected by simple pattern matching." -ForegroundColor Gray
