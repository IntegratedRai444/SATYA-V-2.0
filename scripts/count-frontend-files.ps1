# count-frontend-files.ps1
# Script to count and categorize frontend files in the project

# Set execution policy for the current session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

$projectRoot = $PSScriptRoot
$clientSrcPath = Join-Path $projectRoot "client\src"

# File extensions to include
$extensions = @("*.tsx", "*.ts", "*.js", "*.jsx", "*.css", "*.scss", "*.json")

# Initialize counters
$fileCount = 0
$fileTypes = @{}
$directoryStats = @{}
$fileList = @()

# Function to write colored output
function Write-ColorOutput($message, $color) {
    if ($Host.UI.RawUI) {
        $originalColor = $Host.UI.RawUI.ForegroundColor
        $Host.UI.RawUI.ForegroundColor = $color
        Write-Output $message
        $Host.UI.RawUI.ForegroundColor = $originalColor
    } else {
        Write-Output $message
    }
}

Write-ColorOutput "`nScanning frontend files in: $clientSrcPath" "Cyan"
Write-Output "----------------------------------------"

# Get all matching files
foreach ($ext in $extensions) {
    $files = Get-ChildItem -Path $clientSrcPath -Recurse -Include $ext -File -ErrorAction SilentlyContinue
    
    foreach ($file in $files) {
        $fileCount++
        $ext = $file.Extension.ToLower()
        
        # Count by file type
        if ($fileTypes.ContainsKey($ext)) {
            $fileTypes[$ext]++
        } else {
            $fileTypes[$ext] = 1
        }
        
        # Count by directory
        $dirName = $file.Directory.Name
        if ($directoryStats.ContainsKey($dirName)) {
            $directoryStats[$dirName]++
        } else {
            $directoryStats[$dirName] = 1
        }
        
        # Add to file list with relative path
        $relativePath = $file.FullName.Substring($projectRoot.Length + 1)
        $fileList += [PSCustomObject]@{
            Name = $file.Name
            Path = $relativePath
            SizeKB = [math]::Round($file.Length / 1KB, 2)
            LastModified = $file.LastWriteTime
        }
    }
}

# Display summary
Write-ColorOutput "`n[FILE TYPE SUMMARY]" "Green"
$fileTypes.GetEnumerator() | Sort-Object Value -Descending | Format-Table @{
    Label="Extension"; Expression={$_.Key}
}, @{
    Label="Count"; Expression={$_.Value}
}, @{
    Label="% of Total"; Expression={"{0:P1}" -f ($_.Value / $fileCount)}
} -AutoSize

# Display directory statistics
Write-ColorOutput "`n[DIRECTORY STATISTICS - TOP 10]" "Green"
$directoryStats.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 10 | Format-Table @{
    Label="Directory"; Expression={$_.Key}
}, @{
    Label="File Count"; Expression={$_.Value}
}, @{
    Label="% of Total"; Expression={"{0:P1}" -f ($_.Value / $fileCount)}
} -AutoSize

# Display largest files
Write-ColorOutput "`n[LARGEST FILES - TOP 10]" "Green"
$fileList | Sort-Object SizeKB -Descending | Select-Object -First 10 | Format-Table @{
    Label="Size (KB)"; Expression={"{0:N2}" -f $_.SizeKB}
}, @{
    Label="File"; Expression={$_.Name}
}, @{
    Label="Path"; Expression={$_.Path}
} -AutoSize

# Save detailed report
$reportPath = Join-Path $projectRoot "frontend-files-report-$(Get-Date -Format 'yyyyMMdd-HHmmss').csv"
$fileList | Select-Object @{
    Name="Size (KB)"; Expression={"{0:N2}" -f $_.SizeKB}
}, Name, Path, LastModified | Export-Csv -Path $reportPath -NoTypeInformation

# Calculate statistics
$tsxCount = $fileTypes[".tsx"] ?? 0
$tsCount = $fileTypes[".ts"] ?? 0
$jsxCount = $fileTypes[".jsx"] ?? 0
$jsCount = $fileTypes[".js"] ?? 0
$cssCount = $fileTypes[".css"] ?? 0
$scssCount = $fileTypes[".scss"] ?? 0
$jsonCount = $fileTypes[".json"] ?? 0

$totalTs = $tsxCount + $tsCount
$totalJs = $jsxCount + $jsCount
$totalCss = $cssCount + $scssCount

$tsPercentage = [math]::Round(($totalTs / $fileCount) * 100, 1)
$jsPercentage = [math]::Round(($totalJs / $fileCount) * 100, 1)
$cssPercentage = [math]::Round(($totalCss / $fileCount) * 100, 1)

$testFileCount = ($fileList | Where-Object { $_.Name -match '\.test\.(tsx?|jsx?)$' }).Count
$avgFileSize = [math]::Round(($fileList | Measure-Object -Property SizeKB -Average).Average, 2)

# Display summary
Write-ColorOutput "`n[SUMMARY]" "Cyan"
Write-Output "Total files found: $fileCount"
Write-Output "Report saved to: $reportPath"

Write-ColorOutput "`n[LANGUAGE BREAKDOWN]" "Yellow"
Write-Output "TypeScript: $totalTs files ($tsPercentage%)"
Write-Output "  - React components (.tsx): $tsxCount"
Write-Output "  - Utility/Type files (.ts): $tsCount"
Write-Output "JavaScript: $totalJs files ($jsPercentage%)"
Write-Output "  - React components (.jsx): $jsxCount"
Write-Output "  - Plain JS (.js): $jsCount"
Write-Output "Styles: $totalCss files ($cssPercentage%)"
Write-Output "  - CSS: $cssCount"
Write-Output "  - SCSS: $scssCount"
Write-Output "JSON: $jsonCount files"

Write-ColorOutput "`n[OTHER STATISTICS]" "Yellow"
Write-Output "Test files: $testFileCount"
Write-Output "Average file size: $avgFileSize KB"
Write-Output "Last modified: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
