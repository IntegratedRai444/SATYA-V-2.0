# Script to clean up duplicate files
$basePath = "c:\Users\OMEN\dyad-apps\SATYA-V-2.0"

# Files to remove
$filesToRemove = @(
    "client\src\components\upload\MediaUpload.tsx",
    "client\src\components\upload\EnhancedMediaUpload.tsx",
    "client\src\components\ModernUpload.tsx"
)

foreach ($file in $filesToRemove) {
    $fullPath = Join-Path -Path $basePath -ChildPath $file
    if (Test-Path $fullPath) {
        Write-Host "Removing: $fullPath"
        Remove-Item -Path $fullPath -Force
    } else {
        Write-Host "File not found: $fullPath"
    }
}

# Rename EnhancedMediaUploadV2 to MediaUpload
$source = Join-Path -Path $basePath -ChildPath "client\src\components\upload\EnhancedMediaUploadV2.tsx"
$dest = Join-Path -Path $basePath -ChildPath "client\src\components\upload\MediaUpload.tsx"

if (Test-Path $source) {
    if (Test-Path $dest) {
        Remove-Item -Path $dest -Force
    }
    Rename-Item -Path $source -NewName "MediaUpload.tsx"
    Write-Host "Renamed EnhancedMediaUploadV2.tsx to MediaUpload.tsx"
} else {
    Write-Host "Source file not found: $source"
}

Write-Host "Cleanup completed!"
