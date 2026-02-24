#!/usr/bin/env python3

# Fix the JSX syntax error in ImageAnalysis.tsx
with open('ImageAnalysis.tsx', 'r') as f:
    lines = f.readlines()

# Remove the extra </div> line (around line 400)
fixed_lines = []
for i, line in enumerate(lines):
    if i == 399:  # Line 400 (0-indexed)
        # Skip the extra </div> line
        continue
    fixed_lines.append(line)

with open('ImageAnalysis.tsx', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed JSX syntax error by removing extra </div> tag")
