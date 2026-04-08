#!/bin/bash

# Check what will be uploaded to GitHub
# This helps identify large files before pushing

echo "=========================================="
echo "GitHub Upload Size Check"
echo "=========================================="
echo ""

echo "Checking repository size..."
echo ""

# Total size
echo "Total project size:"
du -sh .
echo ""

# Size by folder
echo "Size by folder:"
du -sh */ | sort -hr
echo ""

# Large files (> 10MB)
echo "Files larger than 10MB (may cause issues):"
find . -type f -size +10M -not -path "./.git/*" -not -path "./.venv/*" -exec ls -lh {} \; | awk '{print $5, $9}'
echo ""

# Very large files (> 50MB)
echo "Files larger than 50MB (will be rejected by GitHub):"
find . -type f -size +50M -not -path "./.git/*" -not -path "./.venv/*" -exec ls -lh {} \; | awk '{print $5, $9}'
echo ""

# Count files by type
echo "File count by extension:"
echo "Python files: $(find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*" | wc -l)"
echo "Model checkpoints (.pt): $(find . -name "*.pt" | wc -l)"
echo "Log files (.log): $(find . -name "*.log" | wc -l)"
echo "Images (.png): $(find . -name "*.png" | wc -l)"
echo "CSV files: $(find . -name "*.csv" | wc -l)"
echo ""

# Check .gitignore
echo "Checking .gitignore..."
if [ -f .gitignore ]; then
    echo "✓ .gitignore exists"
    echo ""
    echo "Excluded patterns:"
    grep -v "^#" .gitignore | grep -v "^$"
else
    echo "✗ .gitignore NOT found!"
fi
echo ""

# Estimate upload size (excluding .gitignore patterns)
echo "Estimated upload size (excluding .gitignore patterns):"
echo "This is approximate - actual size may vary"
echo ""

# Size of checkpoints
echo "Checkpoint files (.pt) - EXCLUDED by .gitignore:"
du -sh checkpoints/ 2>/dev/null || echo "No checkpoints folder"
echo ""

# Size of logs
echo "Log files - EXCLUDED by .gitignore:"
find . -name "*.log" -exec du -ch {} + 2>/dev/null | tail -1 || echo "No log files"
echo ""

# Size of __pycache__
echo "Python cache - EXCLUDED by .gitignore:"
du -sh __pycache__ 2>/dev/null || echo "No __pycache__ folder"
find . -type d -name "__pycache__" -exec du -sh {} \; 2>/dev/null
echo ""

# Size of .venv
echo "Virtual environment - EXCLUDED by .gitignore:"
du -sh .venv/ 2>/dev/null || echo "No .venv folder"
echo ""

echo "=========================================="
echo "Recommendations:"
echo "=========================================="
echo ""
echo "1. GitHub free tier limits:"
echo "   - File size: 100 MB per file"
echo "   - Repository: 1 GB recommended"
echo "   - Push size: 2 GB maximum"
echo ""
echo "2. If you have large model checkpoints:"
echo "   - Option A: Use Git LFS (Large File Storage)"
echo "   - Option B: Upload to Zenodo/HuggingFace"
echo "   - Option C: Provide download links in README"
echo ""
echo "3. Files currently excluded by .gitignore:"
echo "   - Model checkpoints (*.pt)"
echo "   - Virtual environment (.venv/)"
echo "   - Python cache (__pycache__/)"
echo "   - Log files (*.log)"
echo "   - IDE settings (.vscode/)"
echo ""
echo "4. To include checkpoints, use Git LFS:"
echo "   brew install git-lfs"
echo "   git lfs install"
echo "   git lfs track '*.pt'"
echo "   git add .gitattributes"
echo ""
echo "=========================================="
