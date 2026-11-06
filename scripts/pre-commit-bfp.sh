#!/usr/bin/env bash
# Git pre-commit hook: Build-Failure Predictor (BFP)
# Evaluates code churn and test-to-code ratios of staged files before committing.

echo "🔍 Running Build-Failure Predictor pre-commit inspection..."

# Extract staged diff statistics
DIFF_STAT=$(git diff --cached --shortstat 2>/dev/null)

if [ -z "$DIFF_STAT" ]; then
    echo "No staged changes detected. Proceeding..."
    exit 0
fi

# Parse files changed, insertions, deletions
FILES_CHANGED=$(echo "$DIFF_STAT" | grep -o -E '[0-9]+ file' | awk '{print $1}')
LINES_ADDED=$(echo "$DIFF_STAT" | grep -o -E '[0-9]+ insertion' | awk '{print $1}')
LINES_DELETED=$(echo "$DIFF_STAT" | grep -o -E '[0-9]+ deletion' | awk '{print $1}')

FILES_CHANGED=${FILES_CHANGED:-1}
LINES_ADDED=${LINES_ADDED:-0}
LINES_DELETED=${LINES_DELETED:-0}
TOTAL_CHURN=$((LINES_ADDED + LINES_DELETED))

# Detect test files touched in staged diff
TEST_CHURN=$(git diff --cached -- "*test*" "*spec*" 2>/dev/null | grep -E '^\+[^+]' | wc -l)
SRC_CHURN=$((TOTAL_CHURN - TEST_CHURN))
if [ "$SRC_CHURN" -le 0 ]; then
    SRC_CHURN=1
fi

TEST_RATIO=$(awk "BEGIN {printf \"%.2f\", $TEST_CHURN / $SRC_CHURN}")
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "feature")

# Run BFP predictor
PYTHON_BIN=".venv/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

$PYTHON_BIN src/predict.py \
    --files "$FILES_CHANGED" \
    --churn "$TOTAL_CHURN" \
    --test-ratio "$TEST_RATIO" \
    --branch "$CURRENT_BRANCH"

PRED_EXIT_CODE=$?
echo "✅ Pre-commit build risk inspection completed."
exit 0

