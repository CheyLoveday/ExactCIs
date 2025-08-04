#!/bin/bash

# Script to add PR review feedback to PR #2
# This script assumes that:
# 1. GitHub CLI (gh) is installed and authenticated
# 2. The PR review feedback file (pr_review_feedback.md) exists
# 3. PR #2 exists in the repository

# Set -e to exit on error
set -e

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is not installed. Please install it first."
    exit 1
fi

# Check if the PR review feedback file exists
if [ ! -f "pr_review_feedback.md" ]; then
    echo "PR review feedback file (pr_review_feedback.md) does not exist."
    exit 1
fi

# Add the PR review feedback to PR #2
echo "Adding PR review feedback to PR #2..."
gh pr review 2 --body-file pr_review_feedback.md --comment

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "PR review feedback successfully added to PR #2."
else
    echo "Failed to add PR review feedback to PR #2."
    echo "Please add the PR review feedback manually using the following steps:"
    echo "1. Go to the PR #2 page on GitHub"
    echo "2. Click on 'Add your review'"
    echo "3. Copy the content of pr_review_feedback.md"
    echo "4. Paste it into the review comment box"
    echo "5. Click on 'Comment' to submit the review"
    exit 1
fi