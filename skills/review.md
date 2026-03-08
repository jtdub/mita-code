---
name: review
description: "Review code changes for bugs and improvements"
args:
  - name: target
    type: string
    description: "File path, git ref, or 'staged' to review"
    required: false
    default: "staged"
---

Review the code changes in {{target}} for potential issues.

Steps:
1. If target is "staged", run `git diff --cached`. Otherwise, read the specified file or diff.
2. Look for:
   - Bugs and logic errors
   - Security vulnerabilities
   - Missing error handling
   - Performance concerns
   - Code style issues
3. Provide a structured review with:
   - A summary of the changes
   - Issues found (ranked by severity)
   - Suggestions for improvement
4. Be specific — reference file names and line numbers.
