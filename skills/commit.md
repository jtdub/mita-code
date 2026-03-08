---
name: commit
description: "Generate a git commit message from staged changes"
args: []
trigger: "^/commit$"
---

Look at the current git diff (staged changes) and generate a clear, concise commit message.

Steps:
1. Run `git diff --cached` to see staged changes. If nothing is staged, run `git diff` to see unstaged changes.
2. Analyze the changes and determine the type: feat, fix, refactor, docs, test, chore, etc.
3. Write a commit message following conventional commit format:
   - First line: `<type>: <short summary>` (max 72 chars)
   - Blank line
   - Optional body explaining the "why" (not the "what")
4. Show the proposed commit message and ask for confirmation before committing.
