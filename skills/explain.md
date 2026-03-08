---
name: explain
description: "Explain how code works"
args:
  - name: target
    type: string
    description: "File path or symbol to explain"
    required: true
---

Explain how {{target}} works.

Steps:
1. Read the file or find the symbol in the codebase.
2. Provide a clear explanation including:
   - What the code does at a high level
   - How the key parts work
   - Important design decisions or patterns used
   - How it connects to the rest of the codebase
3. Use simple language. Include code references when helpful.
