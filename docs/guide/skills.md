# Skills

Skills are reusable, parameterized prompt templates stored as Markdown files. Use them to standardize common workflows like committing, code review, or explaining code.

## Using Skills

In the interactive REPL, type `/` followed by the skill name:

```
mita> /commit
mita> /review
mita> /explain src/auth.py
```

The skill template is rendered with your arguments and sent to the agent as a prompt.

## Built-in Skills

Mita ships with three built-in skills:

| Skill | Description | Usage |
|---|---|---|
| `commit` | Generate a git commit message from staged changes | `/commit` |
| `review` | Review code changes for bugs and improvements | `/review` or `/review src/auth.py` |
| `explain` | Explain how code works | `/explain src/auth.py` |

## Listing Skills

```bash
mita skills list
```

## Viewing Skill Details

```bash
mita skills show commit
```

## Creating Custom Skills

```bash
mita skills create my-skill
```

This creates a template file you can edit. Skill files are Markdown with YAML frontmatter:

```markdown
---
name: my-skill
description: "What this skill does"
args:
  - name: target
    type: string
    description: "The target file or symbol"
    required: true
  - name: format
    type: string
    description: "Output format"
    required: false
    default: "markdown"
trigger: "^/my-skill$"
---

Analyze {{target}} and output in {{format}} format.

Steps:
1. Read the target file
2. Perform analysis
3. Output results
```

### Frontmatter Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Skill name (used for `/name` invocation) |
| `description` | string | no | Human-readable description |
| `args` | list | no | Accepted parameters |
| `trigger` | regex | no | Regex pattern for matching invocations |

### Argument Substitution

Use `{{arg_name}}` placeholders in the template body. Arguments can be passed positionally or with `--name value` / `--name=value` syntax.

If no args are declared, the entire user input after the skill name is available as `{{input}}`.

## Skill Search Paths

Skills are discovered from configured paths:

```bash
mita skills path
```

Default paths:

- `~/.config/mita/skills/` — Global user skills
- `.mita/skills/` — Project-local skills

Earlier paths take priority when skills share the same name.

## Configuration

```toml
skills_paths = ["~/.config/mita/skills", ".mita/skills"]
```
