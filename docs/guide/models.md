# Model Management

Mita Code runs LLMs locally via Ollama. The `mita models` commands help you find, install, and manage models.

## Recommendations

```bash
mita models recommend
```

Detects your hardware (RAM, VRAM, GPU) and recommends models that will run well on your machine.

## List Installed Models

```bash
mita models list
```

## Pull a Model

```bash
mita models pull qwen2.5-coder:7b
```

## Set Default Model

```bash
mita models default qwen2.5-coder:14b
```

The default model is used for all chat and ask commands unless overridden.

## Model Info

```bash
mita models info qwen2.5-coder:7b
```

## Remove a Model

```bash
mita models remove qwen2.5-coder:7b
```

## Hardware Info

```bash
mita models hardware
```

Shows detected RAM, VRAM, CPU, and GPU information.

## Configuration

```toml
[model]
default = "qwen2.5-coder:7b"
embedding = "nomic-embed-text"
temperature = 0.1
max_tokens = 4096
context_window = 32768
```
