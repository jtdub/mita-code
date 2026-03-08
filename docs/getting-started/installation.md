# Installation

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

## Install with pipx (recommended)

```bash
pipx install mita-code
```

## Install with pip

```bash
pip install mita-code
```

## Development Install

```bash
git clone https://github.com/jtdub/mita-code.git
cd mita-code
poetry install
```

## Verify Installation

```bash
mita --version
```

## Install Ollama

Mita Code requires Ollama to be installed and running on your machine. Follow the instructions at [ollama.com](https://ollama.com) for your platform.

Once installed, start the Ollama server:

```bash
ollama serve
```

Then pull a coding model:

```bash
mita models pull qwen2.5-coder:7b
```

!!! tip
    Run `mita models recommend` to see which models fit your hardware best.
