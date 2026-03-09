# Optimizing Ollama for Apple Silicon

Mita Code runs LLMs locally via Ollama. On Apple Silicon Macs (M1/M2/M3/M4), Ollama can use the Metal GPU framework for dramatically faster inference. This guide covers installation, verification, and tuning.

## Installing Ollama for Metal Support

!!! warning "Do not use Homebrew"
    If your terminal runs under Rosetta 2 (common on Macs migrated from Intel), `brew install ollama` may install an x86 binary that **cannot use Metal GPU acceleration**. This results in 10-100x slower inference.

**Recommended: Install the macOS app from [ollama.com](https://ollama.com/download)**

The Ollama macOS app:

- Ships as a universal binary (ARM + x86)
- Runs its server as an ARM-native process regardless of your terminal architecture
- Includes Metal GPU support out of the box
- Runs as a background service via the menu bar

### Verifying Your Installation

Check that Metal is active:

```bash
# Pull a model if you haven't already
mita models pull qwen2.5-coder:7b

# Check GPU memory usage
curl -s http://localhost:11434/api/ps | python3 -m json.tool
```

Look for the `size_vram` field:

- **`size_vram > 0`** — Metal is working, model is on the GPU
- **`size_vram: 0`** — CPU only, Metal is not available (see Troubleshooting below)

### Checking for Rosetta

If you suspect your terminal is running under Rosetta:

```bash
uname -m
```

- `arm64` — native ARM, Metal will work in all contexts
- `x86_64` — Rosetta emulation. The Ollama macOS app will still use Metal (it runs its own ARM server process), but CLI-only installs will not

## Performance Expectations

Benchmarks on Apple Silicon with Metal enabled (qwen2.5-coder:7b, Q4_K_M):

| Metric | First Call | Subsequent Calls |
|--------|-----------|-----------------|
| Model load | 15-20s | <0.1s |
| Prompt eval (30 tokens) | 0.5-1.0s | 0.3s |
| Generation (10 tokens) | 1.5-2.0s | 1.5-2.0s |
| **Total** | **~20s** | **~2-3s** |

The first call is slower because Ollama loads the model into GPU memory. The model stays loaded for 5 minutes (configurable) so subsequent calls are fast.

Without Metal (CPU-only), expect 10-100x slower performance.

## Tuning Ollama Runtime Options

Mita exposes Ollama's runtime parameters via configuration. Add these to your project `.mita/settings.toml` or global `~/.config/mita/config.toml`:

```toml
[model.ollama_options]
num_gpu = 99            # Number of layers to offload to GPU (99 = all)
num_ctx = 4096          # Context window size for inference
num_batch = 512         # Batch size for prompt processing
flash_attention = true  # Enable flash attention (faster, less memory)
```

### Key Options Explained

#### `num_gpu`

Number of model layers to run on the GPU. Set to `99` to offload everything. Reduce if you get out-of-memory errors.

```toml
[model.ollama_options]
num_gpu = 99   # All layers on GPU (default behavior for Apple Silicon)
num_gpu = 20   # Partial offload — remaining layers run on CPU
num_gpu = 0    # Force CPU-only (useful for debugging)
```

#### `num_ctx`

Context window size in tokens. Larger values use more memory. Ollama defaults to 4096 even if the model supports more.

```toml
[model.ollama_options]
num_ctx = 4096    # Default — good balance of speed and context
num_ctx = 8192    # More context, uses more GPU memory
num_ctx = 32768   # Full qwen2.5-coder context — requires significant memory
```

Memory usage scales linearly with context size. On 16GB Macs, stick with 4096-8192 for 7B models.

#### `num_thread`

Number of CPU threads for any CPU-bound computation. Defaults to the number of performance cores.

```toml
[model.ollama_options]
num_thread = 8   # Match your core count
```

#### `flash_attention`

Enables flash attention for more efficient memory usage during inference. Recommended on all Apple Silicon.

```toml
[model.ollama_options]
flash_attention = true
```

#### `use_mmap` / `use_mlock`

Memory-mapped I/O and memory locking. `use_mmap = true` is the default and recommended. `use_mlock = true` prevents the OS from swapping the model out of memory but requires sufficient free RAM.

```toml
[model.ollama_options]
use_mmap = true    # Default — recommended
use_mlock = false  # Set true if you have RAM to spare and want consistent latency
```

#### `low_vram`

Reduces VRAM usage at the cost of speed. Useful on 8GB Macs running larger models.

```toml
[model.ollama_options]
low_vram = true
```

## Model Selection by Hardware

| Mac | RAM | Recommended Models |
|-----|-----|-------------------|
| M1/M2/M3 Air | 8GB | qwen2.5-coder:3b, deepseek-coder:1.3b |
| M1/M2/M3 Air | 16GB | qwen2.5-coder:7b, codellama:7b |
| M1/M2/M3 Pro | 18-36GB | qwen2.5-coder:14b, codellama:13b |
| M1/M2/M3 Max | 32-96GB | qwen2.5-coder:32b, codellama:34b |

Use `mita models recommend` for hardware-specific suggestions.

## Ollama Server Management

Mita can auto-start and manage the Ollama server:

```toml
[ollama]
host = "http://localhost:11434"
timeout = 120
auto_manage = true   # Auto-start Ollama if not running
```

Manual control:

```bash
mita ollama start    # Start the server
mita ollama stop     # Stop the server (only if started by Mita)
mita ollama status   # Show server status, binary path, and config
```

!!! note
    If you use the Ollama macOS app, the server runs automatically via the menu bar. Set `auto_manage = false` since Mita doesn't need to manage it.

## Troubleshooting

### `size_vram: 0` — Model running on CPU

**Cause:** Ollama can't access Metal GPU.

**Fix:** Install the macOS app from [ollama.com](https://ollama.com/download) instead of Homebrew. The app runs its server as an ARM-native process with Metal support.

### Very slow responses (30+ seconds for simple prompts)

**Cause:** Likely CPU-only inference. Check `size_vram` as described above.

**Quick diagnosis:**

```bash
curl -s http://localhost:11434/api/ps | python3 -c "
import sys, json
d = json.load(sys.stdin)
for m in d.get('models', []):
    vram = m.get('size_vram', 0) / (1024**3)
    total = m.get('size', 0) / (1024**3)
    print(f'{m[\"name\"]}: {vram:.1f}/{total:.1f} GB on GPU')
"
```

### `Failed to load MLX dynamic library`

**Cause:** The Ollama CLI is running as x86 under Rosetta and can't load the ARM-only MLX framework.

**Fix:** This is harmless if using the macOS app — the app's server process runs natively. If you see this from `ollama --version`, it just means the CLI binary is running under Rosetta, but the server (which does the actual inference) is ARM-native.

### Model takes 15-20 seconds on first prompt

**This is normal.** Ollama loads the model into GPU memory on the first request. Subsequent requests reuse the loaded model and respond in 2-3 seconds. The model stays loaded for 5 minutes by default.

### Out of memory errors

Reduce GPU memory usage:

```toml
[model.ollama_options]
num_ctx = 2048       # Reduce context window
low_vram = true      # Enable low VRAM mode
num_gpu = 20         # Partial GPU offload
```

Or switch to a smaller model: `mita models default qwen2.5-coder:3b`
