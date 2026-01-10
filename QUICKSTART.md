# Soprano TTS Quick Start Guide

This guide will help you get started with Soprano TTS web interface and API server.

## Prerequisites

- **Linux or Windows** with CUDA-enabled GPU
- **Python 3.10+**
- **CUDA-enabled PyTorch**

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/ekwek1/soprano.git
cd soprano
```

2. **Install dependencies:**

For GPU inference (recommended):

```bash
pip install -e ".[gpu]"
pip uninstall -y torch
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

For CPU-only inference (no CUDA required):

```bash
pip install -e ".[onnx]"
```

> **Note**: For the web interface and API server, you'll also need to install the optional dependencies. See "Optional Features" below.

## Using the Web Interface (Gradio)

### Start the Web Interface

```bash
python app.py
```

The interface will be available at: **http://localhost:7860**

### Features

- 🎙️ **Single Text Generation**: Generate speech from a single text with customizable parameters
- 📚 **Batch Generation**: Process multiple texts at once
- 🎛️ **Advanced Controls**: Fine-tune temperature, top-p, and repetition penalty
- 💡 **Built-in Tips**: Usage guidelines and best practices
- 🎨 **Modern Design**: Clean, professional interface with gradient header

### Tips for Best Results

- Keep sentences between 2-15 seconds long
- Convert numbers to words (e.g., "1+1" → "one plus one")
- Use proper grammar and punctuation
- Avoid multiple consecutive spaces

## Using the API Server

### Start the API Server

```bash
python api_server.py
```

The server will be available at: **http://localhost:8000**

### Available Endpoints

- **POST /v1/audio/speech** - Generate speech (OpenAI-compatible)
- **GET /health** - Check server health
- **GET /v1/models** - List available models
- **GET /docs** - Interactive API documentation
- **GET /redoc** - Alternative API documentation

### Example: Generate Speech with cURL

```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "soprano-80m",
    "input": "Hello from Soprano TTS!",
    "voice": "default",
    "temperature": 0.3
  }' \
  --output speech.wav
```

### Example: Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "soprano-80m",
        "input": "Soprano is an ultra-lightweight TTS model.",
        "voice": "default",
        "temperature": 0.3,
        "top_p": 0.95,
        "repetition_penalty": 1.2
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)

print("Audio saved to output.wav")
```

### Example: Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "backend": "soprano-80m"
}
```

## OpenAI API Compatibility

The Soprano API is designed to be compatible with OpenAI's TTS API format. You can use the same request structure:

```python
# OpenAI-compatible request
{
  "model": "soprano-80m",  # or "tts-1", "tts-1-hd"
  "input": "Your text here",
  "voice": "default",
  "response_format": "wav"
}
```

**Supported OpenAI parameters:**
- `model`: "soprano-80m", "tts-1", or "tts-1-hd" (all map to Soprano)
- `input`: Text to synthesize (max 4096 characters)
- `voice`: "default" or OpenAI voices (all map to default)
- `response_format`: "wav" (currently the only supported format)
- `speed`: Accepted but ignored (Soprano doesn't support speed control)

**Additional Soprano-specific parameters:**
- `temperature`: Sampling temperature (0.1-1.0, default: 0.3)
- `top_p`: Top-p sampling (0.5-1.0, default: 0.95)
- `repetition_penalty`: Repetition penalty (1.0-2.0, default: 1.2)

## Running Examples

The `examples/` directory contains sample scripts:

```bash
# Test the API
python examples/api_example.py --test

# Generate speech
python examples/api_example.py --text "Hello world!"

# Custom parameters
python examples/api_example.py \
  --text "Custom TTS example" \
  --temperature 0.5 \
  --top_p 0.9 \
  --output my_audio.wav
```

## Performance Tips

### For Best Speed:
- Increase `cache_size_mb` (default: 10)
- Increase `decoder_batch_size` (default: 1)
- Use longer input texts (10+ seconds of speech)

### For Lower VRAM Usage:
- Decrease `cache_size_mb` to 1
- Keep `decoder_batch_size` at 1
- Use `backend='transformers'` if LMDeploy fails

### Example: High Performance Setup

```python
from soprano import SopranoTTS

model = SopranoTTS(
    backend='auto',
    device='cuda',
    cache_size_mb=50,  # Higher cache
    decoder_batch_size=4  # Larger batches
)
```

### Example: Low VRAM Setup

```python
from soprano import SopranoTTS

model = SopranoTTS(
    backend='transformers',  # Fallback backend
    device='cuda',
    cache_size_mb=1,  # Minimal cache
    decoder_batch_size=1  # Single batch
)
```

## Troubleshooting

### Model Not Loading

**Issue**: "TTS model not loaded" error

**Solutions**:
1. Make sure you have a CUDA-enabled GPU
2. Check GPU is available: `python -c "import torch; print(torch.cuda.is_available())"`
3. Try `backend='transformers'` if LMDeploy fails
4. Check you have enough VRAM (minimum ~1GB required)

### Slow Generation

**Issue**: Audio generation is slow

**Solutions**:
1. Increase `cache_size_mb` parameter
2. Increase `decoder_batch_size` parameter
3. Use longer input texts (Soprano is optimized for longer texts)
4. Ensure you're using `backend='auto'` or `'lmdeploy'` for best speed

### Import Errors

**Issue**: Missing dependencies

**Solution**:

For GPU installation:
```bash
pip install -e ".[gpu]"
```

For web UI:
```bash
pip install -e ".[webui]"
```

For API server:
```bash
pip install -e ".[server]"
```

For all features:
```bash
pip install -e ".[all]"
```

## Next Steps

- 📖 Read the full [README.md](../README.md) for detailed information
- 🔗 Check out [examples/README.md](../examples/README.md) for more examples
- 📚 Visit the API docs at http://localhost:8000/docs (when server is running)
- 🎵 Explore the Gradio interface at http://localhost:7860 (when app is running)

## Support

- **GitHub**: [github.com/ekwek1/soprano](https://github.com/ekwek1/soprano)
- **HuggingFace Model**: [ekwek/Soprano-80M](https://huggingface.co/ekwek/Soprano-80M)
- **HuggingFace Demo**: [ekwek/Soprano-TTS](https://huggingface.co/spaces/ekwek/Soprano-TTS)

## License

Apache-2.0 License - See [LICENSE](../LICENSE) for details
