# Soprano TTS Examples

This directory contains example scripts demonstrating how to use Soprano TTS in different ways.

## API Example (`api_example.py`)

Demonstrates how to interact with the Soprano TTS FastAPI server.

### Prerequisites

Make sure the API server is running:

```bash
python api_server.py
```

The server will start on `http://localhost:8000`.

### Usage

**Test the API server:**

```bash
python examples/api_example.py --test
```

**Generate speech:**

```bash
python examples/api_example.py --text "Hello from Soprano TTS!"
```

**Generate with custom parameters:**

```bash
python examples/api_example.py \
  --text "This is a test with custom settings." \
  --temperature 0.5 \
  --top_p 0.9 \
  --repetition_penalty 1.3 \
  --output custom_output.wav
```

**Use a remote server:**

```bash
python examples/api_example.py \
  --text "Remote TTS generation" \
  --url http://remote-server:8000
```

## Using the API with cURL

You can also use the API directly with cURL:

```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "soprano-80m",
    "input": "Hello from Soprano TTS!",
    "voice": "default",
    "temperature": 0.3,
    "top_p": 0.95,
    "repetition_penalty": 1.2
  }' \
  --output speech.wav
```

## Using the API with Python requests

```python
import requests

# Generate speech
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "soprano-80m",
        "input": "Hello from Soprano TTS!",
        "voice": "default",
        "temperature": 0.3,
        "top_p": 0.95,
        "repetition_penalty": 1.2
    }
)

# Save to file
with open("output.wav", "wb") as f:
    f.write(response.content)
```

## OpenAI SDK Compatibility

The API is designed to be compatible with OpenAI's TTS API format. While you can't use the OpenAI SDK directly (it requires authentication), you can use the same request format:

```python
import requests

# Similar to OpenAI's API
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "soprano-80m",  # or "tts-1", "tts-1-hd"
        "input": "Your text here",
        "voice": "default",  # or OpenAI voices: "alloy", "echo", etc.
    }
)
```

## API Endpoints

- `POST /v1/audio/speech` - Generate speech (OpenAI-compatible)
- `GET /health` - Health check
- `GET /v1/models` - List available models
- `GET /` - API information
- `GET /docs` - Interactive API documentation
- `GET /redoc` - Alternative API documentation
