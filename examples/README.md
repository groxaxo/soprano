# Soprano TTS Examples

This directory contains example scripts demonstrating how to use Soprano TTS with the new API features.

## Files

### `usage_example.py`

Demonstrates basic usage of the Soprano TTS library:
- Text normalization
- FlashSR upsampling
- Sentence splitting
- Direct Python API usage

**Requirements:**
- Soprano TTS installed
- CUDA-enabled GPU (or CPU for slower inference)

**Usage:**
```bash
python examples/usage_example.py
```

### `api_client_example.py`

Demonstrates how to use the OpenAI-compatible REST API:
- Health check endpoint
- List available voices
- Generate speech in different formats (WAV, Opus, MP3)

**Requirements:**
- Soprano server running: `soprano-server`
- `requests` library: `pip install requests`

**Usage:**
```bash
# Terminal 1: Start the server
soprano-server

# Terminal 2: Run the example
python examples/api_client_example.py
```

## Quick Start

1. **Install Soprano TTS:**
   ```bash
   pip install soprano-tts
   ```

2. **Install torch (if not already installed):**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```

3. **Start the API server:**
   ```bash
   soprano-server --port 8000
   ```

4. **Open the web interface:**
   Navigate to `http://localhost:8000` in your browser

5. **Or use the API programmatically:**
   ```bash
   python examples/api_client_example.py
   ```

## API Endpoints

- **Web UI**: `http://localhost:8000/`
- **API Docs**: `http://localhost:8000/docs`
- **Speech Generation**: `POST http://localhost:8000/v1/audio/speech`
- **List Voices**: `GET http://localhost:8000/v1/audio/voices`
- **Health Check**: `GET http://localhost:8000/health`

## Notes

- The API is compatible with OpenAI's TTS API format
- FlashSR upsampling is enabled by default (32kHz → 48kHz)
- Text is automatically normalized for better pronunciation
- Multiple output formats are supported: WAV, Opus, MP3
