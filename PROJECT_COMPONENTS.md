# Soprano TTS Project Components

This document provides a comprehensive overview of all components in the Soprano TTS project, including their architecture, dependencies, and interactions.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Architecture](#core-architecture)
3. [Component Breakdown](#component-breakdown)
4. [Model Architecture](#model-architecture)
5. [Backend Systems](#backend-systems)
6. [API & Server Components](#api--server-components)
7. [Audio Processing Pipeline](#audio-processing-pipeline)
8. [Dependencies](#dependencies)
9. [File Structure](#file-structure)

---

## Project Overview

**Soprano** is an ultra-lightweight, open-source text-to-speech (TTS) model with only **80M parameters** that achieves:
- Real-time factor (RTF) of **~2000×**
- **10 hours of audio in under 20 seconds**
- True real-time synthesis in **<15 ms** via seamless streaming
- **Under 1 GB VRAM usage**
- **32 kHz audio output** (48 kHz with FlashSR upsampling)

---

## Core Architecture

The Soprano system consists of three main components working in a pipeline:

```
Text Input → Language Model (80M params) → Vocos Decoder → Audio Output
                                               ↓
                                          FlashSR (optional)
                                               ↓
                                        48kHz Audio Output
```

### High-Level Pipeline

1. **Text Processing**: Input text is normalized and split into sentences
2. **Language Model**: Generates hidden states (audio tokens at ~15 tokens/sec, 0.2 kbps)
3. **Vocos Decoder**: Converts hidden states to raw audio waveforms (32 kHz)
4. **FlashSR Upsampler** (optional): Upsamples 32kHz → 48kHz at 200-400x realtime

---

## Component Breakdown

### 1. Text Processing Module
**File**: `soprano/text_processing.py`

**Purpose**: Normalizes and preprocesses input text for optimal TTS synthesis

**Key Functions**:
- Text normalization (numbers, URLs, emails, etc.)
- Sentence segmentation
- Character filtering
- ASCII conversion via unidecode

**Features**:
- Handles various text formats (numbers, dates, currency)
- Removes unsupported characters
- Enforces minimum sentence length
- Adds prompt formatting for the language model

---

### 2. Language Model Backends

The project supports multiple inference backends for flexibility and performance.

#### 2.1 Base Backend Interface
**File**: `soprano/backends/base.py`

**Purpose**: Defines the abstract interface for all backend implementations

**Methods**:
- `infer()`: Batch inference
- `stream_infer()`: Streaming inference

#### 2.2 LMDeploy Backend (Default)
**File**: `soprano/backends/lmdeploy.py`

**Purpose**: High-performance inference using LMDeploy acceleration

**Features**:
- CUDA-only support
- Turbomind engine for maximum speed
- Configurable cache size
- Streaming support
- ~2000x realtime factor on long inputs

**Key Dependencies**:
- `lmdeploy` package
- CUDA-enabled GPU

**Configuration**:
- `cache_size_mb`: Memory allocated for KV cache
- `temperature`: Sampling temperature (0.1-1.0)
- `top_p`: Nucleus sampling parameter
- `repetition_penalty`: Prevents repetitive output

#### 2.3 Transformers Backend (Fallback)
**File**: `soprano/backends/transformers.py`

**Purpose**: HuggingFace transformers-based inference (slower but more compatible)

**Features**:
- CUDA and CPU support (planned)
- Standard HuggingFace API
- No streaming support
- Works when LMDeploy unavailable

**Key Dependencies**:
- `transformers` package
- `torch`

---

### 3. Vocos Decoder

The Vocos decoder converts hidden states from the language model into audio waveforms.

#### 3.1 Main Decoder
**File**: `soprano/vocos/decoder.py`

**Architecture**: `SopranoDecoder`

**Components**:
- **Backbone**: VocosBackbone neural network
- **Head**: ISTFT (Inverse Short-Time Fourier Transform) head
- **Upscaling**: 4x upscaling from hidden state to audio samples

**Parameters**:
- Input channels: 512
- Decoder layers: 8
- Decoder dimension: 512
- Intermediate dimension: 1536
- Hop length: 512
- N-FFT: 2048
- Upscale factor: 4
- Token size: 2048 samples per token

**Key Features**:
- Finite receptive field (4 tokens) enables streaming
- Vocoder-based (not diffusion) for extreme speed
- Orders of magnitude faster than diffusion decoders

#### 3.2 Backbone Network
**File**: `soprano/vocos/models.py`

**Purpose**: Core neural network for feature extraction and transformation

**Architecture**:
- Convolutional neural network
- Multiple layers with intermediate dimensions
- Depthwise convolutions for efficiency

#### 3.3 ISTFT Head
**File**: `soprano/vocos/heads.py`

**Purpose**: Converts features to audio using Inverse STFT

**Features**:
- Spectral conversion to time-domain audio
- High-quality waveform reconstruction
- Optimized for 32 kHz output

#### 3.4 Spectral Operations
**File**: `soprano/vocos/spectral_ops.py`

**Purpose**: Spectral processing utilities for STFT/ISTFT operations

#### 3.5 ConvNeXt Modules
**File**: `soprano/vocos/modules.py`

**Purpose**: Modern convolutional building blocks

---

### 4. FlashSR Upsampler
**File**: `soprano/flashsr_upsampler.py`

**Purpose**: Ultra-fast audio super-resolution (32kHz → 48kHz)

**Class**: `FlashSRUpsampler`

**Features**:
- 200-400x realtime processing speed
- Lightweight resampling using librosa
- Optional (can be disabled)
- Minimal performance impact

**Configuration**:
- `device`: Processing device (cuda/cpu)
- `enable`: Toggle upsampling on/off
- Input sample rate: 32,000 Hz
- Output sample rate: 48,000 Hz

**Implementation**:
- Currently uses librosa's kaiser_best resampling
- Placeholder for future neural FlashSR model
- Can be disabled via `--disable-flashsr` flag or `ENABLE_FLASHSR=false` environment variable

---

### 5. Main TTS Interface
**File**: `soprano/tts.py`

**Class**: `SopranoTTS`

**Purpose**: Main entry point for TTS synthesis

**Initialization Parameters**:
- `backend`: 'auto', 'lmdeploy', or 'transformers'
- `device`: 'cuda' (CPU support planned)
- `cache_size_mb`: Cache size for LMDeploy backend (default: 10)
- `decoder_batch_size`: Batch size for decoder inference (default: 1)

**Methods**:

1. **`infer(text, out_path=None, **params)`**
   - Single text-to-speech generation
   - Returns audio tensor
   - Optional file saving

2. **`infer_batch(texts, out_dir=None, **params)`**
   - Batch processing of multiple texts
   - Optimized for throughput
   - Can achieve 2000x realtime with large batches

3. **`infer_stream(text, chunk_size=1, **params)`**
   - Streaming inference with ultra-low latency
   - First chunk in <15 ms
   - Returns iterator of audio chunks
   - Acoustically identical to offline synthesis

**Internal Methods**:
- `_preprocess_text()`: Text normalization and sentence splitting

**Key Features**:
- Automatic warmup on initialization
- Sentence-level streaming for infinite context
- Efficient batching and caching
- Memory-optimized processing

---

## API & Server Components

### 6. FastAPI Server
**File**: `soprano/api.py`

**Purpose**: Production-ready API server with OpenAI compatibility

**Endpoints**:

1. **`POST /v1/audio/speech`** - OpenAI-compatible TTS
   - Generates speech from text
   - Supports multiple output formats (opus, wav, mp3)
   - FlashSR upsampling enabled by default

2. **`GET /v1/audio/voices`** - List available voices
   - Returns voice metadata
   - Current: single default voice

3. **`GET /health`** - Health check
   - Service status
   - Model loaded status

**Request Model**: `OpenAISpeechRequest`
- `model`: Model identifier (e.g., "tts-1")
- `input`: Text to synthesize
- `voice`: Voice ID (default: "soprano-default")
- `response_format`: "opus", "wav", or "mp3"
- `speed`: Playback speed (accepted but ignored)

**Audio Format Support**:
- **Opus**: Compressed, best for streaming (requires pydub)
- **WAV**: Uncompressed, universal compatibility
- **MP3**: Compressed, widely supported (requires pydub)

**Environment Variables**:
- `MODEL_DEVICE`: Device to use (default: "cuda")
- `BACKEND`: Backend to use (default: "auto")
- `CACHE_SIZE_MB`: Cache size (default: "10")
- `DECODER_BATCH_SIZE`: Decoder batch size (default: "1")
- `ENABLE_FLASHSR`: Enable FlashSR upsampling (default: "true")

### 7. CLI Server Entry Point
**File**: `soprano/server.py`

**Purpose**: Command-line interface for starting the API server

**Command**: `soprano-server`

**Options**:
- `--host`: Bind address (default: 0.0.0.0)
- `--port`: Port number (default: 8000)
- `--device`: Device for inference (default: cuda)
- `--backend`: Backend to use (default: auto)
- `--cache-size-mb`: Cache size (default: 10)
- `--decoder-batch-size`: Decoder batch size (default: 1)
- `--disable-flashsr`: Disable FlashSR upsampling
- `--reload`: Enable auto-reload for development

### 8. Standalone API Server
**File**: `api_server.py`

**Purpose**: Simple standalone server launcher

**Features**:
- Direct script execution
- Default configuration
- Quick deployment

### 9. Gradio Web Interface
**File**: `app.py`

**Purpose**: User-friendly web UI for interactive TTS

**Features**:
- Single text generation with real-time preview
- Batch generation for multiple texts
- Advanced parameter controls:
  - Temperature (0.1-1.0)
  - Top-p (0.5-1.0)
  - Repetition penalty (1.0-2.0)
- Built-in usage tips
- Modern, responsive design

**Interface Components**:
- Text input area
- Parameter sliders
- Audio output player
- Download button
- Batch processing tab
- Tips and documentation panel

**Default URL**: `http://localhost:7860`

### 10. Command-Line Script
**File**: `run_soprano_tts.py`

**Purpose**: Simple CLI for quick TTS generation

**Usage**:
```bash
python run_soprano_tts.py "Your text here"
```

---

## Audio Processing Pipeline

### Data Flow

1. **Input**: Text string
2. **Text Processing**: 
   - Normalize numbers, URLs, etc.
   - Split into sentences
   - Add prompt formatting
3. **Language Model**:
   - Generate hidden states (512-dimensional vectors)
   - ~15 tokens/second compression
   - 0.2 kbps data rate
4. **Decoder**:
   - Convert hidden states to audio samples
   - 4x upsampling via interpolation
   - ISTFT for waveform generation
   - Output: 32 kHz audio
5. **FlashSR** (optional):
   - Upsample 32 kHz → 48 kHz
   - 200-400x realtime processing
6. **Output**: Audio array or file

### Streaming Pipeline

For streaming inference:

1. Generate tokens incrementally
2. Maintain 4-token receptive field buffer
3. Decode each chunk as tokens arrive
4. Yield audio chunks with <15 ms latency
5. Acoustically lossless streaming

---

## Dependencies

### Core Dependencies

**PyTorch Ecosystem**:
- `torch` (2.8.0 with CUDA 12.6 recommended)
- CUDA-enabled GPU required (CPU support planned)

**Language Model Backends**:
- `lmdeploy` (primary, fastest)
- `transformers` (fallback)

**Audio Processing**:
- `scipy` - WAV file I/O
- `numpy` - Array operations
- `librosa` (optional) - FlashSR upsampling
- `pydub` (optional) - Opus/MP3 encoding

**Text Processing**:
- `unidecode` - ASCII normalization

**Web & API**:
- `gradio` (>=4.0.0, <5.0.0) - Web UI
- `fastapi` (>=0.100.0, <1.0.0) - API server
- `uvicorn[standard]` (>=0.23.0, <1.0.0) - ASGI server
- `pydantic` (>=2.0.0, <3.0.0) - Data validation

**Model Source**:
- `huggingface_hub` - Model downloading
- Model repository: `ekwek/Soprano-80M`

### Optional Dependencies

- `pydub` + `ffmpeg` - For opus/mp3 output formats
- `librosa` - For FlashSR audio upsampling

---

## File Structure

```
soprano/
├── README.md                    # Main documentation
├── QUICKSTART.md               # Quick start guide
├── LICENSE                     # Apache-2.0 license
├── pyproject.toml             # Package configuration
├── app.py                     # Gradio web interface
├── api_server.py              # Standalone API server
├── run_soprano_tts.py         # CLI script
│
├── soprano/                   # Main package
│   ├── __init__.py           # Package initialization
│   ├── tts.py                # Main TTS class
│   ├── api.py                # FastAPI server implementation
│   ├── server.py             # Server CLI entry point
│   ├── text_processing.py    # Text normalization utilities
│   ├── flashsr_upsampler.py  # Audio super-resolution
│   │
│   ├── backends/             # Language model backends
│   │   ├── base.py          # Abstract backend interface
│   │   ├── lmdeploy.py      # LMDeploy implementation
│   │   └── transformers.py  # Transformers implementation
│   │
│   ├── vocos/               # Vocos decoder components
│   │   ├── decoder.py       # Main decoder class
│   │   ├── models.py        # Backbone network
│   │   ├── heads.py         # ISTFT head
│   │   ├── modules.py       # ConvNeXt modules
│   │   └── spectral_ops.py  # Spectral operations
│   │
│   └── static/              # Static assets for web UI
│
└── examples/                # Example scripts
    ├── README.md           # Examples documentation
    ├── usage_example.py    # Basic usage examples
    ├── api_example.py      # API usage examples
    └── api_client_example.py # Client implementation examples
```

---

## Model Details

### Architecture Summary

**Total Parameters**: 80M

**Language Model**:
- Type: Causal language model (autoregressive)
- Context: Text prompt with special tokens
- Output: Hidden states (512-dimensional)
- Token rate: ~15 tokens/second of audio
- Compression: 0.2 kbps (extremely efficient)

**Decoder (Vocos)**:
- Type: Vocoder-based neural decoder
- Architecture: ConvNeXt-style backbone + ISTFT head
- Layers: 8 decoder layers
- Dimensions: 512 channels
- Receptive field: 4 tokens
- Output: 32 kHz raw audio

**Total Pipeline**:
```
Text → 80M LM → Hidden States → Vocos Decoder → 32kHz Audio → [FlashSR] → 48kHz Audio
       (512-dim vectors)      (8-layer CNN)
```

### Performance Characteristics

- **Speed**: ~2000x realtime (10 hours in 20 seconds)
- **Latency**: <15 ms first chunk (streaming)
- **VRAM**: <1 GB typical usage
- **Quality**: 32 kHz base, 48 kHz with FlashSR
- **Stability**: Sentence-level generation for infinite context

---

## Key Design Decisions

### 1. Vocoder-based Decoder (Not Diffusion)
- **Advantage**: Orders of magnitude faster
- **Trade-off**: Comparable quality to diffusion at much higher speed

### 2. Neural Audio Codec
- **Compression**: ~15 tokens/sec at 0.2 kbps
- **Benefit**: Extremely fast generation, low memory usage
- **Stability**: High compression maintains quality

### 3. Sentence-level Streaming
- **Design**: Each sentence generated independently
- **Benefits**: 
  - Infinite generation length
  - Stable output
  - Real-time performance for long-form content

### 4. Multiple Backend Support
- **LMDeploy**: Maximum speed (CUDA only)
- **Transformers**: Broader compatibility
- **Auto**: Intelligent fallback

### 5. Optional FlashSR
- **Default**: Enabled for higher quality
- **Performance**: 200-400x realtime (minimal impact)
- **Flexibility**: Can be disabled for lower latency

---

## Integration Points

### For Developers

**Using as Python Library**:
```python
from soprano import SopranoTTS

model = SopranoTTS(backend='auto', device='cuda')
audio = model.infer("Your text here")
```

**Using API Server**:
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"model": "tts-1", "input": "Your text"}
)
audio_data = response.content
```

**Custom Backend Implementation**:
Extend `BaseModel` from `soprano/backends/base.py`

**Custom Decoder**:
Modify `SopranoDecoder` in `soprano/vocos/decoder.py`

---

## Roadmap Features

From the README, planned enhancements include:

- ✅ Model and inference code
- ✅ Seamless streaming
- ✅ Batched inference
- ✅ Server / API inference (FastAPI with OpenAI compatibility)
- ✅ Web interface (Gradio)
- ⏳ Command-line interface (CLI)
- ⏳ Additional LLM backends
- ⏳ CPU support
- ⏳ Voice cloning
- ⏳ Multilingual support

---

## Performance Tuning

### For Maximum Speed
```python
model = SopranoTTS(
    backend='lmdeploy',      # Fastest backend
    device='cuda',
    cache_size_mb=50,        # Larger cache
    decoder_batch_size=4     # Larger batches
)
```

### For Minimum VRAM
```python
model = SopranoTTS(
    backend='transformers',  # More memory-efficient
    device='cuda',
    cache_size_mb=1,         # Minimal cache
    decoder_batch_size=1     # Single batch
)
```

### For Long-form Content
- Use batched inference: `model.infer_batch()`
- Benefits from sentence-level processing
- Achieves 2000x realtime on long inputs

### For Lowest Latency
- Use streaming: `model.infer_stream(chunk_size=1)`
- First chunk in <15 ms
- Disable FlashSR if needed

---

## Acknowledgements

Soprano uses and/or is inspired by:
- **Vocos**: Vocoder architecture
- **XTTS**: TTS design principles
- **LMDeploy**: Inference acceleration
- **VibeVoice**: Text processing and FlashSR inspiration
- **Kokoro-FastAPI**: API implementation patterns

---

## License

Apache-2.0 License

---

## Summary

Soprano is a highly optimized, modular TTS system with:
- Clean separation between language model and decoder
- Multiple backend support for flexibility
- Production-ready API server with OpenAI compatibility
- User-friendly web interface
- Streaming capabilities for real-time use
- Optional audio upsampling for quality enhancement
- Efficient architecture achieving 2000x realtime speed at <1 GB VRAM

The modular design allows for easy extension, customization, and deployment in various environments.
