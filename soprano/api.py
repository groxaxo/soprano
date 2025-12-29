"""
FastAPI server for Soprano TTS with OpenAI-compatible API endpoints.
"""
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

from scipy.io import wavfile

from .tts import SopranoTTS
from .text_processing import normalize_text
from .flashsr_upsampler import FlashSRUpsampler

app = FastAPI(title="Soprano TTS API", version="0.0.2")

BASE = Path(__file__).parent
SAMPLE_RATE = 32000
UPSAMPLED_RATE = 48000


class OpenAISpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "opus"
    speed: Optional[float] = 1.0


@app.on_event("startup")
async def _startup() -> None:
    """Initialize the TTS service on startup."""
    device = os.environ.get("MODEL_DEVICE", "cuda")
    backend = os.environ.get("BACKEND", "auto")
    cache_size_mb = int(os.environ.get("CACHE_SIZE_MB", "10"))
    decoder_batch_size = int(os.environ.get("DECODER_BATCH_SIZE", "1"))
    
    # FlashSR enabled by default
    enable_flashsr_str = os.environ.get("ENABLE_FLASHSR", "true").lower()
    enable_flashsr = enable_flashsr_str in ("true", "1", "yes", "on")
    
    print(f"[startup] Initializing Soprano TTS (device={device}, backend={backend})")
    
    tts_service = SopranoTTS(
        backend=backend,
        device=device,
        cache_size_mb=cache_size_mb,
        decoder_batch_size=decoder_batch_size
    )
    
    # Initialize FlashSR upsampler
    if enable_flashsr:
        print("[startup] Initializing FlashSR upsampler for 32kHz -> 48kHz super-resolution")
        flashsr = FlashSRUpsampler(device=device, enable=True)
        flashsr.load()
    else:
        print("[startup] FlashSR disabled, audio will remain at 32kHz")
        flashsr = FlashSRUpsampler(device=device, enable=False)
    
    app.state.tts_service = tts_service
    app.state.flashsr = flashsr
    app.state.device = device
    print("[startup] Model ready.")


@app.post("/v1/audio/speech")
async def openai_speech(request: OpenAISpeechRequest):
    """
    OpenAI-compatible text-to-speech endpoint.
    
    Generates audio from text with optional FlashSR super-resolution (32kHz → 48kHz).
    
    Args:
        request: OpenAISpeechRequest with model, input text, voice, response_format, and speed
        
    Returns:
        Audio file in the requested format (opus, wav, or mp3)
    """
    tts_service: SopranoTTS = app.state.tts_service
    flashsr: FlashSRUpsampler = app.state.flashsr
    
    # Normalize and clean the input text
    normalized_text = normalize_text(request.input)
    
    if not normalized_text.strip():
        return Response(status_code=400, content="No valid text provided after normalization")
    
    try:
        # Generate audio using Soprano TTS
        audio = tts_service.infer(normalized_text)
        
        # Convert torch tensor to numpy
        if torch.is_tensor(audio):
            audio = audio.cpu().numpy()
        
        # Apply FlashSR upsampling if enabled
        output_sample_rate = SAMPLE_RATE
        if flashsr and flashsr.enabled:
            audio = flashsr.upsample(audio, sample_rate=SAMPLE_RATE)
            output_sample_rate = UPSAMPLED_RATE
        
        # Ensure audio is in correct format
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        
        # Normalize audio to prevent clipping
        peak = np.max(np.abs(audio)) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak
        
        # Convert to WAV first
        buffer = io.BytesIO()
        wavfile.write(buffer, output_sample_rate, audio)
        wav_data = buffer.getvalue()
        
        # Convert to requested format
        if request.response_format == "mp3":
            if not PYDUB_AVAILABLE:
                print("[Warning] pydub not available. Falling back to WAV format.")
                return Response(content=wav_data, media_type="audio/wav")
            buffer.seek(0)
            audio_segment = AudioSegment.from_wav(buffer)
            mp3_buffer = io.BytesIO()
            audio_segment.export(mp3_buffer, format="mp3")
            return Response(content=mp3_buffer.getvalue(), media_type="audio/mpeg")
        elif request.response_format == "opus":
            if not PYDUB_AVAILABLE:
                print("[Warning] pydub not available. Falling back to WAV format.")
                return Response(content=wav_data, media_type="audio/wav")
            try:
                buffer.seek(0)
                audio_segment = AudioSegment.from_wav(buffer)
                opus_buffer = io.BytesIO()
                # Export as opus using the actual output sample rate
                audio_segment.export(opus_buffer, format="opus", codec="libopus",
                                   parameters=["-ar", str(output_sample_rate)])
                return Response(content=opus_buffer.getvalue(), media_type="audio/opus")
            except Exception as e:
                # Fallback to WAV if opus encoding fails (e.g., ffmpeg not available)
                print(f"[Warning] Opus encoding failed: {e}. Falling back to WAV format.")
                return Response(content=wav_data, media_type="audio/wav")
        else:  # wav or default
            return Response(content=wav_data, media_type="audio/wav")
            
    except Exception as e:
        print(f"[Error] Failed to generate speech: {e}")
        return Response(status_code=500, content=f"Failed to generate speech: {str(e)}")


@app.get("/v1/audio/voices")
def get_voices():
    """
    List available voices.
    
    Returns:
        JSON object with list of available voices (currently Soprano uses a single default voice)
    """
    voices = [
        {
            "id": "soprano-default",
            "name": "Soprano Default Voice",
            "object": "voice",
            "category": "soprano_tts"
        }
    ]
    return {"voices": voices}


@app.get("/")
def index():
    """Serve the web interface."""
    index_path = BASE / "static" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Soprano TTS API", "version": "0.0.2", "docs": "/docs"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "soprano-tts"}
