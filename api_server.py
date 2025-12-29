#!/usr/bin/env python3
"""
FastAPI Server with OpenAI-compatible API endpoint for Soprano TTS

This server provides an OpenAI-compatible /v1/audio/speech endpoint
for text-to-speech synthesis using the Soprano TTS model.
"""
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
import torch
import numpy as np
import io
from scipy.io import wavfile
from soprano import SopranoTTS
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Soprano TTS API",
    description="OpenAI-compatible Text-to-Speech API powered by Soprano TTS",
    version="0.0.2",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model instance
model: Optional[SopranoTTS] = None


class TTSRequest(BaseModel):
    """OpenAI-compatible TTS request model"""
    model: str = Field(
        default="soprano-80m",
        description="TTS model to use. Currently only 'soprano-80m' is supported."
    )
    input: str = Field(
        ...,
        description="The text to generate audio for. Maximum length is 4096 characters.",
        max_length=4096
    )
    voice: str = Field(
        default="default",
        description="Voice to use for generation. Currently only 'default' is supported."
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="wav",
        description="Audio format. Currently only 'wav' is supported."
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed of generated audio. Note: Soprano doesn't support speed control, this is ignored."
    )
    # Soprano-specific parameters
    temperature: Optional[float] = Field(
        default=0.3,
        ge=0.1,
        le=1.0,
        description="Sampling temperature (Soprano-specific)"
    )
    top_p: Optional[float] = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="Top-p sampling parameter (Soprano-specific)"
    )
    repetition_penalty: Optional[float] = Field(
        default=1.2,
        ge=1.0,
        le=2.0,
        description="Repetition penalty (Soprano-specific)"
    )


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    model_loaded: bool
    backend: str


@app.on_event("startup")
async def startup_event():
    """Initialize the TTS model on startup"""
    global model
    try:
        logger.info("Loading Soprano TTS model...")
        model = SopranoTTS(
            backend='auto',
            device='cuda',
            cache_size_mb=10,
            decoder_batch_size=1
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Soprano TTS API",
        "version": "0.0.2",
        "description": "OpenAI-compatible Text-to-Speech API",
        "endpoints": {
            "tts": "/v1/audio/speech",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        backend="soprano-80m"
    )


@app.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    """
    OpenAI-compatible text-to-speech endpoint
    
    Generates audio from the input text using Soprano TTS.
    Compatible with OpenAI's /v1/audio/speech endpoint.
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="TTS model not loaded. Please check server logs."
        )
    
    # Validate input
    if not request.input or not request.input.strip():
        raise HTTPException(
            status_code=400,
            detail="Input text is required and cannot be empty."
        )
    
    # Validate model
    if request.model not in ["soprano-80m", "tts-1", "tts-1-hd"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' not supported. Use 'soprano-80m', 'tts-1', or 'tts-1-hd'."
        )
    
    # Validate voice (Soprano only has one voice)
    if request.voice not in ["default", "alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{request.voice}' not supported. Only 'default' is available."
        )
    
    # Validate response format
    if request.response_format != "wav":
        raise HTTPException(
            status_code=400,
            detail=f"Format '{request.response_format}' not yet supported. Currently only 'wav' is available."
        )
    
    try:
        logger.info(f"Generating speech for text: {request.input[:50]}...")
        
        # Generate audio
        audio = model.infer(
            request.input,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        
        # Convert to WAV format
        audio_np = audio.cpu().numpy()
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, 32000, audio_np)
        wav_buffer.seek(0)
        
        logger.info("Speech generated successfully")
        
        # Return audio as streaming response
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating speech: {str(e)}"
        )


@app.get("/v1/models")
async def list_models():
    """
    List available models (OpenAI-compatible endpoint)
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "soprano-80m",
                "object": "model",
                "created": 1704067200,
                "owned_by": "soprano",
                "permission": [],
                "root": "soprano-80m",
                "parent": None
            }
        ]
    }


def main():
    """Run the FastAPI server"""
    logger.info("Starting Soprano TTS API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
