"""
Example usage of Soprano TTS with the new API features.

This script demonstrates:
1. Basic TTS inference with text normalization
2. Using FlashSR upsampling
3. Text processing utilities

Note: This requires torch and the Soprano model to be available.
"""

# Example 1: Basic usage with text normalization
from soprano import SopranoTTS
from soprano.text_processing import normalize_text

# Create TTS model
model = SopranoTTS(backend='auto', device='cuda')

# Normalize text before processing
text = "Visit www.example.com or call 555-1234. The price is $99.99!"
normalized = normalize_text(text)
print(f"Original: {text}")
print(f"Normalized: {normalized}")

# Generate speech
audio = model.infer(normalized, "output.wav")
print("Audio generated successfully!")


# Example 2: Using FlashSR for upsampling
from soprano.flashsr_upsampler import FlashSRUpsampler
import numpy as np

# Create upsampler
upsampler = FlashSRUpsampler(device='cuda', enable=True)
upsampler.load()

# Upsample audio (32kHz -> 48kHz)
audio_np = audio.cpu().numpy()
upsampled = upsampler.upsample(audio_np, sample_rate=32000)
print(f"Original sample rate: 32000 Hz")
print(f"Upsampled to: 48000 Hz")
print(f"Original shape: {audio_np.shape}")
print(f"Upsampled shape: {upsampled.shape}")


# Example 3: Text sentence splitting
from soprano.text_processing import split_text_into_sentences

long_text = "Hello world. This is a test! How are you? I'm doing great."
sentences = split_text_into_sentences(long_text)
print(f"\nOriginal: {long_text}")
print(f"Sentences: {sentences}")


# Example 4: Using the API programmatically (requires server running)
# This would be used when soprano-server is running
"""
import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "tts-1",
        "input": "Hello from Soprano TTS!",
        "response_format": "wav"
    }
)

with open("api_output.wav", "wb") as f:
    f.write(response.content)
"""

print("\n✓ Examples completed successfully!")
