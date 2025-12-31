"""
Example API client for Soprano TTS server.

This demonstrates how to use the OpenAI-compatible API endpoints.

Usage:
    1. Start the server: soprano-server
    2. Run this script: python examples/api_client_example.py
"""

import requests
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_list_voices():
    """Test the voices listing endpoint."""
    print("Testing /v1/audio/voices endpoint...")
    response = requests.get(f"{BASE_URL}/v1/audio/voices")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Available voices: {len(data['voices'])}")
    for voice in data['voices']:
        print(f"  - {voice['id']}: {voice['name']}")
    print()


def test_speech_generation(text, format="wav"):
    """Test speech generation endpoint."""
    print(f"Testing /v1/audio/speech endpoint with format={format}...")
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": text,
            "voice": "soprano-default",
            "response_format": format
        }
    )
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        # Save the audio file
        filename = f"output_{format}_{int(time.time())}.{format}"
        with open(filename, "wb") as f:
            f.write(response.content)
        
        size_kb = len(response.content) / 1024
        print(f"Audio saved to: {filename}")
        print(f"Size: {size_kb:.2f} KB")
        print(f"Generation time: {elapsed:.2f} seconds")
    else:
        print(f"Error: {response.text}")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Soprano TTS API Client Example")
    print("=" * 60)
    print()
    
    try:
        # Test health endpoint
        test_health()
        
        # Test voices endpoint
        test_list_voices()
        
        # Test speech generation with different formats
        test_text = "Soprano is an extremely lightweight text to speech model capable of generating high-fidelity speech at unprecedented speed."
        
        # WAV format
        test_speech_generation(test_text, format="wav")
        
        # Opus format (if available)
        test_speech_generation(test_text, format="opus")
        
        # MP3 format (if available)
        test_speech_generation(test_text, format="mp3")
        
        print("=" * 60)
        print("✓ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server.")
        print("Make sure the server is running: soprano-server")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
