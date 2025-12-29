#!/usr/bin/env python3
"""
Example usage of Soprano TTS API Server
Demonstrates how to use the OpenAI-compatible API endpoint
"""
import requests
import argparse
import sys


def test_health(base_url: str = "http://localhost:8000"):
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        data = response.json()
        print(f"  Status: {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Backend: {data['backend']}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_list_models(base_url: str = "http://localhost:8000"):
    """Test the models listing endpoint"""
    print("\nListing available models...")
    try:
        response = requests.get(f"{base_url}/v1/models")
        response.raise_for_status()
        data = response.json()
        print(f"  Found {len(data['data'])} model(s):")
        for model in data['data']:
            print(f"    - {model['id']}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def generate_speech(
    text: str,
    output_file: str = "output.wav",
    base_url: str = "http://localhost:8000",
    model: str = "soprano-80m",
    temperature: float = 0.3,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2
):
    """Generate speech using the API"""
    print(f"\nGenerating speech...")
    print(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"  Model: {model}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print(f"  Repetition penalty: {repetition_penalty}")
    
    try:
        response = requests.post(
            f"{base_url}/v1/audio/speech",
            json={
                "model": model,
                "input": text,
                "voice": "default",
                "response_format": "wav",
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty
            },
            timeout=30
        )
        response.raise_for_status()
        
        with open(output_file, "wb") as f:
            f.write(response.content)
        
        print(f"  ✅ Speech generated successfully!")
        print(f"  Saved to: {output_file}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"  Details: {e.response.text}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Example usage of Soprano TTS API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test the API server
  python examples/api_example.py --test

  # Generate speech
  python examples/api_example.py --text "Hello from Soprano TTS!"

  # Generate with custom parameters
  python examples/api_example.py --text "Custom speech" --temperature 0.5 --top_p 0.9

  # Use a different server URL
  python examples/api_example.py --text "Remote TTS" --url http://remote-server:8000
        """
    )
    
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run API tests (health check, list models)"
    )
    parser.add_argument(
        "--text",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output WAV file path (default: output.wav)"
    )
    parser.add_argument(
        "--model",
        default="soprano-80m",
        help="Model to use (default: soprano-80m)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling (default: 0.95)"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition penalty (default: 1.2)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Soprano TTS API Example")
    print("=" * 60)
    print(f"Server: {args.url}")
    print()
    
    # Run tests if requested
    if args.test:
        success = True
        success &= test_health(args.url)
        success &= test_list_models(args.url)
        
        if success:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
        
        if not args.text:
            return
    
    # Generate speech if text provided
    if args.text:
        success = generate_speech(
            text=args.text,
            output_file=args.output,
            base_url=args.url,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        
        if not success:
            sys.exit(1)
    elif not args.test:
        parser.print_help()
        print("\nError: Either --test or --text must be provided")
        sys.exit(1)


if __name__ == "__main__":
    main()
