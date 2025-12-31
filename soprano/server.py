#!/usr/bin/env python3
"""
CLI script to run the Soprano TTS API server.
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run Soprano TTS API server with OpenAI-compatible endpoints"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "lmdeploy", "transformers"],
        help="Backend to use for inference (default: auto)",
    )
    parser.add_argument(
        "--cache-size-mb",
        type=int,
        default=10,
        help="Cache size in MB (default: 10)",
    )
    parser.add_argument(
        "--decoder-batch-size",
        type=int,
        default=1,
        help="Decoder batch size (default: 1)",
    )
    parser.add_argument(
        "--disable-flashsr",
        action="store_true",
        help="Disable FlashSR audio upsampling (32kHz -> 48kHz)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    args = parser.parse_args()

    # Set environment variables for the FastAPI app
    os.environ["MODEL_DEVICE"] = args.device
    os.environ["BACKEND"] = args.backend
    os.environ["CACHE_SIZE_MB"] = str(args.cache_size_mb)
    os.environ["DECODER_BATCH_SIZE"] = str(args.decoder_batch_size)
    os.environ["ENABLE_FLASHSR"] = "false" if args.disable_flashsr else "true"

    print("🚀 Starting Soprano TTS API server...")
    print(f"   Device: {args.device}")
    print(f"   Backend: {args.backend}")
    print(f"   FlashSR: {'disabled' if args.disable_flashsr else 'enabled (32kHz -> 48kHz)'}")
    print(f"\n🌐 Server starting at: http://{args.host}:{args.port}")
    print(f"   API Docs: http://{args.host}:{args.port}/docs")
    print(f"   Web UI: http://{args.host}:{args.port}/")
    print("   Press Ctrl+C to stop the server\n")

    # Import and run uvicorn
    try:
        import uvicorn
    except ImportError:
        print("❌ Error: uvicorn not installed. Please install it with:")
        print("   pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn.run(
        "soprano.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
