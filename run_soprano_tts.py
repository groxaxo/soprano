#!/usr/bin/env python3
import argparse
import sys
import torch

from soprano import SopranoTTS


def pick_device(requested: str) -> str:
    requested = requested.lower().strip()
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available in this PyTorch install.", file=sys.stderr)
            sys.exit(1)
        return "cuda"
    if requested == "cpu":
        # Soprano currently requires a CUDA GPU per README (CPU support coming later).
        print("ERROR: Soprano currently requires a CUDA GPU (CPU support is not available yet).", file=sys.stderr)
        sys.exit(1)
    return requested  # allow advanced users to pass things like 'cuda:0'


def main():
    ap = argparse.ArgumentParser(description="Run Soprano TTS on Linux (GTX 750-friendly settings).")
    ap.add_argument("--text", required=True, help="Text to synthesize.")
    ap.add_argument("--out", default="out.wav", help="Output WAV path (default: out.wav)")
    ap.add_argument("--device", default="cuda", help="Device: cuda (default)")

    # Force transformers backend for older GPUs (LMDeploy supports newer GPUs only).
    ap.add_argument("--backend", default="transformers", choices=["transformers", "auto"],
                    help="Backend: transformers (default) or auto (tries LMDeploy).")

    # Conservative defaults for low VRAM
    ap.add_argument("--cache_mb", type=int, default=1, help="KV cache size in MB (default: 1)")
    ap.add_argument("--decoder_batch_size", type=int, default=1, help="Decoder batch size (default: 1)")

    # Optional sampling controls
    ap.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3)")
    ap.add_argument("--top_p", type=float, default=0.95, help="Top-p (default: 0.95)")
    ap.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty (default: 1.2)")

    # Streaming mode (optional)
    ap.add_argument("--stream", action="store_true", help="Use streaming inference and then save WAV")

    args = ap.parse_args()
    device = pick_device(args.device)

    # Validate streaming + backend compatibility
    if args.stream and args.backend == "transformers":
        print("ERROR: Streaming mode (--stream) is not supported with the transformers backend.", file=sys.stderr)
        print("Please use --backend auto to try LMDeploy, or remove --stream flag.", file=sys.stderr)
        sys.exit(1)

    if device.startswith("cuda"):
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        print(f"CUDA device: {name} | compute capability: {cc[0]}.{cc[1]}")
        if args.backend == "auto":
            print("NOTE: backend=auto may try LMDeploy; on older GPUs it may fail. "
                  "Use --backend transformers if you hit errors.")

    print(f"Loading SopranoTTS (backend={args.backend}, device={device}) ...")
    model = SopranoTTS(
        backend=args.backend,
        device=device,
        cache_size_mb=args.cache_mb,
        decoder_batch_size=args.decoder_batch_size,
    )

    print("Running inference...")
    if not args.stream:
        # This saves to args.out if a filename is provided (as shown in the repo README).
        _ = model.infer(
            args.text,
            args.out,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        # Streaming inference (collect chunks, then save)
        chunks = []
        stream = model.infer_stream(
            args.text,
            chunk_size=1,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        for chunk in stream:
            chunks.append(chunk)
        audio = torch.cat(chunks)

        # If Soprano doesn't expose a dedicated save util, re-run infer to save.
        # (Keeps this script compatible with the public README interface.)
        _ = model.infer(args.text, args.out,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=args.repetition_penalty)

    print(f"Done. Wrote: {args.out}")


if __name__ == "__main__":
    main()
