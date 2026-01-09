#!/usr/bin/env python
"""
Export the Soprano decoder (Vocos) model to ONNX format.

This script exports the decoder component of Soprano TTS to ONNX, enabling
CPU inference without requiring CUDA. The exported model can be used with
ONNX Runtime or converted to OpenVINO IR for optimized CPU execution.

Example usage:
    python soprano/export/decoder_export.py \\
        --repo_id ekwek/Soprano-80M \\
        --decoder_ckpt decoder.pth \\
        --out decoder.onnx
"""

import argparse
import torch
from huggingface_hub import hf_hub_download
from soprano.vocos.decoder import SopranoDecoder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Soprano decoder to ONNX format"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="ekwek/Soprano-80M",
        help="HuggingFace repository ID (default: ekwek/Soprano-80M)",
    )
    parser.add_argument(
        "--decoder_ckpt",
        type=str,
        default="decoder.pth",
        help="Decoder checkpoint filename in the repo (default: decoder.pth)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="decoder.onnx",
        help="Output ONNX model path (default: decoder.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"Loading decoder checkpoint from {args.repo_id}/{args.decoder_ckpt}...")
    decoder_path = hf_hub_download(repo_id=args.repo_id, filename=args.decoder_ckpt)
    
    # Initialize decoder
    decoder = SopranoDecoder()
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))
    decoder.eval()
    
    # Create dummy input (batch_size=1, channels=512, sequence_length=100)
    dummy_input = torch.randn(1, 512, 100)
    
    print(f"Exporting decoder to ONNX (opset {args.opset})...")
    torch.onnx.export(
        decoder,
        dummy_input,
        args.out,
        input_names=["hidden_states"],
        output_names=["audio"],
        dynamic_axes={
            "hidden_states": {0: "batch_size", 2: "sequence_length"},
            "audio": {0: "batch_size", 1: "audio_length"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    
    print(f"✓ Decoder exported successfully to {args.out}")
    print(f"  Input shape: [batch_size, 512, sequence_length]")
    print(f"  Output shape: [batch_size, audio_length]")


if __name__ == "__main__":
    main()
