#!/usr/bin/env python
"""
Export the Soprano language model (LM) to ONNX format for step-by-step inference.

This script exports the language model component of Soprano TTS to ONNX format,
enabling CPU inference. The exported model performs autoregressive generation
one step at a time, suitable for both streaming and non-streaming inference.

Example usage:
    python soprano/export/lm_step_export.py \\
        --repo_id ekwek/Soprano-80M \\
        --lm_ckpt lm.pth \\
        --out lm_step.onnx
"""

import argparse
import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export Soprano language model to ONNX format"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="ekwek/Soprano-80M",
        help="HuggingFace repository ID (default: ekwek/Soprano-80M)",
    )
    parser.add_argument(
        "--lm_ckpt",
        type=str,
        default="lm.pth",
        help="LM checkpoint filename (not used currently, model loaded from repo)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="lm_step.onnx",
        help="Output ONNX model path (default: lm_step.onnx)",
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
    
    print(f"Loading language model from {args.repo_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.repo_id,
        torch_dtype=torch.float32,
    )
    base_model.eval()
    
    # Create a wrapper model that outputs both logits and hidden states
    class ModelWithHiddenStates(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # Return logits and last hidden state
            # logits: [batch, seq_len, vocab_size]
            # hidden_states: [batch, seq_len, hidden_dim]
            return outputs.logits, outputs.hidden_states[-1]
    
    model = ModelWithHiddenStates(base_model)
    model.eval()
    
    # Create dummy inputs for a single forward pass
    # input_ids: (batch_size, sequence_length)
    # attention_mask: (batch_size, sequence_length)
    batch_size = 1
    seq_length = 10
    dummy_input_ids = torch.randint(0, base_model.config.vocab_size, (batch_size, seq_length))
    dummy_attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    
    print(f"Exporting language model to ONNX (opset {args.opset})...")
    print("  Model will output both logits and hidden states...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        args.out,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits", "hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
            "hidden_states": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    
    print(f"✓ Language model exported successfully to {args.out}")
    print(f"  Input: input_ids [batch_size, sequence_length]")
    print(f"  Input: attention_mask [batch_size, sequence_length]")
    print(f"  Output: logits [batch_size, sequence_length, vocab_size]")
    print(f"  Output: hidden_states [batch_size, sequence_length, hidden_dim]")


if __name__ == "__main__":
    main()
