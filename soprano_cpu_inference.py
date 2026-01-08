#!/usr/bin/env python
"""
Run Soprano TTS on the CPU using ONNX models.

This script demonstrates how to synthesize Spanish speech using the
Soprano TTS interface included in the `finetune-hf-vits` repository.  It
assumes that you have already exported the Soprano language model (LM)
and decoder to ONNX format using the export scripts described in the
project documentation.  See `README.md` for full instructions on exporting
the models and converting them to OpenVINO if desired.

Steps to use this script:

1. Install the repository with ONNX support:

   ```bash
   pip install -e ".[onnx]"
   ```

2. Export the models.  Replace the `repo_id` argument with the model
   checkpoint you wish to use (e.g., `ekwek/Soprano-80M`):

   ```bash
   python soprano/export/decoder_export.py --repo_id ekwek/Soprano-80M --decoder_ckpt decoder.pth --out decoder.onnx
   python soprano/export/lm_step_export.py --repo_id ekwek/Soprano-80M --lm_ckpt lm.pth --out lm_step.onnx
   ```

3. Run this script to synthesize speech on the CPU:

   ```bash
   python soprano_cpu_inference.py --lm lm_step.onnx --decoder decoder.onnx --text "Hola, mundo!"
   ```

   The script will produce a WAV file named `tts_output.wav` in the current
   directory.

You can optionally convert the ONNX models to OpenVINO IR for faster
CPU inference as described in the README.  In that case, set
`--backend openvino_cpu` and pass the `.xml` files instead of `.onnx`.

"""

import argparse
from soprano.tts import SopranoTTS
import scipy.io.wavfile


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Synthesize speech with Soprano TTS on CPU")
    parser.add_argument("--lm", required=True, help="Path to the LM ONNX or OpenVINO model")
    parser.add_argument("--decoder", required=True, help="Path to the decoder ONNX or OpenVINO model")
    parser.add_argument("--text", required=True, help="Input text to synthesize")
    parser.add_argument(
        "--backend",
        choices=["onnx_cpu", "openvino_cpu"],
        default="onnx_cpu",
        help="Backend to use for inference; use 'openvino_cpu' for OpenVINO models",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of CPU threads for ONNX/OpenVINO runtime",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Initialise the TTS engine.  For ONNX/OpenVINO backends the device is
    # always CPU.  See `soprano/tts.py` in the repository for details.
    tts = SopranoTTS(
        lm_path=args.lm,
        decoder_path=args.decoder,
        backend=args.backend,
        num_threads=args.num_threads,
    )

    # Synthesize audio.  You can adjust sampling parameters such as
    # temperature and nucleus sampling here.  For deterministic output,
    # leave temperature at 1.0 and top_p at 1.0.
    result = tts.synthesize(
        text=args.text,
        max_new_tokens=100,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=42,
    )

    audio = result["audio"]
    sample_rate = result["sample_rate"]

    # Write the output to a WAV file
    out_path = "tts_output.wav"
    scipy.io.wavfile.write(out_path, sample_rate, audio)
    print(f"✓ Synthesized audio saved to {out_path}")


if __name__ == "__main__":
    main()
