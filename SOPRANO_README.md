# Soprano CPU Inference Guide

This guide explains how to run Soprano TTS on CPU using ONNX or OpenVINO backends.

## Overview

Soprano TTS can be run on CPU without CUDA by exporting the models to ONNX format and using ONNX Runtime or OpenVINO for inference. This enables deployment on CPU-only machines.

## Installation

Install Soprano with ONNX support:

```bash
pip install -e ".[onnx]"
```

For OpenVINO support (optional, for optimized CPU inference):

```bash
pip install openvino
```

## Exporting Models to ONNX

Before running CPU inference, you need to export both the language model and decoder to ONNX format.

### Export the Decoder

The decoder (Vocos) converts hidden states to audio waveforms:

```bash
python soprano/export/decoder_export.py \
    --repo_id ekwek/Soprano-80M \
    --decoder_ckpt decoder.pth \
    --out decoder.onnx
```

**Parameters:**
- `--repo_id`: HuggingFace repository containing the model (default: ekwek/Soprano-80M)
- `--decoder_ckpt`: Checkpoint filename in the repo (default: decoder.pth)
- `--out`: Output path for the ONNX model (default: decoder.onnx)
- `--opset`: ONNX opset version (default: 14)

### Export the Language Model

The language model generates hidden states from text:

```bash
python soprano/export/lm_step_export.py \
    --repo_id ekwek/Soprano-80M \
    --lm_ckpt lm.pth \
    --out lm_step.onnx
```

**Parameters:**
- `--repo_id`: HuggingFace repository containing the model (default: ekwek/Soprano-80M)
- `--lm_ckpt`: LM checkpoint filename (default: lm.pth)
- `--out`: Output path for the ONNX model (default: lm_step.onnx)
- `--opset`: ONNX opset version (default: 14)

## Converting to OpenVINO IR (Optional)

For optimized CPU inference, you can convert the ONNX models to OpenVINO Intermediate Representation (IR):

### Install OpenVINO Development Tools

```bash
pip install openvino-dev
```

### Convert Decoder to OpenVINO

```bash
mo --input_model decoder.onnx --output_dir openvino_models
```

This creates `decoder.xml` and `decoder.bin` files.

### Convert Language Model to OpenVINO

```bash
mo --input_model lm_step.onnx --output_dir openvino_models
```

This creates `lm_step.xml` and `lm_step.bin` files.

**Note:** The Model Optimizer (`mo`) is part of the `openvino-dev` package. Alternatively, you can use the newer `ovc` tool:

```bash
ovc decoder.onnx --output_model openvino_models/decoder.xml
ovc lm_step.onnx --output_model openvino_models/lm_step.xml
```

## Running CPU Inference

### Using the Example Script

The simplest way to synthesize speech on CPU:

```bash
python soprano_cpu_inference.py \
    --lm lm_step.onnx \
    --decoder decoder.onnx \
    --text "Hello, world! This is Soprano running on CPU."
```

**With OpenVINO models:**

```bash
python soprano_cpu_inference.py \
    --lm openvino_models/lm_step.xml \
    --decoder openvino_models/decoder.xml \
    --backend openvino_cpu \
    --text "Hello, world! This is Soprano running on CPU with OpenVINO."
```

**Parameters:**
- `--lm`: Path to the LM model (ONNX or OpenVINO .xml)
- `--decoder`: Path to the decoder model (ONNX or OpenVINO .xml)
- `--text`: Text to synthesize
- `--backend`: Backend to use (`onnx_cpu` or `openvino_cpu`, default: onnx_cpu)
- `--num_threads`: Number of CPU threads (default: 4)

### Using the Python API

```python
from soprano.tts import SopranoTTS
import scipy.io.wavfile

# Initialize with ONNX backend
tts = SopranoTTS(
    lm_path='lm_step.onnx',
    decoder_path='decoder.onnx',
    backend='onnx_cpu',
    num_threads=4,
)

# Synthesize speech
result = tts.synthesize(
    text="Hello from Soprano TTS on CPU!",
    max_new_tokens=100,
    temperature=1.0,
    top_p=1.0,
    repetition_penalty=1.0,
    seed=42,
)

# Save to file
scipy.io.wavfile.write('output.wav', result['sample_rate'], result['audio'])
```

**With OpenVINO backend:**

```python
tts = SopranoTTS(
    lm_path='openvino_models/lm_step.xml',
    decoder_path='openvino_models/decoder.xml',
    backend='openvino_cpu',
    num_threads=4,
)
```

## Performance Considerations

### ONNX Runtime vs. OpenVINO

- **ONNX Runtime**: Fully implemented, easy to use, good cross-platform compatibility
- **OpenVINO**: Partially implemented (model loading only). Full inference support coming in future releases. Expected to be 1.5-3x faster than ONNX Runtime on Intel CPUs once completed.

**Note**: For now, use `backend='onnx_cpu'` for actual inference. The `openvino_cpu` backend can load models but does not yet support inference operations.

### Thread Count

Adjust `--num_threads` based on your CPU:
- For most modern CPUs: 4-8 threads
- For high-end workstations: 8-16 threads
- Experiment to find the optimal value for your hardware

### Memory Usage

CPU inference uses significantly less memory than GPU inference:
- ONNX Runtime: ~500-800 MB
- OpenVINO: ~400-600 MB

## Sampling Parameters

Control the quality and diversity of generated speech:

- **temperature** (default: 1.0): Higher values (e.g., 1.2) increase randomness, lower values (e.g., 0.8) make output more deterministic
- **top_p** (default: 1.0): Nucleus sampling threshold (0.9-0.95 recommended for diverse output)
- **repetition_penalty** (default: 1.0): Penalize repeated tokens (1.1-1.2 helps avoid repetition)
- **seed**: Set for reproducible output
- **max_new_tokens**: Maximum tokens to generate (default: 100)

## Limitations

Current limitations of CPU inference:

1. **Slower than GPU**: CPU inference is typically 5-10x slower than GPU inference
2. **No streaming support**: The CPU backend currently only supports batch inference via `synthesize()`
3. **Model export required**: You must export models before using CPU inference
4. **OpenVINO incomplete**: The OpenVINO backend is partially implemented (model loading only, no inference yet)
5. **ONNX Runtime recommended**: Use `backend='onnx_cpu'` for production use

## Troubleshooting

### ImportError: onnxruntime is required

Install ONNX Runtime:
```bash
pip install onnxruntime
```

### ImportError: openvino is required

Install OpenVINO:
```bash
pip install openvino
```

### ONNX export fails

Ensure you have the required packages:
```bash
pip install torch transformers huggingface_hub
```

### Audio quality issues

- Try adjusting sampling parameters (temperature, top_p)
- Ensure models were exported correctly
- Check that text preprocessing is working correctly

## Future Improvements

Planned enhancements for CPU inference:

- [ ] Streaming support for CPU backends
- [ ] Optimized hidden state extraction
- [ ] Quantized models for faster inference
- [ ] Pre-exported ONNX models on HuggingFace
- [ ] Batch processing support

## See Also

- [Main README](README.md) - General Soprano documentation
- [QUICKSTART](QUICKSTART.md) - Quick start guide
- [HuggingFace Model](https://huggingface.co/ekwek/Soprano-80M) - Pre-trained model
