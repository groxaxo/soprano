# Soprano TTS ONNX Export & CPU Inference Master Plan

This document outlines the comprehensive strategy for converting Soprano TTS to ONNX format for CPU inference, including OpenVINO optimization options.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current Architecture Analysis](#current-architecture-analysis)
3. [ONNX Export Strategy](#onnx-export-strategy)
4. [OpenVINO Integration](#openvino-integration)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Performance Considerations](#performance-considerations)
7. [Challenges & Solutions](#challenges--solutions)
8. [Testing & Validation](#testing--validation)
9. [Deployment Guide](#deployment-guide)
10. [Appendix](#appendix)

---

## Executive Summary

### Goal
Transform Soprano TTS to enable CPU-based inference through ONNX export, with optional OpenVINO optimization for Intel hardware acceleration.

### Current State
- **GPU-only**: CUDA required for both language model and decoder
- **Backend**: LMDeploy (CUDA-only) or Transformers (limited CPU support)
- **Components**: 80M parameter language model + Vocos decoder
- **Performance**: ~2000x realtime on GPU

### Target State
- **ONNX Models**: Separate ONNX exports for LM and decoder
- **CPU Inference**: Pure CPU execution path
- **OpenVINO**: Optional optimization for Intel CPUs
- **Performance Goal**: >50x realtime on modern CPU (acceptable for real-world use)

### Key Benefits
1. **Broader Deployment**: Run on CPU-only servers and edge devices
2. **Lower Costs**: No GPU required for inference
3. **Better Compatibility**: ONNX Runtime supported on all platforms
4. **Intel Optimization**: OpenVINO for maximum CPU performance
5. **Flexibility**: Mix of ONNX Runtime and OpenVINO depending on hardware

---

## Current Architecture Analysis

### System Components

The Soprano pipeline consists of two main neural network components:

```
Text → Language Model (80M) → Hidden States → Vocos Decoder → Audio
       [PyTorch/LMDeploy]    (512-dim)        [PyTorch]      (32kHz)
```

### Component 1: Language Model (80M Parameters)

**File**: Loaded from HuggingFace (`ekwek/Soprano-80M`)

**Type**: Causal language model (autoregressive)

**Current Implementation**:
- **LMDeploy Backend**: CUDA-only, Turbomind engine
- **Transformers Backend**: PyTorch-based, potential CPU support

**Input**: Tokenized text
**Output**: Hidden states (sequence of 512-dimensional vectors)

**Characteristics**:
- Autoregressive generation (sequential token generation)
- KV-cache for efficiency
- Sampling parameters: temperature, top_p, repetition_penalty
- Max sequence length: 512 tokens

**ONNX Export Complexity**: **High**
- Autoregressive generation requires special handling
- Dynamic shapes (sequence length varies)
- KV-cache management
- Sampling operations

### Component 2: Vocos Decoder

**Files**: `soprano/vocos/decoder.py`, `soprano/vocos/models.py`, `soprano/vocos/heads.py`

**Type**: Feed-forward vocoder (ConvNeXt backbone + ISTFT head)

**Current Implementation**: PyTorch CUDA

**Input**: Hidden states (Batch, 512, Sequence)
**Output**: Audio waveform (Batch, Samples)

**Architecture**:
- `VocosBackbone`: ConvNeXt-style CNN (8 layers, 512 dim)
- Linear interpolation upsampling (4x)
- `ISTFTHead`: Spectral conversion to audio

**Characteristics**:
- Feed-forward (non-autoregressive)
- Fixed architecture
- No dynamic control flow
- Receptive field: 4 tokens

**ONNX Export Complexity**: **Low-Medium**
- Mostly standard operations
- ISTFT may need custom implementation
- Linear interpolation supported in ONNX

### Component 3: FlashSR Upsampler

**File**: `soprano/flashsr_upsampler.py`

**Type**: Audio resampling (currently librosa-based)

**Current Implementation**: CPU-based resampling

**ONNX Export Complexity**: **Low**
- Already CPU-compatible
- Can use standard audio processing libraries
- Not critical for ONNX conversion

---

## ONNX Export Strategy

### Overview

We'll export Soprano in **two separate ONNX models**:

1. **Language Model ONNX**: Text → Hidden States
2. **Decoder ONNX**: Hidden States → Audio

This separation provides:
- Easier debugging and validation
- Flexible backend mixing (e.g., PyTorch LM + ONNX Decoder)
- Independent optimization
- Simpler model management

### Export Approach

#### Strategy 1: Decoder-First (Recommended)

**Phase 1**: Export Vocos Decoder to ONNX
- Lower complexity
- Quick wins for CPU inference
- Can still use GPU for LM if needed

**Phase 2**: Export Language Model to ONNX
- More complex due to autoregressive nature
- Multiple approaches available

**Advantages**:
- Incremental progress
- Early validation
- Hybrid deployment options

#### Strategy 2: Full Pipeline Export

Export entire pipeline as single ONNX model

**Advantages**:
- Simpler deployment
- Single model file

**Disadvantages**:
- Much more complex
- Harder to debug
- Less flexible

**Recommendation**: Use Strategy 1 (Decoder-First)

---

## ONNX Export Implementation

### Part 1: Exporting the Vocos Decoder

The Vocos decoder is a feed-forward model, making it ideal for ONNX export.

#### Step 1: Prepare the Decoder Model

```python
import torch
from soprano.vocos.decoder import SopranoDecoder
from huggingface_hub import hf_hub_download

# Load the decoder
decoder = SopranoDecoder()
decoder_path = hf_hub_download(repo_id='ekwek/Soprano-80M', filename='decoder.pth')
decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
decoder.eval()
decoder.cpu()
```

#### Step 2: Create Dummy Input

```python
# Create dummy input matching expected shape
# Shape: (batch_size, channels=512, sequence_length)
batch_size = 1
channels = 512
sequence_length = 100  # Example length

dummy_input = torch.randn(batch_size, channels, sequence_length)
```

#### Step 3: Export to ONNX

```python
import torch.onnx

# Export with dynamic axes for flexibility
torch.onnx.export(
    decoder,
    dummy_input,
    "soprano_decoder.onnx",
    export_params=True,
    opset_version=17,  # Use latest stable opset
    do_constant_folding=True,
    input_names=['hidden_states'],
    output_names=['audio'],
    dynamic_axes={
        'hidden_states': {0: 'batch_size', 2: 'sequence_length'},
        'audio': {0: 'batch_size', 1: 'audio_samples'}
    }
)
```

#### Step 4: Verify Export

```python
import onnx
import onnxruntime as ort

# Load and check the model
onnx_model = onnx.load("soprano_decoder.onnx")
onnx.checker.check_model(onnx_model)

# Test inference
session = ort.InferenceSession("soprano_decoder.onnx")
ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
ort_outputs = session.run(None, ort_inputs)

# Compare with PyTorch
with torch.no_grad():
    pytorch_output = decoder(dummy_input)

# Check numerical accuracy
print(f"Max difference: {torch.max(torch.abs(pytorch_output - torch.tensor(ort_outputs[0])))}")
```

#### Potential Issues with Decoder Export

**Issue 1: ISTFT Operation**

The ISTFT (Inverse Short-Time Fourier Transform) in `soprano/vocos/heads.py` may not be natively supported in ONNX.

**Solution**:
- **Option A**: Replace with ONNX-compatible operations
- **Option B**: Use custom ONNX operator
- **Option C**: Implement ISTFT in post-processing (outside ONNX model)

**Implementation Example** (Option A - ONNX Compatible):
```python
# In soprano/vocos/heads.py, modify ISTFTHead
# Replace torch.istft with supported operations
# Or use separate post-processing
```

**Issue 2: Linear Interpolation**

The `torch.nn.functional.interpolate` in the decoder may need verification.

**Solution**:
- ONNX supports interpolation via Resize operator
- Test with different opset versions if issues arise

**Issue 3: Spectral Operations**

Check `soprano/vocos/spectral_ops.py` for custom operations.

**Solution**:
- Verify all operations are ONNX-compatible
- Replace custom ops with standard operations if needed

### Part 2: Exporting the Language Model

The language model export is more complex due to autoregressive generation.

#### Approach 1: Export with Static KV-Cache (Recommended)

This approach exports the model with a fixed maximum sequence length.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    'ekwek/Soprano-80M',
    torch_dtype=torch.float32,
    device_map='cpu'
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')

# Prepare for export
# Note: This requires model modifications for static shapes
```

**Challenges**:
- Need to modify the model to output hidden states
- KV-cache handling
- Autoregressive loop outside ONNX

#### Approach 2: Use Optimum Library

HuggingFace Optimum provides tools for ONNX export of transformers models.

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# This handles the complexity of exporting causal LM
model = ORTModelForCausalLM.from_pretrained(
    'ekwek/Soprano-80M',
    export=True,
    provider='CPUExecutionProvider'
)

tokenizer = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')

# Use like regular transformers model
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model.generate(**inputs)
```

**Advantages**:
- Handles KV-cache automatically
- Optimized for inference
- Well-tested
- Supports dynamic shapes

**Disadvantages**:
- Need to extract hidden states
- May require custom modeling code

#### Approach 3: Export Only the Forward Pass

Export single forward pass, implement generation loop in Python.

```python
class HiddenStateModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        # Return last hidden states
        return outputs.hidden_states[-1]

# Wrap and export
wrapped_model = HiddenStateModel(model)
dummy_input_ids = torch.randint(0, 1000, (1, 10))
dummy_attention_mask = torch.ones_like(dummy_input_ids)

torch.onnx.export(
    wrapped_model,
    (dummy_input_ids, dummy_attention_mask),
    "soprano_lm_forward.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['hidden_states'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'hidden_states': {0: 'batch', 1: 'sequence', 2: 'hidden_dim'}
    },
    opset_version=17
)
```

**Advantages**:
- Simpler export
- More control over generation

**Disadvantages**:
- Need to implement generation in Python
- Slower than optimized ONNX generation

#### Recommended Approach for Language Model

**Use HuggingFace Optimum** with custom wrapper to extract hidden states:

```python
from optimum.onnxruntime import ORTModelForCausalLM
import torch

class SopranoLMONNX:
    def __init__(self, model_path="ekwek/Soprano-80M"):
        self.model = ORTModelForCausalLM.from_pretrained(
            model_path,
            export=True,
            provider='CPUExecutionProvider'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def infer(self, prompts, temperature=0.3, top_p=0.95, repetition_penalty=1.2):
        """Generate hidden states for prompts"""
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Generate with custom parameters
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            
            # Extract hidden states
            # This requires model modifications or post-processing
            hidden_states = self._extract_hidden_states(outputs)
            
            results.append({
                'finish_reason': 'stop',
                'hidden_state': hidden_states
            })
        
        return results
    
    def _extract_hidden_states(self, outputs):
        # Custom logic to extract and process hidden states
        # May need to modify the ONNX export to include hidden states
        pass
```

---

## OpenVINO Integration

OpenVINO is Intel's toolkit for optimizing and deploying deep learning models on Intel hardware.

### Why OpenVINO?

**Advantages**:
1. **CPU Optimization**: Highly optimized for Intel CPUs (AVX2, AVX-512)
2. **Performance**: Often 2-4x faster than ONNX Runtime on Intel CPUs
3. **Quantization**: INT8 quantization for 4x speedup with minimal quality loss
4. **Hardware Support**: CPU, GPU (Intel), VPU, GNA
5. **Model Optimization**: Graph optimizations and fusion

**Disadvantages**:
1. Intel hardware focus (less optimized for AMD)
2. Additional dependency
3. Learning curve

### OpenVINO Conversion Process

#### Method 1: ONNX → OpenVINO

Convert existing ONNX models to OpenVINO IR format.

```bash
# Install OpenVINO
pip install openvino-dev

# Convert decoder ONNX to OpenVINO
mo --input_model soprano_decoder.onnx \
   --output_dir openvino_models/decoder \
   --input_shape [1,512,100] \
   --data_type FP32

# Convert language model ONNX to OpenVINO
mo --input_model soprano_lm.onnx \
   --output_dir openvino_models/language_model \
   --data_type FP32
```

#### Method 2: PyTorch → OpenVINO Direct

Convert PyTorch models directly to OpenVINO IR.

```python
import openvino as ov
from soprano.vocos.decoder import SopranoDecoder
import torch

# Load decoder
decoder = SopranoDecoder()
decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
decoder.eval()

# Convert to OpenVINO
dummy_input = torch.randn(1, 512, 100)
ov_model = ov.convert_model(decoder, example_input=dummy_input)

# Save
ov.save_model(ov_model, "soprano_decoder_openvino.xml")
```

#### Method 3: Optimum Intel

Use HuggingFace Optimum with Intel backend.

```python
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

# Export and optimize for OpenVINO
model = OVModelForCausalLM.from_pretrained(
    'ekwek/Soprano-80M',
    export=True,
    device='CPU'
)

# Quantize to INT8 for better performance
from optimum.intel import OVQuantizer

quantizer = OVQuantizer.from_pretrained(model)
quantizer.quantize(save_directory="soprano_lm_int8")
```

### OpenVINO Inference

```python
import openvino as ov
import numpy as np

# Load OpenVINO model
core = ov.Core()
model = core.read_model("soprano_decoder_openvino.xml")
compiled_model = core.compile_model(model, "CPU")

# Inference
input_data = np.random.randn(1, 512, 100).astype(np.float32)
results = compiled_model([input_data])
audio_output = results[0]
```

### Performance Optimization with OpenVINO

#### 1. INT8 Quantization

Reduce model size and increase speed with minimal quality loss.

```python
from openvino.tools import mo
from openvino.runtime import Core

# Post-training quantization
# Requires calibration dataset
```

**Expected Speedup**: 2-4x on CPU
**Quality Impact**: Minimal (typically <0.1% quality degradation)

#### 2. Dynamic Shapes Optimization

Optimize for common input shapes.

```bash
mo --input_model soprano_decoder.onnx \
   --output_dir openvino_models/decoder \
   --input_shape [1,512,-1] \
   --dynamic_shape [1,512,50..200]
```

#### 3. CPU Streams

Enable parallel inference for throughput.

```python
# Configure for maximum throughput
config = {"PERFORMANCE_HINT": "THROUGHPUT", "NUM_STREAMS": "4"}
compiled_model = core.compile_model(model, "CPU", config)
```

---

## Implementation Roadmap

### Phase 1: Decoder ONNX Export (Week 1-2)

**Goal**: Export Vocos decoder to ONNX and validate

**Tasks**:
1. ✅ Set up ONNX export environment
2. ✅ Export decoder to ONNX with dynamic shapes
3. ✅ Handle ISTFT operation compatibility
4. ✅ Validate numerical accuracy (PyTorch vs ONNX)
5. ✅ Create ONNX inference wrapper class
6. ✅ Performance benchmarking (CPU)
7. ✅ Integration testing with existing pipeline

**Deliverables**:
- `soprano_decoder.onnx` model file
- `soprano/backends/onnx_decoder.py` backend implementation
- Validation tests
- Performance benchmark results

**Success Criteria**:
- Max numerical difference < 1e-4
- CPU inference works correctly
- Audio quality matches PyTorch decoder

### Phase 2: Language Model ONNX Export (Week 3-4)

**Goal**: Export language model using HuggingFace Optimum

**Tasks**:
1. ✅ Set up Optimum ONNX runtime
2. ✅ Export LM using Optimum
3. ✅ Implement hidden state extraction
4. ✅ Create ONNX LM backend wrapper
5. ✅ Validate generation quality
6. ✅ Test sampling parameters
7. ✅ Integration with existing code

**Deliverables**:
- `soprano_lm.onnx` model file(s)
- `soprano/backends/onnx_lm.py` backend implementation
- Modified TTS class to support ONNX backend
- Validation tests

**Success Criteria**:
- Generated audio quality matches PyTorch
- All sampling parameters work correctly
- CPU inference functional

### Phase 3: OpenVINO Integration (Week 5-6)

**Goal**: Convert ONNX models to OpenVINO and optimize

**Tasks**:
1. ✅ Convert decoder ONNX → OpenVINO IR
2. ✅ Convert LM ONNX → OpenVINO IR
3. ✅ Implement OpenVINO backend wrappers
4. ✅ Test INT8 quantization
5. ✅ Performance optimization
6. ✅ Benchmark comparisons
7. ✅ Documentation

**Deliverables**:
- OpenVINO IR models (.xml/.bin)
- `soprano/backends/openvino_decoder.py`
- `soprano/backends/openvino_lm.py`
- Performance comparison report
- Deployment guide

**Success Criteria**:
- OpenVINO models functional
- Performance improvement vs ONNX Runtime
- Quality maintained with quantization

### Phase 4: Integration & Testing (Week 7)

**Goal**: Integrate all backends and comprehensive testing

**Tasks**:
1. ✅ Update backend selection logic
2. ✅ Add CPU device support in main TTS class
3. ✅ End-to-end testing
4. ✅ Performance benchmarking (all backends)
5. ✅ Documentation updates
6. ✅ Example scripts for ONNX/OpenVINO usage
7. ✅ CI/CD integration

**Deliverables**:
- Updated `soprano/tts.py` with CPU support
- Comprehensive test suite
- Benchmark report (GPU vs CPU, ONNX vs OpenVINO)
- User documentation
- Migration guide

**Success Criteria**:
- All backends work correctly
- Clear performance metrics available
- Easy to use for end users

### Phase 5: Optimization & Documentation (Week 8)

**Goal**: Final optimizations and complete documentation

**Tasks**:
1. ✅ Profile and optimize bottlenecks
2. ✅ Add caching for ONNX/OpenVINO models
3. ✅ Create deployment examples
4. ✅ Write comprehensive documentation
5. ✅ Tutorial notebooks
6. ✅ Docker containers for deployment
7. ✅ Release preparation

**Deliverables**:
- Optimized inference code
- Complete documentation
- Deployment guides
- Docker images
- Release notes

---

## Performance Considerations

### Expected Performance

Based on similar model conversions:

**GPU (Current - CUDA)**:
- Language Model: ~2000x realtime (LMDeploy)
- Decoder: ~5000x realtime
- Total: ~2000x realtime (limited by LM)

**CPU - ONNX Runtime (Predicted)**:
- Language Model: ~50-100x realtime (modern CPU, 16 threads)
- Decoder: ~200-500x realtime
- Total: ~50-100x realtime

**CPU - OpenVINO (Predicted)**:
- Language Model: ~100-200x realtime (Intel CPU with AVX-512)
- Decoder: ~500-1000x realtime
- Total: ~100-200x realtime

**CPU - OpenVINO INT8 (Predicted)**:
- Language Model: ~200-400x realtime
- Decoder: ~1000-2000x realtime
- Total: ~200-400x realtime

### Bottleneck Analysis

**Language Model** will be the primary bottleneck:
- Autoregressive generation is inherently sequential
- 80M parameters require significant computation
- KV-cache memory bandwidth important

**Decoder** is less critical:
- Feed-forward, highly parallelizable
- Smaller than language model
- Already fast on CPU

### Optimization Strategies

#### For Language Model:
1. **INT8 Quantization**: 2-4x speedup
2. **Operator Fusion**: 10-20% speedup
3. **Multi-threading**: Linear scaling up to physical cores
4. **Batch Processing**: Amortize overhead

#### For Decoder:
1. **SIMD Optimization**: Use AVX2/AVX-512
2. **Operator Fusion**: Combine convolutions
3. **Memory Layout**: Optimize for cache locality

#### System-Level:
1. **Pipeline Parallelism**: Overlap LM and decoder
2. **Asynchronous Processing**: Non-blocking inference
3. **Model Caching**: Keep models in memory
4. **Request Batching**: Process multiple requests together

### Hardware Recommendations

**Minimum**:
- CPU: 4+ cores, AVX2 support
- RAM: 4 GB
- OS: Linux, Windows, macOS

**Recommended**:
- CPU: 8+ cores, AVX-512 support (Intel)
- RAM: 8 GB
- OS: Linux (best ONNX/OpenVINO support)

**Optimal**:
- CPU: 16+ cores, Intel Xeon with AVX-512
- RAM: 16 GB
- OS: Linux with optimized BLAS libraries

---

## Challenges & Solutions

### Challenge 1: ISTFT Operation in ONNX

**Problem**: ISTFT may not have native ONNX operator

**Solutions**:
1. **Pre-compute ISTFT Matrix**: Convert to matrix multiplication
2. **Custom ONNX Operator**: Implement custom op
3. **Post-processing**: Remove ISTFT from ONNX, do in Python
4. **Alternative Architecture**: Replace ISTFT with convolutions

**Recommended**: Option 1 (matrix multiplication) or Option 3 (post-processing)

### Challenge 2: Hidden State Extraction

**Problem**: Need to extract hidden states from language model, not just tokens

**Solutions**:
1. **Model Modification**: Modify model to output hidden states
2. **Optimum Custom Logic**: Use Optimum with custom output processing
3. **Separate Export**: Export only the parts that generate hidden states
4. **Hook-based Extraction**: Use forward hooks during generation

**Recommended**: Option 2 (Optimum) or Option 1 (model modification)

### Challenge 3: Dynamic Sequence Lengths

**Problem**: Input text length varies, causing variable hidden state sequences

**Solutions**:
1. **Dynamic Axes**: Use ONNX dynamic axes (may be slower)
2. **Fixed Shapes**: Pad to maximum length
3. **Multiple Models**: Different models for different length ranges
4. **Chunking**: Split long inputs into fixed chunks

**Recommended**: Option 1 (dynamic axes) with optimization for common shapes

### Challenge 4: Sampling Parameters

**Problem**: Temperature, top_p, repetition_penalty need to be supported

**Solutions**:
1. **ONNX Runtime Extensions**: Use custom sampling ops
2. **Python Post-processing**: Sample in Python, not ONNX
3. **Optimum Integration**: Leverage Optimum's built-in sampling

**Recommended**: Option 3 (Optimum handles this)

### Challenge 5: Performance Gap

**Problem**: CPU will be slower than GPU

**Solutions**:
1. **Quantization**: INT8 for 2-4x speedup
2. **OpenVINO**: Use for Intel CPUs
3. **Batching**: Process multiple requests together
4. **Caching**: Cache models and compiled graphs
5. **Hardware**: Recommend appropriate CPU specs

**Recommended**: Combine all approaches

### Challenge 6: Streaming Support

**Problem**: Streaming inference more complex with ONNX

**Solutions**:
1. **Python-side Streaming**: Implement streaming logic in Python
2. **Multiple Inference Calls**: Call ONNX model iteratively
3. **Stateful Models**: Use ONNX support for stateful models (experimental)

**Recommended**: Option 1 or 2 (Python-side streaming)

---

## Testing & Validation

### Validation Strategy

#### 1. Numerical Accuracy Tests

Compare outputs between PyTorch and ONNX/OpenVINO:

```python
def test_decoder_accuracy():
    # Load both models
    pytorch_decoder = load_pytorch_decoder()
    onnx_decoder = load_onnx_decoder()
    
    # Generate test inputs
    test_inputs = generate_test_hidden_states(num_samples=100)
    
    # Compare outputs
    for input_data in test_inputs:
        pytorch_out = pytorch_decoder(input_data)
        onnx_out = onnx_decoder(input_data)
        
        # Check numerical difference
        max_diff = torch.max(torch.abs(pytorch_out - onnx_out))
        assert max_diff < 1e-4, f"Numerical difference too large: {max_diff}"
        
        # Check audio quality metrics
        snr = compute_snr(pytorch_out, onnx_out)
        assert snr > 40, f"SNR too low: {snr} dB"
```

#### 2. Perceptual Quality Tests

Use audio quality metrics:

```python
def test_perceptual_quality():
    # Generate same text with PyTorch and ONNX
    text = "The quick brown fox jumps over the lazy dog."
    
    pytorch_audio = pytorch_tts.infer(text)
    onnx_audio = onnx_tts.infer(text)
    
    # Compute PESQ (Perceptual Evaluation of Speech Quality)
    pesq_score = pesq(pytorch_audio, onnx_audio, sr=32000)
    assert pesq_score > 4.0, f"PESQ score too low: {pesq_score}"
    
    # Compute STOI (Short-Time Objective Intelligibility)
    stoi_score = stoi(pytorch_audio, onnx_audio, sr=32000)
    assert stoi_score > 0.95, f"STOI score too low: {stoi_score}"
```

#### 3. Performance Benchmarks

```python
def benchmark_inference():
    backends = ['pytorch-cuda', 'onnx-cpu', 'openvino-cpu', 'openvino-int8']
    text_lengths = [50, 100, 200, 500]  # characters
    
    results = {}
    for backend in backends:
        for length in text_lengths:
            text = generate_text(length)
            model = load_model(backend)
            
            # Warmup
            for _ in range(5):
                model.infer(text)
            
            # Benchmark
            times = []
            for _ in range(20):
                start = time.time()
                audio = model.infer(text)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            audio_duration = len(audio) / 32000
            rtf = audio_duration / avg_time
            
            results[(backend, length)] = {
                'time': avg_time,
                'rtf': rtf
            }
    
    return results
```

#### 4. Integration Tests

```python
def test_end_to_end():
    # Test full pipeline with various inputs
    test_cases = [
        "Hello world!",
        "This is a longer sentence with multiple words.",
        "Numbers: 123, 456. URLs: https://example.com",
        "Multiple sentences. Each one independent. Testing stability.",
    ]
    
    for text in test_cases:
        try:
            audio = onnx_tts.infer(text)
            assert audio is not None
            assert len(audio) > 0
            assert audio.dtype == np.float32
            assert -1.0 <= audio.max() <= 1.0
            assert -1.0 <= audio.min() <= 1.0
        except Exception as e:
            pytest.fail(f"Failed on text: {text}\nError: {e}")
```

### Test Coverage

- ✅ Decoder export and inference
- ✅ Language model export and inference
- ✅ End-to-end pipeline
- ✅ Sampling parameters
- ✅ Various text lengths
- ✅ Batch inference
- ✅ Streaming inference
- ✅ Error handling
- ✅ Memory usage
- ✅ Performance benchmarks

---

## Deployment Guide

### Deployment Scenarios

#### Scenario 1: High-Performance CPU Server

**Hardware**: Intel Xeon with AVX-512, 16+ cores

**Recommended Setup**:
- OpenVINO with INT8 quantization
- Multi-threading enabled
- Request batching

**Expected Performance**: ~200-400x realtime

**Use Case**: Production API serving multiple users

#### Scenario 2: Standard CPU Server

**Hardware**: Modern CPU, 8 cores

**Recommended Setup**:
- ONNX Runtime or OpenVINO FP32
- Multi-threading

**Expected Performance**: ~50-100x realtime

**Use Case**: Internal tools, moderate load

#### Scenario 3: Edge Device / Embedded

**Hardware**: ARM or low-power CPU

**Recommended Setup**:
- ONNX Runtime (ARM support)
- Single-threaded or limited threading
- Consider model pruning

**Expected Performance**: ~10-30x realtime

**Use Case**: On-device TTS for privacy

#### Scenario 4: Docker Container

**Hardware**: Variable (cloud deployment)

**Recommended Setup**:
- ONNX Runtime for portability
- Environment-based configuration
- Auto-scaling

**Use Case**: Cloud deployment (AWS, GCP, Azure)

### Installation Guide

#### ONNX Runtime Setup

```bash
# Install Soprano with ONNX support
pip install soprano-tts
pip install onnxruntime  # CPU
# or
pip install onnxruntime-gpu  # GPU (optional)

# Export models
python -m soprano.export.decoder --output soprano_decoder.onnx
python -m soprano.export.language_model --output soprano_lm.onnx
```

#### OpenVINO Setup

```bash
# Install OpenVINO
pip install openvino-dev openvino

# Convert models
mo --input_model soprano_decoder.onnx --output_dir openvino_models/decoder
mo --input_model soprano_lm.onnx --output_dir openvino_models/lm

# Use OpenVINO backend
export USE_OPENVINO=true
python app.py
```

### Usage Examples

#### Using ONNX Backend

```python
from soprano import SopranoTTS

# Initialize with ONNX backend
model = SopranoTTS(
    backend='onnx',  # New backend
    device='cpu',
    onnx_decoder_path='soprano_decoder.onnx',
    onnx_lm_path='soprano_lm.onnx'
)

# Use normally
audio = model.infer("Hello from ONNX backend!")
```

#### Using OpenVINO Backend

```python
from soprano import SopranoTTS

# Initialize with OpenVINO
model = SopranoTTS(
    backend='openvino',
    device='cpu',
    openvino_decoder_path='openvino_models/decoder/soprano_decoder.xml',
    openvino_lm_path='openvino_models/lm/soprano_lm.xml',
    num_threads=8  # CPU threads
)

audio = model.infer("Hello from OpenVINO!")
```

#### API Server with CPU Backend

```bash
# Start API server with ONNX backend
export BACKEND=onnx
export MODEL_DEVICE=cpu
export ONNX_DECODER_PATH=./soprano_decoder.onnx
export ONNX_LM_PATH=./soprano_lm.onnx

soprano-server --device cpu --backend onnx
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install soprano-tts onnxruntime

# Copy ONNX models
COPY soprano_decoder.onnx /app/models/
COPY soprano_lm.onnx /app/models/

# Set environment
ENV BACKEND=onnx
ENV MODEL_DEVICE=cpu
ENV ONNX_DECODER_PATH=/app/models/soprano_decoder.onnx
ENV ONNX_LM_PATH=/app/models/soprano_lm.onnx

# Run server
CMD ["soprano-server", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Appendix

### A. ONNX Opset Compatibility

**Recommended Opset**: 17 (latest stable as of 2024)

**Key Operations**:
- Linear interpolation: Opset 11+
- LayerNormalization: Opset 17 (improved)
- Attention mechanisms: Opset 14+

### B. Model Size Comparison

| Model | Format | Precision | Size | Notes |
|-------|--------|-----------|------|-------|
| Language Model | PyTorch | FP32 | ~320 MB | Base model |
| Language Model | ONNX | FP32 | ~320 MB | Same size |
| Language Model | OpenVINO | FP32 | ~320 MB | Similar size |
| Language Model | OpenVINO | INT8 | ~80 MB | 4x smaller, quantized |
| Decoder | PyTorch | FP32 | ~50 MB | Vocos decoder |
| Decoder | ONNX | FP32 | ~50 MB | Same size |
| Decoder | OpenVINO | FP32 | ~50 MB | Similar size |
| Decoder | OpenVINO | INT8 | ~12 MB | 4x smaller |

### C. CPU Requirements

**Minimum**:
- Architecture: x86_64 or ARM64
- Cores: 4
- RAM: 4 GB
- Instructions: SSE4.2

**Recommended**:
- Architecture: x86_64 (Intel)
- Cores: 8-16
- RAM: 8 GB
- Instructions: AVX2

**Optimal**:
- Architecture: x86_64 (Intel Xeon)
- Cores: 16+
- RAM: 16 GB
- Instructions: AVX-512

### D. Performance Tuning Parameters

```python
# ONNX Runtime
session_options = onnxruntime.SessionOptions()
session_options.intra_op_num_threads = 8  # Parallel ops
session_options.inter_op_num_threads = 2  # Parallel graphs
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

# OpenVINO
config = {
    "PERFORMANCE_HINT": "THROUGHPUT",  # or "LATENCY"
    "NUM_STREAMS": "4",  # Parallel inference streams
    "INFERENCE_NUM_THREADS": "8",
    "ENABLE_CPU_PINNING": "YES",
}
```

### E. Troubleshooting Common Issues

**Issue**: "Unsupported ONNX operator"
- **Solution**: Update ONNX Runtime, or use different opset version

**Issue**: "Numerical mismatch between PyTorch and ONNX"
- **Solution**: Check for non-deterministic operations, set tolerances appropriately

**Issue**: "OpenVINO model very slow"
- **Solution**: Check CPU instructions (AVX-512), enable optimizations, use INT8

**Issue**: "Out of memory"
- **Solution**: Reduce batch size, use quantization, clear cache

### F. References

1. **ONNX Documentation**: https://onnx.ai/
2. **ONNX Runtime**: https://onnxruntime.ai/
3. **OpenVINO**: https://docs.openvino.ai/
4. **HuggingFace Optimum**: https://huggingface.co/docs/optimum/
5. **PyTorch ONNX Export**: https://pytorch.org/docs/stable/onnx.html

### G. Code Repository Structure (After Implementation)

```
soprano/
├── soprano/
│   ├── backends/
│   │   ├── base.py
│   │   ├── lmdeploy.py
│   │   ├── transformers.py
│   │   ├── onnx_decoder.py      # New
│   │   ├── onnx_lm.py           # New
│   │   ├── openvino_decoder.py  # New
│   │   └── openvino_lm.py       # New
│   ├── export/
│   │   ├── __init__.py          # New
│   │   ├── decoder_exporter.py  # New
│   │   └── lm_exporter.py       # New
│   └── ...
├── models/                       # New
│   ├── onnx/
│   │   ├── soprano_decoder.onnx
│   │   └── soprano_lm.onnx
│   └── openvino/
│       ├── decoder/
│       └── lm/
├── tests/
│   ├── test_onnx_export.py      # New
│   ├── test_openvino_export.py  # New
│   └── test_cpu_inference.py    # New
└── ...
```

---

## Conclusion

This master plan provides a comprehensive roadmap for transforming Soprano TTS to support ONNX export and CPU inference with optional OpenVINO optimization. The phased approach ensures incremental progress with validation at each step.

**Key Takeaways**:

1. **Dual Export**: Export language model and decoder separately for flexibility
2. **HuggingFace Optimum**: Use for language model export (handles complexity)
3. **OpenVINO**: Use for Intel CPUs to maximize performance
4. **INT8 Quantization**: Enable for 2-4x speedup with minimal quality loss
5. **Incremental Approach**: Start with decoder (easier), then language model
6. **Thorough Testing**: Validate numerical accuracy and perceptual quality
7. **Performance Targets**: Aim for >50x realtime on modern CPUs

**Expected Outcomes**:

- ✅ CPU-only inference capability
- ✅ Broader deployment options
- ✅ Lower infrastructure costs
- ✅ Maintained audio quality
- ✅ Acceptable performance (50-400x realtime depending on setup)

**Next Steps**:

1. Review and approve this plan
2. Set up development environment
3. Begin Phase 1: Decoder ONNX export
4. Regular progress reviews and adjustments

For questions or clarifications, please refer to the component documentation in `PROJECT_COMPONENTS.md` or open an issue in the repository.
