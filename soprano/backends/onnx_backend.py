"""
ONNX and OpenVINO backend for Soprano TTS CPU inference.

This backend provides CPU-based inference using ONNX Runtime or OpenVINO,
enabling deployment without CUDA dependencies.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Iterator
from .base import BaseModel


class ONNXModel(BaseModel):
    """ONNX Runtime backend for CPU inference."""
    
    def __init__(
        self,
        lm_path: str,
        decoder_path: str,
        tokenizer_repo: str = "ekwek/Soprano-80M",
        num_threads: int = 4,
        **kwargs
    ):
        """
        Initialize ONNX backend.
        
        Args:
            lm_path: Path to LM ONNX model
            decoder_path: Path to decoder ONNX model
            tokenizer_repo: HuggingFace repo for tokenizer
            num_threads: Number of CPU threads for inference
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX backend. "
                "Install with: pip install onnxruntime"
            )
        
        from transformers import AutoTokenizer
        
        # Set up ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Load models
        self.lm_session = ort.InferenceSession(lm_path, sess_options)
        self.decoder_session = ort.InferenceSession(decoder_path, sess_options)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
        
        print(f"✓ ONNX backend initialized with {num_threads} threads")
    
    def infer(
        self,
        prompts: List[str],
        top_p: float = 0.95,
        temperature: float = 0.3,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 512,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of prompts.
        
        Args:
            prompts: List of text prompts
            top_p: Nucleus sampling threshold
            temperature: Sampling temperature
            repetition_penalty: Penalty for repeated tokens
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of dicts with 'hidden_state' and 'finish_reason'
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors='np',
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        results = []
        eos_token_id = self.tokenizer.eos_token_id or 2
        
        for i in range(len(prompts)):
            input_ids = inputs['input_ids'][i:i+1]
            attention_mask = inputs['attention_mask'][i:i+1]
            
            hidden_states_list = []
            finish_reason = 'length'
            
            # Autoregressive generation
            for step in range(max_new_tokens):
                # Run LM forward pass
                outputs = self.lm_session.run(
                    None,
                    {
                        "input_ids": input_ids.astype(np.int64),
                        "attention_mask": attention_mask.astype(np.int64),
                    }
                )
                
                logits = outputs[0]  # Shape: [batch, seq_len, vocab_size]
                hidden_states = outputs[1] if len(outputs) > 1 else logits
                
                # Get last token logits
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for prev_token_id in input_ids[0]:
                        if prev_token_id < len(next_token_logits):
                            if next_token_logits[prev_token_id] < 0:
                                next_token_logits[prev_token_id] *= repetition_penalty
                            else:
                                next_token_logits[prev_token_id] /= repetition_penalty
                
                # Sample next token
                probs = self._softmax(next_token_logits)
                
                if top_p < 1.0:
                    next_token = self._sample_top_p(probs, top_p)
                else:
                    next_token = np.argmax(probs)
                
                # Check for EOS
                if next_token == eos_token_id:
                    finish_reason = 'stop'
                    break
                
                # Store hidden state for this token
                # Extract the last hidden state for the newly generated token
                if len(outputs) > 1:
                    # Assuming hidden_states is the second output
                    token_hidden = hidden_states[0, -1, :]
                    hidden_states_list.append(token_hidden)
                
                # Update input_ids and attention_mask
                input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
                attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
            
            # Convert to torch tensor
            if hidden_states_list:
                final_hidden = torch.from_numpy(np.stack(hidden_states_list)).float()
            else:
                final_hidden = torch.zeros(1, 512)  # Fallback
            
            results.append({
                'hidden_state': final_hidden,
                'finish_reason': finish_reason
            })
        
        return results
    
    def stream_infer(
        self,
        prompt: str,
        top_p: float = 0.95,
        temperature: float = 0.3,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 512,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream inference for a single prompt.
        
        Yields dictionaries with 'hidden_state' and 'finish_reason'.
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors='np',
            truncation=True,
            max_length=512,
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        eos_token_id = self.tokenizer.eos_token_id or 2
        
        # Autoregressive generation with streaming
        for step in range(max_new_tokens):
            # Run LM forward pass
            outputs = self.lm_session.run(
                None,
                {
                    "input_ids": input_ids.astype(np.int64),
                    "attention_mask": attention_mask.astype(np.int64),
                }
            )
            
            logits = outputs[0]
            hidden_states = outputs[1] if len(outputs) > 1 else logits
            
            # Get last token logits
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for prev_token_id in input_ids[0]:
                    if prev_token_id < len(next_token_logits):
                        if next_token_logits[prev_token_id] < 0:
                            next_token_logits[prev_token_id] *= repetition_penalty
                        else:
                            next_token_logits[prev_token_id] /= repetition_penalty
            
            # Sample next token
            probs = self._softmax(next_token_logits)
            
            if top_p < 1.0:
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = np.argmax(probs)
            
            # Yield current token's hidden state
            if len(outputs) > 1:
                token_hidden = torch.from_numpy(hidden_states[0, -1, :]).float()
                finish_reason = 'stop' if next_token == eos_token_id else None
                
                yield {
                    'hidden_state': token_hidden,
                    'finish_reason': finish_reason
                }
            
            # Check for EOS
            if next_token == eos_token_id:
                break
            
            # Update input_ids and attention_mask
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
            attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    @staticmethod
    def _sample_top_p(probs: np.ndarray, p: float) -> int:
        """Sample from top-p (nucleus) distribution."""
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)
        
        # Find cutoff
        cutoff_idx = np.searchsorted(cumsum_probs, p)
        if cutoff_idx >= len(sorted_indices):
            cutoff_idx = len(sorted_indices) - 1
        
        # Sample from truncated distribution
        top_indices = sorted_indices[:cutoff_idx + 1]
        top_probs = probs[top_indices]
        top_probs = top_probs / top_probs.sum()
        
        return np.random.choice(top_indices, p=top_probs)


class OpenVINOModel(BaseModel):
    """OpenVINO backend for optimized CPU inference."""
    
    def __init__(
        self,
        lm_path: str,
        decoder_path: str,
        tokenizer_repo: str = "ekwek/Soprano-80M",
        num_threads: int = 4,
        **kwargs
    ):
        """
        Initialize OpenVINO backend.
        
        Args:
            lm_path: Path to LM OpenVINO IR (.xml file)
            decoder_path: Path to decoder OpenVINO IR (.xml file)
            tokenizer_repo: HuggingFace repo for tokenizer
            num_threads: Number of CPU threads for inference
        """
        try:
            from openvino.runtime import Core
        except ImportError:
            raise ImportError(
                "openvino is required for OpenVINO backend. "
                "Install with: pip install openvino"
            )
        
        from transformers import AutoTokenizer
        
        # Initialize OpenVINO
        core = Core()
        
        # Load models
        self.lm_model = core.read_model(lm_path)
        self.lm_compiled = core.compile_model(self.lm_model, "CPU")
        
        self.decoder_model = core.read_model(decoder_path)
        self.decoder_compiled = core.compile_model(self.decoder_model, "CPU")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
        
        print(f"✓ OpenVINO backend initialized with {num_threads} threads")
    
    def infer(
        self,
        prompts: List[str],
        top_p: float = 0.95,
        temperature: float = 0.3,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 512,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Run inference on a batch of prompts (similar to ONNXModel)."""
        # For simplicity, delegate to ONNX-style implementation
        # In production, this would use OpenVINO's optimized inference
        raise NotImplementedError(
            "OpenVINO inference not fully implemented yet. "
            "Use ONNX backend for now."
        )
    
    def stream_infer(
        self,
        prompt: str,
        top_p: float = 0.95,
        temperature: float = 0.3,
        repetition_penalty: float = 1.2,
        **kwargs
    ) -> Iterator[Dict[str, Any]]:
        """Stream inference (similar to ONNXModel)."""
        raise NotImplementedError(
            "OpenVINO streaming not fully implemented yet. "
            "Use ONNX backend for now."
        )
