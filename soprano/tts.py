from .vocos.decoder import SopranoDecoder
import torch
import numpy as np
import re
from unidecode import unidecode
from scipy.io import wavfile
from huggingface_hub import hf_hub_download
import os
import time


# Constants for Soprano TTS
SAMPLE_RATE = 32000  # Audio sample rate in Hz
HIDDEN_DIM = 512  # Hidden dimension for transformer models


class SopranoTTS:
    def __init__(self,
            backend='auto',
            device='cuda',
            cache_size_mb=10,
            decoder_batch_size=1,
            lm_path=None,
            decoder_path=None,
            num_threads=4):
        """
        Initialize Soprano TTS engine.
        
        Args:
            backend: Backend to use ('auto', 'lmdeploy', 'transformers', 'onnx_cpu', 'openvino_cpu')
            device: Device for PyTorch backends ('cuda' or 'cpu')
            cache_size_mb: Cache size for LMDeploy backend
            decoder_batch_size: Batch size for decoder
            lm_path: Path to LM model (required for ONNX/OpenVINO backends)
            decoder_path: Path to decoder model (required for ONNX/OpenVINO backends)
            num_threads: Number of CPU threads for ONNX/OpenVINO backends
        """
        self.decoder_batch_size = decoder_batch_size
        self.RECEPTIVE_FIELD = 4  # Decoder receptive field
        self.TOKEN_SIZE = 2048  # Number of samples per audio token
        
        # Check if using CPU backends (ONNX/OpenVINO)
        if backend in ['onnx_cpu', 'openvino_cpu']:
            if lm_path is None or decoder_path is None:
                raise ValueError(
                    "For ONNX/OpenVINO backends, both lm_path and decoder_path must be provided"
                )
            self._init_cpu_backend(backend, lm_path, decoder_path, num_threads)
            self.use_cpu_backend = True
            self.backend = backend
            return
        
        # Legacy PyTorch backends
        RECOGNIZED_DEVICES = ['cuda']
        RECOGNIZED_BACKENDS = ['auto', 'lmdeploy', 'transformers']
        assert device in RECOGNIZED_DEVICES, f"unrecognized device {device}, device must be in {RECOGNIZED_DEVICES}"
        if backend == 'auto':
            if device == 'cpu':
                backend = 'transformers'
            else:
                try:
                    import lmdeploy
                    backend = 'lmdeploy'
                except ImportError:
                    backend='transformers'
            print(f"Using backend {backend}.")
        assert backend in RECOGNIZED_BACKENDS, f"unrecognized backend {backend}, backend must be in {RECOGNIZED_BACKENDS}"

        if backend == 'lmdeploy':
            from .backends.lmdeploy import LMDeployModel
            self.pipeline = LMDeployModel(device=device, cache_size_mb=cache_size_mb)
        elif backend == 'transformers':
            from .backends.transformers import TransformersModel
            self.pipeline = TransformersModel(device=device)

        self.decoder = SopranoDecoder().cuda()
        decoder_path_file = hf_hub_download(repo_id='ekwek/Soprano-80M', filename='decoder.pth')
        self.decoder.load_state_dict(torch.load(decoder_path_file))
        self.use_cpu_backend = False
        self.backend = backend

        self.infer("Hello world!") # warmup
    
    def _init_cpu_backend(self, backend, lm_path, decoder_path, num_threads):
        """Initialize ONNX or OpenVINO backend for CPU inference."""
        if backend == 'onnx_cpu':
            try:
                import onnxruntime as ort
            except ImportError:
                raise ImportError(
                    "onnxruntime is required for ONNX backend. "
                    "Install with: pip install onnxruntime"
                )
            
            # Set up ONNX Runtime
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = num_threads
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.lm_session = ort.InferenceSession(lm_path, sess_options)
            self.decoder_session = ort.InferenceSession(decoder_path, sess_options)
            
            print(f"✓ Loaded ONNX models with {num_threads} CPU threads")
            
        elif backend == 'openvino_cpu':
            try:
                from openvino.runtime import Core
            except ImportError:
                raise ImportError(
                    "openvino is required for OpenVINO backend. "
                    "Install with: pip install openvino"
                )
            
            core = Core()
            self.lm_model = core.read_model(lm_path)
            self.lm_compiled = core.compile_model(self.lm_model, "CPU")
            
            self.decoder_model = core.read_model(decoder_path)
            self.decoder_compiled = core.compile_model(self.decoder_model, "CPU")
            
            print(f"✓ Loaded OpenVINO models on CPU")
        
        # Load tokenizer for CPU backends
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')

    def _preprocess_text(self, texts, min_length=30):
        '''
        adds prompt format and sentence/part index
        Enforces a minimum sentence length by merging short sentences.
        '''
        res = []
        for text_idx, text in enumerate(texts):
            text = text.strip()
            sentences = re.split(r"(?<=[.!?])\s+", text)
            processed = []
            for sentence_idx, sentence in enumerate(sentences):
                old_len = len(sentence)
                new_sentence = re.sub(r"[^A-Za-z !\$%&'*+,-./0123456789<>?_]", "", sentence)
                new_sentence = re.sub(r"[<>/_+]", "", new_sentence)
                new_sentence = re.sub(r"\.\.[^\.]", ".", new_sentence)
                new_sentence = re.sub(r"\s+", " ", new_sentence)
                new_len = len(new_sentence)
                if old_len != new_len:
                    print(f"Warning: unsupported characters found in sentence: {sentence}\n\tThese characters have been removed.")
                new_sentence = unidecode(new_sentence.strip())
                processed.append({
                    "text": new_sentence,
                    "text_idx": text_idx,
                })

            if min_length > 0 and len(processed) > 1:
                merged = []
                i = 0
                while i < len(processed):
                    cur = processed[i]
                    if len(cur["text"]) < min_length:
                        if merged: merged[-1]["text"] = (merged[-1]["text"] + " " + cur["text"]).strip()
                        else:
                            if i + 1 < len(processed): processed[i + 1]["text"] = (cur["text"] + " " + processed[i + 1]["text"]).strip()
                            else: merged.append(cur)
                    else: merged.append(cur)
                    i += 1
                processed = merged
            sentence_idxes = {}
            for item in processed:
                if item['text_idx'] not in sentence_idxes: sentence_idxes[item['text_idx']] = 0
                res.append((f'[STOP][TEXT]{item["text"]}[START]', item["text_idx"], sentence_idxes[item['text_idx']]))
                sentence_idxes[item['text_idx']] += 1
        return res

    def infer(self,
            text,
            out_path=None,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        if self.use_cpu_backend:
            raise NotImplementedError(
                "infer() is not available with ONNX/OpenVINO backends. "
                "Use synthesize() instead."
            )
        
        results = self.infer_batch([text],
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            out_dir=None)[0]
        if out_path:
            wavfile.write(out_path, SAMPLE_RATE, results.cpu().numpy())
        return results

    def infer_batch(self,
            texts,
            out_dir=None,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        if self.use_cpu_backend:
            raise NotImplementedError(
                "infer_batch() is not available with ONNX/OpenVINO backends. "
                "Use synthesize() instead."
            )
        
        sentence_data = self._preprocess_text(texts)
        prompts = list(map(lambda x: x[0], sentence_data))
        responses = self.pipeline.infer(prompts,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty)
        hidden_states = []
        for i, response in enumerate(responses):
            if response['finish_reason'] != 'stop':
                print(f"Warning: some sentences did not complete generation, likely due to hallucination.")
            hidden_state = response['hidden_state']
            hidden_states.append(hidden_state)
        combined = list(zip(hidden_states, sentence_data))
        combined.sort(key=lambda x: -x[0].size(0))
        hidden_states, sentence_data = zip(*combined)

        num_texts = len(texts)
        audio_concat = [[] for _ in range(num_texts)]
        for sentence in sentence_data:
            audio_concat[sentence[1]].append(None)
        for idx in range(0, len(hidden_states), self.decoder_batch_size):
            batch_hidden_states = []
            lengths = list(map(lambda x: x.size(0), hidden_states[idx:idx+self.decoder_batch_size]))
            N = len(lengths)
            for i in range(N):
                batch_hidden_states.append(torch.cat([
                    torch.zeros((1, HIDDEN_DIM, lengths[0]-lengths[i]), device='cuda'),
                    hidden_states[idx+i].unsqueeze(0).transpose(1,2).cuda().to(torch.float32),
                ], dim=2))
            batch_hidden_states = torch.cat(batch_hidden_states)
            with torch.no_grad():
                audio = self.decoder(batch_hidden_states)
            
            for i in range(N):
                text_id = sentence_data[idx+i][1]
                sentence_id = sentence_data[idx+i][2]
                audio_concat[text_id][sentence_id] = audio[i].squeeze()[-(lengths[i]*self.TOKEN_SIZE-self.TOKEN_SIZE):]
        audio_concat = [torch.cat(x).cpu() for x in audio_concat]
        
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            for i in range(len(audio_concat)):
                wavfile.write(f"{out_dir}/{i}.wav", SAMPLE_RATE, audio_concat[i].cpu().numpy())
        return audio_concat

    def infer_stream(self,
            text,
            chunk_size=1,
            top_p=0.95,
            temperature=0.3,
            repetition_penalty=1.2):
        if self.use_cpu_backend:
            raise NotImplementedError(
                "infer_stream() is not available with ONNX/OpenVINO backends."
            )
        
        start_time = time.time()
        sentence_data = self._preprocess_text([text])

        first_chunk = True
        for sentence, _, _ in sentence_data:
            responses = self.pipeline.stream_infer(sentence,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty)
            hidden_states_buffer = []
            chunk_counter = chunk_size
            for token in responses:
                finished = token['finish_reason'] is not None
                if not finished: hidden_states_buffer.append(token['hidden_state'][-1])
                hidden_states_buffer = hidden_states_buffer[-(2*self.RECEPTIVE_FIELD+chunk_size):]
                if finished or len(hidden_states_buffer) >= self.RECEPTIVE_FIELD + chunk_size:
                    if finished or chunk_counter == chunk_size:
                        batch_hidden_states = torch.stack(hidden_states_buffer)
                        inp = batch_hidden_states.unsqueeze(0).transpose(1, 2).cuda().to(torch.float32)
                        with torch.no_grad():
                            audio = self.decoder(inp)[0]
                        if finished:
                            audio_chunk = audio[-((self.RECEPTIVE_FIELD+chunk_counter-1)*self.TOKEN_SIZE-self.TOKEN_SIZE):]
                        else:
                            audio_chunk = audio[-((self.RECEPTIVE_FIELD+chunk_size)*self.TOKEN_SIZE-self.TOKEN_SIZE):-(self.RECEPTIVE_FIELD*self.TOKEN_SIZE-self.TOKEN_SIZE)]
                        chunk_counter = 0
                        if first_chunk:
                            print(f"Streaming latency: {1000*(time.time()-start_time):.2f} ms")
                            first_chunk = False
                        yield audio_chunk.cpu()
                    chunk_counter += 1
    
    def synthesize(
        self,
        text,
        max_new_tokens=100,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
        seed=None,
        **kwargs
    ):
        """
        Synthesize speech from text (ONNX/OpenVINO interface).
        
        Args:
            text: Input text to synthesize
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeated tokens
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'audio' (numpy array) and 'sample_rate' (int)
        """
        if not self.use_cpu_backend:
            raise NotImplementedError(
                "synthesize() is only available with ONNX/OpenVINO backends. "
                "Use infer() or infer_batch() for PyTorch backends."
            )
        
        if seed is not None:
            np.random.seed(seed)
        
        # Preprocess text
        processed_text = self._preprocess_text_simple(text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors='np',
            truncation=True,
            max_length=512,
        )
        
        # Generate hidden states
        hidden_states = self._generate_hidden_states(
            inputs['input_ids'],
            inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        # Decode to audio
        audio = self._decode_audio(hidden_states)
        
        return {
            'audio': audio,
            'sample_rate': SAMPLE_RATE,
        }
    
    def _preprocess_text_simple(self, text):
        """Simple text preprocessing for CPU backends."""
        # Remove unsupported characters
        text = re.sub(r"[^A-Za-z !\$%&'*+,-./0123456789<>?_]", "", text)
        text = re.sub(r"[<>/_+]", "", text)
        text = re.sub(r"\.\.[^\.]", ".", text)
        text = re.sub(r"\s+", " ", text)
        text = unidecode(text.strip())
        
        # Add prompt format
        return f'[STOP][TEXT]{text}[START]'
    
    def _generate_hidden_states(
        self,
        input_ids,
        attention_mask,
        max_new_tokens,
        temperature,
        top_p,
        repetition_penalty,
    ):
        """Generate hidden states using LM (ONNX/OpenVINO backend)."""
        eos_token_id = self.tokenizer.eos_token_id or 2
        hidden_states_list = []
        
        for step in range(max_new_tokens):
            # Run LM inference
            if self.backend == 'onnx_cpu':
                outputs = self.lm_session.run(
                    None,
                    {
                        "input_ids": input_ids.astype(np.int64),
                        "attention_mask": attention_mask.astype(np.int64),
                    }
                )
            elif self.backend == 'openvino_cpu':
                result = self.lm_compiled([input_ids.astype(np.int64), attention_mask.astype(np.int64)])
                outputs = [result[key] for key in result.keys()]
            
            logits = outputs[0]  # [batch, seq_len, vocab_size]
            
            # Get next token logits
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            if temperature > 0 and temperature != 1.0:
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
                break
            
            # Extract hidden states from the model output
            # The LM export script (lm_step_export.py) uses a wrapper model to ensure
            # hidden states are properly exported as the second output. The wrapper calls
            # the base model with output_hidden_states=True and returns both logits and
            # the last hidden layer.
            if len(outputs) > 1:
                # Second output contains hidden states from the last layer
                # Shape: [batch, seq_len, hidden_dim]
                token_hidden = outputs[1][0, -1, :]
                if token_hidden.shape[0] == HIDDEN_DIM:  # Verify expected hidden dim
                    hidden_states_list.append(token_hidden)
                else:
                    print(f"Warning: Unexpected hidden state dimension {token_hidden.shape[0]}, expected {HIDDEN_DIM}")
            
            # Update sequences for next iteration
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
            attention_mask = np.concatenate([attention_mask, [[1]]], axis=1)
        
        if hidden_states_list:
            return np.stack(hidden_states_list)
        else:
            # If no hidden states were extracted, raise an error instead of silently failing
            raise RuntimeError(
                "Failed to extract hidden states from the language model. "
                "This likely means the LM export script needs to be updated to properly "
                "export hidden states. Please ensure lm_step_export.py is configured to "
                "output the last hidden layer states as the second output."
            )
    
    def _decode_audio(self, hidden_states):
        """Decode hidden states to audio waveform."""
        # Prepare input for decoder: [batch=1, channels=HIDDEN_DIM, seq_len]
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.T[np.newaxis, :, :]  # [1, HIDDEN_DIM, seq_len]
        
        # Run decoder inference
        if self.backend == 'onnx_cpu':
            audio = self.decoder_session.run(
                None,
                {"hidden_states": hidden_states.astype(np.float32)}
            )[0]
        elif self.backend == 'openvino_cpu':
            result = self.decoder_compiled([hidden_states.astype(np.float32)])
            audio = result[list(result.keys())[0]]
        
        # Return audio as 1D array
        if len(audio.shape) > 1:
            audio = audio[0]  # Remove batch dimension
        
        return audio.astype(np.float32)
    
    @staticmethod
    def _softmax(x):
        """Compute softmax with numerical stability."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    @staticmethod
    def _sample_top_p(probs, p):
        """Sample from top-p (nucleus) distribution."""
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumsum_probs = np.cumsum(sorted_probs)
        
        cutoff_idx = np.searchsorted(cumsum_probs, p)
        if cutoff_idx >= len(sorted_indices):
            cutoff_idx = len(sorted_indices) - 1
        
        top_indices = sorted_indices[:cutoff_idx + 1]
        top_probs = probs[top_indices]
        top_probs = top_probs / top_probs.sum()
        
        return np.random.choice(top_indices, p=top_probs)
