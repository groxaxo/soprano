#!/usr/bin/env python3
"""
Gradio Web Interface for Soprano TTS
A modern, user-friendly interface for text-to-speech synthesis
"""
import gradio as gr
import torch
import numpy as np
from soprano import SopranoTTS
import tempfile
import os

# Initialize the TTS model
print("Loading Soprano TTS model...")
try:
    model = SopranoTTS(backend='auto', device='cuda', cache_size_mb=10, decoder_batch_size=1)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def generate_speech(
    text: str,
    temperature: float = 0.3,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
) -> tuple:
    """
    Generate speech from text using Soprano TTS
    
    Args:
        text: Input text to synthesize
        temperature: Sampling temperature (lower = more deterministic)
        top_p: Top-p sampling parameter
        repetition_penalty: Penalty for repeating tokens
        
    Returns:
        Tuple of (sample_rate, audio_array) for Gradio Audio component
    """
    if not text or not text.strip():
        return None, "Please enter some text to synthesize."
    
    if model is None:
        return None, "Model failed to load. Please check the server logs."
    
    try:
        # Generate audio
        audio = model.infer(
            text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        # Convert to numpy array for Gradio
        audio_np = audio.cpu().numpy()
        
        # Return sample rate and audio data
        # Soprano uses 32kHz sample rate
        return (32000, audio_np), "✅ Audio generated successfully!"
        
    except Exception as e:
        error_msg = f"❌ Error generating audio: {str(e)}"
        print(error_msg)
        return None, error_msg


def generate_speech_batch(texts: str) -> tuple:
    """
    Generate speech for multiple texts (one per line)
    
    Args:
        texts: Multiple texts separated by newlines
        
    Returns:
        Tuple of (sample_rate, audio_array) for first text and status message
    """
    if not texts or not texts.strip():
        return None, "Please enter some text to synthesize."
    
    if model is None:
        return None, "Model failed to load. Please check the server logs."
    
    try:
        # Split by newlines and filter empty lines
        text_list = [t.strip() for t in texts.split('\n') if t.strip()]
        
        if not text_list:
            return None, "No valid text found."
        
        # Generate audio for all texts
        audio_list = model.infer_batch(text_list)
        
        # For simplicity, return the first one (could be enhanced to return all)
        audio_np = audio_list[0].cpu().numpy()
        
        status = f"✅ Generated audio for {len(text_list)} text(s). Showing first one."
        return (32000, audio_np), status
        
    except Exception as e:
        error_msg = f"❌ Error generating audio: {str(e)}"
        print(error_msg)
        return None, error_msg


# Custom CSS for modern design
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 1200px;
    margin: auto;
}

.header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.header h1 {
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
}

.header p {
    font-size: 1.2rem;
    margin-top: 0.5rem;
    opacity: 0.9;
}

.feature-box {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}

.tips {
    background: #fff3cd;
    border: 1px solid #ffc107;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

footer {
    text-align: center;
    padding: 2rem 0;
    color: #6c757d;
    border-top: 1px solid #dee2e6;
    margin-top: 3rem;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Soprano TTS") as demo:
    
    # Header
    gr.HTML("""
        <div class="header">
            <h1>🎵 Soprano TTS</h1>
            <p>Instant, Ultra-Realistic Text-to-Speech</p>
            <p style="font-size: 0.9rem;">2000× Real-time • 32kHz Audio • <15ms Latency</p>
        </div>
    """)
    
    # Main content
    with gr.Tabs():
        # Tab 1: Single Text Generation
        with gr.TabItem("🎙️ Single Text"):
            gr.Markdown("""
            ### Generate speech from a single text
            Enter your text below and customize the generation parameters.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        label="Text to Synthesize",
                        placeholder="Enter your text here... (works best with 2-15 second sentences)",
                        lines=5,
                        value="Soprano is an extremely lightweight text to speech model capable of generating high-quality audio in real-time."
                    )
                    
                    with gr.Accordion("🎛️ Advanced Settings", open=False):
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.3,
                            step=0.05,
                            label="Temperature",
                            info="Lower = more deterministic, Higher = more varied"
                        )
                        top_p = gr.Slider(
                            minimum=0.5,
                            maximum=1.0,
                            value=0.95,
                            step=0.05,
                            label="Top P",
                            info="Nucleus sampling threshold"
                        )
                        repetition_penalty = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.2,
                            step=0.1,
                            label="Repetition Penalty",
                            info="Penalty for repeating tokens"
                        )
                    
                    generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    audio_output = gr.Audio(
                        label="Generated Audio",
                        type="numpy",
                        autoplay=False
                    )
                    status_output = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
            
            generate_btn.click(
                fn=generate_speech,
                inputs=[text_input, temperature, top_p, repetition_penalty],
                outputs=[audio_output, status_output]
            )
        
        # Tab 2: Batch Text Generation
        with gr.TabItem("📚 Batch Generation"):
            gr.Markdown("""
            ### Generate speech for multiple texts
            Enter multiple texts, one per line. Each will be synthesized independently.
            """)
            
            with gr.Row():
                with gr.Column():
                    batch_text_input = gr.Textbox(
                        label="Texts to Synthesize (one per line)",
                        placeholder="Enter multiple texts, one per line...",
                        lines=10
                    )
                    batch_generate_btn = gr.Button("🎵 Generate All", variant="primary", size="lg")
                    
                with gr.Column():
                    batch_audio_output = gr.Audio(
                        label="Generated Audio (First Text)",
                        type="numpy",
                        autoplay=False
                    )
                    batch_status_output = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
            
            batch_generate_btn.click(
                fn=generate_speech_batch,
                inputs=[batch_text_input],
                outputs=[batch_audio_output, batch_status_output]
            )
        
        # Tab 3: Information & Tips
        with gr.TabItem("ℹ️ Information"):
            gr.Markdown("""
            ## About Soprano TTS
            
            **Soprano** is an ultra-lightweight, open-source text-to-speech model designed for real-time, 
            high-fidelity speech synthesis at unprecedented speed.
            
            ### ✨ Key Features
            
            - **⚡ 2000× Real-time**: Generate 10 hours of audio in under 20 seconds
            - **🎵 High-fidelity 32 kHz audio**: Superior quality compared to 24 kHz models
            - **🚀 Ultra-low latency**: Streaming with <15 ms first-chunk latency
            - **💾 Efficient**: Only 80M parameters, under 1 GB VRAM usage
            - **🔊 Vocoder-based decoder**: Orders of magnitude faster than diffusion models
            
            ### 💡 Usage Tips
            
            - **Optimal sentence length**: 2-15 seconds per sentence works best
            - **Numbers & special characters**: Convert to phonetic form (e.g., "1+1" → "one plus one")
            - **Regeneration**: If results are unsatisfactory, try regenerating or adjusting sampling settings
            - **Grammar**: Use proper grammar and avoid multiple spaces
            
            ### 🎛️ Parameter Guide
            
            - **Temperature** (0.1-1.0): Controls randomness. Lower values (0.2-0.4) produce more consistent results
            - **Top P** (0.5-1.0): Nucleus sampling. Keep at 0.95 for best results
            - **Repetition Penalty** (1.0-2.0): Prevents repetition. Default 1.2 works well
            
            ### 🔗 Links
            
            - [GitHub Repository](https://github.com/ekwek1/soprano)
            - [HuggingFace Model](https://huggingface.co/ekwek/Soprano-80M)
            - [HuggingFace Demo](https://huggingface.co/spaces/ekwek/Soprano-TTS)
            
            ### 📄 License
            
            This project is licensed under the Apache-2.0 license.
            """)
    
    # Footer
    gr.HTML("""
        <footer>
            <p>Built with ❤️ using <a href="https://github.com/ekwek1/soprano" target="_blank">Soprano TTS</a></p>
            <p style="font-size: 0.9rem;">Open-source • Apache-2.0 License</p>
        </footer>
    """)


def main():
    """Launch the Gradio interface"""
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
