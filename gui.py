import gradio as gr
import torch
import time
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_music(prompt, duration, model_type):
    # Laad model
    model = MusicGen.get_pretrained(model_type)
    model.set_generation_params(duration=int(duration))

    # Genereer
    wav = model.generate([prompt])

    # Opslaan
    filename = f"generated_music_{int(time.time())}.wav"
    filepath = os.path.join(os.getcwd(), filename)
    audio_write(filename, wav[0].cpu(), model.sample_rate, strategy='loudness', loudness_compressor=True)

    return filepath

# Gradio interface
iface = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Prompt", value="hardcore uptempo"),
        gr.Slider(minimum=5, maximum=60, value=30, label="Duration (seconds)"),
        gr.Dropdown(choices=["facebook/musicgen-small", "facebook/musicgen-medium", "facebook/musicgen-large"], value="facebook/musicgen-large", label="Model Type")
    ],
    outputs=gr.Audio(label="Generated Music"),
    title="MusicGen GUI",
    description="Genereer muziek met MusicGen via een eenvoudige interface."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)