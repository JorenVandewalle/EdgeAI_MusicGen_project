import argparse
import logging
import os
import subprocess as sp
import sys
import time
import gc
import shutil
import torch
import gradio as gr
from transformers import pipeline
from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio

# --- CONFIGURATIE ---
# Map waar bestanden permanent worden opgeslagen
OUTPUT_DIR = "generated"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Maakt de map aan als hij niet bestaat

# Geheugen optimalisatie
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

MODEL = None
MBD = None
PROMPT_AI = None
INTERRUPTING = False
USE_DIFFUSION = False

# Fix FFmpeg logs
_old_call = sp.call
def _call_nostderr(*args, **kwargs):
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)
sp.call = _call_nostderr

def interrupt():
    global INTERRUPTING
    INTERRUPTING = True

def free_memory():
    """Maakt geheugen vrij tussen generaties door"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

# --- AI PROMPT HELPER ---
def enhance_prompt(user_text):
    global PROMPT_AI
    if not user_text: return "Please type something..."
    
    if PROMPT_AI is None:
        print("Loading Prompt AI...")
        PROMPT_AI = pipeline('text-generation', model='distilgpt2', device=0 if torch.cuda.is_available() else -1)
    
    base = f"Music description: {user_text} -> {user_text}, high fidelity, stereo, "
    res = PROMPT_AI(base, max_new_tokens=40, num_return_sequences=1)[0]['generated_text']
    output = res.split('->')[-1].strip().split('\n')[0]
    return output

# --- SAVING HELPER ---
def save_track(current_file, filename_input):
    if current_file is None:
        return "⚠️ Genereer eerst muziek voordat je opslaat!"
    
    if not filename_input:
        filename_input = f"track_{int(time.time())}"
    
    # 1. Maak de bestandsnaam veilig (verwijder rare tekens)
    safe_name = "".join([c for c in filename_input if c.isalnum() or c in " _-"])
    if not safe_name: safe_name = "unnamed_track"
    
    # 2. Bepaal het pad
    target_path = os.path.join(OUTPUT_DIR, f"{safe_name}.wav")
    
    # 3. Kopieer het bestand
    try:
        shutil.copy(current_file, target_path)
        return f"✅ Opgeslagen in map '{OUTPUT_DIR}' als: {safe_name}.wav"
    except Exception as e:
        return f"❌ Fout bij opslaan: {e}"

# --- MODEL LADEN ---
def load_model(version):
    global MODEL
    print(f"Loading MusicGen Model: {version}...")
    
    if MODEL is not None and MODEL.name != version:
        del MODEL
        MODEL = None
        free_memory()
    
    if MODEL is None:
        try:
            MODEL = MusicGen.get_pretrained(version)
        except Exception as e:
            raise gr.Error(f"Error loading model: {e}")

def load_diffusion():
    global MBD
    if MBD is None:
        print("Loading MultiBand Diffusion Decoder...")
        MBD = MultiBandDiffusion.get_mbd_musicgen()

# --- GENERATIE LOGICA ---
def predict(model_name, decoder, text, melody, duration, topk, topp, temperature, cfg_coef):
    global INTERRUPTING, USE_DIFFUSION
    INTERRUPTING = False
    
    print(f"\n--- START: {text} ({duration}s) ---")
    free_memory()

    if melody is not None and "melody" not in model_name:
        raise gr.Error(f"Model '{model_name}' kan geen audio input gebruiken. Kies een 'melody' model.")

    load_model(model_name)
    
    if decoder == "MultiBand_Diffusion":
        USE_DIFFUSION = True
        load_diffusion()
    else:
        USE_DIFFUSION = False

    MODEL.set_generation_params(
        duration=duration, top_k=int(topk), top_p=topp, temperature=temperature, cfg_coef=cfg_coef
    )
    
    def _progress(generated, to_generate):
        if INTERRUPTING: raise gr.Error("Gestopt door gebruiker.")
    MODEL.set_custom_progress_callback(_progress)

    melody_wavs = None
    if melody is not None:
        sr, audio = melody
        audio = torch.from_numpy(audio).to(MODEL.device).float().t()
        if audio.dim() == 1: audio = audio[None]
        audio = audio[..., :int(sr * duration)]
        melody_wavs = [convert_audio(audio, sr, MODEL.sample_rate, MODEL.audio_channels)]

    try:
        if melody_wavs:
            wav = MODEL.generate_with_chroma(descriptions=[text], melody_wavs=melody_wavs, 
                                           melody_sample_rate=MODEL.sample_rate, progress=True, return_tokens=USE_DIFFUSION)
        else:
            wav = MODEL.generate([text], progress=True, return_tokens=USE_DIFFUSION)
    except RuntimeError as e:
        if "out of memory" in str(e):
            free_memory()
            raise gr.Error("GPU Geheugen Vol! Probeer een kortere duur.")
        raise e

    if USE_DIFFUSION and MBD:
        try:
            wav = MBD.tokens_to_wav(wav[1])
        except RuntimeError:
            print("Diffusion OOM, fallback to default")
            wav = MODEL.compression_model.decode(wav[1])

    # Sla tijdelijk bestand op in /tmp (zodat Gradio het kan tonen)
    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as tfile:
        audio_write(tfile.name, wav[0].cpu(), MODEL.sample_rate, strategy="peak", loudness_compressor=False, add_suffix=False)
        filename = tfile.name
    
    print("--- KLAAR ---")
    return filename

# --- GUI ---
def ui_full(launch_kwargs):
    with gr.Blocks(title="MusicGen Ultimate", theme=gr.themes.Base()) as interface:
        gr.Markdown("# 🎹 MusicGen Ultimate")
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    text = gr.Textbox(label="Beschrijving", placeholder="Typ hier (bv: 'Techno')...", lines=2)
                    magic_btn = gr.Button("✨ Verbeter Prompt met AI")
                    magic_btn.click(enhance_prompt, inputs=text, outputs=text)

                with gr.Tab("Instellingen"):
                    model = gr.Dropdown(
                        [
                            "facebook/musicgen-stereo-medium",
                            "facebook/musicgen-stereo-large",
                            "facebook/musicgen-stereo-melody",
                            "facebook/musicgen-large",
                        ],
                        label="Kies Model", value="facebook/musicgen-stereo-medium"
                    )
                    duration = gr.Slider(5, 60, value=30, step=5, label="Duur")
                    decoder = gr.Radio(["Default", "MultiBand_Diffusion"], label="Kwaliteit", value="Default")

                with gr.Tab("Audio Uploaden"):
                    melody = gr.Audio(source="upload", type="numpy", label="Upload melodie")

                with gr.Accordion("Expert", open=False):
                    cfg_coef = gr.Slider(1.0, 10.0, value=3.0, label="Guidance")
                    temperature = gr.Slider(0.1, 2.0, value=1.0, label="Temp")
                    topk = gr.Number(value=250, label="Top-k")
                    topp = gr.Number(value=0, label="Top-p")

                with gr.Row():
                    submit = gr.Button("🚀 Genereer", variant="primary")
                    stop = gr.Button("🛑 Stop")

            with gr.Column(scale=1):
                # De Audio Output Speler
                output = gr.Audio(label="Resultaat", type="filepath")
                
                # --- NIEUW: OPSLAAN SECTIE ---
                gr.Markdown("### 💾 Opslaan op Server")
                with gr.Row():
                    filename_input = gr.Textbox(label="Bestandsnaam", placeholder="Mijn_Techno_Track", scale=3)
                    save_btn = gr.Button("Opslaan", scale=1)
                
                save_status = gr.Label(label="Status")
                
                # Koppel de save knop
                # inputs=[output, filename_input] -> pakt het bestand uit de speler + de tekst
                save_btn.click(save_track, inputs=[output, filename_input], outputs=save_status)
                # -----------------------------

        submit.click(predict, inputs=[model, decoder, text, melody, duration, topk, topp, temperature, cfg_coef], outputs=[output])
        stop.click(interrupt, queue=False)

        interface.queue().launch(**launch_kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen', type=str, default='0.0.0.0')
    parser.add_argument('--server_port', type=int, default=7860)
    args = parser.parse_args()

    ui_full({'server_name': args.listen, 'server_port': args.server_port})