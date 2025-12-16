import argparse
import logging
import os
import subprocess as sp
import sys
import time
import gc
import shutil
import json
import random
import glob
import re
from tempfile import NamedTemporaryFile
import torch
import gradio as gr
from audiocraft.models import MusicGen, MultiBandDiffusion
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import convert_audio

# --- GLOBAL CONFIGURATION ---
OUTPUT_DIR = "generated"
SEPARATED_DIR = "generated/separated"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEPARATED_DIR, exist_ok=True)

# Optimize PyTorch memory allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

MODEL = None
MBD = None
INTERRUPTING = False
USE_DIFFUSION = False
LAST_METADATA = {} 

# --- UTILITIES ---

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
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

# --- STEM SEPARATION LOGIC ---

def separate_audio(audio_path):
    global MODEL, MBD
    if not audio_path: raise gr.Error("No audio file provided.")

    original_filename = os.path.splitext(os.path.basename(audio_path))[0]
    cleaner_name = re.sub(r'-\d+-\d+$', '', original_filename)
    safe_name = "".join([c for c in cleaner_name if c.isalnum() or c in " _-"]).strip().replace(" ", "_")
    if not safe_name: safe_name = "audio"

    print(f"\n--- START SEPARATION PROCESS: {safe_name} ---")
    
    if MODEL is not None: del MODEL; MODEL = None
    if MBD is not None: del MBD; MBD = None
    free_memory()

    folder_name = f"{safe_name}_stems"
    target_folder = os.path.join(SEPARATED_DIR, "htdemucs", folder_name)

    if os.path.exists(target_folder):
        try: shutil.rmtree(target_folder)
        except Exception as e: print(f"Warning: {e}")
    
    command = [
        "demucs", "-n", "htdemucs", "-o", SEPARATED_DIR,
        "--filename", f"{folder_name}/{safe_name}_{{stem}}.{{ext}}",
        audio_path
    ]
    
    try: sp.run(command, check=True)
    except sp.CalledProcessError as e: raise gr.Error(f"Demucs execution failed: {e}")

    drums = os.path.join(target_folder, f"{safe_name}_drums.wav")
    bass = os.path.join(target_folder, f"{safe_name}_bass.wav")
    other = os.path.join(target_folder, f"{safe_name}_other.wav")
    vocals = os.path.join(target_folder, f"{safe_name}_vocals.wav")

    print(f"--- SEPARATION COMPLETED ---")
    return [
        drums if os.path.exists(drums) else None,
        bass if os.path.exists(bass) else None,
        other if os.path.exists(other) else None,
        vocals if os.path.exists(vocals) else None
    ]

# --- SAVE LOGIC (VM ONLY) ---

def save_track(current_file, filename_input):
    global LAST_METADATA
    
    if current_file is None: return "⚠️ Generate audio first."
    if not filename_input: filename_input = f"track_{int(time.time())}"
    
    safe_name = "".join([c for c in filename_input if c.isalnum() or c in " _-"])
    if not safe_name: safe_name = "unnamed"
    
    wav_path = os.path.join(OUTPUT_DIR, f"{safe_name}.wav")
    json_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    
    try:
        shutil.copy(current_file, wav_path)
        with open(json_path, 'w') as f:
            json.dump(LAST_METADATA, f, indent=4)
        return f"✅ Saved to VM: {safe_name}.wav"
    except Exception as e:
        return f"❌ Error: {e}"

# --- MODEL LOADING ---

def load_model(version):
    global MODEL
    print(f"Loading MusicGen Model: {version}...")
    if MODEL is not None and MODEL.name != version:
        del MODEL; MODEL = None; free_memory()
    if MODEL is None:
        try: MODEL = MusicGen.get_pretrained(version)
        except Exception as e: raise gr.Error(f"Error loading model: {e}")

def load_diffusion():
    global MBD
    if MBD is None:
        print("Loading MultiBand Diffusion Decoder...")
        MBD = MultiBandDiffusion.get_mbd_musicgen()

# --- GENERATION LOGIC ---

def predict(model_name, decoder, text, melody, duration, topk, topp, temperature, cfg_coef, seed):
    global INTERRUPTING, USE_DIFFUSION, LAST_METADATA
    INTERRUPTING = False
    
    if "lo-fi" not in text.lower() and "high fidelity" not in text:
        text += ", high fidelity, wide stereo, 44kHz, crisp quality, well produced"

    if seed == -1 or seed is None: seed = random.randint(0, 2**32 - 1)
    set_all_seeds(seed)
    
    print(f"\n--- START GENERATION: {text} ---")
    free_memory()

    if melody is not None and "melody" not in model_name:
        raise gr.Error(f"Model '{model_name}' does not support melody input.")

    load_model(model_name)
    
    if "stereo" in model_name and decoder == "MultiBand_Diffusion":
        print("WARNING: Diffusion incompatible with Stereo. Fallback to Default.")
        decoder = "Default"; USE_DIFFUSION = False
    else:
        USE_DIFFUSION = (decoder == "MultiBand_Diffusion")
    
    if USE_DIFFUSION: load_diffusion()

    MODEL.set_generation_params(duration=duration, top_k=int(topk), top_p=topp, temperature=temperature, cfg_coef=cfg_coef)
    
    def _progress(generated, to_generate):
        if INTERRUPTING: raise gr.Error("Interrupted by user.")
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
            wav = MODEL.generate_with_chroma(descriptions=[text], melody_wavs=melody_wavs, melody_sample_rate=MODEL.sample_rate, progress=True, return_tokens=USE_DIFFUSION)
        else:
            wav = MODEL.generate([text], progress=True, return_tokens=USE_DIFFUSION)
    except RuntimeError as e:
        if "out of memory" in str(e):
            free_memory()
            raise gr.Error("GPU OOM error. Try reducing duration.")
        raise e

    if USE_DIFFUSION and MBD:
        try: wav = MBD.tokens_to_wav(wav[1])
        except Exception: wav = MODEL.compression_model.decode(wav[1])

    with NamedTemporaryFile("wb", suffix=".wav", delete=False) as tfile:
        audio_write(tfile.name, wav[0].cpu(), MODEL.sample_rate, strategy="peak", loudness_compressor=False, add_suffix=False)
        filename = tfile.name
    
    LAST_METADATA = { "prompt": text, "model": model_name, "seed": seed, "duration": duration, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S") }
    
    print("--- COMPLETED ---")
    return filename, seed

# --- GUI INTERFACE ---

def ui_full(launch_kwargs):
    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")
    
    with gr.Blocks(title="MusicGen Pro", theme=theme) as interface:
        gr.Markdown("# MusicGen Pro & Stem Splitter")
        
        with gr.Tabs():
            # --- TAB 1: MUSIC GENERATION ---
            with gr.TabItem("✨ Generate Music"):
                with gr.Row():
                    with gr.Column(scale=1):
                        text = gr.Textbox(label="Description", placeholder="E.g. 'Techno, 140 BPM'...", lines=2)
                        with gr.Tab("Settings"):
                            model = gr.Dropdown(["facebook/musicgen-small", "facebook/musicgen-stereo-medium", "facebook/musicgen-stereo-large", "facebook/musicgen-stereo-melody", "facebook/musicgen-large"], label="Model", value="facebook/musicgen-stereo-medium")
                            duration = gr.Slider(5, 60, value=30, step=5, label="Duration (s)")
                            decoder = gr.Radio(["Default", "MultiBand_Diffusion"], label="Decoder Quality", value="Default")
                            seed_input = gr.Number(label="Seed (-1 = Random)", value=-1, precision=0)
                        with gr.Tab("Melody Input"):
                            melody = gr.Audio(source="upload", type="numpy", label="Upload Melody (Optional)")
                        with gr.Accordion("Advanced Parameters", open=False):
                            cfg_coef = gr.Slider(1.0, 10.0, value=3.0, label="CFG Guidance")
                            temperature = gr.Slider(0.1, 2.0, value=1.0, label="Temperature")
                            topk = gr.Number(value=250, label="Top-k")
                            topp = gr.Number(value=0, label="Top-p")
                        with gr.Row():
                            submit = gr.Button("Generate", variant="primary")
                            stop = gr.Button("Stop")

                    with gr.Column(scale=1):
                        # show_download_button=False verwijdert de PC download optie
                        output = gr.Audio(label="Result", type="filepath", interactive=False, show_download_button=False)
                        used_seed_output = gr.Number(label="Seed Used", interactive=False)
                        
                        gr.Markdown("### 💾 Save to VM")
                        with gr.Row():
                            filename_input = gr.Textbox(label="Filename", placeholder="My_Track", scale=3)
                            save_btn = gr.Button("Save", scale=1)
                        
                        save_status = gr.Label(label="Status")
                        save_btn.click(save_track, inputs=[output, filename_input], outputs=save_status)

                submit.click(predict, inputs=[model, decoder, text, melody, duration, topk, topp, temperature, cfg_coef, seed_input], outputs=[output, used_seed_output])
                stop.click(interrupt, queue=False)

            # --- TAB 2: STEM SEPARATION ---
            with gr.TabItem("✂️ Stem Splitter"):
                gr.Markdown("Upload audio to separate into stems: **Drums, Bass, Other, Vocals**.")
                with gr.Row():
                    with gr.Column():
                        input_separator = gr.Audio(type="filepath", label="Input Audio", show_edit_button=False)
                        btn_separate = gr.Button("Separate Audio", variant="primary")
                    
                    with gr.Column():
                        # Geen download knoppen, alleen luisteren
                        out_drums = gr.Audio(label="Drums", type="filepath", interactive=False, show_download_button=False)
                        out_bass = gr.Audio(label="Bass", type="filepath", interactive=False, show_download_button=False)
                        out_other = gr.Audio(label="Other", type="filepath", interactive=False, show_download_button=False)
                        out_vocals = gr.Audio(label="Vocals", type="filepath", interactive=False, show_download_button=False)

                btn_separate.click(separate_audio, inputs=[input_separator], outputs=[out_drums, out_bass, out_other, out_vocals])

        interface.queue().launch(**launch_kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen', type=str, default='0.0.0.0')
    parser.add_argument('--server_port', type=int, default=7860)
    args = parser.parse_args()
    ui_full({'server_name': args.listen, 'server_port': args.server_port})