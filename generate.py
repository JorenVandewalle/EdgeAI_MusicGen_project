import torch
import time
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# --- INSTELLINGEN ---
PROMPT = "hardcore uptempo "
DURATION = 30
MODEL_TYPE = 'facebook/musicgen-large' 

# --- INFO ---
if torch.cuda.is_available():
    print(f"HARDWARE: {torch.cuda.get_device_name(0)}")
    print("GPU is active! Generating will be fast.")
else:
    print("WARNING: Running on CPU. This will be slow.")

print(f"Loading model: {MODEL_TYPE}...")
model = MusicGen.get_pretrained(MODEL_TYPE)

# VERWIJDERD: model.to(device) -> AudioCraft doet dit automatisch!

model.set_generation_params(duration=DURATION)

print(f"Generating {DURATION}s of audio...")
start_time = time.time()

# Genereren (dit gebeurt nu op de GPU)
wav = model.generate([PROMPT])

end_time = time.time()
print(f"Generation took: {end_time - start_time:.2f} seconds")

# Opslaan
filename = "klj muziek"
print(f"Saving to {filename}.wav ...")
audio_write(filename, wav[0].cpu(), model.sample_rate, strategy='loudness', loudness_compressor=True)

print("------------------------------------------------")
print(f"DONE! File saved as: {os.getcwd()}/generated/{filename}.wav")
print("------------------------------------------------")