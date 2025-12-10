import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import os

print("--- START CI TEST ---")
print("Checking imports...")
import gradio, transformers, av
print("Imports OK.")

print("Loading small model for smoke test (CPU)...")
try:
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    model.set_generation_params(duration=1) # 1 seconde is genoeg
    wav = model.generate(['test run'])
    audio_write('ci_test_output', wav[0].cpu(), model.sample_rate)
    
    if os.path.exists('ci_test_output.wav'):
        print("SUCCESS: File generated.")
    else:
        raise Exception("File not found")
except Exception as e:
    print(f"TEST FAILED: {e}")
    exit(1)

print("--- CI TEST PASSED ---")
exit(0)