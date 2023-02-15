import time
import os

from diffusers import DDPMPipeline
from fire import Fire

from resolve import resolve_device

def gen(num_skins=1, prefix="skin"):
    os.makedirs("outputs", exist_ok=True)

    device = resolve_device()

    print(f"Using device: {device}")

    pipeline = DDPMPipeline.from_pretrained('WiNE-iNEFF/Minecraft-Skin-Diffusion')
    pipeline = pipeline.to(device)

    pipeline.enable_attention_slicing()

    # "warm up" the pipeline
    pipeline(num_inference_steps=1)

    for i in range(num_skins):
        print(f"Generating skin {i}")
        start = time.time()
        image = pipeline().images[0].convert('RGBA')
        image.save(f"outputs/{prefix}_{i}.png")
        end = time.time()
        print(f"Generated skin {i} in {round(end-start, 2)} seconds")

if __name__ == "__main__":
    Fire(gen)
