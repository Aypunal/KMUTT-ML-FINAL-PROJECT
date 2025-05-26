from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

prompts = [
    # Animal photography
    "wildlife animal",
    # "zoo animal",
    # "pet dog",
    # "cat portrait",

    # # Food photography
    # "cooked food",
    # "street food",
    # "dessert",
    # "fruit",

    # # Plant photography
    # "real plant",
    # "flower macro",
    # "tree landscape",
    # "forest",
]

def image_generation(prompt):
    """
    Generate an AI image using Stable Diffusion and save it to a specified folder.
    """
    # ---- CONFIGURATION ----
    output_folder = "/ai_image"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"ai_image_{timestamp}.png"
    model_name = "runwayml/stable-diffusion-v1-5"
    # -----------------------

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the Stable Diffusion model (CPU only)
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    pipe = pipe.to(device)

    # Generate image
    image = pipe(prompt).images[0]

    # Save image
    output_path = os.path.join(output_folder, output_filename)
    image.save(output_path)

    print(f"Image saved to: {output_path}")


for prompt in prompts:
     image_generation(prompt)