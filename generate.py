import os
import torch
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video
import uuid
import argparse
import PIL

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prompt')
parser.add_argument('-i', '--image')
# this duration is `temp`, in [1, 31] <=> frame in [1, 241] <=> duration in [0, 10s]
parser.add_argument('-d', '--duration', type=int, default=1)
parser.add_argument('-l', '--latents')
args = parser.parse_args()

model = PyramidDiTForVideoGeneration(
    model_path="pyramid_flow_model",
    model_name = "pyramid_flux",
    model_dtype="bf16",
    model_variant="diffusion_transformer_384p",
)
#height=768,
#width=1280,
height = 384
width = 640

device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

model.text_encoder.to(device)
model.dit.to(device)
model.vae.to(device)
model.vae.enable_tiling()

def resize_crop_image(img, width, height):
    ori_width, ori_height = img.width, img.height
    scale = max(width / ori_width, height / ori_height)
    resized_width = round(ori_width * scale)
    resized_height = round(ori_height * scale)
    img = img.resize((resized_width, resized_height), resample=PIL.Image.LANCZOS)

    left = (resized_width - width) / 2
    top = (resized_height - height) / 2
    right = (resized_width + width) / 2
    bottom = (resized_height + height) / 2

    img = img.crop((left, top, right, bottom))

    return img

def generate(prompt, image, duration):
    if image is not None:
        image = PIL.Image.open(image).convert("RGB")
        image = resize_crop_image(image, width, height)
        return model.generate_i2v(
            prompt=prompt,
            input_image=image,
            temp=duration,
            num_inference_steps=[10, 10, 10],
            guidance_scale=7.0,
            video_guidance_scale=5.0,
            output_type="latent",
            save_memory=True,
        )
    else:
        return model.generate(
            prompt=prompt,
            num_inference_steps=[10, 10, 10],
            video_num_inference_steps=[5, 5, 5],
            height=height,
            width=width,
            temp=duration,
            guidance_scale=7.0,
            video_guidance_scale=5.0,
            output_type="latent",
            save_memory=True,
        )

def decode(latents, basename):
    frames = model.decode_latent(latents, save_memory=True)
    export_to_video(frames, f"{basename}.mp4", fps=24)
    print(f"Exported {basename}.mp4")

with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
    if args.latents is not None:
        basename, _ = os.path.splitext(os.path.basename(args.latents))
        latents = torch.load(args.latents, weights_only=True).to(device)
    else:
        latents = generate(args.prompt, args.image, args.duration)
        basename = str(uuid.uuid4())
        torch.save(latents, f"{basename}.pt")
        print(f"Exported {basename}.pt")

    decode(latents, basename)
