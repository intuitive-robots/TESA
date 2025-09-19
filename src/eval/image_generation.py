import torch
from torch_geometric.loader import DataLoader

from src.models.component_handler import logits_from_config
from src.util.util_siglip import siglip_encode_textList
from src.util.util_clip import clip_encode_text_list

from diffusers import StableDiffusionXLPipeline

from src.util.device import DEVICE

from transformers import AutoTokenizer


def eval_image_generation(model, dataset_to_eval, config):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(DEVICE)
        
    prompt = ["a futuristic city skyline at sunset, ultra-detailed, 8k"]
    tokenizer_1 = pipe.tokenizer  # for text_encoder
    tokenizer_2 = AutoTokenizer.from_pretrained(pipe.text_encoder_2.config._name_or_path)

    text_inputs_1 = tokenizer_1(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_1.model_max_length,
        return_tensors="pt"
    ).to(DEVICE)

    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_2.model_max_length,
        return_tensors="pt"
    ).to(DEVICE)

    # ---------------------------
    # ENCODE PROMPT
    # ---------------------------
    # First text encoder (token-level)
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(
            text_inputs_1.input_ids,
            attention_mask=text_inputs_1.attention_mask
        )[0].to(DEVICE)  # shape [B, 77, 1024]

    # Second text encoder (pooled)
    with torch.no_grad():
        pooled_prompt_embeds = pipe.text_encoder_2(
            text_inputs_2.input_ids,
            attention_mask=text_inputs_2.attention_mask
        ).pooler_output.to(DEVICE)  # shape [B, 1280]

    image = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=30,
        guidance_scale=7.5,
        height=1024,
        width=1024,
    ).images[0]
    
    path = "/home/vquapil/TESA/"
    image.save(path + "name_img.png")
    print(f"✅ Saved image at {path}")