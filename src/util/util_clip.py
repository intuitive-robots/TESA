import clip
import torch
from PIL import Image

from src.util.device import DEVICE

clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, jit=True)
clip_model.eval()


def clip_encode_text_list(text_list):
    with torch.no_grad():
        token = clip.tokenize(text_list).to(DEVICE)
        return clip_model.encode_text(token)


class ClipProcessor:

    def __init__(self):
        # import clip here since it might not be installed if not needed.
        import clip

        self.model, self.preprocess = clip.load(
            "ViT-B/32", device=DEVICE
        )  # gibt auch /16, /14.

    def run(self, jpg_path):
        image = torch.stack([self.preprocess(Image.open(jpg_path))]).to(DEVICE)
        return self.model.encode_image(image)


def clip_encode_image_list(image_list):
    p = ClipProcessor()
    return torch.cat([p.run(img) for img in image_list])
