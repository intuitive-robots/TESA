import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from preprocessor import To3Channel

from src.util.device import DEVICE

model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")


def siglip_encode_textList(textList):
    inputs = tokenizer(
        textList, padding="max_length", return_tensors="pt", truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        return model.get_text_features(**inputs)


class SigLIPProcessor:
    def __init__(self):
        self.model = AutoModel.from_pretrained("google/siglip-base-patch16-224").to(
            DEVICE
        )
        self.processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-224", do_rescale=False
        )
        self.transform = T.Compose(
            [
                T.ToTensor(),
                To3Channel(),
            ]
        )

    def run(self, jpg_path):
        with torch.no_grad():
            img = Image.open(jpg_path)
            img = self.transform(img).to(DEVICE)
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = inputs.to(DEVICE)
            embedding = self.model.get_image_features(**inputs)
            return embedding


def siglip_encode_imageList(imageList):
    p = SigLIPProcessor()
    return torch.cat([p.run(img) for img in imageList])
