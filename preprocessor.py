r"This script should be run to preprocess images. Output to /processed."

import os

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from dotenv import load_dotenv
from PIL import Image
from torchvision.io import read_image
from tqdm import tqdm

#from src.util.util_clip import ClipProcessor
#from src.util.util_siglip import SigLIPProcessor
from src.util.device import DEVICE
load_dotenv()
data_dir = os.getenv("DATA_DIR")


def main():
    # voltron: use python 3.8 (https://github.com/MahmoudAshraf97/whisper-diarization/issues/65)
    # ["clip", "DINOv2", "ResNet-50", "SigLIP"], mit python 3.8: ["voltron-cond"]
    models = ["SigLIP"]
    datasets = [
        "vg/VG_100K",
        "vg/VG_100K_2",
        "coco/train2017",
        "coco/val2017",
        "gqa/images",
    ]
    # ["vg/VG_100K", "vg/VG_100K_2","coco/train2017", "coco/val2017", "gqa/images"]
    for m in models:
        for d in datasets:
            preprocess(m, d)
    print("done.")


def preprocess(model, dataset_str, padding_to_length=None):
    if model == "clip":
        image_processor = ClipProcessor()
    elif model == "voltron-cond":
        image_processor = VoltronCondProcessor()
    elif model == "DINOv2":
        image_processor = DINOv2Processor()
    elif model == "ResNet-50":
        image_processor = ResNet50Processor()
    elif model == "SigLIP":
        image_processor = SigLIPProcessor()
    else:
        raise NotImplementedError(f"model {model} unknown.")

    print(
        f"preprocessing using model {model} for dataset {data_dir}/raw/{dataset_str}")
    in_path = f"{data_dir}/raw/{dataset_str}"
    out_path = f"{data_dir}/processed/{dataset_str}/{model}_out"

    guarantee_dir(out_path)

    jpg_files = [file for file in os.listdir(in_path) if file.endswith(".jpg")]
    print(f"INFO: {len(jpg_files)} .jpg files.")
    with torch.no_grad():
        for f in tqdm(jpg_files):
            try:
                img_features = image_processor.run(f"{in_path}/{f}")
            except Exception as e:
                print(e)
                print(f"WARNING: Error with image {in_path}/{f}!")
                continue
            # shape: [1,x]
            v = img_features.to(torch.float32)
            if padding_to_length:
                padding_len = padding_to_length - v.shape[1]
                if padding_len < 0:
                    raise Exception(
                        f"{model} embedding of {f} has {v.shape[1]} dimensions, which is larger than desired amount of dimensions ({padding_to_length})."
                    )
                padding = torch.zeros(
                    (1, padding_len), dtype=v.dtype, device=v.device)
                v = torch.cat((v, padding), dim=1)
            np.save(f"{out_path}/{f}", v.cpu().numpy())


def guarantee_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created dir at {path}")


class ResNet50Processor:
    def __init__(self):
        self.backbone = torch.hub.load(
            "pytorch/vision",
            "resnet50",
            weights="ResNet50_Weights.IMAGENET1K_V1",
        )
        self.backbone = self.backbone.to(DEVICE)
        self.backbone.eval()
        self.transform = T.Compose(
            [
                T.ToTensor(),
                To3Channel(),
            ]
        )

    def run(self, jpg_path):
        img = Image.open(jpg_path)
        img = self.transform(img)
        embedding = self.backbone(img.unsqueeze(0).to(DEVICE))
        return embedding


class VoltronCondProcessor:
    def __init__(self):
        from voltron import instantiate_extractor, load

        self.vcond, self.preprocess = load(
            "v-cond", device=DEVICE, freeze=True)
        self.vector_extractor = instantiate_extractor(self.vcond)().to(DEVICE)

    def run(self, jpg_path):
        img = read_image(jpg_path)
        img = img.expand(3, -1, -1)  # for handling grayscale.
        img = self.preprocess(img)[None, ...].to(DEVICE)
        e = self.vcond(img, mode="visual")
        return self.vector_extractor(e)


class DINOv2Processor:
    def __init__(self):
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14")
        self.backbone.eval()
        self.backbone.to(DEVICE)
        self.transform = T.Compose(
            [  # needed?
                T.ToTensor(),
                To3Channel(),
                T.Resize(224),
                PadToMultipleOf(14),
            ]
        )
        # height and width need to be multiple of 14.

    def run(self, jpg_path):
        img = Image.open(jpg_path)
        img = self.transform(img)
        embedding = self.backbone(img.unsqueeze(0).to(DEVICE))
        return embedding


class To3Channel:
    def __call__(self, img):
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        return img


class PadToMultipleOf:
    def __init__(self, patch_length):
        self.p = patch_length

    def __call__(self, img):
        _, h, w = img.shape

        pad_h = (self.p - h % self.p) % self.p  # Amount to add to height
        pad_w = (self.p - w % self.p) % self.p  # Amount to add to width

        # Apply padding on the right and bottom sides
        padding = (0, 0, pad_w, pad_h)  # (left, top, right, bottom)
        return F.pad(img, padding, fill=0)  # Zero padding


if __name__ == "__main__":
    main()
