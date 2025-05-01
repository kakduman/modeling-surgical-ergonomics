# batch_run_segformer.py

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import SegformerForSemanticSegmentation

from train_segformer import get_device

MODEL_CHECKPOINT  = "nvidia/mit-b4"
FINETUNED_WEIGHTS = "trained_models/segformer/segformer_finetuned.pth"

# where to write your segformer pred_maps
OUT_BASE = "evaluation_results/segformer/test_scans/pred_map"

INPUT_SIZE = (640, 480)   # (width, height)

# class to RGB
PALETTE = {
    0: [255,   0,   0],   # class 0  red
    1: [  0, 255,   0],   # class 1  green
    2: [  0,   0, 255],   # class 2  blue
    3: [  0,   0,   0],   # class 3  black (background)
}

CASES = [
    ("patient5",  "evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient5.png"),
    ("patient6",  "evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient6.png"),
    ("patient1",  "evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient1.png"),
    ("patient2",  "evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient2.png"),
    ("patient3",  "evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient3.png"),
    ("patient4",  "evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient4.png"),
    ("patient14", "evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient14.png"),
    ("patient15", "evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient15.png"),
    ("patient16", "evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient16.png"),
    ("patient17", "evaluation_results/u-net/unet_traintest/test_scans/scan_color/scan_color_patient17.png"),
]
def load_model():
    device = get_device()
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=4
    )
    ckpt = torch.load(FINETUNED_WEIGHTS, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, device

def preprocess(pil_img):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])(pil_img)

def colorize(mask):
    """Turn HxW label mask into an HxWx3 RGB image via PALETTE."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, col in PALETTE.items():
        out[mask == cls] = col
    return out

def run_case(model, device, img_path, out_path):
    # load & resize
    img = Image.open(img_path).convert("RGB")
    img = img.resize(INPUT_SIZE, Image.BILINEAR)

    # inference
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).logits
        logits = F.interpolate(
            logits,
            size=(INPUT_SIZE[1], INPUT_SIZE[0]),
            mode="bilinear", align_corners=False
        )
        pred = logits.argmax(dim=1)[0].cpu().numpy()

    # colorize & save
    pred_rgb = colorize(pred)
    Image.fromarray(pred_rgb).save(out_path)
    print(f"Saved {out_path}")

def main():
    model, device = load_model()
    os.makedirs(OUT_BASE, exist_ok=True)
    for pid, ipath in CASES:
        out_path = os.path.join(OUT_BASE, f"pred_map_{pid}.png")
        print(f"Processing {pid}")
        run_case(model, device, ipath, out_path)

if __name__ == "__main__":
    main()
