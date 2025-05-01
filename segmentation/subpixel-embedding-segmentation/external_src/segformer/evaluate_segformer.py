print("Importing libraries...")

from train_segformer import SegmentationDataset, get_device
from augment import get_seg_overlay
import random
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "evaluation_results/segformer_brightness"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(1)

# Parameters & paths
MODEL_CHECKPOINT = "nvidia/mit-b4"
FINE_TUNED_PATH = "trained_models/segformer/segformer_finetuned.pth"
SCAN_AUG_FILE    = "training/ergonomics/train_scans_augmented.txt"
MASK_AUG_FILE    = "training/ergonomics/train_ground_truths_augmented.txt"

INPUT_SIZE = (640, 480)   # width, height
VAR_INDICES = list(range(11, 22)) 

print("Loading model...")
# Load file lists and split validation
with open(SCAN_AUG_FILE) as f:
    scans = [l.strip() for l in f if l.strip()]
with open(MASK_AUG_FILE) as f:
    masks = [l.strip() for l in f if l.strip()]

# first 99 are validation (same as in train_segformer.py)
val_scans = scans[:99]
val_masks = masks[:99]

# Load model
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=4
)

print("Loading fine-tuned weights...")
model.load_state_dict(torch.load(FINE_TUNED_PATH, map_location=device))
model.to(device).eval()

# same normalization used in training
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# utility to add Gaussian noise
def add_noise(pil_img, sigma):
    arr = np.array(pil_img).astype(np.int16)
    noise = np.random.randn(*arr.shape) * sigma
    arr_n = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr_n)

print("Processing samples...")
# Loop over the selected validation samples
for idx in VAR_INDICES:
    print(f"Processing sample {idx} / {len(VAR_INDICES)}...")
    rgb_path = val_scans[idx]
    img0 = Image.open(rgb_path).convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)

    # Build the 5 variants
    variants = [
        ("orig", img0),
        ("bright+40%",  ImageEnhance.Brightness(img0).enhance(1.4)),
        ("dark-20% + noise", add_noise(ImageEnhance.Brightness(img0).enhance(0.8), 10)),
        ("dark-40% + more noise", add_noise(ImageEnhance.Brightness(img0).enhance(0.6), 20)),
        ("dark-60% + even more noise", add_noise(ImageEnhance.Brightness(img0).enhance(0.4), 30)),
    ]

    # Run inference & overlay
    overlays = []
    for name, pil in variants:
        # model input
        x = img_transform(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x).logits
            # upsample logits to INPUT_SIZE
            out = F.interpolate(out, size=INPUT_SIZE[::-1],
                                mode="bilinear", align_corners=False)
            pred = out.argmax(dim=1)[0].cpu().numpy()

        ov = get_seg_overlay(np.array(pil), pred, overlay_opacity=0.5)
        overlays.append((name, ov))

    # Plot 1×5 overlay
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (name, ov) in zip(axs, overlays):
        ax.imshow(ov)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    
    # save
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"segformer_{idx}.png"),
        dpi=300
    )
    print(f"Saved to evaluation_results/segformer_brightness/segformer_{idx}.png")
    plt.close(fig)
