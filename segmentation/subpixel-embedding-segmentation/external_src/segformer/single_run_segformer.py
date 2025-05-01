"""
single_run_segformer.py

Load a fine-tuned SegFormer model, run inference on one validation image,
and display & save overlays of the ground truth and the model prediction.
"""

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

from transformers import SegformerForSemanticSegmentation
from train_segformer import get_device
from augment import get_seg_overlay

def main():
    #config
    rgb_path          = "../../data/scans/SB/SB1Cback/data/image/0000005469-1681416978030.0056.png"
    gt_path           = "../../data/annotated/SB/SB1Cback/SegmentationClass/0000005469-1681416978030.0056.npy"
    model_checkpoint  = "nvidia/mit-b4"
    finetuned_weights = "trained_models/segformer/segformer_finetuned.pth"
    output_dir        = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    input_size = (640, 480)   # (width, height)

    # load image
    img = Image.open(rgb_path).convert("RGB")
    img = img.resize(input_size, Image.BILINEAR)
    img_np = np.array(img)

    # load & convert ground truth
    gt_arr = np.load(gt_path)
    if gt_arr.ndim == 3 and gt_arr.shape[2] == 3:
        summed = gt_arr.sum(axis=2)
        gt_mask = np.argmax(gt_arr, axis=2)
        gt_mask[summed == 0] = 3
    else:
        gt_mask = gt_arr

    # load model
    device = get_device()
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=4
    )
    weights = torch.load(finetuned_weights, map_location=device)
    model.load_state_dict(weights)
    model.to(device).eval()

    # preprocess
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        logits = model(input_tensor).logits
        logits = F.interpolate(
            logits,
            size=(input_size[1], input_size[0]),
            mode="bilinear",
            align_corners=False
        )
        pred_mask = logits.argmax(dim=1)[0].cpu().numpy()

    # overlays
    overlay_gt   = get_seg_overlay(img_np, gt_mask,   overlay_opacity=0.9)
    overlay_pred = get_seg_overlay(img_np, pred_mask, overlay_opacity=0.9)

    # plot results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(overlay_gt)
    axes[0].set_title("Ground Truth Overlay")
    axes[0].axis("off")

    axes[1].imshow(overlay_pred)
    axes[1].set_title("Prediction Overlay")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    # save figure
    out_path = os.path.join(output_dir, "segformer_single_example.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved overlay figure to {out_path}")

if __name__ == "__main__":
    main()
