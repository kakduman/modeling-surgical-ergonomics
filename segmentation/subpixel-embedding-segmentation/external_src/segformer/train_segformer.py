#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from transformers import SegformerForSemanticSegmentation

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=(640, 480), transform=None):
        assert len(image_paths) == len(mask_paths), "Images/masks count mismatch"
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.image_size  = image_size
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # load & resize image
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize(self.image_size, resample=Image.BILINEAR)

        # load mask → label_map
        arr = np.load(self.mask_paths[idx])
        if arr.ndim == 3 and arr.shape[2] == 3:
            psum = arr.sum(axis=2)
            lm   = np.argmax(arr, axis=2).astype(np.int64)
            lm[psum == 0] = 3
        else:
            lm = arr.astype(np.int64)
        mask_pil = Image.fromarray(lm.astype(np.int32), mode="I")
        mask_pil = mask_pil.resize(self.image_size, resample=Image.NEAREST)
        label_map = np.array(mask_pil, dtype=np.int64)

        # image → tensor
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(img)

        return img, torch.as_tensor(label_map, dtype=torch.long)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS detected, falling back to CPU for stability.")
        return torch.device("cpu")
    else:
        return torch.device("cpu")

def parse_args():
    p = argparse.ArgumentParser(description="Train SegFormer on augmented segmentation data")
    p.add_argument("--scan-aug-file",   type=str, default="train_scans_augmented.txt",
                   help="List of augmented image paths")
    p.add_argument("--mask-aug-file",   type=str, default="train_ground_truths_augmented.txt",
                   help="List of augmented mask paths")
    p.add_argument("--model-checkpoint",type=str, default="nvidia/mit-b2",
                   help="HuggingFace model checkpoint")
    p.add_argument("--val-size",        type=int, default=99,
                   help="Number of samples for validation (from top of list)")
    p.add_argument("--batch-size",      type=int, default=8)
    p.add_argument("--num-epochs",      type=int, default=10)
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--step-size",       type=int, default=1,
                   help="Scheduler step size (epochs)")
    p.add_argument("--gamma",           type=float, default=0.7,
                   help="Scheduler gamma")
    p.add_argument("--num-classes",     type=int, default=4,
                   help="Number of segmentation classes")
    p.add_argument("--num-workers",     type=int, default=0)
    return p.parse_args()

def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    # load augmented lists
    with open(args.scan_aug_file) as f:
        scans = [l.strip() for l in f if l.strip()]
    with open(args.mask_aug_file) as f:
        masks = [l.strip() for l in f if l.strip()]
    assert len(scans) == len(masks), "Mismatch between scans and masks"

    # split train/val
    val_size = args.val_size
    assert val_size < len(scans), f"val-size {val_size} >= dataset size {len(scans)}"
    val_scans, val_masks = scans[:val_size], masks[:val_size]
    tr_scans,  tr_masks  = scans[val_size:], masks[val_size:]

    # transforms
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # datasets & loaders
    train_ds = SegmentationDataset(tr_scans, tr_masks, transform=img_transform)
    val_ds   = SegmentationDataset(val_scans, val_masks, transform=img_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)

    # model, optimizer, scheduler, loss
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_checkpoint,
        num_labels=args.num_classes
    )
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_size, gamma=args.gamma
    )

    # training loop
    for epoch in range(1, args.num_epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for imgs, masks_t in train_loader:
            imgs, masks_t = imgs.to(device), masks_t.to(device)
            optimizer.zero_grad()
            logits = model(imgs).logits
            logits = nn.functional.interpolate(
                logits, size=masks_t.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = criterion(logits, masks_t)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks_t in val_loader:
                imgs, masks_t = imgs.to(device), masks_t.to(device)
                logits = model(imgs).logits
                logits = nn.functional.interpolate(
                    logits, size=masks_t.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = criterion(logits, masks_t)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_ds)

        # step scheduler
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch [{epoch}/{args.num_epochs}] "
              f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  LR: {lr:.2e}")

    # save final model
    out_dir = "trained_models/segformer"
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "segformer_finetuned.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
