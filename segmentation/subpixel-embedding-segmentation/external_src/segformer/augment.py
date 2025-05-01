import os
import random
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
from tqdm import tqdm

# overlay function for displaying segmentation results
def get_seg_overlay(image_np, seg_np, overlay_opacity=0.5):
    """
    - image_np: HxWx3 uint8 background
    - seg_np: HxW int labels 0,1,2 (classes) or 3 (transparent background)
    - overlay_opacity: opacity of the overlay (0-1.0)
    Classes 0-2 are overlaid at 70% opacity; class 3 is left unchanged.
    """
    palette = {
        0: np.array([255,   0,   0], dtype=np.uint8),
        1: np.array([  0, 255,   0], dtype=np.uint8),
        2: np.array([  0,   0, 255], dtype=np.uint8),
    }
    overlay = image_np.copy().astype(np.float32)
    alpha = overlay_opacity
    for lbl, color in palette.items():
        mask = (seg_np == lbl)
        overlay[mask] = (alpha * color + (1 - alpha) * image_np[mask]).astype(np.uint8)
    return overlay.astype(np.uint8)

# Augmentation functions
def augment_basic(img, mask):
    if random.random() > 0.5:
        img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
    ang = random.uniform(-15, 15)
    img  = img.rotate(ang,   resample=Image.BILINEAR)
    mask = mask.rotate(ang,   resample=Image.NEAREST, fillcolor=3)
    bf, cf, sf = [random.uniform(0.8, 1.2) for _ in range(3)]
    img = ImageEnhance.Brightness(img).enhance(bf)
    img = ImageEnhance.Contrast(img).enhance(cf)
    img = ImageEnhance.Color(img).enhance(sf)
    return img, mask

def augment_random_crop(img, mask, frac=0.8):
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
    img_c  = img.crop((x0, y0, x0+cw, y0+ch)).resize((w, h), Image.BILINEAR)
    mask_c = mask.crop((x0, y0, x0+cw, y0+ch)).resize((w, h), Image.NEAREST)
    return img_c, mask_c

def augment_gaussian_blur(img, mask, r_range=(1, 3)):
    return img.filter(ImageFilter.GaussianBlur(random.uniform(*r_range))), mask

def augment_noise(img, mask, lvl=25):
    arr   = np.array(img).astype(np.int16)
    noise = np.random.randn(*arr.shape) * lvl
    img_n = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(img_n), mask

def augment_shear(img, mask, max_s=0.3):
    w, h = img.size
    m = random.uniform(-max_s, max_s)
    mat = (1, m, 0, 0, 1, 0)
    img_s  = img.transform((w, h), Image.AFFINE,  mat, resample=Image.BILINEAR)
    mask_s = mask.transform((w, h), Image.AFFINE, mat, resample=Image.NEAREST, fillcolor=3)
    return img_s, mask_s

def find_coeffs(dst, src):
    A, B = [], []
    for (xd, yd), (xs, ys) in zip(dst, src):
        A.append([xd, yd, 1, 0, 0, 0, -xs*xd, -xs*yd])
        A.append([0, 0, 0, xd, yd, 1, -ys*xd, -ys*yd])
        B.extend([xs, ys])
    return np.linalg.solve(np.array(A, float), np.array(B, float)).tolist()

def augment_perspective(img, mask, max_d=0.1):
    w, h = img.size
    src = [(0,0),(w,0),(w,h),(0,h)]
    dst = [(x + random.uniform(-max_d*w, max_d*w),
            y + random.uniform(-max_d*h, max_d*h)) for x,y in src]
    coeffs = find_coeffs(dst, src)
    img_p  = img.transform((w,h), Image.PERSPECTIVE,  coeffs, resample=Image.BILINEAR)
    mask_p = mask.transform((w,h),Image.PERSPECTIVE, coeffs, resample=Image.NEAREST, fillcolor=3)
    return img_p, mask_p

new_augs = [
    augment_random_crop,
    augment_gaussian_blur,
    augment_noise,
    augment_shear,
    augment_perspective,
]

def main():
    p = argparse.ArgumentParser(description="CLI augmentation script for segmentation data")
    p.add_argument("--scan-file",     type=str, required=True, help="Input train_scans.txt")
    p.add_argument("--mask-file",     type=str, required=True, help="Input train_ground_truths.txt")
    p.add_argument("--scan-aug-file", type=str, required=True, help="Output augmented scans list")
    p.add_argument("--mask-aug-file", type=str, required=True, help="Output augmented masks list")
    p.add_argument("--num-basic",     type=int, default=5,   help="Number of basic augs per image")
    p.add_argument("--num-new",       type=int, default=5,   help="Number of new augs per image")
    p.add_argument("--show-images",   action="store_true",   help="Display overlays for one sample")
    p.add_argument("--display-index", type=int, default=4,   help="Sample index to display (0-based)")
    args = p.parse_args()

    # load file lists
    with open(args.scan_file) as f: orig_imgs = [l.strip() for l in f if l.strip()]
    with open(args.mask_file) as f: orig_masks= [l.strip() for l in f if l.strip()]
    assert len(orig_imgs) == len(orig_masks), "Mismatch images/masks"

    # write & augment
    with open(args.scan_aug_file, "w") as so, open(args.mask_aug_file, "w") as mo:
        random.seed(0)
        for img_p, m_p in tqdm(zip(orig_imgs, orig_masks),
                              total=len(orig_imgs),
                              desc="Augmenting"):
            so.write(img_p+"\n"); mo.write(m_p+"\n")
            img0 = Image.open(img_p).convert("RGB").resize((640,480), Image.BILINEAR)
            arr  = np.load(m_p)
            if arr.ndim==3 and arr.shape[2]==3:
                s = arr.sum(axis=2); lm = np.argmax(arr,axis=2); lm[s==0]=3
            else:
                lm = arr
            mask0 = Image.fromarray(lm.astype(np.int32),mode="I")\
                         .resize((640,480),Image.NEAREST)
            bi, ei = os.path.splitext(img_p)
            bm, em = os.path.splitext(m_p)

            # basic
            for j in range(1, args.num_basic+1):
                ai, am = augment_basic(img0, mask0)
                ni, nm = f"{bi}_augment{j}{ei}", f"{bm}_augment{j}{em}"
                ai.save(ni); np.save(nm, np.array(am,dtype=np.int32))
                so.write(ni+"\n"); mo.write(nm+"\n")

            # new
            for k, fn in enumerate(new_augs, start=args.num_basic+1):
                ai, am = fn(img0, mask0)
                ni, nm = f"{bi}_augment{k}{ei}", f"{bm}_augment{k}{em}"
                ai.save(ni); np.save(nm, np.array(am,dtype=np.int32))
                so.write(ni+"\n"); mo.write(nm+"\n")

    print(f"Augmentation lists saved to:\n  {args.scan_aug_file}\n  {args.mask_aug_file}")

    if args.show_images:
        idx = args.display_index
        oi, om = orig_imgs[idx], orig_masks[idx]
        bi, ei = os.path.splitext(oi)
        bm, em = os.path.splitext(om)
        img0 = Image.open(oi).convert("RGB").resize((640,480), Image.BILINEAR)
        arr0 = np.load(om)
        if arr0.ndim==3 and arr0.shape[2]==3:
            s0 = arr0.sum(axis=2); lm0 = np.argmax(arr0,axis=2); lm0[s0==0]=3
        else:
            lm0 = arr0
        img_np0 = np.array(img0)
        overlays = [get_seg_overlay(img_np0, lm0)]
        titles   = ["Original"]
        total    = args.num_basic + args.num_new
        for j in range(1, total+1):
            ai = Image.open(f"{bi}_augment{j}{ei}").resize((640,480),Image.BILINEAR)
            am = np.load(f"{bm}_augment{j}{em}")
            overlays.append(get_seg_overlay(np.array(ai), am))
            titles.append(f"Augmentation {j}")

        fig, axs = plt.subplots(1, len(overlays), figsize=(3*len(overlays),4))
        for ax, ov, t in zip(axs, overlays, titles):
            ax.imshow(ov); ax.set_title(t); ax.axis("off")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
