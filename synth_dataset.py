# core/synth_dataset.py
"""
Generate synthetic per-square images using templates in your templates/ folder.

Output layout:
data/synth/
  empty/
  wP/
  wN/
  ...
  bK/

Run:
python core/synth_dataset.py --out_dir data/synth --samples_per_class 2000
"""

import os
import cv2
import numpy as np
import argparse
import random
from glob import glob

def random_affine(img, max_rot=15, scale_range=(0.8,1.1), translate=0.05):
    h, w = img.shape[:2]
    # rotation + scale
    angle = random.uniform(-max_rot, max_rot)
    scale = random.uniform(scale_range[0], scale_range[1])
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    # translation
    tx = random.uniform(-translate, translate) * w
    ty = random.uniform(-translate, translate) * h
    M[:,2] += (tx, ty)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out

def random_brightness_contrast(img):
    alpha = random.uniform(0.8, 1.2)  # contrast
    beta = random.uniform(-20, 20)    # brightness
    out = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return out

def overlay_piece(bg, piece_rgba):
    # piece_rgba assumed RGBA; bg is BGR
    h, w = bg.shape[:2]
    # resize piece randomly to 40-90% of square
    scale = random.uniform(0.45, 0.9)
    pw = int(w * scale)
    ph = int(h * scale)
    piece = cv2.resize(piece_rgba, (pw, ph), interpolation=cv2.INTER_AREA)

    # random position (center +- jitter)
    x = (w - pw) // 2 + random.randint(-int(0.05*w), int(0.05*w))
    y = (h - ph) // 2 + random.randint(-int(0.05*h), int(0.05*h))

    # split
    if piece.shape[2] == 4:
        alpha = piece[:, :, 3] / 255.0
        alpha = alpha[..., None]
        fg = piece[:, :, :3][:, :, ::-1]  # RGBA -> BGR
        bg_patch = bg[y:y+ph, x:x+pw].astype(float)
        comp = fg.astype(float) * alpha + bg_patch * (1 - alpha)
        bg[y:y+ph, x:x+pw] = comp.astype(np.uint8)
    else:
        # no alpha channel: simple overlay
        bg[y:y+ph, x:x+pw] = piece[:, :, ::-1]
    return bg

def generate(out_dir="data/synth", templates_dir="templates", samples_per_class=1000, size=64):
    os.makedirs(out_dir, exist_ok=True)
    # design background variations: textured / plain colors
    bg_colors = [(240,240,240), (200,200,180), (220,220,220), (255,255,255)]
    # load piece templates
    template_paths = glob(os.path.join(templates_dir, "*.*"))
    pieces = {}
    for p in template_paths:
        name = os.path.splitext(os.path.basename(p))[0]  # wP, bK...
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        pieces[name] = img

    CLASS_NAMES = ["empty"] + sorted([k for k in pieces.keys()])

    # ensure directories
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    for cls in CLASS_NAMES:
        for i in range(samples_per_class):
            # base background
            bg = np.full((size, size, 3), random.choice(bg_colors), dtype=np.uint8)
            # optionally add subtle noise / texture
            if random.random() < 0.6:
                noise = np.random.normal(0, 6, (size, size, 1)).astype(np.int16)
                bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

            if cls != "empty":
                # overlay corresponding piece
                piece = pieces.get(cls)
                if piece is None:
                    continue
                # apply random transforms on piece first
                piece_t = piece.copy()
                piece_t = random_affine(piece_t, max_rot=10, scale_range=(0.9,1.05))
                piece_t = random_brightness_contrast(piece_t)
                img = overlay_piece(bg.copy(), piece_t)
            else:
                # empty: maybe add small artifacts to mimic noise / board specks
                img = bg.copy()
                if random.random() < 0.3:
                    # draw faint circle or line
                    cv2.line(img, (random.randint(0,size), random.randint(0,size)),
                             (random.randint(0,size), random.randint(0,size)), (random.randint(180,240),)*3, 1)
            # optional final augment
            if random.random() < 0.5:
                img = cv2.GaussianBlur(img, (3,3), 0)

            path = os.path.join(out_dir, cls, f"{cls}_{i:05d}.png")
            cv2.imwrite(path, img)
    print("Synthetic dataset generated at:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/synth")
    parser.add_argument("--templates_dir", default="templates")
    parser.add_argument("--samples_per_class", type=int, default=1000)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    generate(args.out_dir, args.templates_dir, args.samples_per_class, args.size)
