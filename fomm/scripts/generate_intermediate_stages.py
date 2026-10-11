#!/usr/bin/env python3
"""
Generate updated intermediate stages visualization with the new Bella Poarch source image.
Produces a clean 5-panel comparison:
1. Source & Keypoints
2. Driving & Keypoints (Frame 286 / Smile)
3. Deformed (Warped Features)
4. Occlusion Mask (Grayscale, dark = unobserved/missing)
5. Generator Inpainting Output
"""

import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import imageio.v2 as imageio
from skimage.transform import resize
from skimage.draw import disk
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
FOMM_DIR = BASE_DIR
sys.path.insert(0, FOMM_DIR)

from demo import load_checkpoints
from animate import normalize_kp

def draw_kp(img, kp_array, kp_size=4):
    img = np.copy(img)
    h, w = img.shape[:2]
    # kp_array in [-1, 1]
    spatial_size = np.array([w, h])[np.newaxis]
    kp_pixel = spatial_size * (kp_array + 1) / 2.0
    
    # 10 distinct, elegant colors
    colors = [
        (230, 57, 70), (244, 162, 97), (233, 196, 106), (42, 157, 143), (38, 70, 83),
        (88, 129, 87), (58, 90, 64), (163, 177, 138), (142, 202, 230), (33, 158, 188)
    ]
    for idx, pt in enumerate(kp_pixel):
        r, c = int(round(pt[1])), int(round(pt[0]))
        color = colors[idx % len(colors)]
        rr, cc = disk((r, c), kp_size, shape=(h, w))
        img[rr, cc] = color
        # White rim
        rr_rim, cc_rim = disk((r, c), kp_size + 1, shape=(h, w))
        # only borders
        mask_border = np.zeros((h, w), dtype=bool)
        mask_border[rr_rim, cc_rim] = True
        mask_border[rr, cc] = False
        img[mask_border] = (255, 255, 255)
    return img

def main():
    config_path = os.path.join(FOMM_DIR, 'config', 'vox-256.yaml')
    cpk_path = os.path.join(FOMM_DIR, 'vox-cpk.pth.tar')
    
    generator, kp_detector = load_checkpoints(config_path, cpk_path, cpu=True)
    
    src_path = os.path.join(BASE_DIR, 'inputs', 'bella_poarch', 'source.png')
    drv_path = os.path.join(BASE_DIR, 'inputs', 'bella_poarch', 'driving.mp4')
    
    src = imageio.imread(src_path)
    src_256 = (resize(src, (256, 256))[..., :3] * 255).astype(np.uint8)
    
    reader = imageio.get_reader(drv_path)
    frames = [f for f in reader]
    reader.close()
    
    # Target frame: frame 286 (Smile)
    target_idx = 286
    drv_f0_256 = (resize(frames[0], (256, 256))[..., :3] * 255).astype(np.uint8)
    drv_target_256 = (resize(frames[target_idx], (256, 256))[..., :3] * 255).astype(np.uint8)
    
    with torch.no_grad():
        s_t = torch.tensor((src_256 / 255.0)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        d0_t = torch.tensor((drv_f0_256 / 255.0)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        dt_t = torch.tensor((drv_target_256 / 255.0)[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        
        kp_s = kp_detector(s_t)
        kp_d0 = kp_detector(d0_t)
        kp_dt = kp_detector(dt_t)
        
        kp_norm = normalize_kp(kp_source=kp_s, kp_driving=kp_dt, kp_driving_initial=kp_d0,
                               use_relative_movement=True, use_relative_jacobian=True, adapt_movement_scale=True)
        out = generator(s_t, kp_source=kp_s, kp_driving=kp_norm)
        
        # 1. Source with keypoints
        kp_s_vals = kp_s['value'][0].cpu().numpy()
        p1 = draw_kp(src_256, kp_s_vals)
        
        # 2. Driving with keypoints
        kp_dt_vals = kp_dt['value'][0].cpu().numpy()
        p2 = draw_kp(drv_target_256, kp_dt_vals)
        
        # 3. Deformed source
        deformed = out['deformed'][0].permute(1, 2, 0).cpu().numpy()
        p3 = np.clip(deformed * 255, 0, 255).astype(np.uint8)
        
        # 4. Occlusion map (clean grayscale: 0 is black = occluded/missing, 1 is white = visible)
        occ = out['occlusion_map'][0, 0].cpu().numpy()
        occ_res = resize(occ, (256, 256), order=1)
        occ_uint8 = np.clip(occ_res * 255, 0, 255).astype(np.uint8)
        p4 = np.stack([occ_uint8, occ_uint8, occ_uint8], axis=-1)
        
        # 5. Prediction (Inpainted)
        pred = out['prediction'][0].permute(1, 2, 0).cpu().numpy()
        p5 = np.clip(pred * 255, 0, 255).astype(np.uint8)
        
    panels = [p1, p2, p3, p4, p5]
    titles = [
        "1. Source & Keypoints",
        f"2. Driving Frame & Keypoints (F:{target_idx})",
        "3. Deformed (Feature Warping)",
        "4. Occlusion Mask (Dark=Missing)",
        "5. Inpainting Output"
    ]
    
    # Render with nice typography and clean border
    panel_size = 512
    margin_top = 60
    margin_side = 10
    total_w = panel_size * 5 + margin_side * 6
    total_h = panel_size + margin_top + 20
    
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    except:
        font = ImageFont.load_default()
        
    for i, (p, title) in enumerate(zip(panels, titles)):
        im_p = Image.fromarray(p).resize((panel_size, panel_size), Image.LANCZOS)
        x = margin_side + i * (panel_size + margin_side)
        y = margin_top
        
        # Draw header text centered over panel
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        tx = x + (panel_size - tw) // 2
        ty = 18
        draw.text((tx, ty), title, fill=(30, 41, 59), font=font)
        
        # Paste image
        canvas.paste(im_p, (x, y))
        
        # Thin border around panel
        draw.rectangle([x, y, x + panel_size - 1, y + panel_size - 1], outline=(203, 213, 225), width=1)
        
    # Save outputs
    out_paths = [
        os.path.join(ROOT_DIR, 'docs', 'slides', 'figures', 'ara_asamalar_gorsellestirme.png'),
        os.path.join(BASE_DIR, 'notes', 'assets', 'ara_asamalar_gorsellestirme.png')
    ]
    for op in out_paths:
        canvas.save(op, quality=95)
        print(f"[✓] Saved updated intermediate visualization: {op}")

if __name__ == '__main__':
    main()
