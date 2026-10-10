#!/usr/bin/env python3
"""
Face Reenactment (FOMM) Pipeline Runner
Tek ve standart Python aracı.

Kullanım:
  python run.py bella_poarch
  python run.py --all
"""

import os
import sys

# 1. Sanal ortamı (fomm_env) otomatik olarak devreye al
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, 'fomm_env')
VENV_PYTHON = os.path.join(VENV_DIR, 'bin', 'python')

if os.path.exists(VENV_PYTHON) and sys.prefix != VENV_DIR:
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

# 2. Ortam ve uyarı ayarları
os.environ['MPLCONFIGDIR'] = os.environ.get('MPLCONFIGDIR', '/tmp/matplotlib_cache')
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 3. FOMM modüllerini içe aktar
FOMM_DIR = os.path.join(BASE_DIR, 'fomm')
if FOMM_DIR not in sys.path:
    sys.path.insert(0, FOMM_DIR)

import argparse
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch

from demo import load_checkpoints, make_animation


def generate_side_by_side(source_img, driving_frames, pred_frames):
    """
    [Source | Driving | Reenacted] yan yana karşılaştırma videosu üretir.
    """
    combined_frames = []
    h, w, c = pred_frames[0].shape
    src_resized = resize(source_img, (h, w))[..., :3]

    for d_frame, p_frame in zip(driving_frames, pred_frames):
        d_resized = resize(d_frame, (h, w))[..., :3]
        combined = np.concatenate([src_resized, d_resized, p_frame], axis=1)
        combined_frames.append(img_as_ubyte(combined))

    return combined_frames


def process_pair(folder_name):
    """Belirtilen girdi klasörü için hem normal hem karşılaştırma videosu üretir."""
    inputs_dir = os.path.join(BASE_DIR, 'inputs', folder_name)
    outputs_dir = os.path.join(BASE_DIR, 'outputs', folder_name)

    source_path = os.path.join(inputs_dir, 'source.png')
    driving_path = os.path.join(inputs_dir, 'driving.mp4')
    checkpoint_path = os.path.join(FOMM_DIR, 'vox-cpk.pth.tar')
    config_path = os.path.join(FOMM_DIR, 'config', 'vox-256.yaml')

    if not os.path.exists(inputs_dir):
        print(f"[!] Hata: 'inputs/{folder_name}' klasörü bulunamadı.")
        return
    if not os.path.exists(source_path):
        print(f"[!] Hata: '{source_path}' bulunamadı.")
        return
    if not os.path.exists(driving_path):
        print(f"[!] Hata: '{driving_path}' bulunamadı.")
        return
    if not os.path.exists(checkpoint_path):
        print(f"[!] Hata: Checkpoint bulunamadı: '{checkpoint_path}'")
        return

    os.makedirs(outputs_dir, exist_ok=True)
    print(f"\n==========================================")
    print(f"[+] İşleniyor: {folder_name}")
    print(f"    - Kaynak:  {source_path}")
    print(f"    - Sürücü:  {driving_path}")
    print(f"    - Çıktı:   {outputs_dir}")
    print(f"==========================================")

    # Verileri oku
    source_image = imageio.imread(source_path)
    reader = imageio.get_reader(driving_path)
    fps = reader.get_meta_data().get('fps', 30)

    driving_video = []
    try:
        for frame in reader:
            driving_video.append(frame)
    except RuntimeError:
        pass
    reader.close()

    total_frames = len(driving_video)
    print(f"[*] Toplam {total_frames} kare işlenecek (FPS: {fps:.2f})...")

    # 256x256 boyutuna getir
    source_resized = resize(source_image, (256, 256))[..., :3]
    driving_resized = [resize(f, (256, 256))[..., :3] for f in driving_video]

    # Model yükle (CPU)
    print("[*] Model yükleniyor (vox-cpk.pth.tar)...")
    generator, kp_detector = load_checkpoints(config_path=config_path, checkpoint_path=checkpoint_path, cpu=True)

    # İnference
    print("[*] Kareler üretiliyor...")
    predictions = make_animation(
        source_image=source_resized,
        driving_video=driving_resized,
        generator=generator,
        kp_detector=kp_detector,
        relative=True,
        adapt_movement_scale=True,
        cpu=True
    )

    # 1. Normal sonuç videosu
    result_path = os.path.join(outputs_dir, 'result.mp4')
    pred_uint8 = [img_as_ubyte(f) for f in predictions]
    imageio.mimsave(result_path, pred_uint8, fps=fps)
    print(f"[✓] Normal çıktı hazır: {result_path}")

    # 2. Yan yana karşılaştırma videosu (Source | Driving | Result)
    comparison_path = os.path.join(outputs_dir, 'comparison.mp4')
    comp_frames = generate_side_by_side(source_image, driving_video, predictions)
    imageio.mimsave(comparison_path, comp_frames, fps=fps)
    print(f"[✓] Karşılaştırma videosu hazır: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description="Tek ve standart Face Reenactment çalıştırıcısı")
    parser.add_argument('target', nargs='?', default=None,
                        help="inputs/ altındaki klasör adı (örn: bella_poarch)")
    parser.add_argument('--all', action='store_true', help="inputs/ altındaki tüm klasörleri sırayla çalıştır")

    args = parser.parse_args()
    inputs_root = os.path.join(BASE_DIR, 'inputs')

    if args.all:
        folders = [f for f in sorted(os.listdir(inputs_root)) if os.path.isdir(os.path.join(inputs_root, f))]
        for folder in folders:
            process_pair(folder)
    elif args.target:
        process_pair(args.target)
    else:
        folders = [f for f in sorted(os.listdir(inputs_root)) if os.path.isdir(os.path.join(inputs_root, f))]
        print("[*] Kullanım:")
        print("    python run.py <klasor_adi>")
        print("    python run.py --all")
        print(f"\nMevcut girdi klasörleri: {folders}")


if __name__ == '__main__':
    main()
