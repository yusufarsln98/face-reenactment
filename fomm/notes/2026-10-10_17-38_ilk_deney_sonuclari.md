# İlk Yerel FOMM Deneyi (Bella Poarch Verisi)
**Tarih / Saat:** 2026-10-10 17:38

## 1. Deney Kurulumu ve Girdiler
- **Model:** First Order Motion Model (VoxCeleb önceden eğitilmiş ağırlıklar: `vox-cpk.pth.tar`).
- **Kaynak (Source Image):** `inputs/bella_poarch/source.png` (256x256).
- **Sürücü (Driving Video):** `inputs/bella_poarch/driving.mp4` (320 kare, ~30 FPS).
- **Donanım:** Apple Silicon (CPU modu).

## 2. Çalıştırma Parametreleri ve Anlamları
```bash
python demo.py --config config/vox-256.yaml \
  --checkpoint vox-cpk.pth.tar \
  --source_image source.png --driving_video driving.mp4 \
  --relative --adapt_scale --cpu \
  --result_video output_bella.mp4
```
- `--relative`: Mutlak koordinatlar yerine başlangıç karesine göre yer değiştirme farkını ($\Delta$) aktarır.
- `--adapt_scale`: Yüz boyut farklılıklarını dışbükey alan (convex hull) oranlamasıyla dengeler.
- `--cpu`: Donanım uyumluluğu için CPU üzerinde çıkarım yapar.

## 3. Sonuçlar ve Performans
- **İşlenen Kare:** 320 / 320 kare başarıyla sentezlendi.
- **Hız:** Ortalama ~2.05 it/s (~2.5 dakika toplam süre).
- **Çıktı:** `output_bella.mp4` (320 kare, 29.97 FPS, 10.7 saniye video).
- **Gözlem:** Baş rotasyonları ve göz kırpma hareketleri pürüzsüz aktarıldı; ağız hareketlerinde occlusion inpainting devreye girdi.

