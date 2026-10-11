---
name: note-taking
description: Standard operating procedure for creating and maintaining chronological research and experiment notes in notes/ and updating the main project report.
---

# Note-Taking & Documentation Skill

## 1. Directory Structure & File Naming

- **Dizin:** Tüm notlar `notes/` altında saklanır.
- **Dosya İsimlendirme:** Kronolojik sıralama için mutlaka tarih ve saat-dakika içermelidir:
  - Format: `YYYY-MM-DD_HH-MM_<topic>.md`
  - Örnek: `2026-10-10_16-29_gan_ve_face_reenactment.md`
- **Asset Dizini:** Notlarda kullanılan yerel görseller `notes/assets/` altına kaydedilir.

## 2. Note Structure & Metadata

- Her notun en başında başlık ve hemen altında tarih/saat bilgisi yer almalıdır:
  ```markdown
  # <Konu Başlığı>

  **Tarih / Saat:** YYYY-MM-DD HH:MM
  ```

## 3. Hard Brevity & Language Constraints

- **Dil:** Not metni Türkçe yazılacaktır.
- **Teknik Terminoloji:** Tüm teknik terimler ve konseptler orijinal İngilizce formlarında bırakılacaktır (örn. _generator_, _discriminator_, _keypoint_, _dense flow_, _occlusion mask_, _warp_, _inpainting_, _latent space_).
- **Uzunluk Sınırı:** Bir not en fazla **50 satır** olabilir. Gereksiz dolgu metinlerinden kaçınılmalı, taranabilir ve maddeli tutulmalıdır.
- **Meta Content Yasağı:** Not içine soru, model monoloğu, prompt kalıntıları veya "bu notta şunlar anlatılmaktadır" gibi meta yorumlar yazılmayacaktır.

## 4. Visuals, Diagrams & Citations

- **Diyagramlar ve Grafikler (Akademik Rapor SVG Standardı):**
  - Tüm mimari ve akış şemaları (Mermaid diyagramları dahil) raporda ve slaytlarda kullanılmak üzere **SVG (`.svg`) formatında** üretilmelidir.
  - **Tasarım Stili:** Şemalar aşırı süslü gradyanlar ve gölgeler yerine **sade ve klasik akademik makale/rapor formatında** olmalıdır (temiz siyah/koyu gri kenarlıklar, beyaz veya çok açık nötr dolgular, sade oklar ve net tipografi).
  - Üretilen tüm SVG dosyaları `notes/assets/` dizinine kaydedilmeli ve not içerisine `![Açıklama](assets/<dosya>.svg)` şeklinde eklenmelidir.
  - Şema metinleri uluslararası akademik standart gereği İngilizce terminolojiyle yazılmalıdır.
- **Görsel Kullanımı:** Kavramları, mimariyi veya deneysel sonuçları somutlaştırmak için gerekli yerlerde görsel eklenmelidir.
- **Yerel Saklama:** Dış kaynaklı görseller link kopmasını önlemek için `notes/assets/` klasörüne indirilip oradan göreceli yol ile (`assets/<dosya>`) çağrılmalıdır.
- **Zorunlu Kaynakça:** Kullanılan her görselin altına mutlaka açık kaynak referansı (_Paper adı, yazar veya proje bağlantısı_) yazılmalıdır.
