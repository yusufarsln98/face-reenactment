# Diş Teli Deneyi ve Hata Analizi (Failure Mode)
**Tarih / Saat:** 2026-10-10 18:36

## 1. Deney Amacı ve Hipotez
- **Amaç:** Kaynak görselde sıra dışı/nadir bir detay (diş teli) varken ağız açılıp kapandığında modelin dokuyu nasıl koruduğunu test etmek.
- **Hipotez:** Eğitim veri setinde (VoxCeleb) diş teli bulunmadığı için model, inpainting devreye girdiğinde teli silip standart diş dokusu çizecektir.

## 2. Zaman Çizelgesi ve Görsel Kanıt (3x3 Grid)

![Diş Teli Kaybolma Analizi](assets/dis_teli_kaybolma_analizi.png)
*Görsel Kaynağı: Yerel deney çıkarımı (outputs/bella_poarch_dis_teli/result.mp4, Frame 10, 195 ve 288 karşılaştırması)*

## 3. Gözlemler ve Teknik Hata Analizi

1. **Ağız Kapanma Uyuşmazlığı:** Sürüş videosunda Bella'nın ağzı kapalı olmasına rağmen, kaynak fotoğrafta çok geniş bir gülümseme olduğu için model ağzı videonun büyük kısmında tam kapatamaz; hafif açık bırakır.
2. **Kapanıp Açılınca Telin Kaybolması (10. Saniye - F:288):**
   - **F:0 - F:40:** İlk karelerde diş telleri ve metal braketler net olarak taşınır (Feature Warping).
   - **F:100 - F:200:** Ağız büzülüp kapandıkça modelin deformasyon alanı değişir.
   - **F:288 (10. Saniyenin Başı):** Ağız yeniden belirgin şekilde açılıp gülümsediğinde **diş telleri tamamen yok olur**. Metal braketler silinmiş, yerini düz beyaz diş dokusu almıştır.

## 4. Rapor İçin Çıkarım (Failure Mode Sebebi)
FOMM kareler arası hafızaya sahip değildir. Occlusion mask yeni açılan ağız bölgesini "eksik veri" olarak işaretlediğinde, Generator veri setindeki genel yüz ön-bilgisine (prior bias) başvurur ve teli unutup standart diş çizer (**Inpainting Artifact**).
