# Diş Teli Deneyi ve Hata Analizi (Failure Mode)
**Tarih / Saat:** 2026-10-10 18:36

## 1. Deney Amacı ve Hipotez
- **Amaç:** Kaynak görselde sıra dışı/nadir bir detay (diş teli) varken ağız açılıp kapandığında modelin dokuyu nasıl koruduğunu test etmek.
- **Hipotez:** Eğitim veri setinde (VoxCeleb) diş teli bulunmadığı için model, inpainting devreye girdiğinde teli silip standart diş dokusu çizecektir.

## 2. Zaman Çizelgesi ve Görsel Kanıt (3x3 Grid)

![Diş Teli Kaybolma Analizi](assets/dis_teli_kaybolma_analizi.png)
*Görsel Kaynağı: Yerel deney çıkarımı (outputs/bella_poarch_dis_teli/result.mp4, Frame 10, 195 ve 288 karşılaştırması)*

## 3. Gözlemler ve Teknik Hata Analizi

1. **Kare Bağımsızlığı (Stateless Pipeline):** FOMM kareler arasında yinelemeli (recurrent) bellek taşımaz. Her kare yalnızca kaynak görsel ($S$) ve o anki sürüş karesi ($D_t$) ile üretilir; ara karelerde ağzın kapanması sonraki kareleri doğrudan etkilemez.
2. **Kare Bazında Warping vs. Inpainting Farkı:**
   - **F:10 (0.3s) — Başarılı Warping ($O \approx 1$):** Sürüş mimiği ilk kareye yakındır; dense motion network kaynak diş piksellerini hedef geometriye eşleştirebilir. Kaynaktaki gerçek pikseller taşınır ve diş telleri korunur.
   - **F:195 (6.5s) — Ağız Kapanması:** Sürüş karesinde dudaklar kapandığı için dişler doğal olarak görünmez.
   - **F:288 (9.6s) — Inpainting Devreye Girişi ($O \approx 0$):** Sürüş yüzünde aşırı ve farklı açılı bir gülümseme oluşur. Model kaynak ağız geometrisini bu aşırı deformasyona eşleştiremeyince ağız içini eksik alan ($O \approx 0$) olarak maskeler.
3. **Diş Telinin Silinme Nedeni (Dataset Bias):**
   - Occlusion mask $O \approx 0$ olduğunda kaynak pikseller terk edilir; Generator ağız içini sıfırdan çizer (*Inpainting*).
   - Eğitim seti (VoxCeleb) diş telli yüz içermediği için Generator'ın ön-bilgisi (prior) standart beyaz diş üretir ve telleri yok eder.

## 4. Rapor İçin Çıkarım (Teknik Sonuç)
- **Warping** kaynaktaki nadir kimlik detaylarını korurken, **Inpainting** çıktıyı eğitim setinin ortalamasına zorlar.
- FOMM'da zamansal bellek (temporal memory) veya kimlik önbelleği (identity cache) bulunmadığı için, inpainting tetiklendiği her anda kaynaktaki nadir kişisel detaylar kaybolur.
