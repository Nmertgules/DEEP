# Digital Image Processing — Week 9: Color Spaces

**Group members: Nmertgules and Iliya**

Renk uzayları (RGB, HSV, HSI, YUV/YCbCr, CIE Lab); 8K görüntüde piksel işlemlerinin performansı (NumPy, Python döngüsü, C++); ten segmentasyonu; web kamerasında hareket tabanlı ROI içinde ten tespiti ve renk uzayları + Hough ile kırmızı çemberli trafik işareti tespiti.

## Dosyalar

| Dosya | İçerik |
| --- | --- |
| [renk_uzaylari_odev.ipynb](renk_uzaylari_odev.ipynb) · [Colab](https://colab.research.google.com/github/iliyadindar/DEEP/blob/main/week9/renk_uzaylari_odev.ipynb) | Bütün görevler; Colab'da çalıştırıldı, çıktılarıyla kaydedildi |
| [renk_uzaylari_odev.html](renk_uzaylari_odev.html) | Notebook'un HTML çıktısı (`jupyter nbconvert --to html`, kaydedilmiş çıktılardan) |

## Görevler

| Görev | Yapılan işlem |
| --- | --- |
| Renk uzayları | RGB, HSV, HSI, YCbCr ve CIE Lab'ın açıklaması, avantaj ve dezavantajları |
| 1–2 | 1000 × 1000 renkli ve gri BMP; dosya boyutu karşılaştırması (3 000 054 / 1 001 078 bayt, ≈ 3 kat) |
| 3 | 8K (7680 × 4320) görüntüde piksel değerlerini 2'ye bölme: NumPy 87 ms, Python iç içe döngü 78 s (≈ 900 kat), C++ iç içe döngü 131 ms |
| 4 | 8K görüntüyü HSV, HSI (formülle elle), YUV ve CIE Lab'a çevirip BMP olarak kaydetme |
| 5 | RGB kuralı, HSV ve YCbCr ile ten segmentasyonu; ten olmayan pikseller kırmızı |
| Haftalık 1 | Web kamerasında ardışık çerçeve farkıyla hareketli ROI, sadece ROI içinde YCrCb ile ten tespiti (kamera olmadığı için sanal kamera kaydıyla) |
| Haftalık 2 | Kaggle trafik işareti veri kümesinde kırmızı maske (Gray / RGB / HSV / Lab / YCrCb) + Hough çember + geometrik doğrulama |

## Sonuçlar

**Ten segmentasyonu:** YCbCr koyu tenler dahil yüzlerin neredeyse tamamını korudu. HSV koyu tenli yüzlerin çoğunu kaçırdı, RGB kuralı sarı/turuncu arka planı da ten saydı. Bu yüzden haftalık görev 1'de YCbCr kullanıldı.

**Hareket tabanlı ROI:** 72 karelik sanal kamera kaydında hareket, kişinin göründüğü 57 karenin hepsinde bulundu, hareketsiz karelerde ROI oluşmadı. ROI ortalama karenin %11,9'u. Ten tespiti sadece ROI'de 0,30 ms, bütün karede 1,46 ms.

**Kırmızı çemberli işaretler (26 pozitif / 19 negatif):**

| Renk uzayı | Doğruluk | Hassasiyet | Duyarlılık | F1 |
| --- | --- | --- | --- | --- |
| Gray (renksiz Hough) | 0,578 | 0,578 | 1,000 | 0,732 |
| RGB | 0,956 | 0,962 | 0,962 | 0,962 |
| **HSV** | **0,978** | **1,000** | **0,962** | **0,980** |
| CIE Lab | 0,956 | 1,000 | 0,923 | 0,960 |
| YCrCb | 0,956 | 0,962 | 0,962 | 0,962 |

Renk bilgisi olmadan Hough her görselde çember buluyor. HSV en iyisi, çünkü Hue kanalı kırmızıyı parlaklıktan bağımsız ayırıyor. Eşikler 45 görsellik küçük sette ayarlandığı için skorlar iyimser.

## Çalıştırma

Notebook Colab için yazıldı. Görev 3 görseli Pexels'ten, haftalık görev 2 veri kümesi `kagglehub` ile indirilir. Görev 5 ve haftalık görev 1, Colab'a yüklenen `/content/görüntüişleme9.jpg` görselini kullanır; bu görsel repoda yok, aynı adla yüklenmesi gerekir. Kameralı bir bilgisayarda gerçek zamanlı sürüm: `python webcam_hareket_ten.py` (notebook bu dosyayı `%%writefile` ile yazar).

`pip install -r week9/requirements.txt`
