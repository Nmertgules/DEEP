# Digital Image Processing — Week 5: Color Histograms

**Group members: Nmertgules and Iliya**

## Görevler

| Görev | Dosya | Yapılan işlem |
| --- | --- | --- |
| 1 | [OpenCV histogramları](01_color_histograms_opencv.ipynb) | Renkli görüntüyü oku ve göster; B, G, R kanallarının histogramlarını `cv2.calcHist` ile ayrı hesapla ve çiz. |
| 2 | [Elle histogram hesabı](02_color_histograms_manual.ipynb) | `calc_hist` fonksiyonunu piksel sayaçlarıyla kendimiz yaz; her kanalda ayrı çalıştır ve çiz. |

| 3 | [Griye dönüşüm ve karşılaştırma](03_grayscale_comparison.ipynb) | OpenCV ve kendi ağırlıklı dönüşümümüzle tek kanallı gri oluştur; piksel farklarını raporla. |

Her histogram 256 kutuludur: yatay eksen 0–255 piksel değerini, dikey eksen bu değere sahip piksel sayısını gösterir. OpenCV kanalları **BGR** sırasıyla okur; fotoğraf ekranda RGB'ye çevrilerek doğru renklerle gösterilir.

## Çalıştırma

Repoyu bütünüyle indirin. `pip install -r week5/requirements.txt` komutundan sonra notebook'ları VS Code/Jupyter'da Python ortamınızı seçerek baştan sona çalıştırın. Repo kökünden veya `week5` klasöründen çalışırlar.

Girdi, Week 4'teki mevcut [renkli rocket.jpg](../week4/assets/rocket.jpg) fotoğrafıdır; [kaynak ve lisans kaydı](../week4/assets/sources.json) aynı yerde tutulur. Notebook'ların kaydedilmiş çıktıları GitHub üzerinde görülebilir.

## Çalıştırılan sonuçlar

640 × 427 renkli görüntüyle iki görev de CPU'da çalıştırıldı. Her kanalın histogram toplamı **273.280 piksel** oldu. Elle yazılan fonksiyon, üç kanalın tamamında OpenCV ile **birebir aynı** sonucu verdi; maksimum kutu farkı **0**.

Elle sonucu bilinen küçük görüntü, tamamen 0, tamamen 255 ve sabit tohumlu rastgele görüntü kontrolleri de geçti. Görev 2'de hazır histogram fonksiyonu yalnızca sonuç doğrulaması için kullanılır; asıl hesap iki döngü ve 256 sayaçla yapılır.

![Görev 1 — OpenCV](results/task1_histograms.svg)

![Görev 2 — Elle hesaplanan](results/task2_histograms.svg)

Bu iki maddede thread/CUDA istenmediği için bu teslim normal CPU uygulamasıdır.

## Görev 3 — Griye dönüşüm sonuçları

`Y = 0.299R + 0.587G + 0.114B` formülü kendi fonksiyonumuzda uygulanır. Kanal sırası BGR olarak ele alınır; en yakın tamsayıya yuvarlanır. İki çıktı da tek kanallı uint8 görüntüdür. Fark hesabından önce int16'ya dönüştürülerek taşma önlenir.

| Ölçüt | Sonuç |
| --- | --- |
| Toplam piksel | 273280 |
| Farklı piksel | 1 (%0.0004) |
| Maksimum mutlak fark | 1 gri seviye |
| Ortalama mutlak fark | 0.000004 |
| RMSE | 0.001913 |

Bir gri seviyelik farklar OpenCV'nin katsayı/tamsayı yaklaşımı ve yuvarlama davranışı ile kendi hesabımızın farklılığından kaynaklanır. Sonuçlar gözle çok benzerdir; aynı oldukları varsayılmadan piksel düzeyinde karşılaştırılmıştır. Siyah/beyaz ve saf BGR renkleri, 256 nötr gri ve sabit tohumlu rastgele görüntü kontrolleri geçti.

![Görev 3 karşılaştırması](results/task3_comparison.png)

Fark paneli 0–1 ölçeğinde görünür kılınmıştır. [Gerçek mutlak fark görüntüsü](results/task3_absolute_difference.png) 0–255 gösterimde neredeyse siyah görünür. [OpenCV çıktısı](results/task3_opencv_gray.png) · [Kendi çıktımız](results/task3_manual_gray.png) · [Sayısal sonuçlar](results/task3_metrics.json).
