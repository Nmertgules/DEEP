"""
SORU 2 - Histogram Esitleme (Histogram Equalization)

Istenen:
  * Dusuk kontrastli tek kanalli goruntuyu OpenCV ile ac
  * 256 kovali (bin) histogramini cikar
  * Histogramdan Kumulatif Dagilim Fonksiyonunu (CDF) hesapla
  * Bu dagilimla histogram esitleme uygula
  * Cikti resmini kaydet
  * Histogram, CDF ve esitleme adimlarinda hazir fonksiyon KULLANMA
    (cv2.calcHist, np.histogram, np.cumsum, cv2.equalizeHist YASAK)

Kullanilan formul:
    h[v]    = v degerine sahip piksel sayisi            (histogram)
    cdf[v]  = toplam_{k=0..v} h[k]                      (kumulatif dagilim)
    cdf_min = sifirdan farkli ilk cdf degeri

                   ( cdf[v] - cdf_min )
    LUT[v] = round( ------------------- * (L-1) ),   L = 256
                   (   N - cdf_min    )

    N = toplam piksel sayisi. cdf_min cikarilmasi, en koyu pikselin
    tam olarak 0'a oturmasini saglar (OpenCV'nin equalizeHist'i ile ayni kural).

Calistirma:
    python3 soru2_histogram_esitleme.py [goruntu_yolu]
"""
import os
import sys
import numpy as np
import cv2

BURASI = os.path.dirname(os.path.abspath(__file__))
CIKTI = os.path.join(BURASI, "cikti")
L = 256  # seviye sayisi


# ---------------------------------------------------------------------------
# 1) 256 kovali histogram - elle sayarak
# ---------------------------------------------------------------------------
def histogram_hesapla(img):
    """Her piksel degerinin kac kez gectigini sayar. Hazir fonksiyon yok."""
    hist = [0] * L
    satir, sutun = img.shape

    for y in range(satir):
        for x in range(sutun):
            hist[int(img[y, x])] += 1

    return hist


# ---------------------------------------------------------------------------
# 2) Kumulatif Dagilim Fonksiyonu (CDF) - elle toplayarak
# ---------------------------------------------------------------------------
def cdf_hesapla(hist):
    """cdf[v] = h[0] + h[1] + ... + h[v]. np.cumsum kullanilmadi."""
    cdf = [0] * L
    toplam = 0

    for v in range(L):
        toplam += hist[v]
        cdf[v] = toplam

    return cdf


# ---------------------------------------------------------------------------
# 3) Esitleme donusum tablosu (LUT) + uygulama
# ---------------------------------------------------------------------------
def esitleme_lut_olustur(cdf, toplam_piksel):
    """CDF'den [0,255] araligina esleyen donusum tablosunu uretir."""
    # Sifirdan farkli ilk cdf degerini bul
    cdf_min = 0
    for v in range(L):
        if cdf[v] > 0:
            cdf_min = cdf[v]
            break

    payda = toplam_piksel - cdf_min
    lut = [0] * L

    if payda <= 0:
        # Goruntude tek bir gri seviye var -> esitleme anlamsiz
        return lut, cdf_min

    for v in range(L):
        if cdf[v] == 0:
            lut[v] = 0
            continue
        deger = (cdf[v] - cdf_min) / payda * (L - 1)
        # Yuvarla ve sinirla
        deger = int(deger + 0.5)
        if deger < 0:
            deger = 0
        elif deger > L - 1:
            deger = L - 1
        lut[v] = deger

    return lut, cdf_min


def lut_uygula(img, lut):
    """Donusum tablosunu her piksele uygular."""
    satir, sutun = img.shape
    cikis = np.zeros((satir, sutun), dtype=np.uint8)

    for y in range(satir):
        for x in range(sutun):
            cikis[y, x] = lut[int(img[y, x])]

    return cikis


# ---------------------------------------------------------------------------
# Histogram grafigi (rapor icin, islemin parcasi degil)
# ---------------------------------------------------------------------------
def grafik_kaydet(img_once, hist_once, cdf_once, img_sonra, hist_sonra, cdf_sonra, yol):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib yok, grafik atlandi)")
        return

    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    N = img_once.size

    for satir, (im, h, c, baslik) in enumerate([
        (img_once, hist_once, cdf_once, "ONCE (dusuk kontrast)"),
        (img_sonra, hist_sonra, cdf_sonra, "SONRA (histogram esitlenmis)"),
    ]):
        ax[satir, 0].imshow(im, cmap="gray", vmin=0, vmax=255)
        ax[satir, 0].set_title(baslik)
        ax[satir, 0].axis("off")

        ax[satir, 1].bar(range(L), h, width=1.0, color="#444")
        ax[satir, 1].set_title("Histogram (256 kova)")
        ax[satir, 1].set_xlim(0, 255)

        ax[satir, 2].plot([v / N for v in c], color="#c0392b")
        ax[satir, 2].set_title("Kumulatif Dagilim (CDF)")
        ax[satir, 2].set_xlim(0, 255)
        ax[satir, 2].set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(yol, dpi=110)
    plt.close()
    print(f"Grafik       : {yol}")


def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CIKTI, "dusuk_kontrast.png")

    img = cv2.imread(yol, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"HATA: goruntu acilamadi -> {yol}")

    N = img.size
    print(f"Goruntu      : {yol}")
    print(f"Boyut        : {img.shape[0]} x {img.shape[1]} = {N} piksel")

    # --- 1) Histogram ---
    hist = histogram_hesapla(img)
    print(f"\nHistogram    : 256 kova, toplam = {sum(hist)} (piksel sayisina esit olmali: {N})")
    assert sum(hist) == N
    dolu_kova = sum(1 for h in hist if h > 0)
    print(f"               dolu kova sayisi = {dolu_kova}/256  (dusuk kontrastin gostergesi)")

    # --- 2) CDF ---
    cdf = cdf_hesapla(hist)
    print(f"CDF          : cdf[255] = {cdf[255]} (toplam piksel sayisina esit olmali)")
    assert cdf[L - 1] == N

    # --- 3) Esitleme ---
    lut, cdf_min = esitleme_lut_olustur(cdf, N)
    print(f"               cdf_min = {cdf_min}")
    esitlenmis = lut_uygula(img, lut)

    hist_sonra = histogram_hesapla(esitlenmis)
    cdf_sonra = cdf_hesapla(hist_sonra)
    dolu_sonra = sum(1 for h in hist_sonra if h > 0)

    print(f"\nSonuc        : min={esitlenmis.min()}, max={esitlenmis.max()}, "
          f"dolu kova={dolu_sonra}/256")
    print(f"Standart sapma: {img.std():.2f} -> {esitlenmis.std():.2f}")

    # --- Dogrulama: cv2.equalizeHist ile karsilastir ---
    ref = cv2.equalizeHist(img)
    maks_fark = int(np.abs(esitlenmis.astype(int) - ref.astype(int)).max())
    ortalama_fark = float(np.abs(esitlenmis.astype(int) - ref.astype(int)).mean())
    ayni_oran = float((esitlenmis == ref).mean() * 100)
    print(f"\nDogrulama (cv2.equalizeHist referansi):")
    print(f"  maksimum fark = {maks_fark} seviye, ortalama fark = {ortalama_fark:.4f} seviye, "
          f"birebir ayni piksel = %{ayni_oran:.2f}")

    cikti_yolu = os.path.join(CIKTI, "soru2_histogram_esitlenmis.png")
    cv2.imwrite(cikti_yolu, esitlenmis)
    print(f"\nKaydedildi   : {cikti_yolu}")

    grafik_kaydet(img, hist, cdf, esitlenmis, hist_sonra, cdf_sonra,
                  os.path.join(CIKTI, "soru2_histogram_grafik.png"))


if __name__ == "__main__":
    main()
