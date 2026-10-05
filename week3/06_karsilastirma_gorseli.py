"""
06 - Tum sonuclari iki karsilastirma gorselinde toplar (rapor icin).
Once soru1..soru5 betiklerini calistirin.
"""
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CIKTI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cikti")


def oku(ad):
    img = cv2.imread(os.path.join(CIKTI, ad), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  UYARI: {ad} bulunamadi, atlandi")
    return img


def histogram(img):
    h = np.zeros(256, dtype=np.int64)
    for p in img.reshape(-1):
        h[p] += 1
    return h


def panel_ciz(paneller, dosya, histogramli=True, baslik=None):
    veriler = [(oku(a), b) for a, b in paneller]
    veriler = [(im, b) for im, b in veriler if im is not None]
    if not veriler:
        return

    n = len(veriler)
    satir = 2 if histogramli else 1
    fig, ax = plt.subplots(satir, n, figsize=(2.5 * n, 3.3 * satir + 0.4),
                           squeeze=False)

    for k, (im, bas) in enumerate(veriler):
        ax[0][k].imshow(im, cmap="gray", vmin=0, vmax=255)
        ax[0][k].set_title(bas, fontsize=9)
        ax[0][k].axis("off")

        if histogramli:
            ax[1][k].bar(range(256), histogram(im), width=1.0, color="#333")
            ax[1][k].set_xlim(0, 255)
            ax[1][k].set_yticks([])
            ax[1][k].tick_params(labelsize=7)
            if k == 0:
                ax[1][k].set_ylabel("histogram", fontsize=8)

    if baslik:
        fig.suptitle(baslik, fontsize=11)
        plt.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        plt.tight_layout()

    yol = os.path.join(CIKTI, dosya)
    plt.savefig(yol, dpi=115)
    plt.close()
    print(f"Kaydedildi: {yol}")


# --- Soru 1-2-3: kontrast iyilestirme ---
panel_ciz([
    ("dusuk_kontrast.png", "GIRDI\ndusuk kontrast"),
    ("soru1_dogrusal_olcekli.png", "SORU 1\ndogrusal olcekleme"),
    ("soru2_histogram_esitlenmis.png", "SORU 2\nhistogram esitleme"),
    ("soru3_clahe_manuel.png", "SORU 3\nCLAHE (manuel)"),
    ("soru3_ahe_sinirlamasiz.png", "SORU 3 ek\nAHE (sinirlamasiz)"),
], "00_kontrast_karsilastirma.png",
    baslik="Soru 1-2-3: Kontrast iyilestirme")

# --- Soru 4-5: gurultu bastirma ---
panel_ciz([
    ("gurultulu_gauss.png", "SORU 4 girdi\nGauss gurultusu"),
    ("soru4_ortalama_3x3.png", "SORU 4\n3x3 ortalama (konvolusyon)"),
    ("tuz_biber.png", "SORU 5 girdi\ntuz-biber %10"),
    ("soru5_medyan_manuel.png", "SORU 5\n5x5 medyan (manuel)"),
    ("soru5_ortalama5x5_karsilastirma.png", "SORU 5 karsilastirma\n5x5 ortalama (yetersiz)"),
], "00_gurultu_karsilastirma.png", histogramli=False,
    baslik="Soru 4-5: Gurultu bastirma")
