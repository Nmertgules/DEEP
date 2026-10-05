"""
00 - Test goruntulerini uretir.

Uretilen dosyalar (cikti/ klasorune):
  orijinal_temiz.png  -> gurultusuz referans (PSNR olcumu icin)
  dusuk_kontrast.png  -> Soru 1, 2, 3 icin dusuk kontrastli tek kanalli goruntu
  gurultulu_gauss.png -> Soru 4 icin Gauss gurultulu goruntu
  tuz_biber.png       -> Soru 5 icin tuz-biber (salt & pepper) gurultulu goruntu

Elinizde hazir bir goruntu varsa bu adimi atlayip dosya yolunu dogrudan
diger betiklere argüman olarak verebilirsiniz.
"""
import os
import numpy as np
import cv2

CIKTI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cikti")
os.makedirs(CIKTI, exist_ok=True)

H, W = 384, 512
rng = np.random.default_rng(1907)


def sahne_olustur():
    """Yapay ama 'fotograf gibi' davranan tek kanalli bir sahne."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # Yumusak arka plan gradyani
    img = 120 + 70 * np.cos(xx / W * np.pi * 0.8) + 40 * (1 - yy / H)

    # Genis parlak disk
    img += 60 * np.exp(-(((xx - 150) ** 2 + (yy - 120) ** 2) / (2 * 70.0 ** 2)))

    # Koyu dikdortgen blok, ic detayli
    img[230:340, 60:210] = 55 + 12 * np.sin(xx[230:340, 60:210] / 4.0)

    # Ince cizgili doku (yuksek frekans -> filtre testinde ise yarar)
    img[40:130, 300:470] = 150 + 45 * np.sin(xx[40:130, 300:470] / 2.5)

    # Dama tahtasi yamasi
    kare = 8
    blok = (((xx // kare).astype(int) + (yy // kare).astype(int)) % 2)
    maske = (xx > 290) & (xx < 460) & (yy > 220) & (yy < 340)
    img[maske] = (100 + 60 * blok)[maske]

    # Halkalar
    r = np.sqrt((xx - 400) ** 2 + (yy - 80) ** 2)
    img += 25 * np.sin(r / 6.0) * np.exp(-r / 120.0)

    return np.clip(img, 0, 255)


def kontrasti_dusur(img):
    """
    Goruntuyu DAR bir parlaklik araligina sikistirir.
    Konuma bagli taban parlaklik eklenir: boylece global histogram esitleme
    yetersiz kalir ve CLAHE'nin yerel ustunlugu Soru 3'te net gorulur.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    n = (img - img.min()) / (img.max() - img.min())
    taban = 45 + 70 * (xx / W)      # sol karanlik, sag aydinlik
    return np.clip(taban + 55 * n, 0, 255).astype(np.uint8)


def gauss_gurultusu(img, sigma=20.0):
    """Soru 4 icin: yalnizca toplanir (additive) Gauss gurultusu."""
    g = img.astype(np.float64) + rng.normal(0, sigma, size=img.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def tuz_biber_gurultusu(img, oran=0.05):
    """
    Soru 5 icin: yalnizca impulse (tuz-biber) gurultusu.
    Piksellerin %oran kadari 0 (biber), %oran kadari 255 (tuz) yapilir.
    """
    cikis = img.copy()
    maske = rng.random(img.shape)
    cikis[maske < oran] = 0                  # biber
    cikis[maske > 1.0 - oran] = 255          # tuz
    return cikis


if __name__ == "__main__":
    temiz = sahne_olustur().astype(np.uint8)
    dusuk = kontrasti_dusur(sahne_olustur())
    gauss = gauss_gurultusu(temiz, sigma=20.0)
    tuzbiber = tuz_biber_gurultusu(temiz, oran=0.05)

    for ad, im in [("orijinal_temiz.png", temiz),
                   ("dusuk_kontrast.png", dusuk),
                   ("gurultulu_gauss.png", gauss),
                   ("tuz_biber.png", tuzbiber)]:
        cv2.imwrite(os.path.join(CIKTI, ad), im)

    bozuk = int((tuzbiber != temiz).sum())
    print("Uretilen goruntuler (hepsi tek kanalli, 384x512):")
    print(f"  orijinal_temiz.png  min={temiz.min():>3} max={temiz.max():>3}")
    print(f"  dusuk_kontrast.png  min={dusuk.min():>3} max={dusuk.max():>3}  "
          f"dinamik aralik={int(dusuk.max()) - int(dusuk.min())}/255")
    print(f"  gurultulu_gauss.png Gauss sigma=20")
    print(f"  tuz_biber.png       bozulan piksel={bozuk} "
          f"(%{100 * bozuk / temiz.size:.1f}), yaklasik yarisi 0 yarisi 255")
