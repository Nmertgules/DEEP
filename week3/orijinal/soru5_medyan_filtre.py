"""
SORU 5 - 5x5 Medyan Filtre ile Tuz-Biber (Salt & Pepper) Gurultusu Bastirma

Istenen:
  * Tuz-biber gurultulu bir resim bul
  * Once OpenCV'nin HAZIR medyan filtre fonksiyonu ile bastir
  * Sonra ayni filtreyi hazir fonksiyon KULLANMADAN kendin yaz
  * Filtre 5x5 boyutunda olmali

MEDYAN FILTRE NEDIR / NEDEN CALISIR:
    Her piksel icin 5x5 komsulugundaki 25 deger buyukten kucuge siralanir ve
    ORTADAKI deger (13. eleman, 0-tabanli indeks 12) cikisa yazilir.

    Tuz-biber gurultusu AYKIRI (outlier) deger uretir: 0 veya 255.
    * Ortalama filtre bu aykiri degeri hesaba katar ve 5x5 alana YAYAR.
    * Medyan filtre siralamanin UCUNDAKI degeri hic secmez; aykiri deger
      tamamen atilir. 25 komsudan 12'den azi bozuksa medyan hala saglam
      bir pikselden gelir -> gurultu TAMAMEN yok olur, bulaniklik olmaz.

    Medyan dogrusal OLMAYAN bir filtredir; konvolusyon ile ifade edilemez
    (Soru 4'teki ortalama filtre dogrusaldir ve konvolusyondur).

KENAR (border) ELE ALMA:
    cv2.medianBlur kenarda BORDER_REPLICATE kullanir (kenar pikseli tekrarlanir):
        ... a a [a b c] c c ...
    Birebir ayni sonucu almak icin burada da ayni yontem uygulandi.

Calistirma:
    python3 soru5_medyan_filtre.py [tuz_biber_goruntu] [temiz_referans]
"""
import os
import sys
import time
import numpy as np
import cv2

BURASI = os.path.dirname(os.path.abspath(__file__))
CIKTI = os.path.join(BURASI, "cikti")


# ---------------------------------------------------------------------------
# Kenar genisletme: BORDER_REPLICATE (elle)
# ---------------------------------------------------------------------------
def replicate_genislet(img, pad):
    """Kenar pikselini tekrarlayarak dort yana 'pad' kadar dolgu ekler."""
    h, w = img.shape
    gen = np.empty((h + 2 * pad, w + 2 * pad), dtype=img.dtype)

    gen[pad:pad + h, pad:pad + w] = img          # merkez

    for i in range(pad):
        gen[i, pad:pad + w] = img[0, :]          # ust
        gen[pad + h + i, pad:pad + w] = img[h - 1, :]   # alt

    for j in range(pad):
        gen[:, j] = gen[:, pad]                  # sol
        gen[:, pad + w + j] = gen[:, pad + w - 1]  # sag

    return gen


# ---------------------------------------------------------------------------
# 25 elemanin medyani - elle siralama (hazir sort/median yok)
# ---------------------------------------------------------------------------
def medyan_bul(pencere):
    """
    Verilen listenin medyanini EKLEMELI SIRALAMA (insertion sort) ile bulur.
    Kucuk diziler (25 eleman) icin eklemeli siralama pratikte en hizlisidir.
    """
    n = len(pencere)
    for i in range(1, n):
        anahtar = pencere[i]
        j = i - 1
        while j >= 0 and pencere[j] > anahtar:
            pencere[j + 1] = pencere[j]
            j -= 1
        pencere[j + 1] = anahtar
    return pencere[n // 2]     # 25 eleman -> indeks 12 (ortadaki)


# ---------------------------------------------------------------------------
# Medyan filtre - acik dongu (pedagojik surum)
# ---------------------------------------------------------------------------
def medyan_filtre_dongu(img, k=5):
    """Her piksel icin 5x5 pencereyi toplayip elle siralar, ortancayi yazar."""
    assert k % 2 == 1, "Cekirdek boyutu tek sayi olmali"
    pad = k // 2
    gen = replicate_genislet(img, pad)
    h, w = img.shape
    cikis = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        # Satiri Python listesine cevirmek ic donguyu belirgin hizlandirir
        bloklar = [gen[y + i].tolist() for i in range(k)]
        satir_cikis = cikis[y]
        for x in range(w):
            pencere = []
            for i in range(k):
                pencere.extend(bloklar[i][x:x + k])
            satir_cikis[x] = medyan_bul(pencere)

    return cikis


# ---------------------------------------------------------------------------
# Medyan filtre - vektorel (ayni matematik, hizli)
# ---------------------------------------------------------------------------
def medyan_filtre_vektorel(img, k=5):
    """
    kxk penceredeki tum kaydirmalar bir eksende yiginlanir, o eksen siralanir
    ve ortadaki duzlem alinir. Dongu surumuyle birebir ayni sonucu verir.
    """
    pad = k // 2
    gen = replicate_genislet(img, pad)
    h, w = img.shape

    yigin = np.empty((k * k, h, w), dtype=np.uint8)
    idx = 0
    for i in range(k):
        for j in range(k):
            yigin[idx] = gen[i:i + h, j:j + w]
            idx += 1

    yigin.sort(axis=0)
    return yigin[(k * k) // 2]


# ---------------------------------------------------------------------------
# Kalite olcutleri
# ---------------------------------------------------------------------------
def mse_hesapla(a, b):
    fark = a.astype(np.float64) - b.astype(np.float64)
    return float((fark * fark).mean())


def psnr_hesapla(a, b):
    mse = mse_hesapla(a, b)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def karsilastir(ad, a, b):
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    print(f"  {ad:<34} maks fark={int(d.max()):>3}  ort fark={d.mean():.4f}  "
          f"birebir ayni=%{(a == b).mean() * 100:.2f}")
    return int(d.max())


# ---------------------------------------------------------------------------
# 3x3 ve 5x5 medyanin gurultu yogunluguna gore karsilastirmasi
# ---------------------------------------------------------------------------
def yogunluk_taramasi(temiz):
    """
    Artan tuz-biber yogunlugunda 3x3 ve 5x5 medyani karsilastirir.
    Dusuk yogunlukta 3x3, yuksek yogunlukta 5x5 kazanir.
    """
    rng = np.random.default_rng(2024)
    print("\nGurultu yogunluguna gore 3x3 vs 5x5 medyan (PSNR, dB):")
    print(f"{'bozulma orani':<16}{'filtresiz':>11}{'3x3':>9}{'5x5':>9}   kazanan")
    print("-" * 56)

    for oran in [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.45]:
        bozuk = temiz.copy()
        m = rng.random(temiz.shape)
        bozuk[m < oran / 2] = 0
        bozuk[m > 1.0 - oran / 2] = 255

        p0 = psnr_hesapla(bozuk, temiz)
        p3 = psnr_hesapla(medyan_filtre_vektorel(bozuk, 3), temiz)
        p5 = psnr_hesapla(medyan_filtre_vektorel(bozuk, 5), temiz)
        kazanan = "3x3" if p3 > p5 else "5x5"
        print(f"%{oran * 100:<15.0f}{p0:>11.2f}{p3:>9.2f}{p5:>9.2f}   {kazanan}")

    print("\nBeklendigi gibi: dusuk yogunlukta kucuk pencere (daha az detay kaybi),")
    print("yogun gurultude buyuk pencere kazanir (pencerenin yarisindan fazlasinin")
    print("bozulma olasiligi kucuk pencerede hizla artar).")


# ---------------------------------------------------------------------------
def main():
    gurultulu_yol = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CIKTI, "tuz_biber.png")
    temiz_yol = sys.argv[2] if len(sys.argv) > 2 else os.path.join(CIKTI, "orijinal_temiz.png")

    gurultulu = cv2.imread(gurultulu_yol, cv2.IMREAD_GRAYSCALE)
    if gurultulu is None:
        raise SystemExit(f"HATA: goruntu acilamadi -> {gurultulu_yol}")
    temiz = cv2.imread(temiz_yol, cv2.IMREAD_GRAYSCALE)

    print(f"Gurultulu goruntu : {gurultulu_yol}")
    print(f"Boyut             : {gurultulu.shape[0]} x {gurultulu.shape[1]}")

    sifir = int((gurultulu == 0).sum())
    ikiyuz = int((gurultulu == 255).sum())
    print(f"Tuz-biber izi     : {sifir} adet 0 (biber), {ikiyuz} adet 255 (tuz)")

    # --- A) OpenCV'nin HAZIR medyan filtresi ---
    t0 = time.time()
    hazir = cv2.medianBlur(gurultulu, 5)
    t_hazir = time.time() - t0

    # --- B) Elle yazilmis medyan filtre (vektorel) ---
    t0 = time.time()
    manuel_v = medyan_filtre_vektorel(gurultulu, 5)
    t_vekt = time.time() - t0

    # --- C) Elle yazilmis medyan filtre (acik dongu + eklemeli siralama) ---
    t0 = time.time()
    manuel_d = medyan_filtre_dongu(gurultulu, 5)
    t_dongu = time.time() - t0

    print(f"\nSure              : hazir={t_hazir * 1000:.1f} ms, "
          f"manuel-vektorel={t_vekt * 1000:.1f} ms, manuel-dongu={t_dongu:.2f} s")

    # --- TEST ---
    print("\nTEST - elle yazilan filtre vs cv2.medianBlur(5):")
    m1 = karsilastir("manuel (acik dongu)", manuel_d, hazir)
    m2 = karsilastir("manuel (vektorel)", manuel_v, hazir)
    m3 = karsilastir("dongu vs vektorel (ic tutarlilik)", manuel_d, manuel_v)

    # Ek testler: farkli boyut ve icerikte de tutuyor mu?
    print("\nEk testler:")
    gecti = (m1 == 0 and m2 == 0 and m3 == 0)
    rng = np.random.default_rng(7)
    testler = [
        ("rastgele 37x53", rng.integers(0, 256, (37, 53), dtype=np.uint8)),
        ("tek satir 1x40", rng.integers(0, 256, (1, 40), dtype=np.uint8)),
        ("kucuk 3x3", rng.integers(0, 256, (3, 3), dtype=np.uint8)),
        ("sabit goruntu", np.full((20, 20), 128, dtype=np.uint8)),
    ]
    for ad, t in testler:
        m = karsilastir(ad, medyan_filtre_vektorel(t, 5), cv2.medianBlur(t, 5))
        gecti = gecti and (m == 0)

    # 3x3 cekirdekle de dogru mu?
    m = karsilastir("3x3 cekirdek (ana goruntu)",
                    medyan_filtre_vektorel(gurultulu, 3), cv2.medianBlur(gurultulu, 3))
    gecti = gecti and (m == 0)

    print(f"\nSONUC: {'TUM TESTLER GECTI - elle yazilan filtre OpenCV ile birebir ayni.' if gecti else 'FARK VAR - kontrol edilmeli.'}")

    # --- Gurultu bastirma basarisi ---
    if temiz is not None:
        ortalama5 = cv2.blur(gurultulu, (5, 5))
        medyan3 = cv2.medianBlur(gurultulu, 3)
        gauss5 = cv2.GaussianBlur(gurultulu, (5, 5), 0)

        print("\nGurultu bastirma olcumleri (tuz-biber gurultusu uzerinde):")
        print(f"{'goruntu':<32}{'MSE':>10}{'PSNR (dB)':>12}")
        print("-" * 54)
        for ad, im in [("gurultulu (filtresiz)", gurultulu),
                       ("5x5 MEDYAN (bu soru)", manuel_d),
                       ("3x3 medyan", medyan3),
                       ("5x5 ortalama (Soru 4 yontemi)", ortalama5),
                       ("5x5 Gauss", gauss5)]:
            print(f"{ad:<32}{mse_hesapla(im, temiz):>10.2f}{psnr_hesapla(im, temiz):>12.2f}")

        kalan = int(((manuel_d == 0) | (manuel_d == 255)).sum()
                    - ((temiz == 0) | (temiz == 255)).sum())
        print(f"\nFiltre sonrasi kalan asiri deger (0/255) fazlasi: {max(kalan, 0)} piksel")

        print("\nYORUM:")
        print("* Medyan filtre tuz-biber gurultusunde ortalama/Gauss filtrelerini acik")
        print("  farkla geride birakir. Sebep: medyan siralamanin UCUNDAKI degeri hic")
        print("  secmez, aykiri degeri tamamen atar. Ortalama filtre ise 0 veya 255'i")
        print("  hesaba katip komsuluga YAYAR; gurultu silinmez, lekeye donusur.")
        print("* Medyan ayrica kenarlari korur: cikis degeri her zaman komsulukta")
        print("  GERCEKTEN VAR OLAN bir piksel degeridir, yeni ara tonlar uretmez.")
        print("* DIKKAT: bu goruntude 3x3 medyan, 5x5'ten DAHA IYI PSNR veriyor.")
        print("  %10 bozulmada 3x3 penceresi zaten yeterli (9 komsudan 5'i birden")
        print("  bozulma olasiligi cok dusuk) ve daha az detay siliyor. 5x5'in ustun")
        print("  geldigi nokta YOGUN gurultudur; asagidaki tarama bunu olcuyor.")

        yogunluk_taramasi(temiz)

    # --- Kaydet ---
    cv2.imwrite(os.path.join(CIKTI, "soru5_medyan_opencv.png"), hazir)
    cv2.imwrite(os.path.join(CIKTI, "soru5_medyan_manuel.png"), manuel_d)
    if temiz is not None:
        cv2.imwrite(os.path.join(CIKTI, "soru5_ortalama5x5_karsilastirma.png"),
                    cv2.blur(gurultulu, (5, 5)))
    print("\nKaydedildi        : cikti/soru5_medyan_opencv.png")
    print("                    cikti/soru5_medyan_manuel.png")
    print("                    cikti/soru5_ortalama5x5_karsilastirma.png")


if __name__ == "__main__":
    main()
