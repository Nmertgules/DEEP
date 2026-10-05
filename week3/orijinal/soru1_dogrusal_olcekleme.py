"""
SORU 1 - Dogrusal Kontrast Germe (Min-Max Contrast Stretching)

Istenen:
  * Dusuk kontrastli, tek kanalli bir goruntuyu OpenCV ile ac
  * Goruntudeki minimum ve maksimum degerleri bul
  * Bu degerleri kullanarak parlaklik degerlerini [0, 255] araligina
    DOGRUSAL olarak olcekle
  * Olcekleme icin hazir fonksiyon (cv2.normalize vb.) KULLANMA

Kullanilan formul:
                     (I(x,y) - min)
    O(x,y) = 255 * -------------------
                      (max - min)

    min == max ise (tek renkli goruntu) bolme tanimsiz olur; bu durum
    ayrica ele alinir.

Calistirma:
    python3 soru1_dogrusal_olcekleme.py [goruntu_yolu]
"""
import os
import sys
import numpy as np
import cv2

BURASI = os.path.dirname(os.path.abspath(__file__))
CIKTI = os.path.join(BURASI, "cikti")


# ---------------------------------------------------------------------------
# 1) Min / Max bulma - acik dongu ile (hazir fonksiyon yok)
# ---------------------------------------------------------------------------
def min_max_bul_dongu(img):
    """Goruntudeki en kucuk ve en buyuk piksel degerini dongu ile bulur."""
    satir, sutun = img.shape
    enkucuk = int(img[0, 0])
    enbuyuk = int(img[0, 0])

    for y in range(satir):
        for x in range(sutun):
            deger = int(img[y, x])
            if deger < enkucuk:
                enkucuk = deger
            if deger > enbuyuk:
                enbuyuk = deger

    return enkucuk, enbuyuk


# ---------------------------------------------------------------------------
# 2) Dogrusal olcekleme - acik dongu ile (pedagojik, yavas)
# ---------------------------------------------------------------------------
def dogrusal_olcekle_dongu(img, enkucuk, enbuyuk):
    """Her pikseli tek tek gezerek [0,255] araligina tasir."""
    satir, sutun = img.shape
    cikis = np.zeros((satir, sutun), dtype=np.uint8)

    if enbuyuk == enkucuk:
        # Tum pikseller ayni: germe tanimsiz, sabit goruntu dondur
        return cikis

    olcek = 255.0 / (enbuyuk - enkucuk)

    for y in range(satir):
        for x in range(sutun):
            yeni = (int(img[y, x]) - enkucuk) * olcek
            # Yuvarlama + [0,255] sinirlama (saturasyon)
            yeni = int(yeni + 0.5)
            if yeni < 0:
                yeni = 0
            elif yeni > 255:
                yeni = 255
            cikis[y, x] = yeni

    return cikis


# ---------------------------------------------------------------------------
# 3) Dogrusal olcekleme - vektorel (ayni matematik, hizli surum)
# ---------------------------------------------------------------------------
def dogrusal_olcekle_vektorel(img, enkucuk, enbuyuk):
    """Dongu surumuyle bire bir ayni sonucu uretir, sadece NumPy ile hizli."""
    if enbuyuk == enkucuk:
        return np.zeros_like(img, dtype=np.uint8)

    f = img.astype(np.float64)
    cikis = (f - enkucuk) * (255.0 / (enbuyuk - enkucuk))
    cikis = np.floor(cikis + 0.5)          # yuvarlama (dongu ile ayni kural)
    cikis = np.clip(cikis, 0, 255)         # saturasyon
    return cikis.astype(np.uint8)


def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CIKTI, "dusuk_kontrast.png")

    # --- Goruntuyu TEK KANALLI olarak ac ---
    img = cv2.imread(yol, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"HATA: goruntu acilamadi -> {yol}")

    print(f"Goruntu      : {yol}")
    print(f"Boyut        : {img.shape[0]} x {img.shape[1]}  (tek kanal, {img.dtype})")

    # --- Min / Max ---
    enkucuk, enbuyuk = min_max_bul_dongu(img)
    print(f"\nBulunan min  : {enkucuk}")
    print(f"Bulunan max  : {enbuyuk}")
    print(f"Dinamik aralik (once): {enbuyuk - enkucuk} / 255")

    # Dogrulama: NumPy'nin min/max'i ile ayni mi?
    assert enkucuk == int(img.min()) and enbuyuk == int(img.max()), "min/max uyusmuyor!"
    print("  [OK] Dongu ile bulunan min/max, NumPy min/max ile ayni.")

    # --- Olcekleme ---
    sonuc_dongu = dogrusal_olcekle_dongu(img, enkucuk, enbuyuk)
    sonuc_vekt = dogrusal_olcekle_vektorel(img, enkucuk, enbuyuk)

    fark = int(np.abs(sonuc_dongu.astype(int) - sonuc_vekt.astype(int)).max())
    print(f"\n  [OK] Dongu surumu ile vektorel surum arasindaki maks. fark: {fark}")

    print(f"\nSonuc min    : {sonuc_dongu.min()}")
    print(f"Sonuc max    : {sonuc_dongu.max()}")
    print(f"Dinamik aralik (sonra): {int(sonuc_dongu.max()) - int(sonuc_dongu.min())} / 255")
    print(f"Standart sapma: {img.std():.2f} -> {sonuc_dongu.std():.2f}  (kontrast artisi)")

    # --- Sadece DOGRULAMA amacli: cv2.normalize ile karsilastir ---
    # (Cozumun kendisi yukaridaki manuel koddur; bu satir sadece kontrol icin)
    ref = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    maks_fark = int(np.abs(sonuc_dongu.astype(int) - ref.astype(int)).max())
    ortalama_fark = float(np.abs(sonuc_dongu.astype(int) - ref.astype(int)).mean())
    print(f"\nDogrulama (cv2.normalize referansi):")
    print(f"  maksimum fark = {maks_fark} seviye, ortalama fark = {ortalama_fark:.4f} seviye")
    print("  (1 seviyelik farklar yalnizca yuvarlama kuralindan kaynaklanir.)")

    cikti_yolu = os.path.join(CIKTI, "soru1_dogrusal_olcekli.png")
    cv2.imwrite(cikti_yolu, sonuc_dongu)
    print(f"\nKaydedildi   : {cikti_yolu}")


if __name__ == "__main__":
    main()
