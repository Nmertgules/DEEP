"""
SORU 4 - Konvolusyon ile 3x3 Ortalama (Mean / Box) Filtre

Istenen:
  * Gurultulu bir resim uzerinde 3x3 ortalama alan bir filtreyi
    KONVOLUSYON ile gezdir ve gurultuyu bastir.

Cekirdek (kernel):
            1  | 1 1 1 |
    K  =   --- | 1 1 1 |
            9  | 1 1 1 |

KORELASYON vs KONVOLUSYON:
    Korelasyon : O(x,y) = SUM_{i,j} K(i,j) * I(x+j, y+i)
    Konvolusyon: O(x,y) = SUM_{i,j} K(i,j) * I(x-j, y-i)
                 yani cekirdek once 180 derece DONDURULUR.

    Ortalama cekirdegi simetrik oldugu icin iki sonuc ayni cikar; yine de
    asagidaki kodda cekirdek acikca dondurulerek gercek konvolusyon yapilir.
    (OpenCV'nin cv2.filter2D fonksiyonu aslinda korelasyon uygular; bu yuzden
    asimetrik cekirdeklerde cv2.flip(kernel,-1) gerekir.)

KENAR (border) ELE ALMA:
    reflect101 : kenar pikseli tekrarlanmadan yansitma  (OpenCV varsayilani)
                 ... c b [a b c] b a ...
    replicate  : kenar pikseli tekrarlanir
                 ... a a [a b c] c c ...

Calistirma:
    python3 soru4_ortalama_filtre.py [gurultulu_goruntu] [temiz_referans]
"""
import os
import sys
import time
import numpy as np
import cv2

BURASI = os.path.dirname(os.path.abspath(__file__))
CIKTI = os.path.join(BURASI, "cikti")


# ---------------------------------------------------------------------------
# Kenar genisletme (dolgu) - elle
# ---------------------------------------------------------------------------
def kenar_genislet(img, pad, mod="reflect101"):
    """Goruntunun dort yanina 'pad' kadar dolgu ekler."""
    h, w = img.shape
    cikis = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float64)
    cikis[pad:pad + h, pad:pad + w] = img

    def esle(i, n):
        """Disari tasan i indeksini [0, n-1] araligina katlar."""
        if mod == "replicate":
            return min(max(i, 0), n - 1)
        # reflect101: kenar pikseli tekrarlanmaz
        if n == 1:
            return 0
        while i < 0 or i >= n:
            if i < 0:
                i = -i
            if i >= n:
                i = 2 * (n - 1) - i
        return i

    for y in range(h + 2 * pad):
        for x in range(w + 2 * pad):
            if pad <= y < pad + h and pad <= x < pad + w:
                continue
            cikis[y, x] = img[esle(y - pad, h), esle(x - pad, w)]

    return cikis


# ---------------------------------------------------------------------------
# Konvolusyon - acik dongu (hazir fonksiyon yok)
# ---------------------------------------------------------------------------
def konvolusyon_dongu(img, cekirdek, mod="reflect101"):
    """
    Cekirdegi 180 derece dondurup goruntu uzerinde gezdirir.
    Pedagojik surum: her piksel ve her cekirdek elemani icin ic ice dongu.
    """
    k = cekirdek.shape[0]
    assert cekirdek.shape[0] == cekirdek.shape[1] and k % 2 == 1, "Tek boyutlu kare cekirdek gerekli"
    pad = k // 2

    # --- KONVOLUSYON: cekirdegi 180 derece dondur ---
    kd = cekirdek[::-1, ::-1]

    gen = kenar_genislet(img, pad, mod)
    h, w = img.shape
    cikis = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            toplam = 0.0
            for i in range(k):
                for j in range(k):
                    toplam += kd[i, j] * gen[y + i, x + j]
            # saturate_cast<uchar>: en yakina yuvarla + [0,255] sinirla
            v = int(np.rint(toplam))
            if v < 0:
                v = 0
            elif v > 255:
                v = 255
            cikis[y, x] = v

    return cikis


# ---------------------------------------------------------------------------
# Konvolusyon - vektorel (ayni matematik, hizli)
# ---------------------------------------------------------------------------
def konvolusyon_vektorel(img, cekirdek, mod="reflect101"):
    """Kaydirilmis dilimleri toplayarak ayni sonucu hizlica uretir."""
    k = cekirdek.shape[0]
    pad = k // 2
    kd = cekirdek[::-1, ::-1]

    gen = kenar_genislet(img, pad, mod)
    h, w = img.shape

    birikim = np.zeros((h, w), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            birikim += kd[i, j] * gen[i:i + h, j:j + w]

    return np.clip(np.rint(birikim), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Kalite olcutleri - elle
# ---------------------------------------------------------------------------
def mse_hesapla(a, b):
    fark = a.astype(np.float64) - b.astype(np.float64)
    return float((fark * fark).mean())


def psnr_hesapla(a, b):
    """Tepe Sinyal/Gurultu Orani (dB). Yuksek = referansa daha yakin."""
    mse = mse_hesapla(a, b)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


# ---------------------------------------------------------------------------
def main():
    gurultulu_yol = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CIKTI, "gurultulu_gauss.png")
    temiz_yol = sys.argv[2] if len(sys.argv) > 2 else os.path.join(CIKTI, "orijinal_temiz.png")

    gurultulu = cv2.imread(gurultulu_yol, cv2.IMREAD_GRAYSCALE)
    if gurultulu is None:
        raise SystemExit(f"HATA: goruntu acilamadi -> {gurultulu_yol}")
    temiz = cv2.imread(temiz_yol, cv2.IMREAD_GRAYSCALE)  # varsa PSNR icin

    print(f"Gurultulu goruntu : {gurultulu_yol}")
    print(f"Boyut             : {gurultulu.shape[0]} x {gurultulu.shape[1]}")

    # --- 3x3 ortalama cekirdegi ---
    cekirdek = np.ones((3, 3), dtype=np.float64) / 9.0
    print("\n3x3 ortalama cekirdegi (her eleman = 1/9 = 0.1111):")
    for satir in cekirdek:
        print("   [" + "  ".join(f"{v:.4f}" for v in satir) + "]")
    print(f"   cekirdek toplami = {cekirdek.sum():.4f}  (1.0 olmali -> parlaklik korunur)")

    # --- Konvolusyon: dongu surumu ---
    t0 = time.time()
    sonuc_dongu = konvolusyon_dongu(gurultulu, cekirdek, mod="reflect101")
    t_dongu = time.time() - t0

    # --- Konvolusyon: vektorel surum ---
    t0 = time.time()
    sonuc_vekt = konvolusyon_vektorel(gurultulu, cekirdek, mod="reflect101")
    t_vekt = time.time() - t0

    fark = int(np.abs(sonuc_dongu.astype(int) - sonuc_vekt.astype(int)).max())
    print(f"\nSure              : dongu={t_dongu:.2f} s, vektorel={t_vekt * 1000:.1f} ms")
    print(f"  [OK] Dongu ve vektorel surum arasindaki maks. fark: {fark}")

    # --- Dogrulama: OpenCV'nin hazir filtreleriyle karsilastir ---
    ref_blur = cv2.blur(gurultulu, (3, 3))                      # BORDER_REFLECT_101
    ref_f2d = cv2.filter2D(gurultulu, -1, cekirdek.astype(np.float32))

    print("\nDogrulama (OpenCV referanslari):")
    for ad, ref in [("cv2.blur(3,3)", ref_blur), ("cv2.filter2D", ref_f2d)]:
        d = np.abs(sonuc_dongu.astype(int) - ref.astype(int))
        print(f"  {ad:<16} maks fark={int(d.max()):>2}  ort fark={d.mean():.4f}  "
              f"birebir ayni=%{(sonuc_dongu == ref).mean() * 100:.2f}")

    # --- Gurultu bastirma basarisi ---
    print("\nGurultu bastirma olcumleri:")
    if temiz is not None:
        print(f"{'goruntu':<28}{'MSE':>10}{'PSNR (dB)':>12}")
        print("-" * 50)
        print(f"{'gurultulu (filtresiz)':<28}{mse_hesapla(gurultulu, temiz):>10.2f}"
              f"{psnr_hesapla(gurultulu, temiz):>12.2f}")
        print(f"{'3x3 ortalama filtre':<28}{mse_hesapla(sonuc_dongu, temiz):>10.2f}"
              f"{psnr_hesapla(sonuc_dongu, temiz):>12.2f}")

        # Ek karsilastirmalar (soru disi, yorum icin)
        cekirdek5 = np.ones((5, 5), dtype=np.float64) / 25.0
        sonuc5 = konvolusyon_vektorel(gurultulu, cekirdek5, mod="reflect101")
        medyan = cv2.medianBlur(gurultulu, 3)
        gauss = cv2.GaussianBlur(gurultulu, (3, 3), 0)
        print(f"{'5x5 ortalama filtre':<28}{mse_hesapla(sonuc5, temiz):>10.2f}"
              f"{psnr_hesapla(sonuc5, temiz):>12.2f}")
        print(f"{'3x3 Gauss (karsilastirma)':<28}{mse_hesapla(gauss, temiz):>10.2f}"
              f"{psnr_hesapla(gauss, temiz):>12.2f}")
        print(f"{'3x3 medyan (karsilastirma)':<28}{mse_hesapla(medyan, temiz):>10.2f}"
              f"{psnr_hesapla(medyan, temiz):>12.2f}")
        cv2.imwrite(os.path.join(CIKTI, "soru4_ortalama_5x5.png"), sonuc5)
        cv2.imwrite(os.path.join(CIKTI, "soru4_medyan_3x3.png"), medyan)

        print("\nYORUM:")
        print("* Ortalama filtre Gauss gurultusu icin dogru aractir: bagimsiz gurultu")
        print("  orneklerinin ortalamasi alindigi icin gurultu standart sapmasi ~1/sqrt(9)")
        print("  oraninda duser. Burada 3x3 ortalama ~+6.9 dB kazandiriyor.")
        print("* 5x5 daha cok gurultu bastirir ama kenarlari da daha cok bulaniklastirir;")
        print("  bu goruntude iki etki birbirini goturdugu icin PSNR neredeyse ayni kaldi.")
        print("* Medyan filtre burada ortalamanin biraz GERISINDE. Medyanin ustun oldugu")
        print("  durum tuz-biber (impulse) gurultusudur -> Soru 5'e bakiniz. Ortalama filtre")
        print("  impulse gurultude aykiri degeri yok etmez, 3x3 alana YAYAR.")
    else:
        print("  (temiz referans bulunamadi, PSNR hesaplanamadi)")

    print(f"\nStandart sapma    : {gurultulu.std():.2f} -> {sonuc_dongu.std():.2f}")

    cikti_yolu = os.path.join(CIKTI, "soru4_ortalama_3x3.png")
    cv2.imwrite(cikti_yolu, sonuc_dongu)
    print(f"\nKaydedildi        : {cikti_yolu}")


if __name__ == "__main__":
    main()
