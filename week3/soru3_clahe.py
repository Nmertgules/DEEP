"""
SORU 3 - CLAHE (Contrast Limited Adaptive Histogram Equalization)

Istenen:
  * Dusuk kontrastli tek kanalli goruntuyu OpenCV ile ac
  * OpenCV'nin HAZIR CLAHE fonksiyonu ile duzelt
  * Ayni islemi hazir fonksiyon KULLANMADAN kendin gerceklestir ve test et

Referans: https://en.wikipedia.org/wiki/Adaptive_histogram_equalization

ALGORITMA (OpenCV'nin cv::CLAHE gerceklemesiyle bire bir ayni adimlar):

  1. KARO BOLME (tiling)
     Goruntu tilesX x tilesY (varsayilan 8x8) karoya bolunur.
     Boyut tam bolunmuyorsa goruntu BORDER_REFLECT_101 ile genisletilir.

  2. KARO HISTOGRAMI
     Her karo icin 256 kovali histogram cikarilir.

  3. KONTRAST SINIRLAMA (contrast limiting) -- CLAHE'yi AHE'den ayiran adim
     clipLimit_gercek = max(1, clipLimit * karo_alani / 256)
     Bu esigi asan kovalar kesilir; kesilen toplam miktar ("clipped")
     TUM kovalara esit dagitilir:
         redistBatch = clipped // 256         -> her kovaya eklenir
         residual    = clipped - redistBatch*256
         kalan 'residual' adet +1, kovalara esit araliklarla serpistirilir
     Amac: duz bolgelerde gurultunun asiri yukseltilmesini engellemek.

  4. KARO LUT'U
     Sinirlanmis histogramin kumulatif toplami alinip [0,255]'e olceklenir:
         lutScale = 255 / karo_alani
         LUT[v] = round( cdf[v] * lutScale )

  5. BILINEER INTERPOLASYON
     Her piksel icin, komsu 4 karonun LUT'u karo MERKEZLERINE gore
     bilineer agirliklandirilir. Bu adim olmazsa karo sinirlarinda
     gorunur blok artefaktlari olusur.

Calistirma:
    python3 soru3_clahe.py [goruntu_yolu]
"""
import os
import sys
import time
import numpy as np
import cv2

BURASI = os.path.dirname(os.path.abspath(__file__))
CIKTI = os.path.join(BURASI, "cikti")
HIST_BOYUT = 256


# ---------------------------------------------------------------------------
# Yardimci: BORDER_REFLECT_101 ile kenar genisletme (elle)
# ---------------------------------------------------------------------------
def reflect101_genislet(img, alt, sag):
    """
    Alt ve saga 'yansimali' dolgu ekler (kenar pikseli tekrarlanmaz).
    Ornek satirlar: ... h-3, h-2, [h-1], h-2, h-3, ...
    """
    h, w = img.shape
    cikis = np.empty((h + alt, w + sag), dtype=img.dtype)
    cikis[:h, :w] = img

    for i in range(alt):
        kaynak = h - 2 - i
        if kaynak < 0:
            kaynak = 0
        cikis[h + i, :w] = img[kaynak, :]

    for j in range(sag):
        kaynak = w - 2 - j
        if kaynak < 0:
            kaynak = 0
        cikis[:, w + j] = cikis[:, kaynak]

    return cikis


# ---------------------------------------------------------------------------
# Adim 2 + 3 + 4 : Her karo icin LUT uret
# ---------------------------------------------------------------------------
def karo_lutlari_hesapla(gen_img, tilesX, tilesY, clip_limit):
    """
    Genisletilmis goruntuden (tilesY, tilesX, 256) boyutunda LUT tablosu uretir.
    """
    gh, gw = gen_img.shape
    karo_w = gw // tilesX
    karo_h = gh // tilesY
    karo_alan = karo_w * karo_h

    # clipLimit'i karo alanina gore gercek piksel sayisina cevir
    if clip_limit > 0.0:
        gercek_limit = int(clip_limit * karo_alan / HIST_BOYUT)
        gercek_limit = max(gercek_limit, 1)
    else:
        gercek_limit = 0

    # OpenCV tek duyarlikli (float32) aritmetik kullanir; birebir ayni
    # sonucu almak icin burada da float32 kullaniyoruz.
    lut_olcek = np.float32(HIST_BOYUT - 1) / np.float32(karo_alan)
    lutlar = np.zeros((tilesY, tilesX, HIST_BOYUT), dtype=np.uint8)

    for ty in range(tilesY):
        for tx in range(tilesX):
            karo = gen_img[ty * karo_h:(ty + 1) * karo_h,
                           tx * karo_w:(tx + 1) * karo_w]

            # --- 2) Karo histogrami (elle sayim) ---
            hist = np.zeros(HIST_BOYUT, dtype=np.int64)
            duz = karo.reshape(-1)
            for p in duz:
                hist[p] += 1

            # --- 3) Kontrast sinirlama + yeniden dagitim ---
            if gercek_limit > 0:
                kesilen = 0
                for i in range(HIST_BOYUT):
                    if hist[i] > gercek_limit:
                        kesilen += hist[i] - gercek_limit
                        hist[i] = gercek_limit

                if kesilen > 0:
                    pay = kesilen // HIST_BOYUT
                    artik = int(kesilen - pay * HIST_BOYUT)

                    for i in range(HIST_BOYUT):
                        hist[i] += pay

                    if artik != 0:
                        adim = max(HIST_BOYUT // artik, 1)
                        i = 0
                        while i < HIST_BOYUT and artik > 0:
                            hist[i] += 1
                            artik -= 1
                            i += adim

            # --- 4) Kumulatif toplam -> LUT ---
            toplam = 0
            for i in range(HIST_BOYUT):
                toplam += int(hist[i])
                deger = np.float32(toplam) * lut_olcek
                # cvRound davranisi: bankaci yuvarlamasi (half-to-even)
                yuvarlak = int(np.rint(np.float64(deger)))
                if yuvarlak < 0:
                    yuvarlak = 0
                elif yuvarlak > 255:
                    yuvarlak = 255
                lutlar[ty, tx, i] = yuvarlak

    return lutlar, karo_w, karo_h, gercek_limit


# ---------------------------------------------------------------------------
# Adim 5 : Bilineer interpolasyon ile ciktiyi uret
# ---------------------------------------------------------------------------
def bilineer_uygula(img, lutlar, karo_w, karo_h, tilesX, tilesY):
    """
    Her piksel icin komsu 4 karonun LUT'unu bilineer harmanlar.
    Karo merkezleri (tx + 0.5) * karo_w konumundadir; bu yuzden -0.5 kaymasi var.
    """
    h, w = img.shape
    BIR = np.float32(1.0)
    YARIM = np.float32(0.5)

    # --- Yatay eksen: pikselin hangi iki karo sutunu arasinda kaldigi ---
    # OpenCV bolme yerine tersiyle carpar (x * inv_tw); float32'de sonuc
    # farkli cikabildigi icin ayni sirayi izliyoruz.
    ters_w = BIR / np.float32(karo_w)
    x = np.arange(w, dtype=np.float32)
    txf = x * ters_w - YARIM        # -0.5 -> karo MERKEZINE gore konum
    tx1 = np.floor(txf).astype(np.int32)
    tx2 = tx1 + 1
    xa = txf - tx1.astype(np.float32)   # agirlik, KIRPMADAN once hesaplanir
    xa1 = BIR - xa
    tx1 = np.clip(tx1, 0, tilesX - 1)
    tx2 = np.clip(tx2, 0, tilesX - 1)

    # --- Dikey eksen ---
    ters_h = BIR / np.float32(karo_h)
    y = np.arange(h, dtype=np.float32)
    tyf = y * ters_h - YARIM
    ty1 = np.floor(tyf).astype(np.int32)
    ty2 = ty1 + 1
    ya = tyf - ty1.astype(np.float32)
    ya1 = BIR - ya
    ty1 = np.clip(ty1, 0, tilesY - 1)
    ty2 = np.clip(ty2, 0, tilesY - 1)

    # 2 boyuta yay
    TY1, TY2 = ty1[:, None], ty2[:, None]
    TX1, TX2 = tx1[None, :], tx2[None, :]
    XA, XA1 = xa[None, :], xa1[None, :]
    YA, YA1 = ya[:, None], ya1[:, None]

    V = img.astype(np.int32)        # piksel degeri = LUT indeksi

    # Dort komsu karonun LUT cikislari
    ust_sol = lutlar[TY1, TX1, V].astype(np.float32)
    ust_sag = lutlar[TY1, TX2, V].astype(np.float32)
    alt_sol = lutlar[TY2, TX1, V].astype(np.float32)
    alt_sag = lutlar[TY2, TX2, V].astype(np.float32)

    # Once yatayda, sonra dikeyde harmanla (OpenCV ile ayni carpanli form):
    #   res = (U1*xa1 + U2*xa) * ya1 + (A1*xa1 + A2*xa) * ya
    ust = ust_sol * XA1 + ust_sag * XA
    alt = alt_sol * XA1 + alt_sag * XA
    sonuc = ust * YA1 + alt * YA

    # saturate_cast<uchar> -> cvRound -> en yakina, berabere kalirsa cift sayiya
    return np.clip(np.rint(sonuc.astype(np.float64)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ana CLAHE fonksiyonu (hazir fonksiyon kullanmadan)
# ---------------------------------------------------------------------------
def clahe_manuel(img, clip_limit=2.0, tile_grid=(8, 8)):
    """OpenCV'nin cv2.createCLAHE(...).apply(img) karsiligi, elle gerceklenmis."""
    tilesX, tilesY = tile_grid
    h, w = img.shape

    # --- 1) Karo bolme icin gerekirse genislet ---
    alt = (tilesY - h % tilesY) % tilesY
    sag = (tilesX - w % tilesX) % tilesX
    gen = reflect101_genislet(img, alt, sag) if (alt or sag) else img

    lutlar, karo_w, karo_h, gercek_limit = karo_lutlari_hesapla(
        gen, tilesX, tilesY, clip_limit)

    cikis = bilineer_uygula(img, lutlar, karo_w, karo_h, tilesX, tilesY)
    return cikis, {"karo_w": karo_w, "karo_h": karo_h,
                   "gercek_limit": gercek_limit, "dolgu": (alt, sag)}


# ---------------------------------------------------------------------------
def yerel_kontrast(img, pencere=15):
    """
    Yerel kontrast olcusu: kucuk pencerelerdeki standart sapmanin ortalamasi.
    std(X) = sqrt( E[X^2] - E[X]^2 ) esitligi kutu filtresiyle hizlica hesaplanir.
    """
    f = img.astype(np.float64)
    ort = cv2.boxFilter(f, -1, (pencere, pencere))
    ort_kare = cv2.boxFilter(f * f, -1, (pencere, pencere))
    varyans = np.maximum(ort_kare - ort * ort, 0.0)
    return float(np.sqrt(varyans).mean())


def karsilastir(ad, a, b):
    fark = np.abs(a.astype(np.int32) - b.astype(np.int32))
    print(f"  {ad:<28} maks fark={int(fark.max()):>3}  "
          f"ort fark={fark.mean():.4f}  birebir ayni=%{(a == b).mean() * 100:.2f}")
    return int(fark.max())


def main():
    yol = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CIKTI, "dusuk_kontrast.png")

    img = cv2.imread(yol, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"HATA: goruntu acilamadi -> {yol}")

    print(f"Goruntu      : {yol}")
    print(f"Boyut        : {img.shape[0]} x {img.shape[1]}")

    # NOT: clipLimit secimi goruntuye baglidir. Bu goruntunun histogrami cok
    # dar oldugu icin clipLimit=2.0 kesme esigini o kadar dusurur ki donusum
    # neredeyse birim fonksiyon olur (hicbir sey degismez). Asagidaki tarama
    # grafigi bu etkiyi gosterir; burada dengeli bir deger secildi.
    CLIP = 8.0
    IZGARA = (8, 8)

    # --- A) OpenCV'nin hazir CLAHE'si ---
    t0 = time.time()
    clahe_nesne = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=IZGARA)
    hazir = clahe_nesne.apply(img)
    t_hazir = time.time() - t0

    # --- B) Elle gerceklenmis CLAHE ---
    t0 = time.time()
    manuel, bilgi = clahe_manuel(img, clip_limit=CLIP, tile_grid=IZGARA)
    t_manuel = time.time() - t0

    print(f"\nParametreler : clipLimit={CLIP}, tileGridSize={IZGARA}")
    print(f"               karo boyutu = {bilgi['karo_w']} x {bilgi['karo_h']} piksel")
    print(f"               gercek kesme esigi = {bilgi['gercek_limit']} piksel/kova")
    print(f"               eklenen dolgu (alt, sag) = {bilgi['dolgu']}")
    print(f"Sure         : hazir={t_hazir * 1000:.1f} ms, manuel={t_manuel * 1000:.1f} ms")

    # --- TEST: iki sonuc ayni mi? ---
    print("\nTEST - manuel gercekleme vs cv2.createCLAHE:")
    maks = karsilastir("ana goruntu", manuel, hazir)

    # Farkli parametrelerle de test et
    print("\nEk testler (farkli parametre ve boyutlar):")
    tum_gecti = (maks == 0)
    for clip, izgara in [(1.0, (4, 4)), (2.0, (8, 8)), (3.0, (8, 8)),
                         (4.0, (16, 16)), (40.0, (2, 2)), (0.0, (8, 8))]:
        ref = cv2.createCLAHE(clipLimit=clip, tileGridSize=izgara).apply(img)
        ben, _ = clahe_manuel(img, clip_limit=clip, tile_grid=izgara)
        m = karsilastir(f"clip={clip}, izgara={izgara}", ben, ref)
        tum_gecti = tum_gecti and (m == 0)

    # Bolunmeyen boyutta (dolgu yolunu test eder)
    kirpik = img[:377, :500]
    ref = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(kirpik)
    ben, _ = clahe_manuel(kirpik, clip_limit=2.0, tile_grid=(8, 8))
    m = karsilastir("500x377 (tam bolunmeyen)", ben, ref)
    tum_gecti = tum_gecti and (m == 0)

    print(f"\nSONUC: {'TUM TESTLER GECTI - manuel gercekleme OpenCV ile birebir ayni.' if tum_gecti else 'FARK VAR - kontrol edilmeli.'}")

    # --- Karsilastirma icin global histogram esitleme de uretelim ---
    global_he = cv2.equalizeHist(img)

    cv2.imwrite(os.path.join(CIKTI, "soru3_clahe_opencv.png"), hazir)
    cv2.imwrite(os.path.join(CIKTI, "soru3_clahe_manuel.png"), manuel)
    print(f"\nKaydedildi   : cikti/soru3_clahe_opencv.png")
    print(f"               cikti/soru3_clahe_manuel.png")

    # AHE (sinirlama yok) ile farki da gosterelim - gurultu yukseltme etkisi
    ahe, _ = clahe_manuel(img, clip_limit=0.0, tile_grid=IZGARA)
    cv2.imwrite(os.path.join(CIKTI, "soru3_ahe_sinirlamasiz.png"), ahe)
    print(f"               cikti/soru3_ahe_sinirlamasiz.png (clipLimit=0, yani sinirlama yok)")

    print("\nKontrast karsilastirmasi:")
    print(f"{'yontem':<22}{'global std':>12}{'yerel kontrast':>17}")
    print("-" * 51)
    for ad, im in [("orijinal", img), ("global HE", global_he),
                   ("CLAHE (manuel)", manuel), ("AHE (sinirlamasiz)", ahe)]:
        print(f"{ad:<22}{im.std():>12.2f}{yerel_kontrast(im):>17.2f}")
    print("\nNOT: Global std CLAHE icin yaniltici bir olcuttur. CLAHE buyuk olcekli")
    print("parlaklik farkini bilerek bastirir (global std duser) ama YEREL detay")
    print("kontrastini yukseltir. Asil bakilmasi gereken sutun 'yerel kontrast'tir.")
    print("AHE'nin yerel kontrasti en yuksektir; ancak duz bolgelerdeki gurultuyu de")
    print("yukselttigi icin CLAHE tercih edilir (kontrast sinirlamasinin amaci budur).")

    clip_taramasi(img, IZGARA)


def clip_taramasi(img, izgara):
    """clipLimit'in etkisini gosteren tarama grafigi uretir (rapor icin)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    clipler = [1.0, 2.0, 4.0, 8.0, 20.0, 40.0, 0.0]
    fig, ax = plt.subplots(1, len(clipler), figsize=(2.3 * len(clipler), 3.4))

    for k, c in enumerate(clipler):
        cikis, _ = clahe_manuel(img, clip_limit=c, tile_grid=izgara)
        etiket = "0.0 (sinirsiz=AHE)" if c == 0.0 else f"{c}"
        ax[k].imshow(cikis, cmap="gray", vmin=0, vmax=255)
        ax[k].set_title(f"clipLimit={etiket}\nyerel kontrast={yerel_kontrast(cikis):.1f}",
                        fontsize=8)
        ax[k].axis("off")

    plt.tight_layout()
    yol = os.path.join(CIKTI, "soru3_cliplimit_taramasi.png")
    plt.savefig(yol, dpi=105)
    plt.close()
    print(f"\nclipLimit tarama grafigi: {yol}")
    print("  Dusuk clipLimit -> donusum birim fonksiyona yaklasir (etki az).")
    print("  Yuksek clipLimit -> AHE'ye yaklasir, gurultu de yukselir.")


if __name__ == "__main__":
    main()
