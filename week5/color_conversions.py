"""Week 5 / Task 4: uint8 BGR -> renk uzayları, hazır cvtColor kullanılmaz.

Formüller: dersteki Renk_Uzaylari_Karsilastirma.pdf.
8-bit OpenCV eşleşmesi için kaynakta açıklanan sabit nokta ölçekleri:
https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/color_hsv.simd.hpp
https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/color_yuv.simd.hpp
https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/color_lab.cpp
Bu dosya OpenCV'yi import etmez; cvtColor yalnızca notebook'ta referanstır.
"""
import numpy as np


def validate(image):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('uint8, H×W×3 BGR görüntü bekleniyor.')
    return image.astype(np.int32)


def to_ycrcb(image):
    """BT.601 tam aralık, kanal sırası Y/Cr/Cb; 14-bit sabit nokta."""
    b, g, r = np.moveaxis(validate(image), -1, 0)
    y = (1868*b + 9617*g + 4899*r + 8192) >> 14
    cr = ((r-y)*11682 + (128 << 14) + 8192) >> 14
    cb = ((b-y)*9241 + (128 << 14) + 8192) >> 14
    return np.clip(np.stack((y, cr, cb), -1), 0, 255).astype(np.uint8)


def to_hsv(image):
    """H 0..179, S/V 0..255; 12-bit bölme tablolarıyla eşdeğer HSV hesabı."""
    b, g, r = np.moveaxis(validate(image), -1, 0)
    v = np.maximum(np.maximum(r, g), b)
    delta = v - np.minimum(np.minimum(r, g), b)
    sdiv = np.zeros(256, np.int32); hdiv = np.zeros(256, np.int32)
    i = np.arange(1, 256)
    sdiv[1:] = np.rint((255 << 12) / i).astype(np.int32)
    hdiv[1:] = np.rint((180 << 12) / (6*i)).astype(np.int32)
    s = (delta*sdiv[v] + 2048) >> 12
    h = np.where(v == r, g-b, np.where(v == g, b-r+2*delta, r-g+4*delta))
    h = (h*hdiv[delta] + 2048) >> 12
    h = np.where(h < 0, h+180, h)
    return np.stack((h, s, v), -1).astype(np.uint8)


def to_lab(image):
    """sRGB -> CIELAB D65; 8-bit L*255/100, a+128, b+128.

    Gamma ve küpkök tabloları formüllerden üretilir. 12-bit XYZ ve
    15-bit Lab sabit nokta ölçekleri OpenCV'nin uint8 temsilini eşler.
    """
    validate(image)
    x = np.arange(256, dtype=np.float32)/np.float32(255)
    linear = np.where(x <= 0.04045, x.astype(float)/12.92,
                      ((x.astype(float)+0.055)/1.055)**2.4).astype(np.float32)
    gamma = np.rint(np.float32(2040)*linear).astype(np.int32)
    t = np.float32(1/2040)*np.arange(3072, dtype=np.float32)
    scale, bias = np.float32(841/108), np.float32(16/116)
    low = (t.astype(float)*float(scale)+float(bias)).astype(np.float32)
    f = np.where(t < np.float32(216/24389), low, np.cbrt(t))
    lut = np.rint(np.float32(32768)*f).astype(np.int32)
    matrix = np.array([[0.412453, 0.357580, 0.180423],
                       [0.212671, 0.715160, 0.072169],
                       [0.019334, 0.119193, 0.950227]])
    coeff = np.rint(4096*matrix/np.array([0.950456, 1, 1.088754])[:, None]).astype(np.int32)[:, ::-1]
    xyz = (gamma[image] @ coeff.T + 2048) >> 12
    fx, fy, fz = np.moveaxis(lut[xyz], -1, 0)
    L = (296*fy - ((16*255*32768+50)//100) + 16384) >> 15
    a = (500*(fx-fy) + 128*32768 + 16384) >> 15
    b = (200*(fy-fz) + 128*32768 + 16384) >> 15
    return np.clip(np.stack((L, a, b), -1), 0, 255).astype(np.uint8)


def to_normalized_rgb(image):
    """r+g+b=1; siyahın tanımsız kromatikliği için (0,0,0) seçilir."""
    rgb = validate(image)[..., ::-1].astype(np.float64)
    total = rgb.sum(-1, keepdims=True)
    return np.divide(rgb, total, out=np.zeros_like(rgb), where=total != 0)


def to_hsi(image):
    """H derece 0..360, S/I 0..1; gri için H=0, siyah için S=0."""
    r, g, b = np.moveaxis(validate(image)[..., ::-1].astype(float)/255, -1, 0)
    total = r+g+b
    denominator = np.sqrt((r-g)**2 + (r-b)*(g-b))
    cosine = np.divide((2*r-g-b)/2, denominator, out=np.ones_like(r), where=denominator > 1e-15)
    theta = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    h = np.where(b <= g, theta, 360-theta)
    h = np.where(denominator > 1e-15, h, 0)
    s = 1 - np.divide(3*np.minimum(np.minimum(r, g), b), total, out=np.ones_like(r), where=total != 0)
    return np.stack((h, s, total/3), -1)
