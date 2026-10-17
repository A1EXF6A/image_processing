"""Procesamiento de imágenes: histogramas RGB, tablas de transformación y
ecualización progresiva por canal (R, G, B) independiente."""

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

CHANNELS = ("r", "g", "b")
IDENTITY_LUT = np.arange(256, dtype=np.uint8)


def load_image(data: bytes) -> np.ndarray:
    """Decodifica bytes (JPG/PNG/...) a un arreglo RGB uint8 de shape (H, W, 3)."""
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def to_data_url(img: np.ndarray, fmt: str = "PNG") -> str:
    """Convierte un arreglo RGB uint8 a data-URL (base64) listo para <img>/canvas."""
    pil = Image.fromarray(img, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def channel_histogram(channel: np.ndarray) -> np.ndarray:
    """Histograma de 256 niveles de un canal uint8."""
    return np.bincount(np.asarray(channel).ravel(), minlength=256).astype(np.int64)


def compute_histograms(img: np.ndarray) -> dict:
    """Histogramas de 256 bins para los canales R, G y B (datos reales de los píxeles)."""
    return {name: channel_histogram(img[:, :, i]) for i, name in enumerate(CHANNELS)}


def equalization_lut(hist) -> np.ndarray:
    """Tabla de transformación completa T_eq(k) a partir de un histograma de 256 bins.

    T_eq(k) = floor((cdf(k) - cdf_min) / (N - cdf_min) * 255), recortado a [0, 255].

    Si no existe variación suficiente (imagen constante, histograma vacío o todo
    concentrado en un único nivel) se devuelve la transformación de identidad.
    """
    h = np.asarray(hist, dtype=np.float64).ravel()
    if h.size != 256:
        raise ValueError("El histograma debe tener 256 niveles.")
    if np.any(h < 0):
        raise ValueError("El histograma no puede contener valores negativos.")

    cdf = h.cumsum()
    n_pixels = cdf[-1]
    if n_pixels <= 0:
        return IDENTITY_LUT.copy()

    nonzero = np.flatnonzero(cdf)
    cdf_min = cdf[nonzero[0]] if nonzero.size else 0.0
    denom = n_pixels - cdf_min
    if denom <= 0:
        return IDENTITY_LUT.copy()

    lut = np.floor((cdf - cdf_min) / denom * 255.0)
    return np.clip(lut, 0, 255).astype(np.uint8)


def interpolate_lut(lut, p: float) -> np.ndarray:
    """T_p(k) = (1 - p) * k + p * T_eq(k), redondeado y recortado a [0, 255].

    p es el porcentaje del slider normalizado entre 0 y 1.
    """
    p = float(p)
    if not np.isfinite(p):
        raise ValueError("El porcentaje debe ser un número finito.")
    p = min(max(p, 0.0), 1.0)
    base = np.asarray(lut, dtype=np.uint8).ravel()
    if base.size != 256:
        raise ValueError("La tabla de transformación debe tener 256 entradas.")

    k = np.arange(256, dtype=np.float64)
    out = np.rint((1.0 - p) * k + p * base.astype(np.float64))
    return np.clip(out, 0, 255).astype(np.uint8)


def build_luts(img: np.ndarray) -> dict:
    """Tablas T_eq de ecualización completa para R, G y B calculadas SIEMPRE
    sobre el histograma de la imagen original."""
    return {
        name: equalization_lut(channel_histogram(img[:, :, i]))
        for i, name in enumerate(CHANNELS)
    }


def apply_lut(channel: np.ndarray, lut) -> np.ndarray:
    """Aplica una tabla de transformación a un canal uint8."""
    table = np.asarray(lut, dtype=np.uint8).ravel()
    if table.size != 256:
        raise ValueError("La tabla de transformación debe tener 256 entradas.")
    return table[np.asarray(channel, dtype=np.uint8)]


def process_rgb(img: np.ndarray, p_r: float, p_g: float, p_b: float) -> np.ndarray:
    """Ecualiza progresivamente los tres canales RGB de forma independiente.

    p_r, p_g, p_b son porcentajes (0..100). El cálculo parte siempre de los
    píxeles originales de `img`, por lo que no acumula transformaciones.
    """
    luts = build_luts(img)
    percents = {"r": p_r, "g": p_g, "b": p_b}

    out = np.empty_like(img)
    for i, name in enumerate(CHANNELS):
        table = interpolate_lut(luts[name], percents[name] / 100.0)
        out[:, :, i] = apply_lut(img[:, :, i], table)
    return out


def prepare_image(img: np.ndarray) -> dict:
    """Histogramas originales y tablas T_eq completas (se calculan una sola vez)."""
    hist = compute_histograms(img)
    return {
        "hist_original": {name: hist[name].tolist() for name in CHANNELS},
        "luts": {name: equalization_lut(hist[name]).tolist() for name in CHANNELS},
    }


def generate_sample_image(width: int = 480, height: int = 360) -> np.ndarray:
    """Genera una imagen de prueba con bajo contraste (ideal para demostrar la ecualización)."""
    y = np.linspace(0, 1, height, dtype=np.float64)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float64)[None, :]

    r = 70 + 40 * y + 20 * x
    g = 90 + 30 * y + 15 * (1 - x)
    b = 110 + 25 * (1 - y) + 0 * x
    img = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)

    pil = Image.fromarray(img, mode="RGB")
    draw = ImageDraw.Draw(pil)

    draw.ellipse((330, 40, 400, 110), fill=(160, 150, 120))
    draw.polygon([(0, 280), (140, 160), (280, 280)], fill=(85, 95, 100))
    draw.polygon([(180, 280), (320, 180), (460, 280)], fill=(100, 105, 110))
    draw.rectangle((0, 280, 480, 360), fill=(75, 85, 80))
    draw.ellipse((60, 300, 180, 345), fill=(110, 100, 90))

    return np.asarray(pil, dtype=np.uint8)
