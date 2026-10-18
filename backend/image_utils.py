import base64
import io
from collections.abc import Sequence
from typing import TypedDict

import numpy as np
from PIL import Image, ImageDraw

HistogramLike = np.ndarray | Sequence[int] | Sequence[float]


class PeakInfo(TypedDict):
    inicio: int
    fin: int
    umbral: float


class PreparedImage(TypedDict):
    hist_original: dict[str, list[int]]
    peaks: dict[str, PeakInfo]

CHANNELS: tuple[str, str, str] = ("r", "g", "b")

DEFAULT_NIVEL: float = 20.0

DEFAULT_PICO: float = 40.0

_POSITION_HASH: np.uint64 = np.uint64(2654435761)


def load_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def to_data_url(img: np.ndarray, fmt: str = "PNG") -> str:
    pil = Image.fromarray(img, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    if fmt.upper() == "PNG":
        mime = "image/png"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64,{b64}"


def channel_histogram(channel: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(channel).ravel(), minlength=256).astype(np.int64)


def compute_histograms(img: np.ndarray) -> dict[str, np.ndarray]:
    hists: dict[str, np.ndarray] = {}
    for i, name in enumerate(CHANNELS):
        hists[name] = channel_histogram(img[:, :, i])
    return hists


def _as_histogram(hist: HistogramLike) -> np.ndarray:
    h = np.asarray(hist, dtype=np.float64).ravel()
    if h.size != 256:
        raise ValueError("El histograma debe tener 256 niveles.")
    if np.any(h < 0):
        raise ValueError("El histograma no puede contener valores negativos.")
    return h


def _clamp_percent(valor: float) -> float:
    if valor < 0.0:
        return 0.0
    if valor > 100.0:
        return 100.0
    return valor


def _nivel_fraction(nivel: float) -> float:
    return _clamp_percent(float(nivel)) / 100.0


def level_threshold(hist: HistogramLike, nivel: float = DEFAULT_NIVEL) -> float:
    h = _as_histogram(hist)
    return float(h.max()) * _nivel_fraction(nivel)


def level_mask(hist: HistogramLike, nivel: float = DEFAULT_NIVEL) -> np.ndarray:
    h = _as_histogram(hist)
    umbral = h.max() * _nivel_fraction(nivel)
    mask = h > umbral
    if not mask.any():
        mask = h == h.max()
    if h.sum() > 0:
        return mask
    return np.zeros(256, dtype=bool)


def _round_histogram(hist: np.ndarray, total: int) -> np.ndarray:
    base = np.floor(hist).astype(np.int64)
    resto = int(total) - int(base.sum())
    if resto > 0:
        sobrantes = np.argsort(-(hist - base))[:resto]
        base[sobrantes] += 1
    return base


def _soft_clip(exceso: np.ndarray, nivel: float, pico: float) -> np.ndarray:
    ex = np.asarray(exceso, dtype=np.float64)
    if nivel <= 0:
        return np.zeros_like(ex)
    p = _clamp_percent(float(pico))
    k = 100.0 / max(p, 1e-6) - 1.0
    return ex / (1.0 + k * ex / nivel)


def sand_target_histogram(
    hist: HistogramLike,
    nivel: float = DEFAULT_NIVEL,
    pico: float = DEFAULT_PICO,
) -> np.ndarray:
    h = _as_histogram(hist)
    n = int(h.sum())
    if n <= 0:
        return h.astype(np.int64)

    p = _clamp_percent(float(pico))
    mask = level_mask(h, nivel)
    fuera = ~mask
    if p >= 100.0 or not fuera.any():
        return h.astype(np.int64)

    umbral = h.max() * _nivel_fraction(nivel)
    exceso = np.maximum(h - umbral, 0.0)
    objetivo = h.copy()
    objetivo[mask] = umbral + _soft_clip(exceso[mask], umbral, p)
    arena = h[mask].sum() - objetivo[mask].sum()
    objetivo += arena / objetivo.size
    return _round_histogram(objetivo, int(h.sum()))


def _assignment_order(flat: np.ndarray) -> np.ndarray:
    posiciones = np.arange(flat.size, dtype=np.uint64)
    hash_pos = (posiciones * _POSITION_HASH) & np.uint64(0xFFFFFFFF)
    return np.lexsort((hash_pos, flat))


def sand_channel(
    channel: np.ndarray,
    nivel: float = DEFAULT_NIVEL,
    pico: float = DEFAULT_PICO,
) -> np.ndarray:
    ch = np.asarray(channel, dtype=np.uint8)
    hist = channel_histogram(ch)
    objetivo = sand_target_histogram(hist, nivel, pico)
    if np.array_equal(objetivo, hist):
        return ch.copy()

    destinos = np.repeat(np.arange(256, dtype=np.uint8), objetivo)
    orden = _assignment_order(ch.ravel())

    salida = np.empty(ch.size, dtype=np.uint8)
    salida[orden] = destinos
    return salida.reshape(ch.shape)


def sand_rgb(
    img: np.ndarray,
    n_r: float = DEFAULT_NIVEL,
    n_g: float = DEFAULT_NIVEL,
    n_b: float = DEFAULT_NIVEL,
    pico: float = DEFAULT_PICO,
) -> np.ndarray:
    niveles = (n_r, n_g, n_b)
    salida = np.empty_like(img)
    for i, nivel in enumerate(niveles):
        salida[:, :, i] = sand_channel(img[:, :, i], nivel, pico)
    return salida


def prepare_image(img: np.ndarray, nivel: float = DEFAULT_NIVEL) -> PreparedImage:
    hist = compute_histograms(img)
    peaks: dict[str, PeakInfo] = {}
    for name in CHANNELS:
        mask = np.flatnonzero(level_mask(hist[name], nivel))
        if mask.size == 0:
            inicio = 0
            fin = 0
        else:
            inicio = int(mask[0])
            fin = int(mask[-1])
        peaks[name] = {
            "inicio": inicio,
            "fin": fin,
            "umbral": level_threshold(hist[name], nivel),
        }

    hist_original: dict[str, list[int]] = {}
    for name in CHANNELS:
        hist_original[name] = hist[name].tolist()

    return {"hist_original": hist_original, "peaks": peaks}


def generate_sample_image(width: int = 480, height: int = 360) -> np.ndarray:
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
