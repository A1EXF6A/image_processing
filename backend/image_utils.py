"""Procesamiento de imágenes: histogramas RGB y efecto "arena" por canal.

Efecto arena (por canal, de forma independiente)
------------------------------------------------
1. ZONA DEL PICO: se marca cada intensidad cuya cuenta supera el
   ``PEAK_FRACTION`` (45 %) del máximo del canal. Ese es el "montón de arena".
2. El slider ``percent`` (0-100) define cuánto se DESINFLA el pico: se le
   quitan ese porcentaje de su *exceso* (la cuenta por encima del umbral).
3. Los píxeles quitados se reparten POR IGUAL entre TODAS las intensidades
   que no son pico: "la arena se esparce por la base" y todo lo demás se
   infla de la misma forma.
4. Los píxeles se reasignan de forma determinista: se ordenan por intensidad
   y, dentro de cada intensidad, por un hash de su posición (así el reparto
   no forma bandas). El histograma de la imagen resultante es, exactamente,
   el histograma objetivo calculado en los pasos 2 y 3.

El total de píxeles se conserva siempre y ``percent=0`` devuelve la imagen
original sin cambios. Para calcular cualquier estado se parte siempre del
histograma ORIGINAL: el proceso no acumula.
"""

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

CHANNELS = ("r", "g", "b")

#: Una intensidad pertenece a la zona del pico si su cuenta supera este
#: porcentaje del máximo del canal.
PEAK_FRACTION = 0.45

#: Multiplicador impar (Knuth) para mezclar posiciones de píxel de forma
#: determinista al reasignar.
_POSITION_HASH = np.uint64(2654435761)


# --------------------------------------------------------------- imagen ---


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


# ---------------------------------------------------------- histogramas ---


def channel_histogram(channel: np.ndarray) -> np.ndarray:
    """Histograma de 256 niveles de un canal uint8."""
    return np.bincount(np.asarray(channel).ravel(), minlength=256).astype(np.int64)


def compute_histograms(img: np.ndarray) -> dict:
    """Histogramas de 256 bins para los canales R, G y B (datos reales de los píxeles)."""
    return {name: channel_histogram(img[:, :, i]) for i, name in enumerate(CHANNELS)}


def _as_histogram(hist) -> np.ndarray:
    h = np.asarray(hist, dtype=np.float64).ravel()
    if h.size != 256:
        raise ValueError("El histograma debe tener 256 niveles.")
    if np.any(h < 0):
        raise ValueError("El histograma no puede contener valores negativos.")
    return h


# --------------------------------------------------------- zona del pico ---


def peak_threshold(hist) -> float:
    """Cuenta mínima para pertenecer a la zona del pico: 45 % del máximo."""
    return float(_as_histogram(hist).max()) * PEAK_FRACTION


def peak_mask(hist, fraction: float = PEAK_FRACTION) -> np.ndarray:
    """Booleanos: qué intensidades forman la zona del pico (cuenta > umbral).

    Si ninguna supera el umbral (histograma vacío o casi plano) se toma como
    pico la intensidad con la cuenta máxima, para no quedarnos sin arena.
    """
    h = _as_histogram(hist)
    mask = h > h.max() * fraction
    if not mask.any():
        mask = h == h.max()
    return mask if h.sum() > 0 else np.zeros(256, dtype=bool)


# ------------------------------------------- histograma objetivo (arena) ---


def _round_histogram(hist: np.ndarray, total: int) -> np.ndarray:
    """Redondea un histograma fraccionario a enteros conservando el total."""
    base = np.floor(hist).astype(np.int64)
    resto = int(total) - int(base.sum())
    if resto > 0:  # las partes fraccionarias más grandes se llevan +1
        sobrantes = np.argsort(-(hist - base))[:resto]
        base[sobrantes] += 1
    return base


def sand_target_histogram(hist, percent: float, fraction: float = PEAK_FRACTION) -> np.ndarray:
    """Histograma objetivo tras desinflar el pico y esparcir la arena.

    - Se quita del pico ``percent`` % de su exceso (la cuenta sobre el umbral).
    - Esa cantidad se reparte POR IGUAL entre las intensidades que no son pico.
    """
    h = _as_histogram(hist)
    n = int(h.sum())
    if n <= 0:
        return h.astype(np.int64)

    p = min(max(float(percent), 0.0), 100.0) / 100.0
    mask = peak_mask(h, fraction)
    fuera = ~mask
    if p == 0.0 or not fuera.any():
        return h.astype(np.int64)

    umbral = h.max() * fraction
    exceso = np.maximum(h - umbral, 0.0)  # arena disponible en el pico
    arena = p * exceso.sum()              # píxeles que se desprenden

    objetivo = h - p * exceso             # el pico se desinfla
    objetivo[fuera] += arena / fuera.sum()  # y la base se infla por igual
    return _round_histogram(objetivo, n)


# --------------------------------------------------- reasignación (píxel) ---


def _assignment_order(flat: np.ndarray) -> np.ndarray:
    """Orden determinista de los píxeles: por intensidad y, dentro de cada
    intensidad, por un hash de su posición (evita bandas espaciales)."""
    posiciones = np.arange(flat.size, dtype=np.uint64)
    hash_pos = (posiciones * _POSITION_HASH) & np.uint64(0xFFFFFFFF)
    return np.lexsort((hash_pos, flat))


def sand_channel(channel: np.ndarray, percent: float, fraction: float = PEAK_FRACTION) -> np.ndarray:
    """Aplica el efecto arena a un canal uint8 y devuelve el canal resultante.

    El histograma de salida coincide exactamente con ``sand_target_histogram``.
    """
    ch = np.asarray(channel, dtype=np.uint8)
    hist = channel_histogram(ch)
    objetivo = sand_target_histogram(hist, percent, fraction)
    if np.array_equal(objetivo, hist):  # 0 % (o sin arena): nada que hacer
        return ch.copy()

    # Lista de destinos: cada intensidad repetida tantas veces como píxeles
    # debe tener en el resultado.
    destinos = np.repeat(np.arange(256, dtype=np.uint8), objetivo)
    orden = _assignment_order(ch.ravel())

    salida = np.empty(ch.size, dtype=np.uint8)
    salida[orden] = destinos  # el píxel más bajo recibe el destino más bajo
    return salida.reshape(ch.shape)


def sand_rgb(img: np.ndarray, p_r: float, p_g: float, p_b: float) -> np.ndarray:
    """Aplica el efecto arena a los tres canales RGB de forma independiente."""
    salida = np.empty_like(img)
    for i, percent in enumerate((p_r, p_g, p_b)):
        salida[:, :, i] = sand_channel(img[:, :, i], percent)
    return salida


# ---------------------------------------------------------- preparación ---


def prepare_image(img: np.ndarray) -> dict:
    """Histogramas originales y zona del pico de cada canal (se calculan una vez)."""
    hist = compute_histograms(img)
    peaks = {}
    for name in CHANNELS:
        mask = np.flatnonzero(peak_mask(hist[name]))
        peaks[name] = {
            "inicio": int(mask[0]),
            "fin": int(mask[-1]),
            "umbral": peak_threshold(hist[name]),
        }
    return {
        "hist_original": {name: hist[name].tolist() for name in CHANNELS},
        "peaks": peaks,
    }


# ------------------------------------------------------ imagen de prueba ---


def generate_sample_image(width: int = 480, height: int = 360) -> np.ndarray:
    """Genera una imagen de prueba con bajo contraste (ideal para demostrar el efecto)."""
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
