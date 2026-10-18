"""Procesamiento de imágenes: histogramas RGB y efecto "arena" por canal.

Efecto arena (por canal, de forma independiente)
-----------------------------------------------
1. NIVEL DE CORTE: ``nivel`` (0-100) fija una línea al ``nivel`` % del máximo
   del canal. Todo lo que queda por encima de esa línea es la "montaña".
2. RECORTE CURVO: ``pico`` (0-100, por defecto 40) decide cuánto baja la cima.
   - ``pico = 100``  -> no se recorta nada (imagen original).
   - ``pico -> 0``   -> la montaña se aplana justo al nivel.
   - valores medios  -> recorte CURVO: los bins más altos pierden
     proporcionalmente más, así el pico deja de ser un pincho sin volverse
     una línea recta. Solo se tocan los bins por encima del nivel, nunca los
     vecinos (no se reparte a los bins de al lado).
3. La arena recortada se reparte POR IGUAL entre las intensidades que quedan
   por debajo del nivel: la base se infla de forma uniforme.
4. Los píxeles se reasignan de forma determinista: se ordenan por intensidad
   y, dentro de cada intensidad, por un hash de su posición (así el reparto
   no forma bandas). El histograma de la imagen resultante es, exactamente,
   el histograma objetivo calculado en los pasos 2 y 3.

El total de píxeles se conserva siempre y ``pico = 100`` (o ``nivel = 100``)
devuelve la imagen original sin cambios. Para calcular cualquier estado se
parte siempre del histograma ORIGINAL: el proceso no acumula.
"""

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

CHANNELS = ("r", "g", "b")

#: Nivel de corte por defecto, en % del máximo del canal.
DEFAULT_NIVEL = 20.0

#: "Pico" por defecto: 100 = pico original, 0 = recorte total al nivel.
DEFAULT_PICO = 40.0

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


# ------------------------------------------------------- nivel de corte ---


def _nivel_fraction(nivel) -> float:
    """Convierte el nivel (0-100 %) a fracción del máximo, recortando al rango."""
    return min(max(float(nivel), 0.0), 100.0) / 100.0


def level_threshold(hist, nivel: float = DEFAULT_NIVEL) -> float:
    """Cuenta del nivel de corte: ``nivel`` % del máximo del canal."""
    h = _as_histogram(hist)
    return float(h.max()) * _nivel_fraction(nivel)


def level_mask(hist, nivel: float = DEFAULT_NIVEL) -> np.ndarray:
    """Booleanos: qué intensidades quedan por encima del nivel de corte.

    Si ninguna lo supera (histograma vacío, plano o ``nivel = 100``) se toma
    la intensidad de cuenta máxima, para no quedarnos sin zona alta.
    """
    h = _as_histogram(hist)
    umbral = h.max() * _nivel_fraction(nivel)
    mask = h > umbral
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


def _soft_clip(exceso, nivel: float, pico: float):
    """Recorta el exceso con una curva: los valores más altos pierden más.

    Devuelve el exceso que SOBREVIVE al recorte. Con ``pico = 100`` no se
    recorta nada (devuelve el exceso tal cual); cuanto menor sea ``pico``,
    más se acerca el resultado a 0 (la cima baja hasta el nivel). Como la
    curva es monótona, nunca invierte el orden de los bins.
    """
    ex = np.asarray(exceso, dtype=np.float64)
    if nivel <= 0:
        return np.zeros_like(ex)
    p = min(max(float(pico), 0.0), 100.0)
    k = 100.0 / max(p, 1e-6) - 1.0
    return ex / (1.0 + k * ex / nivel)


def sand_target_histogram(hist, nivel: float = 20.0, pico: float = 40.0) -> np.ndarray:
    """Histograma objetivo tras recortar la montaña y esparcir la arena.

    - Se recorta el exceso sobre el nivel con la curva de ``_soft_clip``.
    - La arena recortada se reparte POR IGUAL entre las intensidades que
      quedan por debajo del nivel.
    """
    h = _as_histogram(hist)
    n = int(h.sum())
    if n <= 0:
        return h.astype(np.int64)

    p = min(max(float(pico), 0.0), 100.0)
    mask = level_mask(h, nivel)
    fuera = ~mask
    if p >= 100.0 or not fuera.any():  # sin recorte o sin base: nada que hacer
        return h.astype(np.int64)

    umbral = h.max() * _nivel_fraction(nivel)
    exceso = np.maximum(h - umbral, 0.0)
    objetivo = h.copy()
    objetivo[mask] = umbral + _soft_clip(exceso[mask], umbral, p)
    arena = h[mask].sum() - objetivo[mask].sum()  # píxeles que se desprenden
    objetivo[fuera] += arena / fuera.sum()        # la base se infla por igual
    return _round_histogram(objetivo, int(h.sum()))


# --------------------------------------------------- reasignación (píxel) ---


def _assignment_order(flat: np.ndarray) -> np.ndarray:
    """Orden determinista de los píxeles: por intensidad y, dentro de cada
    intensidad, por un hash de su posición (evita bandas espaciales)."""
    posiciones = np.arange(flat.size, dtype=np.uint64)
    hash_pos = (posiciones * _POSITION_HASH) & np.uint64(0xFFFFFFFF)
    return np.lexsort((hash_pos, flat))


def sand_channel(channel: np.ndarray, nivel: float = 20.0, pico: float = 40.0) -> np.ndarray:
    """Aplica el efecto arena a un canal uint8 y devuelve el canal resultante.

    El histograma de salida coincide exactamente con ``sand_target_histogram``.
    """
    ch = np.asarray(channel, dtype=np.uint8)
    hist = channel_histogram(ch)
    objetivo = sand_target_histogram(hist, nivel, pico)
    if np.array_equal(objetivo, hist):  # sin recorte (o sin base): nada que hacer
        return ch.copy()

    # Lista de destinos: cada intensidad repetida tantas veces como píxeles
    # debe tener en el resultado.
    destinos = np.repeat(np.arange(256, dtype=np.uint8), objetivo)
    orden = _assignment_order(ch.ravel())

    salida = np.empty(ch.size, dtype=np.uint8)
    salida[orden] = destinos  # el píxel más bajo recibe el destino más bajo
    return salida.reshape(ch.shape)


def sand_rgb(
    img: np.ndarray,
    n_r: float = 20.0,
    n_g: float = 20.0,
    n_b: float = 20.0,
    pico: float = 40.0,
) -> np.ndarray:
    """Aplica el efecto arena a los tres canales RGB de forma independiente."""
    salida = np.empty_like(img)
    for i, nivel in enumerate((n_r, n_g, n_b)):
        salida[:, :, i] = sand_channel(img[:, :, i], nivel, pico)
    return salida


# ---------------------------------------------------------- preparación ---


def prepare_image(img: np.ndarray, nivel: float = 20.0) -> dict:
    """Histogramas originales y zona de corte de cada canal (se calculan una vez)."""
    hist = compute_histograms(img)
    peaks = {}
    for name in CHANNELS:
        mask = np.flatnonzero(level_mask(hist[name], nivel))
        if mask.size == 0:  # imagen vacía: sin zona
            inicio = fin = 0
        else:
            inicio = int(mask[0])
            fin = int(mask[-1])
        peaks[name] = {
            "inicio": inicio,
            "fin": fin,
            "umbral": level_threshold(hist[name], nivel),
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
