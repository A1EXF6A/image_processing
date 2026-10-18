"""Backend FastAPI: efecto "arena" sobre el histograma RGB.

El nivel de corte de cada canal (0-100 % del máximo) marca la "montaña"; el
recorte curvo la baja según el ``pico`` (fijo en 40 por defecto) y la arena se
esparce por igual entre las intensidades por debajo del nivel. Devuelve además
los histogramas originales y actualizados de R, G y B.
"""

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

import image_utils as iu

app = FastAPI(title="Procesador de imágenes RGB · efecto arena")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


async def _read_image(file: UploadFile):
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="La imagen supera los 20 MB.")
    try:
        img = iu.load_image(data)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")
    return data, img


def _check_percent(name: str, value: float) -> float:
    if not 0 <= value <= 100:
        raise HTTPException(status_code=422, detail=f"{name} debe estar entre 0 y 100.")
    return value


@app.get("/api/sample")
def get_sample():
    """Imagen de prueba (bajo contraste) para demostrar el efecto arena."""
    img = iu.generate_sample_image()
    buf = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/api/prepare")
async def prepare(file: UploadFile = File(...)):
    """Calcula una sola vez, para cada imagen cargada:

    - los histogramas originales reales de R, G y B, y
    - la zona de corte de cada canal (rango de intensidades y umbral) al
      nivel por defecto, para saber de dónde sale la arena.

    Así el cliente conoce la imagen antes de mover los sliders.
    """
    _, img = await _read_image(file)
    payload = iu.prepare_image(img, iu.DEFAULT_NIVEL)
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "image_before": iu.to_data_url(img),
        **payload,
    }


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    nivel_r: float = Form(iu.DEFAULT_NIVEL),
    nivel_g: float = Form(iu.DEFAULT_NIVEL),
    nivel_b: float = Form(iu.DEFAULT_NIVEL),
    pico: float = Form(iu.DEFAULT_PICO),
    format: str = Form("png"),
):
    """Aplica el efecto arena a cada canal RGB.

    Cada canal se procesa de forma independiente a partir de su histograma
    original: se recorta con una curva todo lo que supera su ``nivel``
    (0-100 % del máximo) según ``pico`` (100 = pico original, 0 = plano) y esa
    arena se reparte por igual entre las intensidades por debajo del nivel.

    ``format`` elige cómo se devuelve ``image_after``: ``png`` (sin pérdida,
    por defecto, para descargar) o ``jpeg`` (vista previa, mucho más liviana).
    No devuelve ``image_before`` (el cliente ya la recibió en /api/prepare).
    """
    _, img = await _read_image(file)
    nivel_r = _check_percent("nivel_r", nivel_r)
    nivel_g = _check_percent("nivel_g", nivel_g)
    nivel_b = _check_percent("nivel_b", nivel_b)
    pico = _check_percent("pico", pico)
    fmt = format.strip().lower()
    if fmt not in ("png", "jpeg"):
        raise HTTPException(status_code=422, detail="format debe ser 'png' o 'jpeg'.")

    procesada = iu.sand_rgb(img, nivel_r, nivel_g, nivel_b, pico)
    hist_original = iu.compute_histograms(img)
    hist_after = iu.compute_histograms(procesada)

    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "image_after": iu.to_data_url(procesada, fmt.upper()),
        "hist_original": {name: hist_original[name].tolist() for name in ("r", "g", "b")},
        "hist_after": {name: hist_after[name].tolist() for name in ("r", "g", "b")},
        "peaks": iu.prepare_image(img, iu.DEFAULT_NIVEL)["peaks"],
        "nivel": {"r": nivel_r, "g": nivel_g, "b": nivel_b},
        "pico": pico,
    }
