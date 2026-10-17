"""Backend FastAPI: efecto "arena" sobre el histograma RGB.

El pico de cada canal se desinfla y la arena se esparce por igual entre las
demás intensidades (porcentajes 0-100 por canal). Devuelve además los
histogramas originales y actualizados de R, G y B.
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
    - la zona del pico de cada canal (rango de intensidades y umbral).

    Así el cliente conoce de dónde sale la arena antes de mover los sliders.
    """
    _, img = await _read_image(file)
    payload = iu.prepare_image(img)
    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "image_before": iu.to_data_url(img),
        **payload,
    }


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    cut_r: float = Form(0),
    cut_g: float = Form(0),
    cut_b: float = Form(0),
    format: str = Form("png"),
):
    """Aplica el efecto arena (0-100 %) a cada canal RGB.

    Cada canal se procesa de forma independiente a partir de su histograma
    original: el pico pierde ``cut`` % de su exceso y esos píxeles se reparten
    por igual entre las demás intensidades.

    ``format`` elige cómo se devuelve ``image_after``: ``png`` (sin pérdida,
    por defecto, para descargar) o ``jpeg`` (vista previa, mucho más liviana).
    No devuelve ``image_before`` (el cliente ya la recibió en /api/prepare).
    """
    _, img = await _read_image(file)
    _check_percent("cut_r", cut_r)
    _check_percent("cut_g", cut_g)
    _check_percent("cut_b", cut_b)
    fmt = format.strip().lower()
    if fmt not in ("png", "jpeg"):
        raise HTTPException(status_code=422, detail="format debe ser 'png' o 'jpeg'.")

    procesada = iu.sand_rgb(img, cut_r, cut_g, cut_b)
    hist_original = iu.compute_histograms(img)
    hist_after = iu.compute_histograms(procesada)

    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "image_after": iu.to_data_url(procesada, fmt.upper()),
        "hist_original": {name: hist_original[name].tolist() for name in ("r", "g", "b")},
        "hist_after": {name: hist_after[name].tolist() for name in ("r", "g", "b")},
        "peaks": iu.prepare_image(img)["peaks"],
        "cut": {"r": cut_r, "g": cut_g, "b": cut_b},
    }
