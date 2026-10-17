"""Backend FastAPI: ecualización interactiva RGB (porcentajes 0-100 por canal)
y histogramas originales/actualizados."""

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

import image_utils as iu

app = FastAPI(title="Procesador de imágenes RGB")

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
    """Imagen de prueba (bajo contraste) para demostrar la ecualización."""
    img = iu.generate_sample_image()
    buf = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/api/prepare")
async def prepare(file: UploadFile = File(...)):
    """Calcula una sola vez, para cada imagen cargada:

    - los histogramas originales reales de R, G y B, y
    - las tablas de transformación completas T_eq de cada canal.

    El cliente reutiliza estos datos para actualizar la imagen y los ocho
    histogramas sin volver a enviar la imagen en cada movimiento de slider.
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
    eq_r: float = Form(0),
    eq_g: float = Form(0),
    eq_b: float = Form(0),
):
    """Aplica la ecualización progresiva (0-100 %) de cada canal RGB.

    Cada canal se ecualiza de forma independiente a partir de su histograma
    original: T_p(k) = (1 - p) * k + p * T_eq(k).
    """
    _, img = await _read_image(file)
    _check_percent("eq_r", eq_r)
    _check_percent("eq_g", eq_g)
    _check_percent("eq_b", eq_b)

    equalized = iu.process_rgb(img, eq_r, eq_g, eq_b)
    prepared = iu.prepare_image(img)
    hist_after = iu.compute_histograms(equalized)

    return {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "image_before": iu.to_data_url(img),
        "image_after": iu.to_data_url(equalized),
        "hist_original": prepared["hist_original"],
        "hist_equalized": {name: hist_after[name].tolist() for name in ("r", "g", "b")},
        "eq": {"r": eq_r, "g": eq_g, "b": eq_b},
    }
