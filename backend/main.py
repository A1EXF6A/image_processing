import io
from typing import TypedDict

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image

import image_utils as iu


class PrepareResponse(iu.PreparedImage):
    width: int
    height: int
    image_before: str


class ProcessResponse(TypedDict):
    width: int
    height: int
    image_after: str
    hist_original: dict[str, list[int]]
    hist_after: dict[str, list[int]]
    peaks: dict[str, iu.PeakInfo]
    nivel: dict[str, float]
    pico: float


app = FastAPI(title="Procesador de imágenes RGB · efecto arena")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE: int = 20 * 1024 * 1024


async def _read_image(file: UploadFile) -> tuple[bytes, np.ndarray]:
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
    if value < 0 or value > 100:
        raise HTTPException(status_code=422, detail=f"{name} debe estar entre 0 y 100.")
    return value


@app.get("/api/sample")
def get_sample() -> Response:
    img = iu.generate_sample_image()
    buf = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/api/prepare")
async def prepare(file: UploadFile = File(...)) -> PrepareResponse:
    _, img = await _read_image(file)
    preparado = iu.prepare_image(img, iu.DEFAULT_NIVEL)
    respuesta: PrepareResponse = {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "image_before": iu.to_data_url(img),
        "hist_original": preparado["hist_original"],
        "peaks": preparado["peaks"],
    }
    return respuesta


@app.post("/api/process")
async def process(
    file: UploadFile = File(...),
    nivel_r: float = Form(iu.DEFAULT_NIVEL),
    nivel_g: float = Form(iu.DEFAULT_NIVEL),
    nivel_b: float = Form(iu.DEFAULT_NIVEL),
    pico: float = Form(iu.DEFAULT_PICO),
    format: str = Form("png"),
) -> ProcessResponse:
    _, img = await _read_image(file)
    nivel_r = _check_percent("nivel_r", nivel_r)
    nivel_g = _check_percent("nivel_g", nivel_g)
    nivel_b = _check_percent("nivel_b", nivel_b)
    pico = _check_percent("pico", pico)
    fmt = format.strip().lower()
    if fmt != "png" and fmt != "jpeg":
        raise HTTPException(status_code=422, detail="format debe ser 'png' o 'jpeg'.")

    procesada = iu.sand_rgb(img, nivel_r, nivel_g, nivel_b, pico)
    hist_original = iu.compute_histograms(img)
    hist_after = iu.compute_histograms(procesada)

    hist_original_json: dict[str, list[int]] = {}
    hist_after_json: dict[str, list[int]] = {}
    for name in iu.CHANNELS:
        hist_original_json[name] = hist_original[name].tolist()
        hist_after_json[name] = hist_after[name].tolist()

    respuesta: ProcessResponse = {
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "image_after": iu.to_data_url(procesada, fmt.upper()),
        "hist_original": hist_original_json,
        "hist_after": hist_after_json,
        "peaks": iu.prepare_image(img, iu.DEFAULT_NIVEL)["peaks"],
        "nivel": {"r": nivel_r, "g": nivel_g, "b": nivel_b},
        "pico": pico,
    }
    return respuesta
