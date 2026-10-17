"""Pruebas del efecto "arena" (recorte del pico + reparto uniforme).

Cobertura:
  1. 0 % -> imagen idéntica a la original
  2. detección de la zona del pico (umbral = 45 % del máximo)
  3. el histograma objetivo conserva el total y no tiene negativos
  4. el pico se desinfla y la base se infla (reparto por igual)
  5. el histograma del canal procesado coincide EXACTAMENTE con el objetivo
  6. el procesamiento es determinista
  7. el pico solo baja y la base solo sube al mover el slider (monotonía)
  8. independencia de los canales (función y API)
  9. los histogramas actualizados coinciden con los píxeles de salida
 10. el orden de las peticiones no afecta el resultado
 11. el general sincroniza los tres canales
 12. /api/prepare devuelve imagen, histogramas y zonas del pico
 13. imágenes constantes
 14. formatos de entrada admitidos (PNG/JPEG/BMP, RGBA, escala de grises)
 15. validaciones y utilidades existentes (sample, data URLs)
"""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import image_utils as iu
from main import app

client = TestClient(app)
CHANNELS = ("r", "g", "b")


# ---------------------------------------------------------------- helpers ---


def make_image(width: int = 64, height: int = 48) -> np.ndarray:
    """Imagen RGB determinista con un pico claro en el histograma.

    Mitad superior casi constante (crea el pico) y mitad inferior con un
    degradado suave (crea la base).
    """
    x = np.linspace(0, 1, width)[None, :]
    y = np.linspace(0, 1, height)[:, None]
    top = np.broadcast_to(np.array([40.0, 120.0, 200.0]), (height, width, 3))
    grad = np.stack(
        [60 + 120 * x + 0 * y, 90 + 100 * y + 0 * x, 140 + 80 * (1 - x) + 0 * y],
        axis=-1,
    )
    mezcla = y > 0.5
    img = np.where(mezcla[:, :, None], grad, top)
    return np.clip(img, 0, 255).astype(np.uint8)


def decode_data_url(url: str) -> np.ndarray:
    _, b64 = url.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def post_process(img: np.ndarray, r=0, g=0, b=0) -> dict:
    res = client.post(
        "/api/process",
        files={"file": ("test.png", png_bytes(img), "image/png")},
        data={"cut_r": str(r), "cut_g": str(g), "cut_b": str(b)},
    )
    assert res.status_code == 200, res.text
    return res.json()


# ------------------------------------------------- 1. 0 % = original ---


def test_zero_percent_is_identical_to_original():
    img = make_image()
    out = iu.sand_rgb(img, 0, 0, 0)
    assert np.array_equal(out, img)

    data = post_process(img, 0, 0, 0)
    assert np.array_equal(decode_data_url(data["image_after"]), img)
    assert data["hist_original"] == data["hist_after"]


# ------------------------------------------------ 2. zona del pico ---


def test_peak_threshold_is_45_percent_of_maximum():
    hist = np.zeros(256, dtype=np.int64)
    hist[70:80] = 100
    hist[75] = 500  # máximo
    assert iu.peak_threshold(hist) == pytest.approx(0.45 * 500)

    mask = iu.peak_mask(hist)
    assert mask[75] and not mask[70]  # 100 < 225
    assert mask.sum() == 1


def test_peak_mask_falls_back_to_the_maximum_when_nothing_exceeds():
    hist = np.zeros(256, dtype=np.int64)
    hist[10] = 1000
    hist[11] = 1000
    mask = iu.peak_mask(hist, fraction=1.0)  # nadie supera 1000
    assert mask[10] and mask[11] and mask.sum() == 2  # empate con el máximo

    assert not iu.peak_mask(np.zeros(256, dtype=np.int64)).any()


def test_prepare_reports_peak_ranges():
    img = make_image()
    prepared = iu.prepare_image(img)
    for name in CHANNELS:
        hist = iu.channel_histogram(img[:, :, CHANNELS.index(name)])
        mask = np.flatnonzero(iu.peak_mask(hist))
        info = prepared["peaks"][name]
        assert info["inicio"] == int(mask[0])
        assert info["fin"] == int(mask[-1])
        assert info["umbral"] == pytest.approx(iu.peak_threshold(hist))
        assert len(prepared["hist_original"][name]) == 256


# ------------------------------------- 3, 4. histograma objetivo ---


@pytest.mark.parametrize("percent", [0, 10, 33, 50, 100])
def test_target_histogram_conserves_total_and_is_non_negative(percent):
    hist = iu.channel_histogram(make_image()[:, :, 1])
    objetivo = iu.sand_target_histogram(hist, percent)
    assert objetivo.sum() == hist.sum()
    assert (objetivo >= 0).all()
    assert len(objetivo) == 256


def test_target_histogram_deflates_peak_and_inflates_the_base_equally():
    hist = iu.channel_histogram(make_image()[:, :, 1])
    mask = iu.peak_mask(hist)
    fuera = ~mask

    objetivo = iu.sand_target_histogram(hist, 100)

    # el pico baja (hasta el umbral) y toda la base sube lo mismo
    assert (objetivo[mask] <= hist[mask]).all()
    assert objetivo[mask].sum() < hist[mask].sum()
    ganancias = objetivo[fuera] - hist[fuera]
    assert (ganancias > 0).all()
    assert np.allclose(ganancias, ganancias[0], atol=1)  # reparto por igual
    assert objetivo.sum() == hist.sum()


def test_zero_percent_target_equals_original():
    hist = iu.channel_histogram(make_image()[:, :, 1])
    assert np.array_equal(iu.sand_target_histogram(hist, 0), hist)


# ------------------------------------------ 5, 6, 7. reasignación ---


@pytest.mark.parametrize("percent", [5, 50, 100])
def test_sand_channel_reproduces_the_target_histogram_exactly(percent):
    canal = make_image()[:, :, 1]
    out = iu.sand_channel(canal, percent)
    hist = iu.channel_histogram(canal)
    assert np.array_equal(iu.channel_histogram(out), iu.sand_target_histogram(hist, percent))


def test_sand_channel_is_deterministic_and_keeps_shape_and_dtype():
    canal = make_image()[:, :, 2]
    a = iu.sand_channel(canal, 70)
    b = iu.sand_channel(canal, 70)
    assert np.array_equal(a, b)
    assert a.shape == canal.shape
    assert a.dtype == np.uint8


def test_peak_only_decreases_and_base_only_increases_monotonically():
    canal = make_image()[:, :, 1]
    hist = iu.channel_histogram(canal)
    mask = iu.peak_mask(hist)
    picos, bases = [], []
    for percent in (0, 25, 50, 75, 100):
        out = iu.sand_channel(canal, percent)
        h = iu.channel_histogram(out)
        picos.append(int(h[mask].sum()))
        bases.append(int(h[~mask].sum()))
    assert picos == sorted(picos, reverse=True)
    assert bases == sorted(bases)


# --------------------------------------- 8. independencia de canales ---


@pytest.mark.parametrize("moved,kept", [(0, (1, 2)), (1, (0, 2)), (2, (0, 1))])
def test_moving_one_channel_does_not_touch_the_others(moved, kept):
    img = make_image()
    base_pct = [30, 55, 80]
    changed_pct = list(base_pct)
    changed_pct[moved] = 95

    base = iu.sand_rgb(img, *base_pct)
    changed = iu.sand_rgb(img, *changed_pct)

    assert not np.array_equal(base[:, :, moved], changed[:, :, moved])
    for idx in kept:
        assert np.array_equal(base[:, :, idx], changed[:, :, idx])


def test_api_moving_one_channel_does_not_touch_the_others():
    img = make_image()
    a = decode_data_url(post_process(img, 60, 0, 0)["image_after"])
    b = decode_data_url(post_process(img, 0, 0, 0)["image_after"])
    assert not np.array_equal(a[:, :, 0], b[:, :, 0])
    assert np.array_equal(a[:, :, 1], b[:, :, 1])
    assert np.array_equal(a[:, :, 2], b[:, :, 2])


def test_general_percent_applies_the_same_to_all_channels():
    img = make_image()
    out = iu.sand_rgb(img, 40, 40, 40)
    for ch in range(3):
        assert np.array_equal(out[:, :, ch], iu.sand_channel(img[:, :, ch], 40))


# ------------------------------------ 9, 10, 11, 12. API y orden ---


def test_api_returns_histograms_and_they_match_real_output_pixels():
    img = make_image()
    data = post_process(img, 100, 60, 25)
    after = decode_data_url(data["image_after"])
    real = iu.compute_histograms(after)

    for source in ("hist_original", "hist_after"):
        for channel in CHANNELS:
            assert len(data[source][channel]) == 256
            assert sum(data[source][channel]) == img.shape[0] * img.shape[1]
    for channel in CHANNELS:
        assert data["hist_after"][channel] == real[channel].tolist()
    assert data["cut"] == {"r": 100.0, "g": 60.0, "b": 25.0}


def test_original_histograms_do_not_change_when_sliders_move():
    img = make_image()
    a = post_process(img, 0, 0, 0)
    b = post_process(img, 90, 10, 55)
    c = post_process(img, 100, 100, 100)
    assert a["hist_original"] == b["hist_original"] == c["hist_original"]

    real = iu.compute_histograms(img)
    for channel in CHANNELS:
        assert a["hist_original"][channel] == real[channel].tolist()


def test_results_do_not_depend_on_the_order_of_the_requests():
    img = make_image()
    direct = post_process(img, 60, 20, 100)
    post_process(img, 0, 0, 0)
    post_process(img, 100, 100, 100)
    post_process(img, 15, 85, 40)
    after = post_process(img, 60, 20, 100)
    assert direct["image_after"] == after["image_after"]
    assert direct["hist_after"] == after["hist_after"]
    assert direct["hist_original"] == after["hist_original"]


def test_prepare_returns_image_histograms_and_peaks():
    img = make_image()
    res = client.post("/api/prepare", files={"file": ("test.png", png_bytes(img), "image/png")})
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["width"] == img.shape[1]
    assert data["height"] == img.shape[0]
    assert decode_data_url(data["image_before"]).shape == img.shape

    real = iu.compute_histograms(img)
    for channel in CHANNELS:
        assert data["hist_original"][channel] == real[channel].tolist()
        assert set(data["peaks"][channel]) == {"inicio", "fin", "umbral"}


def test_format_param_selects_the_image_type():
    img = make_image()
    jpeg = client.post(
        "/api/process",
        files={"file": ("t.png", png_bytes(img), "image/png")},
        data={"cut_r": "50", "cut_g": "50", "cut_b": "50", "format": "jpeg"},
    )
    assert jpeg.status_code == 200
    assert jpeg.json()["image_after"].startswith("data:image/jpeg;base64,")

    png = post_process(img, 50, 50, 50)
    assert png["image_after"].startswith("data:image/png;base64,")

    bad = client.post(
        "/api/process",
        files={"file": ("t.png", png_bytes(img), "image/png")},
        data={"format": "gif"},
    )
    assert bad.status_code == 422


def test_image_before_and_after_are_downloadable_data_urls():
    img = make_image()
    prepared = client.post(
        "/api/prepare", files={"file": ("test.png", png_bytes(img), "image/png")}
    ).json()
    assert prepared["image_before"].startswith("data:image/png;base64,")
    assert decode_data_url(prepared["image_before"]).ndim == 3

    data = post_process(img, 45, 45, 45)
    assert data["image_after"].startswith("data:image/png;base64,")
    assert decode_data_url(data["image_after"]).ndim == 3


# ------------------------------------------------- 13. constantes ---


@pytest.mark.parametrize("value", [0, 128, 255])
def test_constant_images(value):
    img = np.full((32, 40, 3), value, dtype=np.uint8)

    # 0 % = identidad; con un porcentaje > 0 la única intensidad es un pico
    # y la arena se esparce (la imagen deja de ser constante).
    assert np.array_equal(iu.sand_rgb(img, 0, 0, 0), img)

    out = iu.sand_rgb(img, 100, 100, 100)
    assert out.shape == img.shape and out.dtype == np.uint8
    for ch in range(3):
        hist = iu.channel_histogram(out[:, :, ch])
        assert hist.sum() == 32 * 40
        assert hist[value] < 32 * 40

    data = post_process(img, 0, 0, 0)
    assert np.array_equal(decode_data_url(data["image_after"]), img)


# ------------------------------------------------- 14. formatos ---


def test_supported_input_formats():
    img = make_image()
    casos = [("image/png", "PNG", "png"), ("image/jpeg", "JPEG", "jpg"), ("image/bmp", "BMP", "bmp")]
    for mime, pil_format, ext in casos:
        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format=pil_format)
        res = client.post(
            "/api/process",
            files={"file": (f"img.{ext}", buf.getvalue(), mime)},
            data={"cut_r": "50", "cut_g": "50", "cut_b": "50"},
        )
        assert res.status_code == 200, f"{mime}: {res.text}"
        assert decode_data_url(res.json()["image_after"]).shape == img.shape

    for modo, nombre in (("RGBA", "alpha.png"), ("L", "gray.png")):
        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").convert(modo).save(buf, format="PNG")
        res = client.post(
            "/api/process",
            files={"file": (nombre, buf.getvalue(), "image/png")},
            data={"cut_r": "10", "cut_g": "20", "cut_b": "30"},
        )
        assert res.status_code == 200
        assert decode_data_url(res.json()["image_after"]).shape == img.shape


# ------------------------------------------------- 15. validaciones ---


def test_sample_endpoint_returns_a_valid_png():
    res = client.get("/api/sample")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(res.content))
    assert img.mode == "RGB"
    assert img.size == (480, 360)


@pytest.mark.parametrize("valor", [101, -1])
def test_percent_out_of_range_is_rejected(valor):
    img = make_image()
    res = client.post(
        "/api/process",
        files={"file": ("t.png", png_bytes(img), "image/png")},
        data={"cut_r": str(valor), "cut_g": "0", "cut_b": "0"},
    )
    assert res.status_code == 422


def test_invalid_files_are_rejected():
    res = client.post("/api/process", files={"file": ("t.png", b"", "image/png")})
    assert res.status_code == 400
    res = client.post("/api/process", files={"file": ("t.png", b"no soy imagen", "image/png")})
    assert res.status_code == 400


def test_histogram_validation():
    with pytest.raises(ValueError):
        iu.peak_threshold(np.zeros(3))
    with pytest.raises(ValueError):
        iu.sand_target_histogram(-np.ones(256), 50)
