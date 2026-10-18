"""Tests del efecto "arena" (nivel de corte + recorte curvo del pico).

Cubre:

1. nivel de corte (``level_threshold`` / ``level_mask``).
2. histograma objetivo: conserva el total, no es negativo, recorta la zona y
   reparte la arena POR IGUAL entre la base.
2b. el recorte curvo: ``pico=100`` es identidad, ``pico=0`` aplana la cima al
    nivel, y bajar ``pico`` baja la zona de forma monótona.
3. reasignación por píxel: reproduce el histograma objetivo exactamente,
   es determinista y conserva forma y tipo.
4. independencia de canales y sincronización del control general.
5. API: /api/prepare y /api/process (histogramas, nivel, pico, format, orden).
6. imágenes constantes.
7. formatos de entrada admitidos (PNG/JPEG/BMP, RGBA, escala de grises).
8. validaciones y utilidades existentes (sample, data URLs).
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

    Mitad superior casi constante (crea la montaña) y mitad inferior con un
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


def post_process(img: np.ndarray, r=20, g=20, b=20, pico=40) -> dict:
    res = client.post(
        "/api/process",
        files={"file": ("test.png", png_bytes(img), "image/png")},
        data={
            "nivel_r": str(r),
            "nivel_g": str(g),
            "nivel_b": str(b),
            "pico": str(pico),
        },
    )
    assert res.status_code == 200, res.text
    return res.json()


# --------------------------------------------------- 1. nivel de corte ---


def test_level_threshold_is_nivel_percent_of_maximum():
    hist = np.zeros(256, dtype=np.int64)
    hist[70:80] = 100
    hist[75] = 500  # máximo
    assert iu.level_threshold(hist, 20) == pytest.approx(0.20 * 500)

    mask = iu.level_mask(hist, 20)
    assert mask[75] and not mask[70]  # 100 < 100 no supera el umbral
    assert mask.sum() == 1


def test_level_mask_falls_back_to_the_maximum_bin():
    hist = np.zeros(256, dtype=np.int64)
    hist[70:80] = 100
    hist[75] = 500
    mask = iu.level_mask(hist, 100)  # umbral = 500: nada lo supera
    assert mask[75] and mask.sum() == 1

    assert not iu.level_mask(np.zeros(256, dtype=np.int64)).any()


def test_lower_nivel_brings_in_more_of_the_mountain():
    hist = iu.channel_histogram(make_image()[:, :, 1])
    assert iu.level_mask(hist, 5).sum() >= iu.level_mask(hist, 20).sum()
    assert iu.level_threshold(hist, 5) < iu.level_threshold(hist, 20)


# ---------------------------------------- 2. histograma objetivo ---


@pytest.mark.parametrize("nivel,pico", [(20, 40), (20, 0), (5, 80), (50, 20), (80, 5)])
def test_target_histogram_conserves_total_and_is_non_negative(nivel, pico):
    hist = iu.channel_histogram(make_image()[:, :, 1])
    objetivo = iu.sand_target_histogram(hist, nivel, pico)
    assert objetivo.sum() == hist.sum()
    assert (objetivo >= 0).all()
    assert len(objetivo) == 256


def test_pico_100_or_nivel_100_is_identity():
    hist = iu.channel_histogram(make_image()[:, :, 1])
    assert np.array_equal(iu.sand_target_histogram(hist, 20, 100), hist)
    assert np.array_equal(iu.sand_target_histogram(hist, 100, 40), hist)


def test_curved_cut_deflates_the_zone_and_inflates_the_base_equally():
    hist = iu.channel_histogram(make_image()[:, :, 1])
    mask = iu.level_mask(hist, 20)
    fuera = ~mask

    objetivo = iu.sand_target_histogram(hist, 20, 0)  # pico=0: recorte total

    # la zona baja y toda la base sube lo mismo (reparto por igual)
    assert (objetivo[mask] <= hist[mask]).all()
    assert objetivo[mask].sum() < hist[mask].sum()
    ganancias = objetivo[fuera] - hist[fuera]
    assert (ganancias > 0).all()
    assert np.allclose(ganancias, ganancias[0], atol=1)
    assert objetivo.sum() == hist.sum()


def test_pico_zero_flattens_the_zone_to_the_level():
    hist = iu.channel_histogram(make_image()[:, :, 1])
    mask = iu.level_mask(hist, 20)
    umbral = iu.level_threshold(hist, 20)
    objetivo = iu.sand_target_histogram(hist, 20, 0)
    assert np.allclose(objetivo[mask], umbral, atol=1)


def test_the_zone_lowers_as_pico_decreases():
    hist = iu.channel_histogram(make_image()[:, :, 1])
    mask = iu.level_mask(hist, 20)
    zona = [
        int(iu.sand_target_histogram(hist, 20, p)[mask].sum())
        for p in (100, 75, 50, 25, 0)
    ]
    assert zona == sorted(zona, reverse=True)


def test_curved_cut_reduces_the_prominence_of_the_peak():
    hist = iu.channel_histogram(make_image()[:, :, 1])
    i = int(np.argmax(hist))

    def prominencia(h: np.ndarray) -> float:
        vecinos = max(h[i - 1], h[i + 1])
        return float((h[i] - vecinos) / h[i])

    original = prominencia(hist)
    for pico in (75, 60, 40, 20, 0):
        assert prominencia(iu.sand_target_histogram(hist, 20, pico)) < original


# ------------------------------------------ 2, 7. reasignación por píxel ---


@pytest.mark.parametrize("nivel,pico", [(20, 40), (5, 0), (50, 80), (10, 20)])
def test_sand_channel_reproduces_the_target_histogram_exactly(nivel, pico):
    canal = make_image()[:, :, 1]
    hist = iu.channel_histogram(canal)
    out = iu.sand_channel(canal, nivel, pico)
    assert np.array_equal(
        iu.channel_histogram(out), iu.sand_target_histogram(hist, nivel, pico)
    )


def test_sand_channel_pico_100_is_the_original():
    canal = make_image()[:, :, 1]
    assert np.array_equal(iu.sand_channel(canal, 20, 100), canal)


def test_sand_channel_is_deterministic_and_keeps_shape_and_dtype():
    canal = make_image()[:, :, 2]
    a = iu.sand_channel(canal, 20, 40)
    b = iu.sand_channel(canal, 20, 40)
    assert np.array_equal(a, b)
    assert a.shape == canal.shape
    assert a.dtype == np.uint8


def test_zone_only_decreases_and_base_only_increases_monotonically():
    canal = make_image()[:, :, 1]
    hist = iu.channel_histogram(canal)
    mask = iu.level_mask(hist, 20)
    zona, base = [], []
    for pico in (100, 75, 50, 25, 0):
        out = iu.sand_channel(canal, 20, pico)
        h = iu.channel_histogram(out)
        zona.append(int(h[mask].sum()))
        base.append(int(h[~mask].sum()))
    assert zona == sorted(zona, reverse=True)
    assert base == sorted(base)


# --------------------------------------- 2. independencia de canales ---


def test_moving_one_channel_does_not_touch_the_others():
    img = make_image()
    base_niveles = [30, 55, 80]
    changed = list(base_niveles)
    changed[1] = 5

    a = iu.sand_rgb(img, *base_niveles, 40)
    b = iu.sand_rgb(img, *changed, 40)

    assert not np.array_equal(a[:, :, 1], b[:, :, 1])
    assert np.array_equal(a[:, :, 0], b[:, :, 0])
    assert np.array_equal(a[:, :, 2], b[:, :, 2])


def test_api_moving_one_channel_does_not_touch_the_others():
    img = make_image()
    a = decode_data_url(post_process(img, 5, 20, 20)["image_after"])
    b = decode_data_url(post_process(img, 20, 20, 20)["image_after"])
    assert not np.array_equal(a[:, :, 0], b[:, :, 0])
    assert np.array_equal(a[:, :, 1], b[:, :, 1])
    assert np.array_equal(a[:, :, 2], b[:, :, 2])


def test_shared_nivel_applies_the_same_to_all_channels():
    img = make_image()
    out = iu.sand_rgb(img, 35, 35, 35, 40)
    for ch in range(3):
        assert np.array_equal(out[:, :, ch], iu.sand_channel(img[:, :, ch], 35, 40))


# ---------------------------------------- 3. API: historiales y orden ---


def test_api_returns_histograms_and_they_match_real_output_pixels():
    img = make_image()
    data = post_process(img, 20, 5, 45, pico=60)
    after = decode_data_url(data["image_after"])
    real = iu.compute_histograms(after)

    for source in ("hist_original", "hist_after"):
        for channel in CHANNELS:
            assert len(data[source][channel]) == 256
            assert sum(data[source][channel]) == img.shape[0] * img.shape[1]
    for channel in CHANNELS:
        assert data["hist_after"][channel] == real[channel].tolist()
    assert data["nivel"] == {"r": 20.0, "g": 5.0, "b": 45.0}
    assert data["pico"] == 60.0


def test_api_uses_defaults_nivel_20_and_pico_40():
    img = make_image()
    res = client.post(
        "/api/process", files={"file": ("t.png", png_bytes(img), "image/png")}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["nivel"] == {"r": 20.0, "g": 20.0, "b": 20.0}
    assert data["pico"] == 40.0


def test_original_histograms_do_not_change_when_sliders_move():
    img = make_image()
    a = post_process(img, 20, 20, 20)
    b = post_process(img, 5, 90, 50)
    c = post_process(img, 100, 100, 100)
    assert a["hist_original"] == b["hist_original"] == c["hist_original"]

    real = iu.compute_histograms(img)
    for channel in CHANNELS:
        assert a["hist_original"][channel] == real[channel].tolist()


def test_results_do_not_depend_on_the_order_of_the_requests():
    img = make_image()
    direct = post_process(img, 60, 20, 5, 40)
    post_process(img, 100, 100, 100, 100)
    post_process(img, 5, 5, 5, 0)
    post_process(img, 80, 30, 15, 70)
    after = post_process(img, 60, 20, 5, 40)
    assert direct["image_after"] == after["image_after"]
    assert direct["hist_after"] == after["hist_after"]
    assert direct["hist_original"] == after["hist_original"]


def test_prepare_returns_image_histograms_and_zone():
    img = make_image()
    res = client.post("/api/prepare", files={"file": ("test.png", png_bytes(img), "image/png")})
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["image_before"].startswith("data:image/png;base64,")
    real = iu.compute_histograms(img)
    for channel in CHANNELS:
        assert data["hist_original"][channel] == real[channel].tolist()
        info = data["peaks"][channel]
        mask = np.flatnonzero(iu.level_mask(real[channel], iu.DEFAULT_NIVEL))
        assert info["inicio"] == int(mask[0])
        assert info["fin"] == int(mask[-1])
        assert info["umbral"] == pytest.approx(
            iu.level_threshold(real[channel], iu.DEFAULT_NIVEL)
        )


def test_format_chooses_png_or_jpeg_and_rejects_others():
    img = make_image()

    jpeg = client.post(
        "/api/process",
        files={"file": ("t.png", png_bytes(img), "image/png")},
        data={"nivel_r": "20", "nivel_g": "20", "nivel_b": "20", "format": "jpeg"},
    )
    assert jpeg.status_code == 200
    assert jpeg.json()["image_after"].startswith("data:image/jpeg;base64,")

    png = post_process(img, 20, 20, 20)
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

    data = post_process(img, 40, 40, 40)
    assert data["image_after"].startswith("data:image/png;base64,")
    assert decode_data_url(data["image_after"]).ndim == 3


# ------------------------------------------------- 13. constantes ---


@pytest.mark.parametrize("value", [0, 128, 255])
def test_constant_images(value):
    img = np.full((32, 40, 3), value, dtype=np.uint8)

    # pico=100 (sin recorte) o nivel=100 (sin zona) devuelven la original
    assert np.array_equal(iu.sand_rgb(img, 20, 20, 20, 100), img)
    assert np.array_equal(iu.sand_rgb(img, 100, 100, 100, 0), img)

    # con recorte, la única intensidad es la zona y la arena se esparce
    out = iu.sand_rgb(img, 20, 20, 20, 0)
    assert out.shape == img.shape and out.dtype == np.uint8
    for ch in range(3):
        hist = iu.channel_histogram(out[:, :, ch])
        assert hist.sum() == 32 * 40
        assert hist[value] < 32 * 40

    data = post_process(img, 100, 100, 100, 40)
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
            data={"nivel_r": "20", "nivel_g": "20", "nivel_b": "20"},
        )
        assert res.status_code == 200, f"{mime}: {res.text}"
        assert decode_data_url(res.json()["image_after"]).shape == img.shape

    for modo, nombre in (("RGBA", "alpha.png"), ("L", "gray.png")):
        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").convert(modo).save(buf, format="PNG")
        res = client.post(
            "/api/process",
            files={"file": (nombre, buf.getvalue(), "image/png")},
            data={"nivel_r": "10", "nivel_g": "20", "nivel_b": "30"},
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
    for campo in ("nivel_r", "pico"):
        data = {"nivel_r": "20", "nivel_g": "20", "nivel_b": "20", "pico": "40"}
        data[campo] = str(valor)
        res = client.post(
            "/api/process",
            files={"file": ("t.png", png_bytes(img), "image/png")},
            data=data,
        )
        assert res.status_code == 422, f"{campo}={valor}"


def test_invalid_files_are_rejected():
    res = client.post("/api/process", files={"file": ("t.png", b"", "image/png")})
    assert res.status_code == 400
    res = client.post("/api/process", files={"file": ("t.png", b"no soy imagen", "image/png")})
    assert res.status_code == 400


def test_histogram_validation():
    with pytest.raises(ValueError):
        iu.level_threshold(np.zeros(3))
    with pytest.raises(ValueError):
        iu.sand_target_histogram(-np.ones(256), 20, 40)
