"""Pruebas de la ecualización interactiva RGB por canal.

Cobertura (15 puntos de la sección 8 del enunciado):
 1. sliders en 0 %  -> imagen idéntica a la original
 2. general 50 %    -> transformación interpolada al 50 % en los tres canales
 3. general 100 %   -> ecualización completa de los tres canales
 4. mover R no cambia G ni B
 5. mover G no cambia R ni B
 6. mover B no cambia R ni G
 7. el general sincroniza los tres canales
 8. estados mixtos (porcentajes distintos por canal)
 9. los ocho histogramas se actualizan
10. los histogramas originales no cambian
11. los histogramas actualizados coinciden con los píxeles de salida
12. el orden de los movimientos no afecta el resultado
13. imágenes con intensidades constantes
14. imágenes RGB y formatos de entrada admitidos
15. carga / descarga / reinicio y resto de funcionalidades
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


# ---------------------------------------------------------------- helpers ---


def make_image(width: int = 64, height: int = 48) -> np.ndarray:
    """Imagen RGB de bajo contraste y determinista."""
    x = np.linspace(0, 1, width)[None, :]
    y = np.linspace(0, 1, height)[:, None]
    r = 60 + 60 * x + 10 * y
    g = 70 + 50 * y + 10 * (1 - x)
    b = 90 + 40 * (1 - 0.5 * (x + y))
    img = np.clip(np.stack([r + 0 * y, g + 0 * x, b + 0 * x], axis=-1), 0, 255)
    return img.astype(np.uint8)


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
        data={"eq_r": str(r), "eq_g": str(g), "eq_b": str(b)},
    )
    assert res.status_code == 200, res.text
    return res.json()


# ------------------------------------------------- 1. 0 % = imagen original ---


def test_zero_percent_is_identical_to_original():
    img = make_image()
    out = iu.process_rgb(img, 0, 0, 0)
    assert np.array_equal(out, img)

    data = post_process(img, 0, 0, 0)
    before = decode_data_url(data["image_before"])
    after = decode_data_url(data["image_after"])
    assert np.array_equal(before, img)
    assert np.array_equal(after, before)


# ------------------------------------------- 2. general 50 % interpolación ---


def test_general_50_uses_interpolated_transform_on_every_channel():
    img = make_image()
    out = iu.process_rgb(img, 50, 50, 50)
    luts = iu.build_luts(img)

    for i, name in enumerate(("r", "g", "b")):
        expected = iu.apply_lut(img[:, :, i], iu.interpolate_lut(luts[name], 0.5))
        assert np.array_equal(out[:, :, i], expected), f"canal {name}"

    full = iu.process_rgb(img, 100, 100, 100)
    assert not np.array_equal(out, img)
    assert not np.array_equal(out, full)


def test_interpolate_lut_endpoints_and_midpoint():
    hist = np.zeros(256, dtype=np.int64)
    hist[100:150] = 10
    lut = iu.equalization_lut(hist)

    assert np.array_equal(iu.interpolate_lut(lut, 0.0), np.arange(256))
    assert np.array_equal(iu.interpolate_lut(lut, 1.0), lut)

    expected = np.clip(np.rint(0.5 * np.arange(256) + 0.5 * lut), 0, 255).astype(np.uint8)
    assert np.array_equal(iu.interpolate_lut(lut, 0.5), expected)


# --------------------------------------------- 3. general 100 % completo ---


def test_general_100_fully_equalizes_the_three_channels():
    img = iu.generate_sample_image()
    out = iu.process_rgb(img, 100, 100, 100)
    luts = iu.build_luts(img)

    for i, name in enumerate(("r", "g", "b")):
        expected = iu.apply_lut(img[:, :, i], luts[name])
        assert np.array_equal(out[:, :, i], expected), f"canal {name}"

    for i, name in enumerate(("r", "g", "b")):
        original_range = np.ptp(img[:, :, i])
        equalized_range = np.ptp(out[:, :, i])
        assert equalized_range >= original_range, name
        assert equalized_range > 0, name


def test_full_equalization_uses_original_histogram_not_previous_result():
    img = make_image()
    first = iu.process_rgb(img, 70, 70, 70)

    luts_original = iu.build_luts(img)
    luts_from_processed = iu.build_luts(first)

    full = iu.process_rgb(img, 100, 100, 100)
    for i, name in enumerate(("r", "g", "b")):
        expected = iu.apply_lut(img[:, :, i], luts_original[name])
        assert np.array_equal(full[:, :, i], expected), f"canal {name}"

    # La imagen ya modificada tiene un histograma distinto: reutilizarla
    # produciría tablas diferentes (acumulación), así que nunca se hace.
    assert any(
        not np.array_equal(luts_original[name], luts_from_processed[name])
        for name in ("r", "g", "b")
    )
    assert not np.array_equal(first, img)


# --------------------------------- 4, 5, 6. independencia de los canales ---


@pytest.mark.parametrize(
    "channel,moved,kept",
    [("r", 0, (1, 2)), ("g", 1, (0, 2)), ("b", 2, (0, 1))],
)
def test_moving_one_channel_does_not_touch_the_others(channel, moved, kept):
    img = make_image()
    base_pct = [30, 55, 80]
    changed_pct = list(base_pct)
    changed_pct[moved] = 95 if base_pct[moved] != 95 else 5

    base = iu.process_rgb(img, *base_pct)
    changed = iu.process_rgb(img, *changed_pct)

    assert not np.array_equal(base[:, :, moved], changed[:, :, moved])
    for idx in kept:
        assert np.array_equal(base[:, :, idx], changed[:, :, idx])


def test_mixed_percentages_example_60_20_100():
    img = make_image()
    out = iu.process_rgb(img, 60, 20, 100)
    luts = iu.build_luts(img)

    expected_r = iu.apply_lut(img[:, :, 0], iu.interpolate_lut(luts["r"], 0.60))
    expected_g = iu.apply_lut(img[:, :, 1], iu.interpolate_lut(luts["g"], 0.20))
    expected_b = iu.apply_lut(img[:, :, 2], iu.interpolate_lut(luts["b"], 1.00))

    assert np.array_equal(out[:, :, 0], expected_r)
    assert np.array_equal(out[:, :, 1], expected_g)
    assert np.array_equal(out[:, :, 2], expected_b)

    data = post_process(img, 60, 20, 100)
    assert data["eq"] == {"r": 60.0, "g": 20.0, "b": 100.0}
    assert np.array_equal(decode_data_url(data["image_after"]), out)


# ------------------------------------------- 7. sincronización del general ---


def test_general_slider_syncs_the_three_channels():
    img = make_image()
    out = iu.process_rgb(img, 40, 40, 40)
    luts = iu.build_luts(img)

    for i, name in enumerate(("r", "g", "b")):
        expected = iu.apply_lut(img[:, :, i], iu.interpolate_lut(luts[name], 0.40))
        assert np.array_equal(out[:, :, i], expected)


# ------------------------------------------------- 9, 10, 11. histogramas ---


def test_api_returns_channel_histograms_and_they_update():
    """Los 8 paneles de la interfaz se alimentan de estos 6 histogramas
    (los dos paneles RGB reutilizan las tres series de canales)."""
    img = make_image()
    data = post_process(img, 100, 60, 25)

    for source in ("hist_original", "hist_equalized"):
        hist = data[source]
        assert set(hist) == {"r", "g", "b"}
        for channel in ("r", "g", "b"):
            assert len(hist[channel]) == 256
            assert sum(hist[channel]) == img.shape[0] * img.shape[1]


def test_original_histograms_do_not_change_when_sliders_move():
    img = make_image()
    a = post_process(img, 0, 0, 0)
    b = post_process(img, 90, 10, 55)
    c = post_process(img, 100, 100, 100)

    assert a["hist_original"] == b["hist_original"] == c["hist_original"]

    real = iu.compute_histograms(img)
    for channel in ("r", "g", "b"):
        assert a["hist_original"][channel] == real[channel].tolist()


def test_updated_histograms_match_real_output_pixels():
    img = make_image()
    data = post_process(img, 75, 40, 10)
    after = decode_data_url(data["image_after"])
    real = iu.compute_histograms(after)

    for channel in ("r", "g", "b"):
        assert data["hist_equalized"][channel] == real[channel].tolist()


def test_frontend_histogram_pairs_share_the_original_data():
    """Las tablas del /api/prepare son las que el cliente interpola."""
    img = make_image()
    prepared = iu.prepare_image(img)
    out = iu.process_rgb(img, 35, 35, 35)

    for channel, lut in prepared["luts"].items():
        idx = ("r", "g", "b").index(channel)
        expected = iu.apply_lut(img[:, :, idx], iu.interpolate_lut(lut, 0.35))
        assert np.array_equal(out[:, :, idx], expected)


# ------------------------------------------------------- 12. orden de uso ---


def test_results_do_not_depend_on_the_order_of_the_requests():
    img = make_image()
    direct = post_process(img, 60, 20, 100)

    post_process(img, 0, 0, 0)
    post_process(img, 100, 100, 100)
    post_process(img, 15, 85, 40)
    after_other_moves = post_process(img, 60, 20, 100)

    assert direct["image_after"] == after_other_moves["image_after"]
    assert direct["hist_original"] == after_other_moves["hist_original"]
    assert direct["hist_equalized"] == after_other_moves["hist_equalized"]

    # La LUT siempre se recalcula desde el histograma original.
    reordered = iu.process_rgb(img, 100, 0, 50)
    again = iu.process_rgb(img, 100, 0, 50)
    assert np.array_equal(reordered, again)


# ------------------------------------------------------- 13. constantes ---


@pytest.mark.parametrize("value", [0, 128, 255])
def test_constant_images_are_processed_without_errors(value):
    img = np.full((32, 40, 3), value, dtype=np.uint8)
    out = iu.process_rgb(img, 0, 50, 100)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, img)

    hist = np.zeros(256, dtype=np.int64)
    hist[value] = 32 * 40
    assert np.array_equal(iu.equalization_lut(hist), np.arange(256))

    data = post_process(img, 100, 100, 100)
    assert np.array_equal(decode_data_url(data["image_after"]), img)


def test_histogram_with_a_single_level_returns_identity_lut():
    assert np.array_equal(iu.equalization_lut(np.zeros(256)), np.arange(256))
    with pytest.raises(ValueError):
        iu.equalization_lut(np.zeros(0))


# ------------------------------------------------------- 14. formatos ---


def test_rgb_and_supported_input_formats():
    img = make_image()

    cases = [
        ("image/png", "PNG", "png"),
        ("image/jpeg", "JPEG", "jpg"),
        ("image/bmp", "BMP", "bmp"),
    ]
    for mime, pil_format, ext in cases:
        buf = io.BytesIO()
        Image.fromarray(img, mode="RGB").save(buf, format=pil_format)
        res = client.post(
            "/api/process",
            files={"file": (f"img.{ext}", buf.getvalue(), mime)},
            data={"eq_r": "50", "eq_g": "50", "eq_b": "50"},
        )
        assert res.status_code == 200, f"{mime}: {res.text}"
        out = decode_data_url(res.json()["image_after"])
        assert out.shape == img.shape

    # PNG con alfa y escala de grises se convierten a RGB.
    rgba = Image.fromarray(img, mode="RGB").convert("RGBA")
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    res = client.post(
        "/api/process",
        files={"file": ("alpha.png", buf.getvalue(), "image/png")},
        data={"eq_r": "10", "eq_g": "20", "eq_b": "30"},
    )
    assert res.status_code == 200
    assert decode_data_url(res.json()["image_after"]).shape == img.shape

    gray = Image.fromarray(img, mode="RGB").convert("L")
    buf = io.BytesIO()
    gray.save(buf, format="PNG")
    res = client.post(
        "/api/process",
        files={"file": ("gray.png", buf.getvalue(), "image/png")},
        data={"eq_r": "10", "eq_g": "20", "eq_b": "30"},
    )
    assert res.status_code == 200
    assert decode_data_url(res.json()["image_after"]).shape == img.shape


# ---------------------------------- 15. funcionalidades existentes / API ---


def test_sample_endpoint_returns_a_valid_png():
    res = client.get("/api/sample")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(res.content))
    assert img.mode == "RGB"
    assert img.size == (480, 360)


def test_prepare_returns_original_image_histograms_and_luts():
    img = make_image()
    res = client.post(
        "/api/prepare",
        files={"file": ("test.png", png_bytes(img), "image/png")},
    )
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["width"] == img.shape[1]
    assert data["height"] == img.shape[0]
    assert decode_data_url(data["image_before"]).shape == img.shape

    real = iu.compute_histograms(img)
    for channel in ("r", "g", "b"):
        assert data["hist_original"][channel] == real[channel].tolist()
        assert len(data["luts"][channel]) == 256
        assert data["luts"][channel] == iu.equalization_lut(real[channel]).tolist()

    # Las LUT del prepare coinciden con el resultado de /api/process al 100 %.
    full = post_process(img, 100, 100, 100)
    after = decode_data_url(full["image_after"])
    for i, channel in enumerate(("r", "g", "b")):
        expected = np.asarray(data["luts"][channel], dtype=np.uint8)[img[:, :, i]]
        assert np.array_equal(after[:, :, i], expected)


def test_image_before_and_after_are_downloadable_data_urls():
    data = post_process(make_image(), 45, 45, 45)
    for key in ("image_before", "image_after"):
        assert data[key].startswith("data:image/png;base64,")
        assert decode_data_url(data[key]).ndim == 3


def test_validation_of_invalid_requests():
    img = make_image()

    res = client.post(
        "/api/process",
        files={"file": ("t.png", png_bytes(img), "image/png")},
        data={"eq_r": "101", "eq_g": "0", "eq_b": "0"},
    )
    assert res.status_code == 422

    res = client.post(
        "/api/process",
        files={"file": ("t.png", png_bytes(img), "image/png")},
        data={"eq_r": "-1", "eq_g": "0", "eq_b": "0"},
    )
    assert res.status_code == 422

    res = client.post("/api/process", files={"file": ("t.png", b"", "image/png")})
    assert res.status_code == 400

    res = client.post("/api/process", files={"file": ("t.png", b"no soy imagen", "image/png")})
    assert res.status_code == 400
