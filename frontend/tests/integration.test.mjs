/**
 * Prueba de integración frontend <-> backend.
 *
 * Reproduce el flujo real del navegador:
 *   1. POST /api/prepare con la imagen original
 *   2. decodificación de la imagen y obtención de los píxeles RGBA
 *   3. aplicación de las LUT interpoladas con los porcentajes de los sliders
 *   4. cálculo de los histogramas actualizados sobre los píxeles de salida
 * y compara el resultado con el cálculo independiente de POST /api/process.
 *
 * Se omite automáticamente si el backend no está levantado en http://127.0.0.1:8000.
 */
import assert from "node:assert/strict";
import test from "node:test";

import { histogramsFromRgba, processRgba } from "../src/lib/equalize.js";

const API = process.env.PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function backendAvailable() {
  try {
    const res = await fetch(`${API}/api/sample`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

const available = await backendAvailable();

function decodeDataUrl(dataUrl) {
  const b64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
  return Buffer.from(b64, "base64");
}

async function rgbaFromPng(buffer) {
  const sharp = (await import("sharp")).default;
  const { data, info } = await sharp(buffer)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  assert.equal(info.channels, 4);
  return { data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength), info };
}

test(
  "integración: /api/prepare + LUTs del cliente == /api/process",
  { skip: available ? false : `backend no disponible en ${API}` },
  async () => {
    const sample = await (await fetch(`${API}/api/sample`)).arrayBuffer();
    const file = new Blob([sample], { type: "image/png" });

    const prepareForm = new FormData();
    prepareForm.append("file", file, "sample.png");
    const prepared = await (await fetch(`${API}/api/prepare`, { method: "POST", body: prepareForm })).json();

    const { data: source, info } = await rgbaFromPng(decodeDataUrl(prepared.image_before));
    assert.equal(info.width, prepared.width);
    assert.equal(info.height, prepared.height);

    const processForm = new FormData();
    processForm.append("file", file, "sample.png");
    processForm.append("eq_r", "60");
    processForm.append("eq_g", "20");
    processForm.append("eq_b", "100");
    const backend = await (await fetch(`${API}/api/process`, { method: "POST", body: processForm })).json();

    const pct = { r: 60, g: 20, b: 100 };
    const out = processRgba(source, prepared.luts, pct);
    const { data: expected } = await rgbaFromPng(decodeDataUrl(backend.image_after));

    assert.equal(out.length, expected.length);
    assert.deepEqual(Array.from(out), Array.from(expected), "los píxeles deben coincidir");

    const histAfter = histogramsFromRgba(out);
    assert.deepEqual(histAfter, backend.hist_equalized, "histogramas actualizados reales");
    assert.deepEqual(prepared.hist_original, backend.hist_original, "histogramas originales intactos");

    // 0 % reproduce la imagen original.
    const identity = processRgba(source, prepared.luts, { r: 0, g: 0, b: 0 });
    assert.deepEqual(Array.from(identity), Array.from(source));

    // El orden de las peticiones no altera el resultado.
    const again = processRgba(source, prepared.luts, pct);
    assert.deepEqual(Array.from(again), Array.from(out));
  },
);
