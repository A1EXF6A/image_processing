/**
 * Prueba de integración contra el backend.
 *
 * Reproduce el flujo real del navegador:
 *   1. GET  /api/sample   (imagen de prueba)
 *   2. POST /api/prepare  (imagen + histogramas originales + zona de corte)
 *   3. POST /api/process  (nivel de corte por canal + pico fijo) y se comprueba
 *      que la respuesta sea coherente: el total de píxeles se conserva, los
 *      histogramas originales no cambian y el proceso es determinista.
 *
 * Todo el algoritmo vive en el backend, así que no hace falta decodificar
 * imágenes en el cliente. Se omite automáticamente si el backend no está
 * levantado en http://127.0.0.1:8000.
 */
import assert from "node:assert/strict";
import test from "node:test";

const API = process.env.PUBLIC_API_URL ?? "http://127.0.0.1:8000";
const CHANNELS = ["r", "g", "b"];

async function backendAvailable() {
  try {
    const res = await fetch(`${API}/api/sample`, { signal: AbortSignal.timeout(2000) });
    return res.ok;
  } catch {
    return false;
  }
}

const available = await backendAvailable();

async function sampleForm() {
  const sample = await (await fetch(`${API}/api/sample`)).arrayBuffer();
  const blob = new Blob([sample], { type: "image/png" });
  const form = new FormData();
  form.append("file", blob, "sample.png");
  return form;
}

function sum(values) {
  return values.reduce((a, b) => a + b, 0);
}

test(
  "integración: /api/prepare + /api/process son coherentes y deterministas",
  { skip: available ? false : `backend no disponible en ${API}` },
  async () => {
    const prepared = await (
      await fetch(`${API}/api/prepare`, { method: "POST", body: await sampleForm() })
    ).json();

    assert.match(prepared.image_before, /^data:image\/png;base64,/);
    assert.ok(prepared.width > 0 && prepared.height > 0);
    const total = prepared.width * prepared.height;
    for (const ch of CHANNELS) {
      assert.equal(prepared.hist_original[ch].length, 256);
      assert.equal(sum(prepared.hist_original[ch]), total, `el original de ${ch} suma los píxeles`);
      assert.ok(prepared.hist_original[ch].some((v) => v > 0));
    }

    async function process(niveles, pico = 40) {
      const form = await sampleForm();
      for (const ch of CHANNELS) form.append(`nivel_${ch}`, String(niveles[ch]));
      form.append("pico", String(pico));
      const res = await fetch(`${API}/api/process`, { method: "POST", body: form });
      assert.ok(res.ok, `process respondió ${res.status}`);
      return res.json();
    }

    const data = await process({ r: 20, g: 20, b: 20 });
    assert.match(data.image_after, /^data:image\/(png|jpeg);base64,/);
    assert.deepEqual(data.nivel, { r: 20, g: 20, b: 20 });
    assert.equal(data.pico, 40);
    for (const ch of CHANNELS) {
      assert.equal(data.hist_after[ch].length, 256);
      assert.equal(sum(data.hist_after[ch]), total, `el total de ${ch} se conserva`);
      assert.deepEqual(data.hist_original[ch], prepared.hist_original[ch], "el original no cambia");
      // el pico baja: la intensidad más poblada tiene menos píxeles que antes
      assert.ok(
        Math.max(...data.hist_after[ch]) < Math.max(...prepared.hist_original[ch]),
        `el pico de ${ch} debe bajar`,
      );
    }

    // determinista: repetir la misma petición da exactamente lo mismo
    const again = await process({ r: 20, g: 20, b: 20 });
    assert.deepEqual(again.hist_after, data.hist_after);

    // nivel = 100 no recorta nada -> histograma original
    const identidad = await process({ r: 100, g: 100, b: 100 });
    for (const ch of CHANNELS) {
      assert.deepEqual(identidad.hist_after[ch], prepared.hist_original[ch]);
    }
  },
);
