import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  CHANNELS,
  areChannelsSynced,
  channelHistogram,
  channelSummary,
  clampPercent,
  generalState,
  histogramsFromRgba,
  interpolateLut,
  processRgba,
  setChannel,
  setGeneral,
} from "../src/lib/equalize.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ASTRO_PAGE = path.join(HERE, "..", "src", "pages", "index.astro");

const SAMPLE_LUTS = {
  r: Array.from({ length: 256 }, (_, k) => (k * 2) % 256),
  g: Array.from({ length: 256 }, (_, k) => (k + 40) % 256),
  b: Array.from({ length: 256 }, (_, k) => 255 - k),
};

function rgbaFromPixels(pixels) {
  const out = new Uint8ClampedArray(pixels.length * 4);
  pixels.forEach(([r, g, b, a = 255], i) => {
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = a;
  });
  return out;
}

function manualProcess(source, luts, pct) {
  const tables = {};
  for (const ch of CHANNELS) {
    const p = clampPercent(pct[ch]) / 100;
    tables[ch] = Array.from({ length: 256 }, (_, k) =>
      Math.min(255, Math.max(0, Math.round((1 - p) * k + p * luts[ch][k]))),
    );
  }
  const out = new Uint8ClampedArray(source.length);
  for (let i = 0; i < source.length; i += 4) {
    out[i] = tables.r[source[i]];
    out[i + 1] = tables.g[source[i + 1]];
    out[i + 2] = tables.b[source[i + 2]];
    out[i + 3] = source[i + 3];
  }
  return out;
}

test("clampPercent limita y redondea al rango 0-100", () => {
  assert.equal(clampPercent(-20), 0);
  assert.equal(clampPercent(130), 100);
  assert.equal(clampPercent("45"), 45);
  assert.equal(clampPercent(33.6), 34);
  assert.equal(clampPercent("no-numero"), 0);
});

test("interpolateLut: 0 % es identidad, 100 % es T_eq y 50 % es el promedio", () => {
  const base = SAMPLE_LUTS.r;
  const identity = interpolateLut(base, 0);
  for (let k = 0; k < 256; k++) assert.equal(identity[k], k);

  const full = interpolateLut(base, 100);
  for (let k = 0; k < 256; k++) assert.equal(full[k], base[k]);

  const half = interpolateLut(base, 50);
  for (let k = 0; k < 256; k++) {
    assert.equal(half[k], Math.min(255, Math.max(0, Math.round(0.5 * k + 0.5 * base[k]))));
  }
});

test("Al 0 % la imagen procesada es idéntica a la original", () => {
  const source = rgbaFromPixels([
    [10, 20, 30],
    [200, 210, 220],
    [128, 64, 32, 128],
  ]);
  const out = processRgba(source, SAMPLE_LUTS, { r: 0, g: 0, b: 0 });
  assert.deepEqual(Array.from(out), Array.from(source));
});

test("Cada porcentaje afecta solo a su canal (ejemplo 60 / 20 / 100)", () => {
  const source = rgbaFromPixels([[100, 100, 100]]);
  const out = processRgba(source, SAMPLE_LUTS, { r: 60, g: 20, b: 100 });

  assert.equal(out[0], Math.round(0.4 * 100 + 0.6 * SAMPLE_LUTS.r[100]));
  assert.equal(out[1], Math.round(0.8 * 100 + 0.2 * SAMPLE_LUTS.g[100]));
  assert.equal(out[2], SAMPLE_LUTS.b[100]);

  const expected = manualProcess(source, SAMPLE_LUTS, { r: 60, g: 20, b: 100 });
  assert.deepEqual(Array.from(out), Array.from(expected));
});

test("El resultado no depende del orden en que se mueven los sliders", () => {
  const source = rgbaFromPixels([
    [12, 130, 250],
    [77, 88, 99],
    [200, 5, 140],
  ]);
  const target = { r: 80, g: 20, b: 60 };

  let viaGeneralFirst = setGeneral({ r: 0, g: 0, b: 0 }, 50);
  viaGeneralFirst = setChannel(viaGeneralFirst, "r", 80);
  viaGeneralFirst = setChannel(viaGeneralFirst, "g", 20);
  viaGeneralFirst = setChannel(viaGeneralFirst, "b", 60);

  let viaChannelFirst = setChannel({ r: 0, g: 0, b: 0 }, "b", 60);
  viaChannelFirst = setChannel(viaChannelFirst, "g", 20);
  viaChannelFirst = setChannel(viaChannelFirst, "r", 80);
  viaChannelFirst = setGeneral(viaChannelFirst, 33);
  viaChannelFirst = setChannel(viaChannelFirst, "r", 80);
  viaChannelFirst = setChannel(viaChannelFirst, "g", 20);
  viaChannelFirst = setChannel(viaChannelFirst, "b", 60);

  assert.deepEqual(viaGeneralFirst, target);
  assert.deepEqual(viaChannelFirst, target);

  const a = processRgba(source, SAMPLE_LUTS, viaGeneralFirst);
  const b = processRgba(source, SAMPLE_LUTS, viaChannelFirst);
  assert.deepEqual(Array.from(a), Array.from(b));
});

test("Mover un slider individual no altera el resto de porcentajes", () => {
  let pct = { r: 30, g: 55, b: 80 };

  pct = setChannel(pct, "r", 95);
  assert.deepEqual(pct, { r: 95, g: 55, b: 80 });

  pct = setChannel(pct, "g", 10);
  assert.deepEqual(pct, { r: 95, g: 10, b: 80 });

  pct = setChannel(pct, "b", 0);
  assert.deepEqual(pct, { r: 95, g: 10, b: 0 });
});

test("El slider general sincroniza los tres canales", () => {
  const pct = setGeneral({ r: 95, g: 10, b: 0 }, 40);
  assert.deepEqual(pct, { r: 40, g: 40, b: 40 });
  assert.ok(areChannelsSynced(pct));
});

test("El estado general distingue sincronizado y mixto", () => {
  const synced = generalState({ r: 50, g: 50, b: 50 });
  assert.equal(synced.synced, true);
  assert.equal(synced.value, 50);
  assert.equal(synced.label, "50%");

  const mixed = generalState({ r: 60, g: 20, b: 100 });
  assert.equal(mixed.synced, false);
  assert.equal(mixed.label, "Mixto");
  assert.equal(mixed.value, 60);
  assert.equal(channelSummary({ r: 60, g: 20, b: 100 }), "R 60% · G 20% · B 100%");
});

test("Los histogramas se calculan con 256 niveles sobre píxeles reales", () => {
  const source = rgbaFromPixels([
    [10, 20, 30],
    [10, 25, 35],
    [200, 210, 220],
  ]);

  assert.equal(source.length, 12);
  assert.equal(channelHistogram(source, "r").length, 256);
  assert.equal(channelHistogram(source, "r")[10], 2);
  assert.equal(channelHistogram(source, "r")[200], 1);
  assert.equal(channelHistogram(source, "g")[25], 1);
  assert.equal(channelHistogram(source, "b")[220], 1);

  const hists = histogramsFromRgba(source);
  assert.deepEqual(Object.keys(hists).sort(), ["b", "g", "r"]);
  for (const ch of CHANNELS) {
    const sum = hists[ch].reduce((a, b) => a + b, 0);
    assert.equal(sum, 3, `el histograma de ${ch} debe sumar los píxeles`);
  }
});

test("El histograma actualizado coincide con los píxeles reales de salida", () => {
  const source = rgbaFromPixels([
    [10, 20, 30],
    [10, 25, 35],
    [200, 210, 220],
    [64, 64, 64],
  ]);
  const pct = { r: 100, g: 35, b: 0 };
  const out = processRgba(source, SAMPLE_LUTS, pct);

  const expected = manualProcess(source, SAMPLE_LUTS, pct);
  assert.deepEqual(Array.from(out), Array.from(expected));

  const histAfter = histogramsFromRgba(out);
  const manual = { r: new Array(256).fill(0), g: new Array(256).fill(0), b: new Array(256).fill(0) };
  const offset = { r: 0, g: 1, b: 2 };
  for (const ch of CHANNELS) {
    for (let i = offset[ch]; i < out.length; i += 4) manual[ch][out[i]]++;
  }
  assert.deepEqual(histAfter, manual);

  const histBefore = histogramsFromRgba(source);
  const histBeforeCopy = JSON.parse(JSON.stringify(histBefore));
  processRgba(source, SAMPLE_LUTS, { r: 100, g: 100, b: 100 });
  assert.deepEqual(histBefore, histBeforeCopy, "los histogramas originales no cambian");
});

test("La página define exactamente 4 sliders de nivel (5-100) y 8 histogramas", () => {
  const html = fs.readFileSync(ASTRO_PAGE, "utf8");

  const ranges = html.match(/<input[^>]*type="range"[^>]*>/g) || [];
  assert.equal(ranges.length, 4, "deben existir exactamente 4 sliders");
  for (const tag of ranges) {
    assert.match(tag, /min="5"/);
    assert.match(tag, /max="100"/);
    assert.match(tag, /value="20"/);
    assert.doesNotMatch(tag, /min="-100"/);
  }
  for (const id of ["eq-all", "eq-r", "eq-g", "eq-b"]) {
    assert.ok(html.includes(`id="${id}"`), `falta el slider ${id}`);
  }

  const canvases = html.match(/<canvas[^>]*id="hist-/g) || [];
  assert.equal(canvases.length, 6, "deben existir exactamente 6 histogramas");
  for (const id of [
    "hist-r-before",
    "hist-r-after",
    "hist-g-before",
    "hist-g-after",
    "hist-b-before",
    "hist-b-after",
  ]) {
    assert.ok(html.includes(`id="${id}"`), `falta el histograma ${id}`);
  }

  assert.doesNotMatch(html, /brightness/i, "no debe quedar control de brillo");
  assert.doesNotMatch(html, /id="equalize"/, "se reemplazó el checkbox de luminancia");
});
