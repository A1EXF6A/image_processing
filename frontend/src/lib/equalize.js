export const CHANNELS = ["r", "g", "b"];

export const CHANNEL_OFFSET = { r: 0, g: 1, b: 2 };

export function clampPercent(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.min(100, Math.max(0, Math.round(n)));
}

export function interpolateLut(baseLut, percent) {
  const p = clampPercent(percent) / 100;
  const out = new Uint8Array(256);
  for (let k = 0; k < 256; k++) {
    const v = Math.round((1 - p) * k + p * Number(baseLut[k]));
    out[k] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  return out;
}

export function processRgba(source, luts, percent) {
  const lutR = interpolateLut(luts.r, percent.r);
  const lutG = interpolateLut(luts.g, percent.g);
  const lutB = interpolateLut(luts.b, percent.b);
  const out = new Uint8ClampedArray(source.length);
  for (let i = 0; i < source.length; i += 4) {
    out[i] = lutR[source[i]];
    out[i + 1] = lutG[source[i + 1]];
    out[i + 2] = lutB[source[i + 2]];
    out[i + 3] = source[i + 3];
  }
  return out;
}

export function channelHistogram(rgba, channel) {
  const offset = CHANNEL_OFFSET[channel];
  if (offset === undefined) throw new Error(`Canal desconocido: ${channel}`);
  const hist = new Array(256).fill(0);
  for (let i = offset; i < rgba.length; i += 4) hist[rgba[i]]++;
  return hist;
}

export function histogramsFromRgba(rgba) {
  const out = {};
  for (const ch of CHANNELS) out[ch] = channelHistogram(rgba, ch);
  return out;
}

export function areChannelsSynced(pct) {
  return pct.r === pct.g && pct.g === pct.b;
}

export function setGeneral(pct, percent) {
  const v = clampPercent(percent);
  return { r: v, g: v, b: v };
}

export function setChannel(pct, channel, percent) {
  if (!CHANNELS.includes(channel)) throw new Error(`Canal desconocido: ${channel}`);
  return { ...pct, [channel]: clampPercent(percent) };
}

export function generalState(pct) {
  const synced = areChannelsSynced(pct);
  return {
    synced,
    value: synced ? pct.r : Math.round((pct.r + pct.g + pct.b) / 3),
    label: synced ? `${pct.r}%` : "Mixto",
  };
}

export function channelSummary(pct) {
  return `R ${pct.r}% · G ${pct.g}% · B ${pct.b}%`;
}
