// Reads a PSD with ag-psd using the exact options StandRig's importer uses
// (packages/core/src/importers.ts) and prints the layer tree as JSON.
// Usage: node read_psd_agpsd.mjs file.psd   (ag-psd must be resolvable)
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(process.cwd() + "/");
const { readPsd, initializeCanvas } = require("ag-psd");
// Node has no canvas: provide a minimal ImageData factory (all StandRig needs to decode pixels).
initializeCanvas(() => { throw new Error("canvas not needed"); },
  (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }));
const buf = readFileSync(process.argv[2]);
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const psd = readPsd(ab, { skipCompositeImageData: true, skipThumbnail: true, skipLinkedFilesData: true,
                          logMissingFeatures: true, useImageData: true });
const walk = (l) => ({ name: l.name, hidden: !!l.hidden, opacity: l.opacity, blendMode: l.blendMode,
  clipping: !!l.clipping, left: l.left, top: l.top, right: l.right, bottom: l.bottom,
  alphaSum: l.imageData ? l.imageData.data.filter((_, i) => i % 4 === 3).reduce((a, b) => a + b, 0) : null,
  firstOpaqueRGBA: l.imageData ? (() => { const d = l.imageData.data; for (let i = 0; i < d.length; i += 4) if (d[i+3] === 255) return [d[i], d[i+1], d[i+2], d[i+3]]; return null; })() : null,
  children: l.children ? l.children.map(walk) : undefined });
console.log(JSON.stringify({ width: psd.width, height: psd.height, children: (psd.children || []).map(walk) }, null, 1));
