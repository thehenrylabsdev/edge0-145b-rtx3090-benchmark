// Render every specs/*.chart.json (an ECharts option plus "width"/"height") to ../assets/<name>.svg and .png.
// ECharts server-side rendering produces the SVG; resvg rasterises it at 2x. No browser is involved.
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as echarts from "echarts";
import { Resvg } from "@resvg/resvg-js";

const here = path.dirname(fileURLToPath(import.meta.url));
const specDir = path.join(here, "specs");
const outDir = path.join(here, "..", "assets");
fs.mkdirSync(outDir, { recursive: true });

for (const file of fs.readdirSync(specDir).filter((f) => f.endsWith(".chart.json")).sort()) {
  const { width = 1200, height = 675, ...option } = JSON.parse(fs.readFileSync(path.join(specDir, file), "utf8"));
  const chart = echarts.init(null, null, { renderer: "svg", ssr: true, width, height });
  chart.setOption({ animation: false, ...option });
  const svg = chart.renderToSVGString();
  chart.dispose();
  const name = file.replace(/\.chart\.json$/, "");
  fs.writeFileSync(path.join(outDir, `${name}.svg`), svg);
  const png = new Resvg(svg, { fitTo: { mode: "zoom", value: 2 }, font: { loadSystemFonts: true } }).render().asPng();
  fs.writeFileSync(path.join(outDir, `${name}.png`), png);
  console.log(`${name}: ${width}x${height} -> assets/${name}.svg, assets/${name}.png (2x)`);
}
