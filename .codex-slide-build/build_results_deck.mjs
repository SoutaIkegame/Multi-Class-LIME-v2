import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const root = "/Users/triumph1118/Github/Multi-Class-LIME v2/Multi-Class-LIME-v2";
const skillDir = "/Users/triumph1118/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const sourcePath = path.join(root, "docs/MIDTERM_PRESENTATION_2026-09-12.pptx");
const outPath = path.join(root, ".codex-slide-build/candidate-results-v3.pptx");
const previewDir = path.join(root, ".codex-slide-build/candidate-results-v3");

const { applyPresentationChartFont } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href,
);

const deck = await PresentationFile.importPptx(await FileBlob.load(sourcePath));
const FONT = "YuGothic";
const C = {
  navy: "#17233D", blue: "#1479C9", lightBlue: "#EAF4FB", orange: "#F28C28",
  green: "#2AA876", red: "#D95C5C", gray: "#687386", lightGray: "#E7EBF0",
  pale: "#F7F9FC", yellow: "#FFF4D6", white: "#FFFFFF", purple: "#7B61A8",
};

function addText(slide, text, left, top, width, height, opts = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: FONT,
    fontSize: opts.fontSize ?? 22,
    bold: opts.bold ?? false,
    color: opts.color ?? C.navy,
    alignment: opts.alignment ?? "left",
    verticalAlignment: opts.verticalAlignment ?? "middle",
    autoFit: opts.autoFit ?? "shrinkText",
  };
  return box;
}

function addRect(slide, left, top, width, height, fill, radius = 0, line = "none") {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill,
    line: line === "none" ? { fill: "none", width: 0 } : { fill: line, width: 1 },
    radius,
  });
}

function baseSlide(title, subtitle, page) {
  const slide = deck.slides.add();
  slide.background.fill = C.white;
  addRect(slide, 54, 43, 12, 50, C.blue);
  addText(slide, title, 80, 36, 1120, 50, { fontSize: 34, bold: true });
  addText(slide, subtitle, 80, 84, 1120, 26, { fontSize: 16, color: C.gray });
  addText(slide, String(page), 1168, 678, 64, 20, { fontSize: 13, color: "#8993A4", alignment: "right" });
  return slide;
}

function addSectionLabel(slide, label, x, y, w, color) {
  addRect(slide, x, y, w, 32, color, 8);
  addText(slide, label, x + 10, y + 1, w - 20, 29, { fontSize: 16, bold: true, color: C.white, alignment: "center" });
}

async function addImage(slide, relativePath, position, alt) {
  const bytes = await fs.readFile(path.join(root, relativePath));
  return slide.images.add({ blob: bytes, contentType: "image/png", alt, fit: "contain", position });
}

function addClaim(slide, text, y = 624, color = C.blue) {
  addRect(slide, 80, y, 1120, 43, C.lightBlue, 8);
  addText(slide, `このスライドの主張：${text}`, 100, y + 5, 1080, 32, {
    fontSize: 18, bold: true, color, alignment: "center",
  });
}

function styleChart(chart, yAxis = {}) {
  chart.xAxis = {
    textStyle: { typeface: FONT, fontSize: 13, fill: C.gray },
    line: { style: "solid", fill: "#CBD2DC", width: 1 },
    majorGridlines: null,
  };
  chart.yAxis = {
    textStyle: { typeface: FONT, fontSize: 13, fill: C.gray },
    line: { style: "solid", fill: "#CBD2DC", width: 1 },
    majorGridlines: { style: "solid", fill: "#E3E7ED", width: 1 },
    ...yAxis,
  };
  chart.legend = {
    position: "bottom", overlay: false,
    textStyle: { typeface: FONT, fontSize: 12, fill: C.navy },
  };
  applyPresentationChartFont(chart, { fontFamily: FONT });
}

// 11: evaluation design
{
  const s = baseSlide("評価設計", "何を説明するかと、どの損失で当てはめるかを分けて比較", 11);
  addRect(s, 80, 111, 1120, 25, C.lightBlue, 6);
  addText(s, "このスライドの主張：同じ説明予算で比べることで、『2クラスに絞る効果』と『回帰方法の効果』を分離する", 96, 112, 1088, 22, { fontSize: 15, bold: true, color: C.blue, alignment: "center" });
  addRect(s, 62, 134, 370, 330, C.pale, 10, C.lightGray);
  addSectionLabel(s, "従来：One-vs-Rest", 94, 156, 300, C.gray);
  addText(s, "各クラス対残りを別々に回帰", 92, 208, 310, 35, { fontSize: 21, bold: true, alignment: "center" });
  addText(s, "p₁(z) ≈ f₁(z)\np₂(z) ≈ f₂(z)", 116, 256, 265, 72, { fontSize: 25, alignment: "center" });
  addText(s, "最後に f₁ − f₂ で\n競合2クラスの方向を作る", 94, 354, 306, 62, { fontSize: 19, color: C.gray, alignment: "center" });

  addText(s, "→", 442, 270, 45, 50, { fontSize: 36, bold: true, color: C.blue, alignment: "center" });

  addRect(s, 492, 134, 726, 330, C.lightBlue, 10, C.blue);
  addSectionLabel(s, "提案の枠組み：Top-1 vs Top-2", 670, 156, 370, C.blue);
  addText(s, "q(z) = p₁(z) / {p₁(z)+p₂(z)}", 592, 212, 526, 45, { fontSize: 25, bold: true, alignment: "center" });
  const variants = [
    ["2クラスLIME", "q を Ridge 回帰", C.green],
    ["Contrastive", "log(p₁/p₂) を Ridge 回帰", C.orange],
    ["OVO Logistic", "q を交差エントロピーで回帰", C.purple],
  ];
  variants.forEach(([name, desc, color], i) => {
    const x = 526 + i * 222;
    addRect(s, x, 286, 200, 122, C.white, 8, color);
    addText(s, name, x + 12, 300, 176, 32, { fontSize: 18, bold: true, color, alignment: "center" });
    addText(s, desc, x + 14, 340, 172, 48, { fontSize: 16, alignment: "center" });
  });

  addText(s, "評価指標", 68, 493, 154, 34, { fontSize: 22, bold: true, color: C.blue });
  const metrics = [
    ["真係数の復元", "線形softmaxの既知係数と\n説明係数の順位相関"],
    ["競合特徴再現率", "本当にクラス差を作る特徴を\n上位K個で拾えた割合"],
    ["共通特徴表示率", "勝敗に関係しない共通特徴へ\n表示枠を使った割合"],
    ["計算時間", "説明1件の学習時間を\n次元数・クラス数別に測定"],
  ];
  metrics.forEach(([name, desc], i) => {
    const x = 68 + i * 296;
    addRect(s, x, 536, 274, 112, C.white, 8, C.lightGray);
    addText(s, name, x + 12, 547, 250, 28, { fontSize: 18, bold: true, color: C.blue, alignment: "center" });
    addText(s, desc, x + 14, 581, 246, 54, { fontSize: 15, color: C.gray, alignment: "center" });
  });
  s.speakerNotes.textFrame.setText("評価設計の出典: docs/OVO_VS_OVR_EXPERIMENT_SUMMARY.md, src/fidelity.py, src/metrics.py");
}

// 12: setup table
{
  const s = baseSlide("実験設定", "すべての手法で摂動点・近接度重み・説明対象を共有", 12);
  const values = [
    ["項目", "設定"],
    ["データ", "make_classification、2,000件、学習70%／評価30%"],
    ["真係数復元", "線形softmax、d={8,14,20}, C={3,4,5}、各条件20 seeds、説明点8件"],
    ["特徴の関連性", "役割既知の線形softmax、d=12, C={3,5}, K=3、共通特徴強度={0,1,3,5}"],
    ["計算時間", "Random Forest 200本、d={8,14,20}, C=3〜10、各条件10 seeds、説明点4件"],
    ["説明対象", "各入力における予測確率Top-1とTop-2。確率差が小さい点を優先"],
    ["摂動と局所重み", "各説明点でGaussian摂動300点。特徴標準偏差と指数カーネルを全手法で共有"],
    ["特徴選択", "関連性・時間実験では同じK。Lasso pathで選択後、選択特徴だけで最終再fit"],
    ["サロゲート", "Ridge α=1.0、Logistic C=1.0。乱数も各条件内で手法間共有"],
  ];
  const table = s.tables.add({ rows: values.length, columns: 2, left: 80, top: 136, width: 1120, height: 486, columnWidths: [245, 875], values });
  table.borders.assign({ style: "solid", fill: "#D9E0E8", width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 2 }).assign({
    fill: C.blue,
    textStyle: { typeface: FONT, fontSize: 17, bold: true, color: C.white },
    margins: { left: 12, right: 12, top: 8, bottom: 8 },
    anchor: "middle",
  });
  table.cells.block({ row: 1, column: 0, rowCount: 8, columnCount: 2 }).assign({
    textStyle: { typeface: FONT, fontSize: 15, color: C.navy },
    margins: { left: 12, right: 12, top: 6, bottom: 6 },
    anchor: "middle",
  });
  table.cells.block({ row: 1, column: 0, rowCount: 8, columnCount: 1 }).assign({
    fill: C.lightBlue,
    textStyle: { typeface: FONT, fontSize: 15, bold: true, color: C.blue },
  });
  for (let r = 1; r < values.length; r += 2) {
    table.getCell(r, 1).fill = "#FBFCFE";
  }
  addText(s, "このスライドの主張：真値・特徴の役割・計算時間という、意味の異なる3つの観点で検証する", 80, 637, 1120, 32, { fontSize: 18, bold: true, color: C.orange, alignment: "center" });
  s.speakerNotes.textFrame.setText("実験設定の出典: src/run_groundtruth_experiment.py, src/run_feature_role_experiment.py, src/run_timing_experiment.py");
}

if (false) {
// Legacy native-chart slides retained in the build source for reference.
// 13: RF main result
{
  const s = baseSlide("主実験：Random Forestでの忠実度", "同じK個の特徴を表示し、未使用摂動でTop-1 / Top-2の大小関係を評価", 13);
  const chart = s.charts.add("bar", {
    position: { left: 66, top: 144, width: 770, height: 425 },
    title: "ペア符号忠実度",
    titleTextStyle: { typeface: FONT, fontSize: 17, fill: C.blue, bold: true },
    categories: ["OVR\nexact-K", "2クラス\nLIME", "Contrastive", "OVO\nLogistic"],
    series: [{
      name: "ペア符号忠実度",
      values: [0.7873022, 0.7913670, 0.7889341, 0.7818017],
      fill: C.blue,
      points: [
        { idx: 0, fill: C.gray }, { idx: 1, fill: C.green },
        { idx: 2, fill: C.orange }, { idx: 3, fill: C.purple },
      ],
    }],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 65, varyColors: true },
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: FONT, fontSize: 15, fill: C.navy, bold: true } },
  });
  styleChart(chart, { min: 0.76, max: 0.80, majorUnit: 0.01, numberFormatCode: "0.00" });
  chart.dataLabels = { showValue: true, position: "outEnd", textStyle: { typeface: FONT, fontSize: 15, fill: C.navy, bold: true } };

  addRect(s, 872, 153, 330, 118, C.lightBlue, 10);
  addText(s, "+0.0041", 896, 164, 282, 48, { fontSize: 34, bold: true, color: C.blue, alignment: "center" });
  addText(s, "2クラスLIME − OVR", 896, 215, 282, 35, { fontSize: 17, color: C.gray, alignment: "center" });
  addRect(s, 872, 289, 330, 113, C.pale, 10, C.lightGray);
  addText(s, "5 / 18 条件", 895, 302, 284, 40, { fontSize: 27, bold: true, color: C.green, alignment: "center" });
  addText(s, "Holm補正後も有意", 895, 348, 284, 31, { fontSize: 17, color: C.gray, alignment: "center" });
  addRect(s, 872, 420, 330, 149, C.yellow, 10);
  addText(s, "忠実度の差は小さい", 895, 435, 284, 34, { fontSize: 22, bold: true, color: C.orange, alignment: "center" });
  addText(s, "+0.41ポイントだけで優位とは言い切らず、\n特徴の関連性と計算時間も合わせて判断する", 895, 478, 284, 69, { fontSize: 16, color: C.navy, alignment: "center" });
  addText(s, "平均値、20 seeds", 70, 590, 760, 22, { fontSize: 13, color: C.gray, alignment: "center" });
  s.speakerNotes.textFrame.setText("出典: results/pairwise_lime_baseline_summary.csv, results/pairwise_lime_baseline_overall_stats.csv");
}

// 14: BB comparison
{
  const s = baseSlide("ブラックボックス別の比較", "softmaxを含むモデルと含まないモデルで、同じ評価を実施", 14);
  const chart = s.charts.add("bar", {
    position: { left: 55, top: 140, width: 900, height: 455 },
    title: "BB別のペア符号忠実度",
    titleTextStyle: { typeface: FONT, fontSize: 17, fill: C.blue, bold: true },
    categories: ["線形\nsoftmax", "非線形NN\nsoftmax", "Random\nForest"],
    series: [
      { name: "OVR exact-K", values: [0.84922, 0.77625, 0.78169], fill: C.gray },
      { name: "2クラスLIME", values: [0.86321, 0.78397, 0.78471], fill: C.green },
      { name: "Contrastive", values: [0.86548, 0.78079, 0.78189], fill: C.orange },
      { name: "OVO Logistic", values: [0.85602, 0.77618, 0.77453], fill: C.purple },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 55 },
    hasLegend: true,
  });
  styleChart(chart, { min: 0.74, max: 0.88, majorUnit: 0.02, numberFormatCode: "0.00" });
  addRect(s, 980, 153, 245, 135, C.lightBlue, 10);
  addText(s, "線形 softmax", 998, 166, 210, 29, { fontSize: 18, bold: true, color: C.blue, alignment: "center" });
  addText(s, "Contrastive が最高\n0.8655", 998, 204, 210, 60, { fontSize: 23, bold: true, color: C.orange, alignment: "center" });
  addRect(s, 980, 307, 245, 135, C.pale, 10, C.lightGray);
  addText(s, "非線形NN・RF", 998, 320, 210, 29, { fontSize: 18, bold: true, color: C.blue, alignment: "center" });
  addText(s, "2クラスLIME が最高", 998, 362, 210, 52, { fontSize: 21, bold: true, color: C.green, alignment: "center" });
  addRect(s, 980, 461, 245, 134, C.yellow, 10);
  addText(s, "示唆", 998, 474, 210, 28, { fontSize: 18, bold: true, color: C.orange, alignment: "center" });
  addText(s, "対数比の利点は\n線形logit構造で最も出る", 998, 510, 210, 62, { fontSize: 17, alignment: "center" });
  addText(s, "平均値、20 seeds", 65, 611, 880, 22, { fontSize: 13, color: C.gray, alignment: "center" });
  s.speakerNotes.textFrame.setText("出典: results/blackbox_comparison_summary.csv, results/blackbox_comparison_stats.csv");
}

// 15: interpretability feature role
{
  const s = baseSlide("解釈性：競合に関係する特徴を選べるか", "2クラスに共通するだけの特徴を強くし、表示上位K個への混入を測定", 15);
  const chart = s.charts.add("line", {
    position: { left: 58, top: 143, width: 850, height: 438 },
    title: "競合特徴再現率",
    titleTextStyle: { typeface: FONT, fontSize: 17, fill: C.blue, bold: true },
    categories: ["0", "1", "3", "5"],
    series: [
      { name: "TopクラスLIME", values: [0.9958, 0.9885, 0.8448, 0.7802], line: { style: "solid", fill: "#9BA3AF", width: 2 }, marker: { symbol: "circle", size: 7 } },
      { name: "OVR exact-K", values: [0.9958, 0.9979, 0.8698, 0.8073], line: { style: "solid", fill: C.blue, width: 3 }, marker: { symbol: "square", size: 7 } },
      { name: "2クラスLIME", values: [1, 1, 1, 1], line: { style: "solid", fill: C.green, width: 3 }, marker: { symbol: "circle", size: 7 } },
      { name: "Contrastive", values: [1, 1, 1, 1], line: { style: "solid", fill: C.orange, width: 3 }, marker: { symbol: "diamond", size: 7 } },
      { name: "OVO Logistic", values: [1, 1, 1, 0.9990], line: { style: "solid", fill: C.purple, width: 3 }, marker: { symbol: "triangle", size: 7 } },
    ],
    lineOptions: { grouping: "standard", smooth: false },
    hasLegend: true,
  });
  styleChart(chart, { min: 0.72, max: 1.02, majorUnit: 0.05, numberFormatCode: "0%", title: "競合特徴再現率" });
  chart.xAxis = {
    title: { text: "共通特徴の強さ", textStyle: { typeface: FONT, fontSize: 14, fill: C.gray } },
    textStyle: { typeface: FONT, fontSize: 13, fill: C.gray },
    line: { style: "solid", fill: "#CBD2DC", width: 1 }, majorGridlines: null,
  };
  addRect(s, 942, 151, 276, 168, C.lightBlue, 10);
  addText(s, "共通特徴の強さ = 5", 959, 166, 242, 30, { fontSize: 18, bold: true, color: C.blue, alignment: "center" });
  addText(s, "OVR　0.807", 968, 210, 224, 34, { fontSize: 24, bold: true, color: C.blue, alignment: "center" });
  addText(s, "2クラス方式　約1.000", 958, 255, 244, 38, { fontSize: 22, bold: true, color: C.green, alignment: "center" });
  addRect(s, 942, 338, 276, 160, C.yellow, 10);
  addText(s, "なぜ差が出るか", 959, 351, 242, 28, { fontSize: 18, bold: true, color: C.orange, alignment: "center" });
  addText(s, "共通特徴は両クラスを同じ方向に動かす。\nOVRでは重要に見えるが、クラス間の差には寄与しない。", 960, 391, 240, 83, { fontSize: 16, alignment: "center" });
  addText(s, "表示枠を『どちらのクラスを支持するか』に集中できる", 936, 520, 292, 54, { fontSize: 19, bold: true, color: C.orange, alignment: "center" });
  addText(s, "平均値、20 seeds", 66, 601, 820, 22, { fontSize: 13, color: C.gray, alignment: "center" });
  s.speakerNotes.textFrame.setText("出典: results/feature_role_summary.csv, results/feature_role_stats.csv");
}

// 16: timing
{
  const s = baseSlide("計算時間", "説明1件あたりのサロゲート学習時間。次元数とクラス数を別々に変化", 16);
  const seriesDim = [
    { name: "OVR exact-K", values: [39.095, 42.053, 43.185], line: { style: "solid", fill: C.gray, width: 3 }, marker: { symbol: "square", size: 7 } },
    { name: "2クラスLIME", values: [6.185, 6.571, 6.875], line: { style: "solid", fill: C.green, width: 3 }, marker: { symbol: "circle", size: 7 } },
    { name: "Contrastive", values: [6.135, 6.529, 6.895], line: { style: "solid", fill: C.orange, width: 3 }, marker: { symbol: "diamond", size: 7 } },
    { name: "OVO Logistic", values: [11.940, 15.713, 20.028], line: { style: "solid", fill: C.purple, width: 3 }, marker: { symbol: "triangle", size: 7 } },
  ];
  const chart1 = s.charts.add("line", {
    position: { left: 55, top: 145, width: 565, height: 402 }, title: "次元数を増やした場合", titleTextStyle: { typeface: FONT, fontSize: 17, fill: C.blue, bold: true }, categories: ["8", "14", "20"], series: seriesDim,
    lineOptions: { grouping: "standard", smooth: false }, hasLegend: false,
  });
  styleChart(chart1, { min: 0, max: 70, majorUnit: 10, numberFormatCode: "0" });
  chart1.xAxis = { title: { text: "次元数 d", textStyle: { typeface: FONT, fontSize: 14, fill: C.gray } }, textStyle: { typeface: FONT, fontSize: 13, fill: C.gray }, majorGridlines: null };
  chart1.yAxis = { title: { text: "時間 [ms]", textStyle: { typeface: FONT, fontSize: 14, fill: C.gray } }, min: 0, max: 70, majorUnit: 10, numberFormatCode: "0", textStyle: { typeface: FONT, fontSize: 13, fill: C.gray }, majorGridlines: { style: "solid", fill: "#E3E7ED", width: 1 } };

  const seriesCls = [
    { name: "OVR exact-K", values: [20.384, 26.826, 33.126, 38.839, 44.642, 49.744, 54.536, 63.457], line: { style: "solid", fill: C.gray, width: 3 }, marker: { symbol: "square", size: 6 } },
    { name: "2クラスLIME", values: [6.671, 6.555, 6.581, 6.553, 6.524, 6.651, 6.332, 6.478], line: { style: "solid", fill: C.green, width: 3 }, marker: { symbol: "circle", size: 6 } },
    { name: "Contrastive", values: [6.544, 6.627, 6.494, 6.602, 6.405, 6.637, 6.340, 6.509], line: { style: "solid", fill: C.orange, width: 3 }, marker: { symbol: "diamond", size: 6 } },
    { name: "OVO Logistic", values: [16.855, 16.294, 16.252, 16.161, 15.484, 15.651, 15.066, 15.388], line: { style: "solid", fill: C.purple, width: 3 }, marker: { symbol: "triangle", size: 6 } },
  ];
  const chart2 = s.charts.add("line", {
    position: { left: 655, top: 145, width: 565, height: 402 }, title: "クラス数を増やした場合", titleTextStyle: { typeface: FONT, fontSize: 17, fill: C.blue, bold: true }, categories: ["3", "4", "5", "6", "7", "8", "9", "10"], series: seriesCls,
    lineOptions: { grouping: "standard", smooth: false }, hasLegend: false,
  });
  styleChart(chart2, { min: 0, max: 70, majorUnit: 10, numberFormatCode: "0" });
  chart2.xAxis = { title: { text: "クラス数 C", textStyle: { typeface: FONT, fontSize: 14, fill: C.gray } }, textStyle: { typeface: FONT, fontSize: 13, fill: C.gray }, majorGridlines: null };
  chart2.yAxis = { min: 0, max: 70, majorUnit: 10, numberFormatCode: "0", textStyle: { typeface: FONT, fontSize: 13, fill: C.gray }, majorGridlines: { style: "solid", fill: "#E3E7ED", width: 1 } };

  const legendItems = [[C.gray, "OVR exact-K"], [C.green, "2クラスLIME"], [C.orange, "Contrastive"], [C.purple, "OVO Logistic"]];
  legendItems.forEach(([color, label], i) => {
    const x = 202 + i * 225;
    addRect(s, x, 569, 22, 8, color, 4);
    addText(s, label, x + 30, 553, 160, 35, { fontSize: 15, color: C.navy });
  });
  addRect(s, 183, 611, 914, 50, C.lightBlue, 9);
  addText(s, "C=10：OVR 63.46 ms ／ 2クラスLIME 6.48 ms ／ Contrastive 6.51 ms", 205, 619, 870, 34, { fontSize: 20, bold: true, color: C.blue, alignment: "center" });
  s.speakerNotes.textFrame.setText("出典: results/timing_plot_summary.csv, results/timing_results.csv");
}

// 17: discussion
{
  const s = baseSlide("結果からの考察", "忠実度・特徴選択・計算量を合わせて提案の位置づけを整理", 17);
  addRect(s, 70, 132, 1140, 96, C.lightBlue, 10);
  addText(s, "2クラスに絞る価値は、忠実度の小幅改善だけではない", 105, 145, 1070, 34, { fontSize: 26, bold: true, color: C.blue, alignment: "center" });
  addText(s, "『なぜ1位が2位より選ばれたか』に関係する特徴へ、限られた表示枠を集中できる", 105, 184, 1070, 30, { fontSize: 20, alignment: "center" });

  const rows = [
    ["汎用的な基準手法", "2クラスLIME", "3種類のBBで安定。実装が単純で、OVRより高速。", C.green],
    ["線形logit構造", "Contrastive", "線形softmaxで最高忠実度。対数比がスコア差と一致する条件を活かせる。", C.orange],
    ["現状の課題", "OVO Logistic", "特徴の関連性は高いが、忠実度と次元方向の計算時間で劣る。正則化の再調整が必要。", C.purple],
  ];
  rows.forEach(([tag, method, body, color], i) => {
    const y = 258 + i * 104;
    addSectionLabel(s, tag, 80, y + 10, 190, color);
    addText(s, method, 298, y, 220, 45, { fontSize: 23, bold: true, color });
    addText(s, body, 520, y - 1, 672, 60, { fontSize: 18, color: C.navy });
    if (i < 2) addRect(s, 295, y + 77, 897, 1, C.lightGray);
  });
  addRect(s, 76, 578, 1132, 70, C.yellow, 9);
  addText(s, "今後：実データ・他の摂動法で再検証し、説明の分かりやすさはユーザ実験で評価する", 102, 593, 1080, 39, { fontSize: 19, bold: true, color: C.orange, alignment: "center" });
  s.speakerNotes.textFrame.setText("考察の出典: docs/DISCUSSION.md。結論は今回の合成データ実験の範囲に限定する。");
}
}

if (false) {
// Earlier extended result section retained for reference.
// 13: RF result as source-backed figure
{
  const s = baseSlide("主実験：Random Forest", "exact-K・held-out評価。点は20 seedsの平均、線は95%ブートストラップ信頼区間", 13);
  await addImage(s, "results/pairwise_lime_baseline_comparison.png", { left: 52, top: 122, width: 870, height: 438 }, "Random Forest主実験の忠実度とBrier誤差");
  addRect(s, 946, 132, 276, 128, C.lightBlue, 10);
  addText(s, "符号忠実度", 964, 146, 240, 28, { fontSize: 18, bold: true, color: C.blue, alignment: "center" });
  addText(s, "2クラスLIME  0.791\nOVR exact-K   0.787", 963, 184, 242, 56, { fontSize: 20, bold: true, alignment: "center" });
  addRect(s, 946, 278, 276, 128, C.pale, 10, C.lightGray);
  addText(s, "確率誤差（Brier）", 964, 292, 240, 28, { fontSize: 18, bold: true, color: C.blue, alignment: "center" });
  addText(s, "2クラスLIME  0.0156\nContrastive   0.0158", 963, 330, 242, 56, { fontSize: 19, bold: true, alignment: "center" });
  addRect(s, 946, 424, 276, 136, C.yellow, 10);
  addText(s, "読み方", 965, 438, 238, 27, { fontSize: 18, bold: true, color: C.orange, alignment: "center" });
  addText(s, "+0.0041は小幅。\n『大幅な精度向上』ではなく、\n同じKでも悪化しない証拠と捉える。", 963, 472, 242, 70, { fontSize: 16, alignment: "center" });
  addClaim(s, "RFではリンク関数の工夫より、説明対象をTop-1対Top-2へ絞る効果が中心", 610);
  s.speakerNotes.textFrame.setText("出典: results/pairwise_lime_baseline_comparison.png。説明: +0.0041は0.41ポイントであり、実用上大きいとは断言しない。5/18条件でHolm補正後も有意だが、主張は同じ特徴予算でも忠実度を保ったこと。");
}

// 14: black-box comparison as source-backed figure
{
  const s = baseSlide("ブラックボックス構造別の結果", "線形softmax、非線形NN＋softmax、Random Forestを同じ設計で比較", 14);
  await addImage(s, "results/blackbox_comparison.png", { left: 56, top: 125, width: 895, height: 398 }, "ブラックボックス構造別の忠実度とBrier誤差");
  addRect(s, 974, 135, 248, 115, C.lightBlue, 10);
  addText(s, "線形 softmax", 992, 149, 212, 25, { fontSize: 17, bold: true, color: C.blue, alignment: "center" });
  addText(s, "Contrastive が最高\n忠実度 0.8655", 991, 184, 214, 52, { fontSize: 20, bold: true, color: C.orange, alignment: "center" });
  addRect(s, 974, 266, 248, 115, C.pale, 10, C.lightGray);
  addText(s, "非線形NN・RF", 992, 280, 212, 25, { fontSize: 17, bold: true, color: C.blue, alignment: "center" });
  addText(s, "2クラスLIME が最高", 990, 318, 216, 42, { fontSize: 20, bold: true, color: C.green, alignment: "center" });
  addRect(s, 974, 397, 248, 126, C.yellow, 10);
  addText(s, "解釈", 992, 411, 212, 25, { fontSize: 17, bold: true, color: C.orange, alignment: "center" });
  addText(s, "softmaxかどうかだけでは不十分。\n局所logit差が線形に近いとき、\n対数比Ridgeが効く。", 991, 445, 214, 62, { fontSize: 15, alignment: "center" });
  addClaim(s, "万能な1方式はなく、黒箱の局所構造に応じて回帰対象を選ぶ必要がある", 610);
  s.speakerNotes.textFrame.setText("出典: results/blackbox_comparison.png。線形softmaxではlog(p1/p2)が線形スコア差と一致するためContrastiveが有利。非線形NNではsoftmaxでもlogit差が入力に対して非線形なので、その利点は弱まる。");
}

// 15: feature-role figure
{
  const s = baseSlide("解釈性：競合に関係する特徴を選べるか", "特徴の役割が既知の線形softmaxで、競合特徴・共通特徴・無関係特徴を分離", 15);
  await addImage(s, "results/feature_role_comparison.png", { left: 52, top: 123, width: 1176, height: 405 }, "競合特徴の再現率、共通特徴の表示率、held-out忠実度");
  addRect(s, 75, 540, 355, 64, C.pale, 8, C.lightGray);
  addText(s, "共通特徴の強さ=5：OVRの競合特徴再現率 0.807", 91, 548, 323, 48, { fontSize: 16, bold: true, color: C.blue, alignment: "center" });
  addRect(s, 461, 540, 355, 64, C.lightBlue, 8);
  addText(s, "2クラス方式：競合特徴再現率 約1.000、共通特徴表示率 約0", 477, 548, 323, 48, { fontSize: 16, bold: true, color: C.green, alignment: "center" });
  addRect(s, 847, 540, 355, 64, C.yellow, 8);
  addText(s, "これは『人の理解しやすさ』ではなく、質問への特徴関連性の定量評価", 863, 548, 323, 48, { fontSize: 16, bold: true, color: C.orange, alignment: "center" });
  addClaim(s, "2クラス説明は、両クラスに共通する証拠を避け、勝敗を分ける特徴へ表示枠を使える", 614);
  s.speakerNotes.textFrame.setText("出典: results/feature_role_comparison.png。共通特徴はAとBの確率を同方向に動かすため、各クラス単独の説明では重要でもA対Bの差には寄与しない。解釈性の直接的な人間評価ではなく、質問関連性を測る代理指標である。");
}

// 16: timing figure
{
  const s = baseSlide("計算時間", "特徴選択＋選択後の最終再学習。BB推論と摂動生成は含めない", 16);
  await addImage(s, "results/timing_scaling.png", { left: 62, top: 119, width: 1156, height: 474 }, "次元数とクラス数によるサロゲート学習時間の推移");
  addClaim(s, "Top-1対Top-2の1ペアだけを作る方式はクラス数に依存せず、C=10ではOVRの約10分の1", 614);
  s.speakerNotes.textFrame.setText("出典: results/timing_scaling.png。C=10ではOVR 63.46ms、2クラスLIME 6.48ms、Contrastive 6.51ms。全クラス対を説明するOVOではなく、説明点のTop-1/Top-2の1ペアだけを作る場合の測定。");
}

// 17: interpretation of selecting two classes
{
  const s = baseSlide("考察1：2クラスに絞ると何が変わるか", "OVRとpairwise説明は、そもそも答えている質問が異なる", 17);
  addRect(s, 80, 128, 1120, 82, C.lightBlue, 10);
  addText(s, "OVRは『なぜクラスAらしいか』、2クラス説明は『なぜAがBより選ばれたか』に答える", 105, 145, 1070, 46, { fontSize: 24, bold: true, color: C.blue, alignment: "center" });
  const rows = [
    ["① 共通証拠を相殺", "AとBを同じ方向に押し上げる特徴は、A単独では重要でもA対Bの勝敗には効かない。qやlog(pA/pB)を説明すると、その共通成分が表示上位から外れやすい。", C.green],
    ["② 表示予算を集中", "説明特徴数Kが小さいほど、関係のない共通特徴が1個入る損失は大きい。pairwise方式は限られた枠を『どちらを支持するか』へ集中できる。", C.orange],
    ["③ 忠実度は維持", "RF主実験の改善は+0.0041と小さいが、同じKでも悪化せず、特徴関連性と計算時間は明確に改善した。価値は3指標を合わせて判断する。", C.purple],
  ];
  rows.forEach(([head, body, color], i) => {
    const y = 238 + i * 112;
    addSectionLabel(s, head, 88, y + 14, 218, color);
    addRect(s, 326, y, 866, 84, C.pale, 8, C.lightGray);
    addText(s, body, 348, y + 8, 822, 68, { fontSize: 18, color: C.navy });
  });
  addClaim(s, "精度向上だけでなく、『質問に合う特徴を出す』ことが2クラス化の主要な意義", 610);
  s.speakerNotes.textFrame.setText("説明時は、OVRとpairwiseが異なる問いへ答える点を明示する。解釈性を断定せず、既知の真値に対する特徴関連性が改善したと述べる。");
}

// 18: interpretation of regression choices
{
  const s = baseSlide("考察2：3つの2クラス方式をどう使い分けるか", "2クラスの選択と、その後のリンク関数・損失の選択を分けて考える", 18);
  const cards = [
    ["2クラスLIME", "qをRidge回帰", "3種類のBBで安定し、実装も最も単純。非線形NNとRFで最高忠実度。まず置くべき汎用ベースライン。", C.green],
    ["Contrastive", "log(p₁/p₂)をRidge回帰", "線形softmaxで忠実度・Brier・真係数復元が最良。局所logit差が線形に近い場合に理論と結果が一致。", C.orange],
    ["OVO Logistic", "qを交差エントロピーで回帰", "極端確率でも目的変数が発散しない利点はあるが、今回の疎な設定では忠実度が低く、次元増加で遅くなった。", C.purple],
  ];
  cards.forEach(([name, target, body, color], i) => {
    const x = 72 + i * 394;
    addRect(s, x, 142, 366, 356, C.white, 10, color);
    addRect(s, x, 142, 366, 60, color, 10);
    addText(s, name, x + 18, 152, 330, 40, { fontSize: 23, bold: true, color: C.white, alignment: "center" });
    addText(s, target, x + 24, 222, 318, 48, { fontSize: 20, bold: true, color, alignment: "center" });
    addRect(s, x + 25, 284, 316, 1, C.lightGray);
    addText(s, body, x + 30, 304, 306, 160, { fontSize: 18, alignment: "left" });
  });
  addRect(s, 102, 526, 1076, 68, C.yellow, 9);
  addText(s, "研究としての主張は『対数比が常に最良』ではなく、2クラス化の効果と回帰方法の条件依存性を分離したこと", 128, 540, 1024, 40, { fontSize: 20, bold: true, color: C.orange, alignment: "center" });
  addClaim(s, "現時点の汎用候補は2クラスLIME、線形logit構造が期待できる場合はContrastive", 614);
  s.speakerNotes.textFrame.setText("OVO Logisticは失敗扱いではなく、正則化C、サンプル数、局所幅との相互作用を今後調べる。今回の結論は合成データとTop-1/Top-2に限定。");
}

// 19: integrated conclusion and limits
{
  const s = baseSlide("結果のまとめと今後の課題", "今回の実験から言えることと言えないことを分ける", 19);
  addSectionLabel(s, "支持された", 82, 132, 180, C.green);
  addText(s, "• 同じKでも、2クラスLIMEはOVRと同等以上のheld-out忠実度を保つ\n• 競合特徴の再現率が高く、両クラス共通特徴への表示枠の消費を避ける\n• Top-1/Top-2の1ペアなら、クラス数が増えても学習時間がほぼ一定", 291, 122, 895, 126, { fontSize: 19 });
  addRect(s, 82, 270, 1104, 1, C.lightGray);
  addSectionLabel(s, "条件付き", 82, 294, 180, C.orange);
  addText(s, "• Contrastiveの利点は、softmaxの有無より局所logit差の線形性に依存する\n• RFでの+0.0041は統計的差がある条件を含むが、効果量としては小さい\n• OVO Logisticの発散回避という理論的利点は、今回の疎な設定では性能向上に結びつかなかった", 291, 284, 895, 126, { fontSize: 19 });
  addRect(s, 82, 432, 1104, 1, C.lightGray);
  addSectionLabel(s, "未検証", 82, 456, 180, C.purple);
  addText(s, "• 実データでも同じ傾向になるか\n• Top-2以外の競合クラスを指定した場合や、入力ごとに競合が変化する場合\n• 人が説明を理解しやすいかというユーザ評価\n• OVO Logisticの正則化、摂動数、局所幅を調整した再評価", 291, 446, 895, 142, { fontSize: 19 });
  addClaim(s, "結論は『OVOが常に高精度』ではなく、特定の競合を説明する設計が関連性と計算量で有望ということ", 614);
  s.speakerNotes.textFrame.setText("最終結論。新規性は既存のpairwise説明やlog-oddsそのものではなく、Top-1/Top-2への適用と、OVR・通常LIME・log-odds Ridge・local logisticを同一条件で分解比較した実証設計に置く。");
}
}

// 13: strongest quantitative recovery result
{
  const s = baseSlide("結果1：真の特徴方向を復元できるか", "係数が既知の線形softmax黒箱で、説明係数の特徴順位を真値と比較", 13);
  await addImage(s, "results/groundtruth_recovery.png", { left: 63, top: 126, width: 865, height: 440 }, "線形softmaxにおける真のペア係数方向の復元結果");
  addRect(s, 957, 145, 255, 142, C.lightBlue, 10);
  addText(s, "Contrastive", 978, 161, 213, 31, { fontSize: 21, bold: true, color: C.orange, alignment: "center" });
  addText(s, "Spearman\n0.9993", 978, 202, 213, 67, { fontSize: 30, bold: true, color: C.orange, alignment: "center" });
  addRect(s, 957, 310, 255, 112, C.pale, 10, C.lightGray);
  addText(s, "OVR：0.9758", 978, 327, 213, 33, { fontSize: 23, bold: true, color: C.gray, alignment: "center" });
  addText(s, "差 +0.0235", 978, 372, 213, 30, { fontSize: 19, bold: true, color: C.blue, alignment: "center" });
  addRect(s, 957, 445, 255, 121, C.yellow, 10);
  addText(s, "9 / 9 条件で有意", 975, 459, 219, 31, { fontSize: 21, bold: true, color: C.green, alignment: "center" });
  addText(s, "対数比が線形スコア差と一致するため、\n真の『AとBを分ける方向』をほぼ完全に復元", 975, 499, 219, 54, { fontSize: 15, alignment: "center" });
  addClaim(s, "線形logit構造では、対数比を回帰するContrastiveが真の特徴方向を最も正確に復元する", 612);
  s.speakerNotes.textFrame.setText("出典: results/groundtruth_recovery.png, results/groundtruth_results.csv, results/groundtruth_stats.csv。Contrastive vs OVRは全9グリッドセルでHolm補正後も有意。線形softmaxに限定した結果である。");
}

// 14: strongest interpretability result
{
  const s = baseSlide("結果2：競合に関係する特徴を選べるか", "A/B競合特徴、A/B共通特徴、他クラス専用・無関係特徴が既知のモデル", 14);
  await addImage(s, "results/feature_role_comparison.png", { left: 52, top: 124, width: 1176, height: 407 }, "競合特徴再現率、共通特徴表示率、held-out忠実度");
  addRect(s, 75, 542, 355, 62, C.pale, 8, C.lightGray);
  addText(s, "共通特徴の強さ=5：OVRの競合特徴再現率 0.807", 91, 549, 323, 47, { fontSize: 16, bold: true, color: C.blue, alignment: "center" });
  addRect(s, 461, 542, 355, 62, C.lightBlue, 8);
  addText(s, "2クラス方式：競合特徴再現率 約1.000、共通特徴表示率 約0", 477, 549, 323, 47, { fontSize: 16, bold: true, color: C.green, alignment: "center" });
  addRect(s, 847, 542, 355, 62, C.yellow, 8);
  addText(s, "共通特徴はA/Bを同方向に動かすため、A対Bの勝敗には寄与しない", 863, 549, 323, 47, { fontSize: 16, bold: true, color: C.orange, alignment: "center" });
  addClaim(s, "2クラス説明は、両クラスに共通する証拠を避け、勝敗を分ける特徴へ表示枠を集中できる", 614);
  s.speakerNotes.textFrame.setText("出典: results/feature_role_comparison.png。これは人の理解しやすさを直接測った結果ではなく、説明質問への特徴関連性を既知の真値で定量化した結果。");
}

// 15: strongest computational result
{
  const s = baseSlide("結果3：クラス数が増えたときの計算時間", "Top-1対Top-2の1ペアだけを説明。特徴選択＋選択後の最終再学習を測定", 15);
  await addImage(s, "results/timing_scaling.png", { left: 62, top: 118, width: 1156, height: 478 }, "次元数とクラス数によるサロゲート学習時間の推移");
  addClaim(s, "C=10ではOVR 63.46msに対し、2クラスLIME 6.48ms・Contrastive 6.51msで約10分の1", 614);
  s.speakerNotes.textFrame.setText("出典: results/timing_scaling.png。BB推論と摂動生成は含まない。全クラス対を作るOVOではなく、説明点のTop-1/Top-2の1ペアだけを作る設定。");
}

// 16: concise conclusion
{
  const s = baseSlide("結果のまとめ", "本編では、提案の価値を最も明確に示す3つの結果に絞る", 16);
  const rows = [
    ["正確さ", "真の特徴方向を復元", "線形softmaxでContrastiveのSpearman相関は0.9993。OVRの0.9758を全9条件で上回った。", C.orange],
    ["関連性", "勝敗を分ける特徴へ集中", "共通特徴を強くしても、2クラス方式は競合特徴を約100%再現し、共通特徴をほぼ表示しなかった。", C.green],
    ["効率", "クラス数に対してほぼ一定", "OVRは全クラス分を学習するためCとともに増加。Top-1/Top-2の1ペアならC=10で約10分の1。", C.blue],
  ];
  rows.forEach(([tag, head, body, color], i) => {
    const y = 148 + i * 137;
    addSectionLabel(s, tag, 82, y + 15, 154, color);
    addText(s, head, 270, y, 310, 46, { fontSize: 23, bold: true, color });
    addRect(s, 596, y - 2, 600, 87, C.pale, 8, C.lightGray);
    addText(s, body, 620, y + 7, 552, 69, { fontSize: 18 });
  });
  addRect(s, 90, 568, 1100, 58, C.yellow, 9);
  addText(s, "主張：特定の競合クラスを説明するなら、OVRよりも2クラスに絞る方が、正確・関連的・高速な説明を作れる条件がある", 115, 579, 1050, 37, { fontSize: 20, bold: true, color: C.orange, alignment: "center" });
  addClaim(s, "特にContrastiveは線形logit構造で強く、2クラス化自体は特徴関連性と計算量で効果が明確", 634);
  s.speakerNotes.textFrame.setText("一般化範囲は合成データ、Top-1/Top-2、線形softmaxまたはRFの今回条件。実データとユーザ評価は今後の課題として口頭で補足する。");
}

await fs.mkdir(previewDir, { recursive: true });
await (await PresentationFile.exportPptx(deck)).save(outPath);
for (let i = 10; i < deck.slides.items.length; i++) {
  const slide = deck.slides.items[i];
  const png = await deck.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(previewDir, `slide-${i + 1}.png`), new Uint8Array(await png.arrayBuffer()));
}
console.log(JSON.stringify({ outPath, slides: deck.slides.items.length, previewDir }));
