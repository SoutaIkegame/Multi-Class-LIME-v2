import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const root = "/Users/triumph1118/Github/Multi-Class-LIME v2/Multi-Class-LIME-v2";
const sourcePath = path.join(root, "pptx/MIDTERM_PRESENTATION_WITH_RESULTS_V2_2026-09-09.pptx");
const outPath = path.join(root, ".codex-slide-build/candidate-selected.pptx");
const previewDir = path.join(root, ".codex-slide-build/candidate-selected");
const deck = await PresentationFile.importPptx(await FileBlob.load(sourcePath));

const FONT = "Hiragino Sans";
const C = {
  navy: "#17233D", blue: "#1479C9", lightBlue: "#EAF4FB", orange: "#F28C28",
  green: "#2AA876", gray: "#687386", lightGray: "#E7EBF0", pale: "#F7F9FC",
  yellow: "#FFF4D6", white: "#FFFFFF",
};

function addText(slide, text, left, top, width, height, opts = {}) {
  const box = slide.shapes.add({
    geometry: "textbox", position: { left, top, width, height },
    fill: "none", line: { fill: "none", width: 0 },
  });
  box.text = text;
  box.text.style = {
    typeface: FONT, fontSize: opts.fontSize ?? 22, bold: opts.bold ?? false,
    color: opts.color ?? C.navy, alignment: opts.alignment ?? "left",
    verticalAlignment: "middle", autoFit: "shrinkText",
  };
  return box;
}

function addRect(slide, left, top, width, height, fill, radius = 0, line = "none") {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect", position: { left, top, width, height }, fill,
    line: line === "none" ? { fill: "none", width: 0 } : { fill: line, width: 1 }, radius,
  });
}

function baseSlide(title, subtitle, page, existingSlide = null) {
  const slide = existingSlide ?? deck.slides.add();
  slide.background.fill = C.white;
  addRect(slide, 54, 43, 12, 50, C.blue);
  addText(slide, title, 80, 36, 1120, 50, { fontSize: 34, bold: true });
  addText(slide, subtitle, 80, 84, 1120, 26, { fontSize: 16, color: C.gray });
  addText(slide, String(page), 1168, 678, 64, 20, { fontSize: 13, color: "#8993A4", alignment: "right" });
  return slide;
}

async function addImage(slide, relativePath, position, alt) {
  const bytes = await fs.readFile(path.join(root, relativePath));
  return slide.images.add({ blob: bytes, contentType: "image/png", alt, fit: "contain", position });
}

function addClaim(slide, text, y = 614) {
  addRect(slide, 80, y, 1120, 43, C.lightBlue, 8);
  addText(slide, `このスライドの主張：${text}`, 100, y + 5, 1080, 32, {
    fontSize: 18, bold: true, color: C.blue, alignment: "center",
  });
}

function addSectionLabel(slide, label, x, y, w, color) {
  addRect(slide, x, y, w, 34, color, 8);
  addText(slide, label, x + 10, y + 2, w - 20, 29, { fontSize: 17, bold: true, color: C.white, alignment: "center" });
}

// Keep the first four result-slide identities and replace their contents.
// Delete only the three surplus discussion slides (22–24).
for (let i = 23; i >= 21; i--) deck.slides.items[i].delete();
const resultSlides = deck.slides.items.slice(17, 21);
for (const slide of resultSlides) {
  slide.shapes.deleteAll();
  for (const image of [...slide.images.items]) slide.images.deleteById(image.id);
  for (const table of [...slide.tables.items]) slide.tables.deleteById(table.id);
  for (const chart of [...slide.charts.items]) slide.charts.deleteById(chart.id);
}

// Update every section divider to the shorter result-section range.
const rangeHits = await deck.inspect({ kind: "textbox", search: "p.18-24", maxChars: 8000 });
for (const line of rangeHits.ndjson.trim().split("\n").filter(Boolean)) {
  const hit = JSON.parse(line);
  if (hit.id) deck.resolve(hit.id).text.replace("p.18-24", "p.18-21");
}

// Align the existing setup slide with the three selected result families.
const setup = deck.resolve("tb/2xsfqlcz");
const setupValues = [
  ["項目", "設定"],
  ["データ", "make_classification、2,000件、学習70%／評価30%"],
  ["局所真値実験", "非線形softmax、相関入力、弱い密ノイズ、d=30, C={3,5,8}, K=5、曲率={0.35,0.80}"],
  ["反復と評価", "各条件20独立BB、各6説明点。真の局所勾配Top-5再現率＋未使用摂動での符号忠実度"],
  ["計算時間", "Random Forest 200本、d={8,14,20}, C=3〜10、各条件10 seeds、説明点4件"],
  ["説明対象", "各入力における予測確率Top-1とTop-2。確率差が小さい点を優先"],
  ["摂動と局所重み", "各説明点でGaussian摂動300点。特徴標準偏差と指数カーネルを全手法で共有"],
  ["特徴選択", "関連性・時間実験では同じK。Lasso pathで選択後、選択特徴だけで最終再fit"],
  ["サロゲート", "Ridge α=1.0、Logistic C=1.0。乱数も各条件内で手法間共有"],
];
for (let r = 0; r < setupValues.length; r++) {
  for (let c = 0; c < 2; c++) setup.cells.set(r, c, setupValues[r][c]);
}
deck.resolve("sh/dcbm583y").text = "このスライドの主張：簡単な線形復元を避け、局所真値・未知摂動・計算時間で検証する";

// 18: challenging local ground-truth recovery
{
  const s = baseSlide("結果1：真の局所特徴を選べるか", "非線形・相関・ノイズを含む30次元BB。各点の真のA/B勾配Top-5を数値計算", 18, resultSlides[0]);
  await addImage(s, "results/challenging_groundtruth_recall.png", { left: 55, top: 126, width: 895, height: 446 }, "曲率別・クラス数別の真の局所Top-5特徴再現率");
  addRect(s, 968, 142, 242, 122, C.pale, 10, C.lightGray);
  addText(s, "全条件平均", 986, 154, 206, 28, { fontSize: 18, bold: true, color: C.gray, alignment: "center" });
  addText(s, "OVR  0.820", 986, 190, 206, 56, { fontSize: 27, bold: true, color: C.gray, alignment: "center" });
  addRect(s, 968, 284, 242, 128, C.lightBlue, 10);
  addText(s, "2クラス通常LIME  0.913\nContrastive  0.926", 986, 302, 206, 88, { fontSize: 20, bold: true, color: C.orange, alignment: "center" });
  addRect(s, 968, 432, 242, 133, C.yellow, 10);
  addText(s, "両方式とも\nOVRより全6条件で有意", 986, 452, 206, 72, { fontSize: 19, bold: true, color: C.green, alignment: "center" });
  addText(s, "Holm補正後", 986, 528, 206, 23, { fontSize: 14, color: C.gray, alignment: "center" });
  addClaim(s, "答えを固定しない非線形条件でも、2クラス方式は真の局所特徴をより多く回収した");
  s.speakerNotes.textFrame.setText("出典: results/challenging_groundtruth_recall.png, challenging_groundtruth_results.csv。真値は各説明点で数値計算したlog(p_A/p_B)の局所勾配。競合特徴数とKを一致させた旧実験とは異なり、弱い密ノイズを含む30次元からTop-5を選ぶ。");
}

// 19: held-out fidelity in the same challenging benchmark
{
  const s = baseSlide("結果2：未知の近傍でも判定を再現できるか", "結果1と同じ非線形BBで、学習に使っていない400摂動上のA/B判定を評価", 19, resultSlides[1]);
  await addImage(s, "results/challenging_groundtruth_fidelity.png", { left: 55, top: 126, width: 895, height: 446 }, "曲率別・クラス数別の未使用摂動におけるA/B符号忠実度");
  addRect(s, 968, 142, 242, 122, C.pale, 10, C.lightGray);
  addText(s, "全条件平均", 986, 154, 206, 28, { fontSize: 18, bold: true, color: C.gray, alignment: "center" });
  addText(s, "OVR  0.887", 986, 190, 206, 56, { fontSize: 27, bold: true, color: C.gray, alignment: "center" });
  addRect(s, 968, 284, 242, 128, C.lightBlue, 10);
  addText(s, "2クラス通常LIME  0.898\nContrastive  0.901", 986, 302, 206, 88, { fontSize: 20, bold: true, color: C.orange, alignment: "center" });
  addRect(s, 968, 432, 242, 133, C.yellow, 10);
  addText(s, "改善は約1ポイント\n大差ではない", 986, 452, 206, 72, { fontSize: 21, bold: true, color: C.orange, alignment: "center" });
  addText(s, "両方式とも6/6条件で有意", 986, 528, 206, 23, { fontSize: 14, color: C.green, alignment: "center" });
  addClaim(s, "2クラス方式は未知摂動でも一貫して改善したが、忠実度の差は約1ポイントに留まる");
  s.speakerNotes.textFrame.setText("出典: results/challenging_groundtruth_fidelity.png。全条件平均はOVR 0.8871、2クラス通常LIME 0.8983、Contrastive 0.9005。20独立BBの対応検定で両方式とも6/6条件有意だが、実用差は約1.1〜1.3ポイント。");
}

// 20: timing
{
  const s = baseSlide("結果3：クラス数が増えたときの計算時間", "Top-1対Top-2の1ペアだけを説明。特徴選択＋選択後の最終再学習を測定", 20, resultSlides[2]);
  await addImage(s, "results/timing_scaling.png", { left: 62, top: 118, width: 1156, height: 478 }, "次元数とクラス数によるサロゲート学習時間の推移");
  addClaim(s, "C=10ではOVR 63.46msに対し、2クラスLIME 6.48ms・Contrastive 6.51msで約10分の1");
  s.speakerNotes.textFrame.setText("出典: results/timing_scaling.png。BB推論と摂動生成は含まない。全クラス対ではなく、Top-1/Top-2の1ペアだけを説明する設定。");
}

// 21: concise conclusion
{
  const s = baseSlide("結果のまとめ", "本編では、提案の価値を最も明確に示す3つの結果に絞る", 21, resultSlides[3]);
  const rows = [
    ["関連性", "真のTop-5を約10pt多く回収", "非線形・相関・ノイズ下でOVR 0.820に対し、2クラス通常LIME 0.913、Contrastive 0.926。", C.orange],
    ["忠実度", "改善は約1ポイント", "未使用摂動でOVR 0.887に対し、2クラス通常LIME 0.898、Contrastive 0.901。差は小さい。", C.green],
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
  addText(s, "2クラス化の主な効果は、予測値の大幅改善より、A/Bを分ける局所特徴を限られた表示枠へ集めること", 115, 579, 1050, 37, { fontSize: 20, bold: true, color: C.orange, alignment: "center" });
  addClaim(s, "特徴関連性の改善を中心に、忠実度の小幅改善と計算効率を併せて主張する", 634);
  s.speakerNotes.textFrame.setText("一般化範囲は今回の合成データとTop-1/Top-2。実データとユーザ評価は今後の課題として口頭で補足する。");
}

await fs.mkdir(previewDir, { recursive: true });
await (await PresentationFile.exportPptx(deck)).save(outPath);
for (let i = 17; i < deck.slides.items.length; i++) {
  const png = await deck.export({ slide: deck.slides.items[i], format: "png", scale: 1 });
  await fs.writeFile(path.join(previewDir, `slide-${i + 1}.png`), new Uint8Array(await png.arrayBuffer()));
}
console.log(JSON.stringify({ outPath, slides: deck.slides.items.length, previewDir }));
