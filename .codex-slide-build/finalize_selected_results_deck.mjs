import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";

const root = "/Users/triumph1118/Github/Multi-Class-LIME v2/Multi-Class-LIME-v2";
const skillDir = "/Users/triumph1118/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const python = "/Users/triumph1118/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const candidatePath = path.join(root, ".codex-slide-build/candidate-selected.pptx");
const finalPath = path.join(root, "pptx/MIDTERM_PRESENTATION_RESULTS_REEXPERIMENT_2026-09-10.pptx");
const stagingDir = path.join(root, ".codex-slide-build/finalizer-selected");
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);

await fs.mkdir(stagingDir, { recursive: true });
const result = await finalizePresentation({
  workspaceDir: root,
  candidatePath,
  finalPath,
  explicitTotalSlideCount: 21,
  requiredNativeTableOwnerSlides: [16],
  requiredNativeChartOwnerSlides: [],
  sourceTemplatePath: path.join(root, "pptx/MIDTERM_PRESENTATION_WITH_RESULTS_V2_2026-09-09.pptx"),
  requiredTemplateReferenceSlides: Array.from({ length: 17 }, (_, i) => i + 1),
  minimumTemplateCoverageRatio: 1,
  nativeChartTargetApplication: "powerpoint",
  pythonExecutable: python,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry", "--validate-heading-fit",
    "--require-native-table-slide", "16",
  ],
  fontPolicy: { basis: "design", families: ["Hiragino Sans", "Cambria Math"], scriptFonts: { ea: "Hiragino Sans" } },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "MIDTERM_PRESENTATION_RESULTS_REEXPERIMENT_2026-09-10.pptx.validation.json"),
});
console.log(JSON.stringify({ finalPath, result }, null, 2));
