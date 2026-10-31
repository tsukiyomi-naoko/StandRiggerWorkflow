#!/usr/bin/env bash
# Full CPU test suite. Needs: a ComfyUI checkout + venv with torch, psd-tools; node; playwright (for UI load).
#   COMFY=/path/to/ComfyUI AGPSD_DIR=/dir/with/node_modules/ag-psd@31.0.2 bash tests/run_all.sh
# ComfyUI must be running on :8188 with these custom_nodes linked:
#   custom_nodes/ComfyUI-StandRig-Bridge, tests/stubs/seethrough_stub, tests/stubs/standrig_test_nodes
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
python -I "$HERE/test_psd_writer.py"
python "$HERE/make_test_character.py" "$COMFY/input/standrig_test_character.png"
cp "$COMFY/input/standrig_test_character.png" "$COMFY/input/character.png"
python "$ROOT/tools/build_workflows.py" --e2e "$HERE/e2e.api.json"
python "$HERE/run_e2e.py" "$HERE/e2e.api.json"                       # full pipeline with stubs: must succeed
(cd "$AGPSD_DIR" && node "$HERE/read_psd_agpsd.mjs" "$COMFY/output/standrig/e2e_test/character.psd" > /dev/null)
for f in "$ROOT"/workflows/api/*.api.json; do                       # real graphs: must pass validation
  out="$(python "$HERE/run_e2e.py" "$f" || true)"; echo "$out" | grep -q "VALIDATION FAILED" && { echo "$out"; exit 1; }
done
NODE_PATH="$(npm root -g)" node "$HERE/check_ui_load.js" http://127.0.0.1:8188 /tmp "$ROOT"/workflows/*.json
echo "ALL TESTS PASSED"
