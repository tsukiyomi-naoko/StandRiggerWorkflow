"""Detection tests for the new Comfy Desktop (multi-install, shared model folders),
mirroring a real layout:
  %LOCALAPPDATA%/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI   (instance)
  %LOCALAPPDATA%/Comfy-Desktop/ComfyUI-Shared/models              (DOWNLOADS)
  ~/Documents/ComfyUI                                             (adopted legacy data)
    python installer/tests/test_desktop2.py <workdir>
"""
import json, os, shutil, sys
from pathlib import Path

W = Path(sys.argv[1]).resolve(); shutil.rmtree(W, ignore_errors=True)
os.environ["STANDRIG_APPDATA"] = str(W / "Roaming")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import standrig_setup as S  # noqa: E402

IS_WIN = S.IS_WIN
exe = "Scripts/python.exe" if IS_WIN else "bin/python3"
la = W / "Local" / "Comfy-Desktop"
inst = la / "ComfyUI-Installs" / "ComfyUI"
inst2 = la / "ComfyUI-Installs" / "ComfyUI-Test"
shared = la / "ComfyUI-Shared" / "models"
docs = W / "Documents" / "ComfyUI"
for d in [inst / "ComfyUI" / "comfy_extras", inst / "ComfyUI" / "custom_nodes", inst / "ComfyUI" / "models",
          inst / "ComfyUI" / ".venv" / Path(exe).parent, inst / "standalone-env",
          inst2 / "ComfyUI" / "comfy_extras", inst2 / "ComfyUI" / ".venv" / Path(exe).parent,
          shared, docs / "custom_nodes", docs / "models" / "vae", docs / "user", docs / ".venv" / Path(exe).parent]:
    d.mkdir(parents=True, exist_ok=True)
for py in [inst / "ComfyUI" / ".venv" / exe, inst2 / "ComfyUI" / ".venv" / exe, docs / ".venv" / exe]:
    py.write_text("")
(inst / "standalone-env" / ("uv.exe" if IS_WIN else "bin")).parent.mkdir(parents=True, exist_ok=True)
uvp = inst / "standalone-env" / ("uv.exe" if IS_WIN else "bin/uv"); uvp.parent.mkdir(parents=True, exist_ok=True); uvp.write_text("")
(inst / "ComfyUI" / "comfyui_version.py").write_text('__version__ = "0.43.1"\n')
(docs / "models" / "vae" / "qwen_image_vae.safetensors").write_text("x")  # already present in a non-download dir

cfg = W / "Roaming" / "Comfy Desktop"; cfg.mkdir(parents=True)
(cfg / "settings.json").write_text(json.dumps({"modelsDirs": [str(shared), str(docs / "models")],
                                                "inputDir": str(docs / "input"), "outputDir": str(docs / "output")}))
adopted = {"id": "inst-1", "name": "ComfyUI", "installPath": str(inst), "sourceId": "standalone", "status": "installed",
           "adopted": True, "adoptedBaseDir": str(docs), "adoptedPythonPath": str(docs / ".venv" / exe),
           "lastLaunchedAt": 200, "useSharedModels": True}
fresh = {"id": "inst-2", "name": "ComfyUI-Test", "installPath": str(inst2), "sourceId": "standalone",
         "status": "installed", "lastLaunchedAt": 100, "useSharedModels": True,
         "modelDirsPrimary": str(inst2 / "ComfyUI" / "models")}
(cfg / "installations.json").write_text(json.dumps([adopted, fresh]))

cat = json.loads((Path(__file__).resolve().parents[1] / "catalog.json").read_text())

# Case 1: adopted install, most recently launched -> chosen by default
env = S.detect(S.Settings())
print("case1:", env.kind, env.instance_name, env.custom_nodes, env.models, env.python, env.uv, env.core_version)
assert env.kind == "desktop2" and env.instance_id == "inst-1"
assert env.custom_nodes == docs / "custom_nodes", "adopted installs keep custom_nodes in the legacy folder"
assert env.base / "user" == docs / "user"
assert env.models == shared, "downloads go to the first shared models dir"
assert env.python == docs / ".venv" / exe
assert env.core == inst / "ComfyUI" and env.core_version == "0.43.1"
assert [i["id"] for i in env.instances] == ["inst-1", "inst-2"]
i = S.Installer(env, cat, S.Settings())
vae = next(m for m in cat["models"] if m["id"] == "qwen-vae")
te = next(m for m in cat["models"] if m["id"] == "qwen-te")
assert i.model_state(vae).startswith("Installed ("), i.model_state(vae)   # found in Documents/ComfyUI/models
assert i.model_state(te) == "Missing"
assert i.model_path(te) == shared / "text_encoders" / te["file"]
i.download_model(vae)  # must not re-download
assert not (shared / "vae" / vae["file"]).exists()
assert i.pip_command(["install", "x"])[1:3] == ["pip", "install"]  # uses bundled uv

# Case 2: pick the fresh install explicitly; its primary dir overrides downloads
s = S.Settings(instance_id="inst-2")
env2 = S.detect(s)
print("case2:", env2.instance_name, env2.custom_nodes, env2.models)
assert env2.custom_nodes == inst2 / "ComfyUI" / "custom_nodes"
assert env2.models == inst2 / "ComfyUI" / "models"
assert env2.python == inst2 / "ComfyUI" / ".venv" / exe
assert shared in env2.all_model_dirs()

# Case 3: manual Browse override falls back to legacy/manual detection
env3 = S.detect(S.Settings(base_path=str(docs)))
assert env3.kind != "desktop2" and env3.base == docs and env3.models == docs / "models"
# Case 4: unusual status + config folder with another name -> still found, with a report
odd = W / "Roaming" / "SomeDesktopName"; odd.mkdir()
shutil.move(str(cfg / "installations.json"), str(odd / "installations.json"))
shutil.move(str(cfg / "settings.json"), str(odd / "settings.json"))
recs = json.loads((odd / "installations.json").read_text())
recs[0]["status"] = "ready"; recs[1]["status"] = "installing"
(odd / "installations.json").write_text(json.dumps(recs))
env4 = S.detect(S.Settings())
print("case4 notes:", env4.notes)
assert env4.kind == "desktop2" and env4.instance_id == "inst-1"
assert any("skipped install 'ComfyUI-Test'" in n for n in env4.notes)

# Case 5: hand-picked folders for a D:-style install -> its core version and its standalone uv
dinst = W / "D" / "Naoko" / "ComfyUI"
(dinst / "ComfyUI" / "comfy_extras").mkdir(parents=True)
(dinst / "ComfyUI" / "custom_nodes").mkdir()
(dinst / "ComfyUI" / "comfyui_version.py").write_text('__version__ = "0.44.0"\n')
(dinst / "ComfyUI" / ".venv" / Path(exe).parent).mkdir(parents=True)
(dinst / "ComfyUI" / ".venv" / exe).write_text("")
duv = dinst / "standalone-env" / ("uv.exe" if IS_WIN else "bin/uv"); duv.parent.mkdir(parents=True); duv.write_text("")
s5 = S.Settings(base_path=str(docs), folders={"custom_nodes": str(dinst / "ComfyUI" / "custom_nodes"),
                                              "python": str(dinst / "ComfyUI" / ".venv" / exe)})
env5 = S.detect(s5)
print("case5:", env5.core, env5.core_version, env5.uv)
assert env5.core == dinst / "ComfyUI" and env5.core_version == "0.44.0" and env5.uv == duv
assert any("Browse" in n for n in env5.notes)
print("ALL DESKTOP2 TESTS PASSED")
