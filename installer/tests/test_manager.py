"""Tests for the general-manager features (StandRig Setup 2.x):
folder overrides, installed-plugin scan, enable/disable, git update, model
library scan/move/delete, download from any URL, .exe hand-off to a newer script.

    python installer/tests/test_manager.py <workdir>
"""
import http.server, json, os, shutil, subprocess, sys, threading
from pathlib import Path

W = Path(sys.argv[1]).resolve(); shutil.rmtree(W, ignore_errors=True); W.mkdir(parents=True)
os.environ["STANDRIG_APPDATA"] = str(W / "Roaming")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import standrig_setup as S  # noqa: E402

cat = json.loads((Path(__file__).resolve().parents[1] / "catalog.json").read_text())
logs = []
base = W / "ComfyUI"
cn = base / "custom_nodes"
cn.mkdir(parents=True)
git = shutil.which("git")
G = ["-c", "user.email=t@t", "-c", "user.name=t", "-c", "init.defaultBranch=main"]

# --- custom_nodes zoo -----------------------------------------------------------
upstream = W / "upstream"
subprocess.run([git, *G, "init", "-q", str(upstream)], check=True)
(upstream / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n")
(upstream / "requirements.txt").write_text("# nothing\n")
subprocess.run([git, *G, "-C", str(upstream), "add", "."], check=True)
subprocess.run([git, *G, "-C", str(upstream), "commit", "-qm", "v1"], check=True)
subprocess.run([git, *G, "clone", "-q", str(upstream), str(cn / "ComfyUI-GitNode")], check=True)
(cn / "ComfyUI-StandRig-Bridge").mkdir()
(cn / "ComfyUI-StandRig-Bridge" / ".standrig_install.json").write_text(json.dumps({"sha": "abcdef1234567"}))
(cn / "manual-node").mkdir()
(cn / "single_file_node.py").write_text("NODE_CLASS_MAPPINGS = {}\n")
(cn / "old-node.disabled").mkdir()
(cn / ".disabled" / "manager-disabled-node").mkdir(parents=True)
(cn / "__pycache__").mkdir()
(cn / "readme.txt").write_text("not a node")

env = S.detect(S.Settings(base_path=str(base)))
inst = S.Installer(env, cat, S.Settings(base_path=str(base)), log=lambda m: (logs.append(m), print("  |", m)))
rows = {e["folder"]: e for e in inst.scan_plugins()}
print({k: (v.get("installed"), v.get("enabled"), v.get("kind")) for k, v in rows.items()})
assert rows["ComfyUI-GitNode"]["kind"] == "git" and len(rows["ComfyUI-GitNode"]["sha"]) == 40
assert rows["ComfyUI-StandRig-Bridge"]["kind"] == "managed" and rows["ComfyUI-StandRig-Bridge"]["sha"].startswith("abcdef")
assert rows["ComfyUI-StandRig-Bridge"]["id"] == "standrig-bridge", "catalog entry merged by folder name"
assert rows["manual-node"]["kind"] == "manual" and rows["single_file_node.py"]["kind"] == "file"
assert rows["old-node"]["enabled"] is False and rows["manager-disabled-node"]["enabled"] is False
assert "__pycache__" not in rows and "readme.txt" not in rows
assert rows["ComfyUI-See-through"]["installed"] is False, "catalog plugins that are missing are listed"

# enable / disable round trip (ComfyUI '.disabled' suffix convention)
inst.set_enabled(rows["manual-node"], False)
assert (cn / "manual-node.disabled").is_dir() and not (cn / "manual-node").exists()
rows = {e["folder"]: e for e in inst.scan_plugins()}
inst.set_enabled(rows["manual-node"], True)
inst.set_enabled(rows["manager-disabled-node"], True)  # out of Manager's .disabled folder
assert (cn / "manual-node").is_dir() and (cn / "manager-disabled-node").is_dir()

# git: upstream gets a commit -> behind 1 -> pull -> up to date
(upstream / "nodes.py").write_text("X = 2\n")
subprocess.run([git, *G, "-C", str(upstream), "add", "."], check=True)
subprocess.run([git, *G, "-C", str(upstream), "commit", "-qm", "v2"], check=True)
e = {x["folder"]: x for x in inst.scan_plugins()}["ComfyUI-GitNode"]
assert inst.git_behind(e) == 1
inst.update_plugin_entry(e)  # git pull + requirements (empty -> skipped)
assert (cn / "ComfyUI-GitNode" / "nodes.py").exists() and inst.git_behind(e) == 0
inst.remove_entry({x["folder"]: x for x in inst.scan_plugins()}["single_file_node.py"])
assert not (cn / "single_file_node.py").exists() and any((base / "standrig_backups").iterdir())

# --- folder overrides ------------------------------------------------------------
dl = W / "SharedModels"; extra = W / "MoreModels"; wfd = W / "MyWorkflows"
s = S.Settings(base_path=str(base), folders={"models_download": str(dl), "extra_model_dirs": [str(extra)],
                                            "workflows": str(wfd), "custom_nodes": str(cn)})
env2 = S.detect(s)
assert env2.models == dl and extra in env2.model_dirs and env2.custom_nodes == cn
assert S.Installer(env2, cat, s).workflows_dest() == wfd
assert set(env2.overridden) == {"models_download", "workflows", "custom_nodes"}
only_nodes = S.detect(S.Settings(folders={"custom_nodes": str(cn)}, base_path=""))
assert only_nodes.custom_nodes == cn

# --- model library ---------------------------------------------------------------
(base / "models" / "loras").mkdir(parents=True)
(base / "models" / "loras" / "style.safetensors").write_bytes(b"x" * 1000)
(base / "models" / "checkpoints" / "sub").mkdir(parents=True)
(base / "models" / "checkpoints" / "sub" / "model.ckpt").write_bytes(b"x" * 3000)
(base / "models" / "checkpoints" / "notes.txt").write_text("skip")
(extra / "vae").mkdir(parents=True)
(extra / "vae" / "v.safetensors").write_bytes(b"x" * 10)
(dl / "loras").mkdir(parents=True)
(dl / "loras" / "half.safetensors.part").write_bytes(b"x" * 5)
i2 = S.Installer(env2, cat, s, log=lambda m: print("  |", m))
lib = i2.scan_models()
print([(r["type"], r["name"], r["size"]) for r in lib])
names = {r["name"]: r for r in lib}
assert set(names) == {"style.safetensors", "model.ckpt", "v.safetensors", "half.safetensors.part"}
assert names["model.ckpt"]["type"] == "checkpoints" and names["half.safetensors.part"]["partial"]
assert sum(r["size"] for r in lib) == 4015
moved = i2.move_model(names["model.ckpt"]["path"], dl)
assert moved == dl / "checkpoints" / "sub" / "model.ckpt" and moved.exists()
i2.delete_model(names["v.safetensors"]["path"])
assert not (extra / "vae" / "v.safetensors").exists()

# --- download from any URL (HF /blob/ link, Content-Disposition name) ---------------
BODY = os.urandom(300_000)


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if "/resolve/" in self.path or self.path.startswith("/dl"):
            rng = self.headers.get("Range")
            hdr = {"Content-Disposition": 'attachment; filename="named_by_server.safetensors"'}
            if rng:
                a, b = rng.split("=")[1].split("-"); a = int(a); b = int(b) if b else len(BODY) - 1
                body, code = BODY[a:b + 1], 206
                hdr["Content-Range"] = f"bytes {a}-{b}/{len(BODY)}"
            else:
                body, code = BODY, 200
            self.send_response(code)
            for k, v in hdr.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
assert S.normalize_model_url("https://huggingface.co/a/b/blob/main/x/y.safetensors") == \
    "https://huggingface.co/a/b/resolve/main/x/y.safetensors"
p1 = i2.download_url(f"http://127.0.0.1:{port}/org/repo/resolve/main/lora_a.safetensors", "loras", dl)
assert p1 == dl / "loras" / "lora_a.safetensors" and p1.read_bytes() == BODY
p2 = i2.download_url(f"http://127.0.0.1:{port}/dl?id=42", "upscale_models", dl)
assert p2.name == "named_by_server.safetensors" and p2.parent.name == "upscale_models"

# --- .exe hand-off to a newer downloaded script --------------------------------------
app = S.tool_dir() / "app"; app.mkdir(parents=True, exist_ok=True)
marker = W / "ran_newer.txt"
(app / "standrig_setup.py").write_text(f'VERSION = "99.0.0"\nopen(r"{marker}", "w").write("ok")\n')
sys.frozen = True
try:
    assert S._run_newer_script_if_any() is True and marker.read_text() == "ok"
    os.environ.pop("STANDRIG_NO_REDIRECT", None)
    (app / "standrig_setup.py").write_text('VERSION = "0.0.1"\n')
    assert S._run_newer_script_if_any() is False, "older downloaded copies are ignored"
finally:
    del sys.frozen
print("ALL MANAGER TESTS PASSED")
