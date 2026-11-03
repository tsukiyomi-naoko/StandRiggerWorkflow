"""Headless tests for standrig_setup.Installer against a fake ComfyUI Desktop and a
mock GitHub / Hugging Face server (HTTP Range supported).

    python installer/tests/test_installer.py <workdir> <comfyui_core_dir> <uv_binary> repo1_dir=owner/name ...
"""
import hashlib, http.server, io, json, os, shutil, subprocess, sys, threading, time, zipfile
from pathlib import Path

WORK = Path(sys.argv[1]).resolve(); CORE = Path(sys.argv[2]).resolve(); UV = Path(sys.argv[3]).resolve()
REPOS = dict(a.split("=", 1)[::-1] for a in sys.argv[4:])  # owner/name -> local dir
shutil.rmtree(WORK, ignore_errors=True); WORK.mkdir(parents=True)
MODEL = os.urandom(5 * 1024 * 1024 + 123)
MODEL_SHA = hashlib.sha256(MODEL).hexdigest()


def repo_zip(path: Path, sha: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        root = f"{path.name}-{sha}/"
        for f in sorted(path.rglob("*")):
            if ".git" in f.parts or f.is_dir():
                continue
            z.write(f, root + str(f.relative_to(path)))
    return buf.getvalue()


SHAS = {r: hashlib.sha1(r.encode()).hexdigest() for r in REPOS}
ZIPS = {r: repo_zip(Path(d), SHAS[r]) for r, d in REPOS.items()}
STATS = {"model_requests": 0, "ranges": []}


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body: bytes, headers=None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path
        if p.startswith("/api/repos/"):
            repo = "/".join(p.split("/")[3:5])
            return self.send(200, SHAS[repo].encode()) if repo in SHAS else self.send(404, b"no")
        if p.startswith("/codeload/"):
            parts = p.split("/")
            repo = f"{parts[2]}/{parts[3]}"
            return self.send(200, ZIPS[repo], {"Content-Type": "application/zip"})
        if p.startswith("/redirect/"):  # HF-style redirect to a CDN path, must not forward auth
            self.send_response(302); self.send_header("Location", "http://localhost:%d/cdn/%s" % (PORT, p.split("/")[-1]))
            self.send_header("Content-Length", "0"); self.end_headers(); return
        if p.startswith("/cdn/"):
            STATS["model_requests"] += 1
            STATS["auth_on_cdn"] = self.headers.get("Authorization")
            rng = self.headers.get("Range")
            STATS["ranges"].append(rng)
            if rng:
                a, b = rng.split("=")[1].split("-")
                a = int(a); b = int(b) if b else len(MODEL) - 1
                body = MODEL[a:b + 1]
                return self.send(206, body, {"Content-Range": f"bytes {a}-{b}/{len(MODEL)}"})
            return self.send(200, MODEL)
        self.send(404, b"?")


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
PORT = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["STANDRIG_GITHUB_API"] = f"http://127.0.0.1:{PORT}/api"
os.environ["STANDRIG_CODELOAD"] = f"http://127.0.0.1:{PORT}/codeload"
os.environ["STANDRIG_APPDATA"] = str(WORK / "AppData" / "Roaming")
os.environ["STANDRIG_LOCALAPPDATA"] = str(WORK / "AppData" / "Local")

# --- fake ComfyUI Desktop -------------------------------------------------------
base = WORK / "Documents" / "ComfyUI"
(base / "custom_nodes").mkdir(parents=True)
(base / "models").mkdir()
subprocess.run([str(UV), "venv", "-q", str(base / ".venv"), "--python", sys.executable], check=True)
cfg = Path(os.environ["STANDRIG_APPDATA"]) / "ComfyUI"
cfg.mkdir(parents=True)
(cfg / "config.json").write_text(json.dumps({"basePath": str(base), "installState": "installed"}))
app = Path(os.environ["STANDRIG_LOCALAPPDATA"]) / "Programs" / "ComfyUI" / "Comfy Desktop"
res = app / "resources"
(res / "uv" / "linux").mkdir(parents=True)
shutil.copy(UV, res / "uv" / "linux" / "uv")
os.symlink(CORE, res / "ComfyUI")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import standrig_setup as S  # noqa: E402

cat = json.loads((Path(__file__).resolve().parents[1] / "catalog.json").read_text())
for m in cat["models"]:  # point every model at the mock HF (via a redirect)
    m["url"] = f"http://127.0.0.1:{PORT}/redirect/{m['file']}"
cat["models"] = [m for m in cat["models"] if m["id"] in ("qwen-edit-int8", "qwen-edit-bf16", "qwen-vae")]

settings = S.Settings.load()
settings["hf_token"] = "hf_secret"
logs = []
env = S.detect(settings)
print("detected:", env.kind, env.base, env.python, env.uv, env.core_version)
assert env.kind == "desktop" and env.base == base and env.python and env.uv and env.core
assert S.core_missing(env, cat) == [], S.core_missing(env, cat)

inst = S.Installer(env, cat, settings, log=lambda m: (logs.append(m), print("  |", m)))
# patch the hf host check so the mock counts as huggingface for auth
S.auth_for = lambda url, s: {"Authorization": "Bearer hf_secret"} if "/redirect/" in url else {}

# plugins
bridge = next(p for p in cat["plugins"] if p["id"] == "standrig-bridge")
see = next(p for p in cat["plugins"] if p["id"] == "see-through")
inst.install_plugin(bridge)
d = base / "custom_nodes" / "ComfyUI-StandRig-Bridge"
assert (d / "standrig_nodes.py").exists() and (d / ".standrig_install.json").exists()
assert inst.plugin_status(bridge, check_remote=True)["state"].startswith("Up to date")
inst.install_plugin(see)  # real requirements install via bundled uv into the fake venv
r = subprocess.run([str(env.python), "-c", "import diffusers, sklearn, cv2; print('deps ok')"], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr[-300:])
assert r.returncode == 0
# update path: change upstream sha -> update available -> reinstall moves old copy to backups
SHAS["jtydhr88/ComfyUI-See-through"] = "f" * 40
ZIPS["jtydhr88/ComfyUI-See-through"] = repo_zip(Path(REPOS["jtydhr88/ComfyUI-See-through"]), "f" * 40)
inst._sha_cache.clear()
assert inst.plugin_status(see, check_remote=True)["state"].startswith("Update available")
inst.install_plugin(see, install_requirements=False)
assert any((base / "standrig_backups").iterdir())
inst.remove_plugin(see)
assert not (base / "custom_nodes" / "ComfyUI-See-through").exists()
# custom plugin from URL
cp = S.parse_github_url("https://github.com/ukr8b3g-cmyk/H3-Character-Sheet-Designer")
assert cp["repo"] == "ukr8b3g-cmyk/H3-Character-Sheet-Designer"
inst.install_plugin(cp, install_requirements=False)
assert (base / "custom_nodes" / "H3-Character-Sheet-Designer" / "__init__.py").exists()

# workflows with int8 patch
settings["qwen_variant"] = "int8"
n = inst.install_workflows()
wfdir = base / "user/default/workflows/StandRig"
full = (wfdir / "standrig_full_pipeline.json").read_text()
def unet_of(text):
    d = json.loads(text)
    u = next(nd for nd in d["nodes"] if nd["type"] == "UNETLoader")
    return u["widgets_values"][0], u["properties"]["models"][0]["url"]
assert n == 5 and unet_of(full)[0] == "qwen_image_edit_2511_int8_convrot.safetensors"
assert unet_of(full)[1].endswith("qwen_image_edit_2511_int8_convrot.safetensors")
settings["qwen_variant"] = "bf16"   # switching variants rewrites model name AND download link
inst._sha_cache.clear()
inst.install_workflows()
name, url = unet_of((wfdir / "standrig_full_pipeline.json").read_text())
assert name == "qwen_image_edit_2511_bf16.safetensors" and url.endswith("/qwen_image_edit_2511_bf16.safetensors"), (name, url)
settings["qwen_variant"] = "int8"
inst.install_workflows()
assert not (wfdir / "api").exists()
print("workflows ok", sorted(p.name for p in wfdir.iterdir()))

# models: int8 variant only, cancel mid-way then resume
names = [m["file"] for m in inst.model_list()]
assert "qwen_image_edit_2511_int8_convrot.safetensors" in names and "qwen_image_edit_2511_bf16.safetensors" not in names
m = inst.model_list()[0]
orig_read_chunk = 1 << 20
calls = {"n": 0}

def prog(done, total):
    calls["n"] += 1
    if done > 2 * 1024 * 1024:
        inst.cancel.set()

try:
    S.download_file(m["url"], inst.model_path(m), settings, prog, inst.cancel, chunk=256 * 1024)
    raise SystemExit("expected cancel")
except S.Cancelled:
    pass
part = inst.model_path(m).with_name(m["file"] + ".part")
assert part.exists() and 0 < part.stat().st_size < len(MODEL)
assert inst.model_state(m).startswith("Partial")
inst.cancel.clear()
inst.download_model(m)
got = inst.model_path(m).read_bytes()
assert hashlib.sha256(got).hexdigest() == MODEL_SHA, "resumed file differs"
assert any(r and r.startswith("bytes=") and r != "bytes=0-" for r in STATS["ranges"]), STATS["ranges"]
assert STATS.get("auth_on_cdn") is None, "Authorization leaked to redirect target"
assert S.remote_size(m["url"], settings) == len(MODEL)
print("models ok: resume ranges", STATS["ranges"])
print("ALL INSTALLER TESTS PASSED")
