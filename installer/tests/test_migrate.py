"""Tests for migrate_data: merge into an existing target copy, live copy wins,
identical duplicates removed from the source, folders replaced by links that
ComfyUI-style paths still resolve through, re-run is a no-op.

    python installer/tests/test_migrate.py <workdir>
"""
import os, shutil, sys, threading
from pathlib import Path

W = Path(sys.argv[1]).resolve(); shutil.rmtree(W, ignore_errors=True); W.mkdir(parents=True)
os.environ["STANDRIG_APPDATA"] = str(W / "Roaming")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import standrig_setup as S  # noqa: E402

C = W / "C" / "Documents" / "ComfyUI"      # live data (what ComfyUI uses)
D = W / "D" / "Naoko" / "ComfyUI" / "ComfyUI"  # an older copy + new installs
BIG = os.urandom(3 << 20)


def w(p: Path, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


# live side
w(C / "models" / "checkpoints" / "nova.safetensors", BIG)                 # identical on D -> dup
w(C / "models" / "loras" / "only_on_c.safetensors", b"c" * 5000)          # moves
w(C / "models" / "vae" / "changed.safetensors", b"new")                   # differs from D -> C wins
w(C / "custom_nodes" / "comfyui-impact-pack" / "__init__.py", b"live")    # on both -> C wins
w(C / "custom_nodes" / "only-c-node" / "__init__.py")
w(C / "user" / "default" / "comfy.settings.json", b"{live}")
w(C / "user" / "comfyui.db", b"db")
w(C / "input" / "pic.png", b"png")
(C / "output").mkdir(parents=True)
w(C / ".venv" / "pyvenv.cfg", b"home = x")                                 # must NOT move
# target side
w(D / "models" / "checkpoints" / "nova.safetensors", BIG)
w(D / "models" / "vae" / "changed.safetensors", b"old")
w(D / "models" / "diffusion_models" / "qwen_image_edit_2511_int8_convrot.safetensors", b"q" * 7000)  # new download
w(D / "custom_nodes" / "comfyui-impact-pack" / "__init__.py", b"stale")
w(D / "custom_nodes" / "ComfyUI-StandRig-Bridge" / "__init__.py")
w(D / "user" / "default" / "workflows" / "StandRig" / "standrig_full_pipeline.json", b"{}")
w(D / "user" / "default" / "comfy.settings.json", b"{old}")

logs = []
cancel = threading.Event()
prev = S.migrate_data(C, D, S.MIGRATE_FOLDERS, logs.append, lambda *a: None, cancel, dry_run=True)
print("preview:", {k: v for k, v in prev.items() if v})
assert prev["dup_files"] == 1 and prev["dup_bytes"] == len(BIG)
assert prev["conflicts"] == 2  # vae/changed + user settings
assert prev["plugins_replaced"] == ["comfyui-impact-pack"] and prev["plugins_moved"] == ["only-c-node"]
assert (C / "models" / "loras" / "only_on_c.safetensors").exists(), "preview must not touch anything"

r = S.migrate_data(C, D, S.MIGRATE_FOLDERS, logs.append, lambda *a: None, cancel, dry_run=False)
print("move:", {k: v for k, v in r.items() if v})
assert r["errors"] == [] and set(r["linked"]) == set(S.MIGRATE_FOLDERS)
for f in S.MIGRATE_FOLDERS:
    assert S.is_link(C / f), f
# ComfyUI keeps using the same paths, now resolved on D
assert (C / "models" / "loras" / "only_on_c.safetensors").read_bytes() == b"c" * 5000
assert (D / "models" / "loras" / "only_on_c.safetensors").exists()
assert (C / "models" / "diffusion_models" / "qwen_image_edit_2511_int8_convrot.safetensors").exists()
assert (C / "models" / "vae" / "changed.safetensors").read_bytes() == b"new", "live copy wins"
assert (D / "_replaced_from_D" / "models" / "vae" / "changed.safetensors").read_bytes() == b"old"
assert (C / "custom_nodes" / "comfyui-impact-pack" / "__init__.py").read_bytes() == b"live"
assert (D / "_replaced_from_D" / "custom_nodes" / "comfyui-impact-pack" / "__init__.py").read_bytes() == b"stale"
assert (C / "custom_nodes" / "ComfyUI-StandRig-Bridge").exists() and (C / "custom_nodes" / "only-c-node").exists()
assert (C / "user" / "default" / "workflows" / "StandRig" / "standrig_full_pipeline.json").exists()
assert (C / "user" / "default" / "comfy.settings.json").read_bytes() == b"{live}"
assert (C / "user" / "comfyui.db").exists() and (C / "input" / "pic.png").exists() and (C / "output").is_dir()
assert (C / ".venv" / "pyvenv.cfg").exists() and not S.is_link(C / ".venv"), ".venv must stay"
assert (D / "models" / "checkpoints" / "nova.safetensors").read_bytes() == BIG

# re-run: nothing left to do
r2 = S.migrate_data(C, D, S.MIGRATE_FOLDERS, logs.append, lambda *a: None, cancel, dry_run=False)
assert set(r2["already_linked"]) == set(S.MIGRATE_FOLDERS) and r2["move_files"] == 0

# cross-device style move path (copy fallback) keeps content
a = W / "a.bin"; a.write_bytes(BIG); b = W / "sub" / "b.bin"
real_replace = os.replace
def fail_once(x, y, _n=[0]):
    if _n[0] == 0 and str(x) == str(a):
        _n[0] += 1
        raise OSError(18, "cross-device")
    return real_replace(x, y)
S.os.replace = fail_once
try:
    S.move_path(a, b, lambda *x: None, cancel)
finally:
    S.os.replace = real_replace
assert b.read_bytes() == BIG and not a.exists()
print("ALL MIGRATION TESTS PASSED")
