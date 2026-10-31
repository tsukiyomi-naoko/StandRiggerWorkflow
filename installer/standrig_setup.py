#!/usr/bin/env python3
"""StandRig Setup: installer / updater for the StandRig ComfyUI workflows.

* Finds ComfyUI Desktop (or a portable / manual ComfyUI) automatically.
* Installs and updates custom-node plugins from GitHub (no git needed) and
  installs their Python requirements into ComfyUI's own environment.
* Downloads models with progress, resume and an optional Hugging Face token.
* Copies the StandRig workflows into ComfyUI's Workflows sidebar.
* Reads an online catalog, so new plugins/models/versions appear without
  reinstalling this tool; can update itself.

Standard library only (tkinter GUI). Run with ComfyUI Desktop's Python:
    <ComfyUI folder>\\.venv\\Scripts\\pythonw.exe standrig_setup.py
or just double-click StandRigSetup.bat.
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import zipfile
import runpy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

APP_NAME = "StandRig Setup"
VERSION = "2.0.0"
CATALOG_URL = "https://raw.githubusercontent.com/tsukiyomi-naoko/StandRiggerWorkflow/main/installer/catalog.json"
# Overridable for testing / mirrors
GITHUB_API = os.environ.get("STANDRIG_GITHUB_API", "https://api.github.com")
CODELOAD = os.environ.get("STANDRIG_CODELOAD", "https://codeload.github.com")
IS_WIN = os.name == "nt"
MARKER = ".standrig_install.json"
USER_AGENT = f"StandRigSetup/{VERSION}"
NO_WINDOW = 0x08000000 if IS_WIN else 0


class Cancelled(Exception):
    pass


# =============================================================== settings / paths

def app_data_dir() -> Path:
    if os.environ.get("STANDRIG_APPDATA"):
        return Path(os.environ["STANDRIG_APPDATA"])
    if IS_WIN and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"])
    return Path.home() / ".config"


def local_app_data_dir() -> Path:
    if os.environ.get("STANDRIG_LOCALAPPDATA"):
        return Path(os.environ["STANDRIG_LOCALAPPDATA"])
    if IS_WIN and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"])
    return Path.home() / ".local" / "share"


def tool_dir() -> Path:
    d = app_data_dir() / "StandRigSetup"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Settings(dict):
    PATH_NAME = "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        p = tool_dir() / cls.PATH_NAME
        try:
            return cls(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return cls()

    def save(self) -> None:
        p = tool_dir() / self.PATH_NAME
        p.write_text(json.dumps(self, indent=2, ensure_ascii=False), encoding="utf-8")


# =================================================================== detection

@dataclass
class ComfyEnv:
    base: Optional[Path] = None          # data folder: models/, custom_nodes/, user/
    kind: str = "unknown"                # desktop | portable | manual | unknown
    python: Optional[Path] = None        # interpreter ComfyUI runs with
    uv: Optional[Path] = None            # uv.exe bundled with Desktop
    core: Optional[Path] = None          # ComfyUI core source (has comfy_extras/)
    core_version: str = "?"
    desktop_app: Optional[Path] = None
    notes: List[str] = field(default_factory=list)
    model_download_dir: Optional[Path] = None   # where new models are saved
    model_dirs: List[Path] = field(default_factory=list)  # every models root ComfyUI searches
    instance_id: Optional[str] = None
    instance_name: Optional[str] = None
    instances: List[dict] = field(default_factory=list)  # Comfy Desktop 2: [{id, name, installPath}]
    custom_nodes_dir: Optional[Path] = None     # user override
    workflows_dir: Optional[Path] = None        # user override
    overridden: List[str] = field(default_factory=list)

    @property
    def custom_nodes(self) -> Path:
        return self.custom_nodes_dir or (self.base / "custom_nodes")

    @property
    def models(self) -> Path:
        return self.model_download_dir or (self.base / "models")

    def all_model_dirs(self) -> List[Path]:
        seen, out = set(), []
        for d in [self.models, *self.model_dirs, self.base / "models" if self.base else None]:
            if d is not None and str(d).lower() not in seen:
                seen.add(str(d).lower())
                out.append(d)
        return out


def _first_existing(paths) -> Optional[Path]:
    for p in paths:
        if p and Path(p).exists():
            return Path(p)
    return None


def desktop_app_candidates(extra: Optional[str] = None) -> List[Path]:
    la = local_app_data_dir() / "Programs"
    c = [Path(extra)] if extra else []
    c += [la / "ComfyUI" / "Comfy Desktop", la / "@comfyorgcomfyui-electron", la / "ComfyUI", la / "comfyui-electron",
          Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "ComfyUI"]
    return c


def find_resources(app_dir: Path) -> Optional[Path]:
    """Locate Electron 'resources' (contains uv/ and ComfyUI/) under an app folder."""
    if not app_dir.exists():
        return None
    for cand in (app_dir / "resources", app_dir):
        if (cand / "ComfyUI").is_dir() or (cand / "uv").is_dir():
            return cand
    for root, dirs, _ in os.walk(app_dir):
        if Path(root).name == "resources" and ("uv" in dirs or "ComfyUI" in dirs):
            return Path(root)
        if root.count(os.sep) - str(app_dir).count(os.sep) > 3:
            dirs[:] = []
    return None


def read_core_version(core: Path) -> str:
    try:
        txt = (core / "comfyui_version.py").read_text(encoding="utf-8")
        m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", txt)
        return m.group(1) if m else "?"
    except Exception:
        return "?"


DESKTOP2_DIR_NAMES = ("Comfy Desktop", "comfyui-desktop-2")


def desktop2_config_dir() -> Optional[Path]:
    """Folder holding Comfy Desktop's settings.json / installations.json."""
    for name in DESKTOP2_DIR_NAMES:
        d = app_data_dir() / name
        if (d / "installations.json").exists():
            return d
    return None


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default


def _find_comfy_dir(install_path: Path) -> Path:
    """Port of comfy-desktop findComfyUIDir()."""
    direct = install_path / "ComfyUI"
    if direct.exists():
        return direct
    if all((install_path / x).exists() for x in ("models", "user", "custom_nodes")):
        return install_path
    try:
        for sub in install_path.iterdir():
            if sub.is_dir() and (sub / "python_embeded").exists():
                return sub / "ComfyUI" if (sub / "ComfyUI").exists() else sub
    except Exception:
        pass
    return direct


def detect_desktop2(settings: Settings) -> Optional[ComfyEnv]:
    """Comfy Desktop (multi-install app: ComfyUI-Installs / ComfyUI-Shared)."""
    cfg = desktop2_config_dir()
    if cfg is None:
        return None
    records = _read_json(cfg / "installations.json", [])
    dsettings = _read_json(cfg / "settings.json", {})
    if not isinstance(records, list):
        return None
    usable = [r for r in records if isinstance(r, dict) and r.get("installPath")
              and r.get("status") in (None, "installed", "updating")
              and Path(r["installPath"]).exists()]
    if not usable:
        return None
    env = ComfyEnv(kind="desktop2")
    env.instances = [{"id": r.get("id"), "name": r.get("name") or Path(r["installPath"]).name,
                      "installPath": r["installPath"]} for r in usable]
    want = settings.get("instance_id")
    rec = next((r for r in usable if r.get("id") == want), None) or \
        max(usable, key=lambda r: r.get("lastLaunchedAt") or 0)
    env.instance_id, env.instance_name = rec.get("id"), rec.get("name")
    ip = Path(rec["installPath"])
    exe = "Scripts/python.exe" if IS_WIN else "bin/python3"
    if rec.get("adopted") is True and rec.get("adoptedBaseDir"):
        base = Path(rec["adoptedBaseDir"])
        env.base = base
        env.python = Path(rec["adoptedPythonPath"]) if rec.get("adoptedPythonPath") else None
        uv = base / ".venv" / ("Scripts/uv.exe" if IS_WIN else "bin/uv")
        env.notes.append("adopted install (data in the legacy ComfyUI folder)")
    else:
        base = _find_comfy_dir(ip)
        env.base = base
        env.python = _first_existing([ip / "ComfyUI" / ".venv" / exe, ip / "envs" / "default" / exe])
        uv = ip / "standalone-env" / ("uv.exe" if IS_WIN else "bin/uv")
    if env.python is not None and not env.python.exists():
        env.python = None
    env.uv = uv if uv.exists() else _first_existing([ip / "standalone-env" / ("uv.exe" if IS_WIN else "bin/uv")])
    core = ip / "ComfyUI"
    env.core = core if (core / "comfy_extras").is_dir() else (base if (base / "comfy_extras").is_dir() else None)
    env.core_version = read_core_version(env.core) if env.core else "?"
    shared = [Path(d) for d in (dsettings.get("modelsDirs") or []) if d] if rec.get("useSharedModels", True) else []
    own = base / "models"
    env.model_dirs = shared + [Path(d) for d in (rec.get("modelDirs") or []) if d] + [own]
    primary = rec.get("modelDirsPrimary")
    env.model_download_dir = Path(primary) if primary else (shared[0] if shared else own)
    env.notes.append(f"Comfy Desktop config: {cfg}")
    return env


FOLDER_KEYS = {
    "custom_nodes": "Custom nodes folder",
    "models_download": "Models download folder",
    "workflows": "Workflows folder",
    "python": "ComfyUI Python (python.exe)",
    "uv": "uv (uv.exe)",
}


def detect(settings: Settings) -> ComfyEnv:
    """Auto-detect, then apply the folder overrides from Settings → Folders."""
    env = _detect_auto(settings)
    ov = settings.get("folders") or {}
    for key, attr in (("custom_nodes", "custom_nodes_dir"), ("models_download", "model_download_dir"),
                      ("workflows", "workflows_dir"), ("python", "python"), ("uv", "uv")):
        if ov.get(key):
            setattr(env, attr, Path(ov[key]))
            env.overridden.append(key)
    for d in ov.get("extra_model_dirs") or []:
        if d and Path(d) not in env.model_dirs:
            env.model_dirs.append(Path(d))
    if env.base is None and env.custom_nodes_dir is not None:
        env.base = env.custom_nodes_dir.parent
        env.kind = "custom"
    return env


def _detect_auto(settings: Settings) -> ComfyEnv:
    if not settings.get("base_path"):
        env2 = detect_desktop2(settings)
        if env2 is not None:
            return env2
    env = ComfyEnv()
    cfg = app_data_dir() / "ComfyUI" / "config.json"
    base = settings.get("base_path")
    if not base and cfg.exists():
        try:
            base = json.loads(cfg.read_text(encoding="utf-8")).get("basePath")
            env.notes.append(f"ComfyUI Desktop config: {cfg}")
        except Exception as e:
            env.notes.append(f"could not read {cfg}: {e}")
    if not base:
        guess = _first_existing([Path.home() / "Documents" / "ComfyUI", Path.home() / "ComfyUI"])
        base = str(guess) if guess else None
    if not base:
        env.notes.append("ComfyUI folder not found. Choose it with Browse…")
        return env
    env.base = Path(base)

    # Desktop app (bundled core + uv)
    res = None
    for c in desktop_app_candidates(settings.get("desktop_app")):
        res = find_resources(c)
        if res:
            env.desktop_app = c
            break
    venv_py = env.base / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")
    if (env.base / "main.py").exists() and (env.base / "comfy_extras").is_dir():
        env.kind, env.core = "manual", env.base
        emb = env.base.parent / "python_embeded" / "python.exe"
        if emb.exists():
            env.kind, env.python = "portable", emb
    elif res is not None or venv_py.exists():
        env.kind = "desktop"
        if res is not None:
            if (res / "ComfyUI" / "comfy_extras").is_dir():
                env.core = res / "ComfyUI"
            uvp = res / "uv" / ("win/uv.exe" if IS_WIN else "linux/uv")
            env.uv = uvp if uvp.exists() else None
    if env.python is None:
        for cand in (venv_py, env.base / "venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")):
            if cand.exists():
                env.python = cand
                break
    if env.uv is None:
        found = shutil.which("uv")
        env.uv = Path(found) if found else None
    if env.core:
        env.core_version = read_core_version(env.core)
    return env


def core_missing(env: ComfyEnv, catalog: dict) -> List[str]:
    if not env.core:
        return []
    return [f for f in catalog.get("comfyui", {}).get("required_core_files", []) if not (env.core / f).exists()]


def comfy_running() -> bool:
    try:
        if IS_WIN:
            out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq ComfyUI.exe", "/NH"], capture_output=True,
                                 text=True, creationflags=NO_WINDOW, timeout=10).stdout
            return "ComfyUI.exe" in out
        out = subprocess.run(["pgrep", "-f", "ComfyUI"], capture_output=True, text=True, timeout=10).stdout
        return bool(out.strip())
    except Exception:
        return False


# ===================================================================== network

class _StripAuthRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects but never forward Authorization to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_opener = urllib.request.build_opener(_StripAuthRedirect)


def http_open(url: str, headers: Optional[dict] = None, timeout: float = 30):
    h = {"User-Agent": USER_AGENT}
    h.update(headers or {})
    return _opener.open(urllib.request.Request(url, headers=h), timeout=timeout)


def http_json(url: str, headers: Optional[dict] = None, timeout: float = 20):
    with http_open(url, headers, timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_text(url: str, headers: Optional[dict] = None, timeout: float = 20) -> str:
    with http_open(url, headers, timeout) as r:
        return r.read().decode("utf-8")


def auth_for(url: str, settings: Settings) -> dict:
    host = urllib.parse.urlparse(url).netloc
    if host.endswith("huggingface.co") and settings.get("hf_token"):
        return {"Authorization": f"Bearer {settings['hf_token']}"}
    if host.endswith("github.com") and settings.get("github_token"):
        return {"Authorization": f"Bearer {settings['github_token']}"}
    return {}


def download_file(url: str, dest: Path, settings: Settings, progress: Callable[[int, int], None],
                  cancel: threading.Event, chunk: int = 1 << 20) -> Path:
    """Resumable download to dest (via dest.part)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = auth_for(url, settings)
    if have:
        headers["Range"] = f"bytes={have}-"
    try:
        r = http_open(url, headers, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:  # already complete
            part.replace(dest)
            return dest
        if e.code in (401, 403):
            raise RuntimeError(f"{e.code} from server: this model may need a Hugging Face token (Settings tab)") from e
        raise
    with r:
        status = getattr(r, "status", 200)
        if status == 200 and have:
            have = 0  # server ignored Range: restart
        total = None
        cr = r.headers.get("Content-Range")
        if cr and "/" in cr and cr.split("/")[-1].isdigit():
            total = int(cr.split("/")[-1])
        elif r.headers.get("Content-Length"):
            total = int(r.headers["Content-Length"]) + have
        with open(part, "ab" if have else "wb") as f:
            done = have
            progress(done, total or 0)
            while True:
                if cancel.is_set():
                    raise Cancelled()
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                progress(done, total or 0)
    if total and part.stat().st_size != total:
        raise RuntimeError(f"incomplete download ({part.stat().st_size} of {total} bytes); run again to resume")
    part.replace(dest)
    return dest


def remote_size(url: str, settings: Settings) -> Optional[int]:
    try:
        with http_open(url, {**auth_for(url, settings), "Range": "bytes=0-0"}, timeout=20) as r:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr and cr.split("/")[-1].isdigit():
                return int(cr.split("/")[-1])
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and getattr(r, "status", 200) == 200 else None
    except Exception:
        return None


def fmt_size(n: Optional[float]) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return "?"


def version_tuple(v: str):
    return tuple(int(x) for x in re.findall(r"\d+", v or "0")[:4])


# ===================================================================== catalog

def load_catalog(settings: Settings, log: Callable[[str], None]) -> dict:
    url = settings.get("catalog_url") or CATALOG_URL
    try:
        cat = json.loads(http_text(url, timeout=15))
        (tool_dir() / "catalog.cache.json").write_text(json.dumps(cat, indent=1), encoding="utf-8")
        log(f"Catalog loaded from {url}")
        return cat
    except Exception as e:
        log(f"Online catalog unavailable ({e}); using local copy")
    here = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    for p in (tool_dir() / "catalog.cache.json", here / "catalog.json", Path(__file__).resolve().parent / "catalog.json"):
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise RuntimeError("No catalog available (offline and no local catalog.json)")


def all_plugins(catalog: dict, settings: Settings) -> List[dict]:
    plugins = [dict(p) for p in catalog.get("plugins", [])]
    known = {p["id"] for p in plugins}
    for p in settings.get("custom_plugins", []):
        if p["id"] not in known:
            plugins.append(dict(p, custom=True))
    return plugins


def parse_github_url(url: str) -> Optional[dict]:
    m = re.match(r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/tree/([^/]+)(?:/(.+))?)?/?$",
                 url.strip())
    if not m:
        return None
    owner, repo, ref, sub = m.groups()
    return {"id": f"custom-{owner}-{repo}".lower(), "name": repo, "repo": f"{owner}/{repo}", "ref": ref or "HEAD",
            "subdir": sub or "", "folder": Path(sub).name if sub else repo, "required": False,
            "description": f"Added from {url.strip()}"}


MODEL_EXTS = {".ckpt", ".pt", ".pt2", ".bin", ".pth", ".safetensors", ".pkl", ".sft", ".gguf", ".onnx"}
MODEL_TYPES = ["checkpoints", "diffusion_models", "unet", "loras", "vae", "text_encoders", "clip", "clip_vision",
               "controlnet", "upscale_models", "embeddings", "background_removal", "sams", "ipadapter",
               "style_models", "photomaker", "LLM"]


def git_head(path: Path) -> Optional[str]:
    """HEAD commit of a git checkout without needing git installed."""
    try:
        head = (path / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(" ", 1)[1].strip()
            f = path / ".git" / ref
            if f.exists():
                return f.read_text(encoding="utf-8").strip()
            packed = path / ".git" / "packed-refs"
            if packed.exists():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(" " + ref):
                        return line.split(" ")[0]
            return None
        return head
    except Exception:
        return None


def normalize_model_url(url: str) -> str:
    """Turn Hugging Face page links (…/blob/…) into direct download links (…/resolve/…)."""
    url = url.strip()
    m = re.match(r"^(https?://(?:www\.)?huggingface\.co/.+?)/blob/(.+)$", url)
    if m:
        return f"{m.group(1)}/resolve/{m.group(2)}"
    return url


def guess_filename(url: str, settings: Settings) -> str:
    name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    if name and "." in name:
        return name
    try:  # ask the server (Content-Disposition)
        with http_open(url, {**auth_for(url, settings), "Range": "bytes=0-0"}, timeout=20) as r:
            cd = r.headers.get("Content-Disposition", "")
            m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', cd)
            if m:
                return urllib.parse.unquote(m.group(1))
    except Exception:
        pass
    raise RuntimeError("could not tell the file name from the URL; type it in the File name box")


def send_to_trash(path: Path) -> None:
    """Delete to the Windows Recycle Bin when possible; plain delete elsewhere."""
    if IS_WIN:
        import ctypes
        from ctypes import wintypes

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT), ("pFrom", wintypes.LPCWSTR),
                        ("pTo", wintypes.LPCWSTR), ("fFlags", ctypes.c_uint16), ("fAnyOperationsAborted", wintypes.BOOL),
                        ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

        op = SHFILEOPSTRUCTW(None, 3, str(path.resolve()) + "\0", None, 0x0040 | 0x0010 | 0x0004 | 0x0400,
                             False, None, None)  # FO_DELETE, ALLOWUNDO|NOCONFIRMATION|SILENT|NOERRORUI
        rc = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        if rc != 0:
            raise RuntimeError(f"could not move {path} to the Recycle Bin (error {rc})")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


# =================================================================== operations

class Installer:
    """All install / update logic. GUI-independent (log + progress callbacks)."""

    def __init__(self, env: ComfyEnv, catalog: dict, settings: Settings,
                 log: Callable[[str], None] = print, progress: Callable[[str, float, str], None] = lambda *a: None):
        self.env, self.catalog, self.settings = env, catalog, settings
        self.log, self.progress = log, progress
        self.cancel = threading.Event()
        self._sha_cache: Dict[str, str] = {}

    # ---------------------------------------------------------------- github
    def latest_sha(self, repo: str, ref: str) -> str:
        key = f"{repo}@{ref}"
        if key not in self._sha_cache:
            url = f"{GITHUB_API}/repos/{repo}/commits/{urllib.parse.quote(ref or 'HEAD', safe='')}"
            self._sha_cache[key] = http_text(url, {"Accept": "application/vnd.github.sha",
                                                   **auth_for("https://api.github.com", self.settings)}).strip()
        return self._sha_cache[key]

    def fetch_repo_zip(self, repo: str, sha: str) -> zipfile.ZipFile:
        url = f"{CODELOAD}/{repo}/zip/{sha}"
        self.log(f"Downloading {repo} @ {sha[:7]}")
        buf = io.BytesIO()
        with http_open(url, auth_for("https://github.com", self.settings), timeout=120) as r:
            total = int(r.headers.get("Content-Length") or 0)
            while True:
                if self.cancel.is_set():
                    raise Cancelled()
                b = r.read(1 << 16)
                if not b:
                    break
                buf.write(b)
                self.progress(f"{repo}", buf.tell() / total if total else -1, fmt_size(buf.tell()))
        return zipfile.ZipFile(buf)

    @staticmethod
    def extract_subdir(zf: zipfile.ZipFile, subdir: str, dest: Path) -> int:
        names = zf.namelist()
        root = names[0].split("/")[0] + "/"
        prefix = root + (subdir.strip("/") + "/" if subdir else "")
        n = 0
        dest_resolved = dest.resolve()
        for name in names:
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            rel = name[len(prefix):]
            target = (dest / rel).resolve()
            if dest_resolved not in target.parents:  # zip-slip guard
                raise RuntimeError(f"unsafe path in archive: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
        if n == 0:
            raise RuntimeError(f"'{subdir}' not found in archive")
        return n

    # --------------------------------------------------------------- plugins
    def plugin_dir(self, p: dict) -> Path:
        return self.env.custom_nodes / p["folder"]

    def plugin_status(self, p: dict, check_remote: bool = False) -> dict:
        d = self.plugin_dir(p)
        st = {"installed": d.exists(), "sha": None, "managed": False, "latest": None, "state": "Not installed"}
        if d.exists():
            m = d / MARKER
            if m.exists():
                try:
                    info = json.loads(m.read_text(encoding="utf-8"))
                    st.update(sha=info.get("sha"), managed=True)
                except Exception:
                    pass
            st["state"] = f"Installed ({st['sha'][:7]})" if st["sha"] else (
                "Installed (git)" if (d / ".git").exists() else "Installed (manual)")
            if (d.parent / (d.name + ".disabled")).exists():
                st["state"] += " +disabled copy"
        elif (d.parent / (d.name + ".disabled")).exists():
            st["state"] = "Disabled"
        if check_remote:
            try:
                st["latest"] = self.latest_sha(p["repo"], p.get("ref") or "HEAD")
                if st["installed"] and st["sha"] and st["sha"] != st["latest"]:
                    st["state"] = f"Update available ({st['sha'][:7]} → {st['latest'][:7]})"
                elif st["installed"] and st["sha"]:
                    st["state"] = f"Up to date ({st['sha'][:7]})"
            except Exception as e:
                st["state"] += " (update check failed)"
                self.log(f"[{p['name']}] update check failed: {e}")
        return st

    def backup_dir(self) -> Path:
        d = self.env.custom_nodes.parent / "standrig_backups"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def install_plugin(self, p: dict, install_requirements: bool = True) -> str:
        if not self.env.base:
            raise RuntimeError("ComfyUI folder not set")
        sha = self.latest_sha(p["repo"], p.get("ref") or "HEAD")
        d = self.plugin_dir(p)
        st = self.plugin_status(p)
        if st["sha"] == sha:
            self.log(f"[{p['name']}] already up to date ({sha[:7]})")
            if install_requirements:
                self.install_requirements(p)
            return sha
        zf = self.fetch_repo_zip(p["repo"], sha)
        self.env.custom_nodes.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="standrig_", dir=self.env.custom_nodes.parent))
        try:
            staged = tmp / p["folder"]
            n = self.extract_subdir(zf, p.get("subdir", ""), staged)
            (staged / MARKER).write_text(json.dumps({"repo": p["repo"], "ref": p.get("ref"), "sha": sha,
                                                     "subdir": p.get("subdir", ""),
                                                     "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                     "tool": VERSION}, indent=2), encoding="utf-8")
            if d.exists():
                bk = self.backup_dir() / f"{p['folder']}-{time.strftime('%Y%m%d-%H%M%S')}"
                shutil.move(str(d), str(bk))
                self.log(f"[{p['name']}] previous version moved to {bk}")
            shutil.move(str(staged), str(d))
            self.log(f"[{p['name']}] installed {n} files at {d} ({sha[:7]})")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if install_requirements:
            self.install_requirements(p)
        return sha

    def remove_plugin(self, p: dict) -> None:
        d = self.plugin_dir(p)
        if not d.exists():
            self.log(f"[{p['name']}] not installed")
            return
        bk = self.backup_dir() / f"{p['folder']}-removed-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(str(d), str(bk))
        self.log(f"[{p['name']}] removed (kept a copy in {bk})")

    def pip_command(self, args: List[str]) -> List[str]:
        py = self.env.python
        if py is None:
            raise RuntimeError("ComfyUI's Python was not found, cannot install requirements")
        if self.env.uv and self.env.kind in ("desktop", "desktop2"):
            return [str(self.env.uv), "pip", *args, "--python", str(py)]
        return [str(py), "-m", "pip", *args]

    def install_requirements(self, p: dict) -> None:
        self.install_requirements_dir(self.plugin_dir(p), p["name"])

    def install_requirements_dir(self, folder: Path, name: str, force_log: bool = False) -> None:
        req = folder / "requirements.txt"
        if not req.exists():
            if force_log:
                self.log(f"[{name}] has no requirements.txt; nothing to install")
            return
        lines = [l.strip() for l in req.read_text(encoding="utf-8", errors="replace").splitlines()]
        if not any(l and not l.startswith("#") for l in lines):
            return
        cmd = self.pip_command(["install", "-r", str(req)])
        self.log(f"[{name}] installing Python requirements…")
        self.run(cmd)

    def run(self, cmd: List[str]) -> None:
        self.log("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        envv = dict(os.environ)
        if self.env.kind == "desktop" and self.env.base:
            envv.setdefault("UV_CACHE_DIR", str(self.env.base / "uv-cache"))
        envv["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=envv, creationflags=NO_WINDOW)
        for line in proc.stdout:
            if self.cancel.is_set():
                proc.terminate()
                raise Cancelled()
            line = line.rstrip()
            if line:
                self.log("  " + line)
        if proc.wait() != 0:
            raise RuntimeError(f"command failed with exit code {proc.returncode}")

    # ------------------------------------------------------ installed plugins
    def scan_plugins(self) -> List[dict]:
        """Everything in custom_nodes (enabled and disabled), merged with the catalog."""
        cn = self.env.custom_nodes
        cat = {p["folder"].lower(): p for p in all_plugins(self.catalog, self.settings)}
        found: Dict[str, dict] = {}

        def add(path: Path, enabled: bool):
            folder = path.name[:-len(".disabled")] if path.name.endswith(".disabled") else path.name
            if path.is_file() and not folder.endswith(".py"):
                return
            if folder in ("__pycache__",) or folder.startswith("."):
                return
            key = folder.lower()
            if key in found and found[key]["enabled"]:
                return  # an enabled copy wins over a disabled one
            entry = dict(cat.get(key) or {"id": "local:" + folder, "name": folder.removesuffix(".py"),
                                          "folder": folder, "description": ""})
            kind = "file" if path.is_file() else "git" if (path / ".git").exists() else \
                "managed" if (path / MARKER).exists() else "manual"
            sha = None
            if kind == "managed":
                sha = _read_json(path / MARKER, {}).get("sha")
            elif kind == "git":
                sha = git_head(path)
            entry.update(path=str(path), enabled=enabled, kind=kind, sha=sha, installed=True)
            found[key] = entry

        if cn.is_dir():
            for child in sorted(cn.iterdir(), key=lambda x: x.name.lower()):
                if child.name == ".disabled" and child.is_dir():  # ComfyUI-Manager's disabled folder
                    for sub in child.iterdir():
                        add(sub, False)
                else:
                    add(child, not child.name.endswith(".disabled"))
        out = list(found.values())
        for key, p in cat.items():
            if key not in found:
                out.append(dict(p, installed=False, enabled=False, kind="-", sha=None, path=None))
        out.sort(key=lambda e: (not e.get("installed"), e.get("id", "").startswith("local:"), e["name"].lower()))
        return out

    @staticmethod
    def describe_plugin(e: dict) -> str:
        if not e.get("installed"):
            return "Not installed"
        state = "Enabled" if e["enabled"] else "Disabled"
        kind = {"git": "git", "managed": "StandRig Setup", "manual": "manual copy", "file": "single file"}[e["kind"]]
        return f"{state} · {kind}" + (f" · {e['sha'][:7]}" if e.get("sha") else "")

    def set_enabled(self, e: dict, enabled: bool) -> None:
        path = Path(e["path"])
        if e["enabled"] == enabled:
            return
        if enabled:
            target = self.env.custom_nodes / e["folder"]
        else:
            target = self.env.custom_nodes / (e["folder"] + ".disabled")
        if target.exists():
            raise RuntimeError(f"cannot rename: {target} already exists")
        shutil.move(str(path), str(target))
        self.log(f"[{e['name']}] {'enabled' if enabled else 'disabled'} ({target.name})")

    def git_update(self, e: dict) -> None:
        git = shutil.which("git")
        if not git:
            raise RuntimeError("git is not installed. Install Git for Windows, or reinstall the plugin from its "
                               "GitHub URL so StandRig Setup can update it without git.")
        self.run([git, "-C", e["path"], "pull", "--ff-only"])

    def git_behind(self, e: dict) -> Optional[int]:
        git = shutil.which("git")
        if not git:
            return None
        try:
            subprocess.run([git, "-C", e["path"], "fetch", "--quiet"], capture_output=True, timeout=60,
                           creationflags=NO_WINDOW)
            out = subprocess.run([git, "-C", e["path"], "rev-list", "--count", "HEAD..@{u}"], capture_output=True,
                                 text=True, timeout=30, creationflags=NO_WINDOW)
            return int(out.stdout.strip()) if out.returncode == 0 else None
        except Exception:
            return None

    def update_plugin_entry(self, e: dict) -> None:
        """Update one scanned plugin the right way for how it was installed."""
        if e.get("installed") and e.get("kind") == "git":
            self.git_update(e)
            self.install_requirements_dir(Path(e["path"]), e["name"])
        elif e.get("repo"):
            self.install_plugin(e)
        else:
            raise RuntimeError(f"{e['name']}: unknown source. Add its GitHub URL in the box below to manage it.")

    def remove_entry(self, e: dict) -> None:
        path = Path(e["path"])
        bk = self.backup_dir() / f"{path.name}-removed-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(str(path), str(bk))
        self.log(f"[{e['name']}] removed (kept a copy in {bk})")

    # --------------------------------------------------------- model library
    def scan_models(self) -> List[dict]:
        rows, seen = [], set()
        for root in self.env.all_model_dirs():
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    partial = fn.endswith(".part")
                    if ext not in MODEL_EXTS and not partial:
                        continue
                    fp = Path(dirpath) / fn
                    key = str(fp.resolve()).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        st = fp.stat()
                    except OSError:
                        continue
                    rel = fp.relative_to(root)
                    rows.append({"path": str(fp), "name": fn, "type": rel.parts[0] if len(rel.parts) > 1 else "(root)",
                                 "root": str(root), "size": st.st_size, "mtime": st.st_mtime, "partial": partial})
        rows.sort(key=lambda r: (r["type"].lower(), r["name"].lower()))
        return rows

    def delete_model(self, path: str) -> None:
        send_to_trash(Path(path))
        self.log(f"Deleted {path}" + (" (moved to the Recycle Bin)" if IS_WIN else ""))

    def move_model(self, path: str, dest_root: Path) -> Path:
        src = Path(path)
        root = next((r for r in self.env.all_model_dirs() if str(src).lower().startswith(str(r).lower())), src.parent)
        try:
            rel = src.relative_to(root)
        except ValueError:
            rel = Path(src.name)
        dest = dest_root / rel
        if dest.exists():
            raise RuntimeError(f"{dest} already exists")
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.log(f"Moving {src} → {dest}")
        shutil.move(str(src), str(dest))
        return dest

    def download_url(self, url: str, type_folder: str, dest_root: Path, filename: str = "") -> Path:
        url = normalize_model_url(url)
        name = filename.strip() or guess_filename(url, self.settings)
        dest = dest_root / type_folder.strip().strip("/\\") / name
        if dest.exists():
            raise RuntimeError(f"{dest} already exists")
        m = {"name": name, "url": url}
        self.log(f"[{name}] downloading → {dest}")
        t0, last = time.time(), [0.0]

        def prog(done, total):
            now = time.time()
            if now - last[0] < 0.25 and done != total:
                return
            last[0] = now
            speed = done / max(now - t0, 1e-3)
            self.progress(name, done / total if total else -1,
                          f"{fmt_size(done)} / {fmt_size(total or None)} at {fmt_size(speed)}/s")

        download_file(m["url"], dest, self.settings, prog, self.cancel)
        self.log(f"[{name}] done ({fmt_size(dest.stat().st_size)})")
        return dest

    # ------------------------------------------------------------- workflows
    def chosen_qwen_file(self) -> Optional[str]:
        mv = self.catalog.get("model_variants", {}).get("qwen_edit")
        if not mv:
            return None
        choice = self.settings.get("qwen_variant") or mv.get("default")
        opt = next((o for o in mv["options"] if o["id"] == choice), mv["options"][0])
        model = next((m for m in self.catalog["models"] if m["id"] == opt["model"]), None)
        return model["file"] if model else None

    def install_workflows(self) -> int:
        wf = self.catalog.get("workflows")
        if not wf or not self.env.base:
            return 0
        sha = self.latest_sha(wf["repo"], wf.get("ref") or "HEAD")
        zf = self.fetch_repo_zip(wf["repo"], sha)
        dest = self.workflows_dest()
        dest.mkdir(parents=True, exist_ok=True)
        root = zf.namelist()[0].split("/")[0] + "/"
        prefix = root + wf["subdir"].strip("/") + "/"
        placeholder = self.catalog.get("model_variants", {}).get("qwen_edit", {}).get("placeholder")
        chosen = self.chosen_qwen_file()
        n = 0
        for name in zf.namelist():
            rel = name[len(prefix):] if name.startswith(prefix) else None
            if not rel or "/" in rel or not fnmatch.fnmatch(rel, "*.json"):
                continue  # top-level UI workflows only (api/ copies are for scripting)
            text = zf.read(name).decode("utf-8")
            if placeholder and chosen and chosen != placeholder:
                text = text.replace(placeholder, chosen)
            (dest / rel).write_text(text, encoding="utf-8")
            n += 1
        self.log(f"Copied {n} workflows to {dest}" + (f" (using {chosen})" if chosen else ""))
        self.log("In ComfyUI open the Workflows sidebar (W) → StandRig.")
        return n

    def workflows_dest(self) -> Optional[Path]:
        if self.env.workflows_dir:
            return self.env.workflows_dir
        wf = self.catalog.get("workflows")
        return self.env.base / wf["dest"] if wf and self.env.base else None

    def workflow_status(self, w: dict) -> dict:
        """Installed? and which plugins/models it still needs."""
        dest = self.workflows_dest()
        installed = bool(dest and (dest / w["file"]).exists())
        plugins = {p["id"]: p for p in all_plugins(self.catalog, self.settings)}
        miss_p = [plugins[i]["name"] for i in w.get("plugins", []) if i in plugins
                  and not self.plugin_dir(plugins[i]).exists()]
        chosen = {m["id"]: m for m in self.model_list()}
        miss_m = []
        for mid in w.get("models", []):
            m = chosen.get(mid) or next((x for x in self.model_list() if x.get("variant_of") == mid), None)
            if m is not None and self.find_model(m) is None:
                miss_m.append(m["name"])
        if not installed:
            state = "Not installed"
        elif miss_p or miss_m:
            state = "Needs " + ", ".join(miss_p + ([f"{len(miss_m)} model(s)"] if miss_m else []))
        else:
            state = "Ready"
        return {"installed": installed, "missing_plugins": miss_p, "missing_models": miss_m, "state": state}

    # ---------------------------------------------------------------- models
    def model_list(self) -> List[dict]:
        """Models to install given the chosen variants."""
        chosen = self.chosen_qwen_file()
        out = []
        for m in self.catalog.get("models", []):
            if m.get("variant_of") and m["file"] != chosen:
                continue
            out.append(m)
        return out

    def model_path(self, m: dict) -> Path:
        return self.env.models / m["dir"] / m["file"]

    def find_model(self, m: dict) -> Optional[Path]:
        for root in self.env.all_model_dirs():
            p = root / m["dir"] / m["file"]
            if p.exists():
                return p
        return None

    def model_state(self, m: dict) -> str:
        p = self.model_path(m)
        found = self.find_model(m)
        if found is not None:
            return "Installed" if found == p else f"Installed ({found.parents[1]})"
        part = p.with_name(p.name + ".part")
        if part.exists():
            return f"Partial ({fmt_size(part.stat().st_size)})"
        return "Missing"

    def download_model(self, m: dict) -> None:
        dest = self.model_path(m)
        found = self.find_model(m)
        if found is not None:
            self.log(f"[{m['name']}] already present: {found}")
            return
        self.log(f"[{m['name']}] downloading → {dest}")
        t0, last = time.time(), [0.0]

        def prog(done, total):
            now = time.time()
            if now - last[0] < 0.25 and done != total:
                return
            last[0] = now
            speed = done / max(now - t0, 1e-3)
            frac = done / total if total else -1
            eta = f", {int((total - done) / speed // 60)} min left" if total and speed > 0 else ""
            self.progress(m["name"], frac, f"{fmt_size(done)} / {fmt_size(total or None)} at {fmt_size(speed)}/s{eta}")

        download_file(m["url"], dest, self.settings, prog, self.cancel)
        self.log(f"[{m['name']}] done ({fmt_size(dest.stat().st_size)})")

    # ------------------------------------------------------------ everything
    def quick_setup(self, plugins: bool = True, workflows: bool = True, models: bool = True,
                    optional_plugins: Optional[List[str]] = None) -> None:
        if plugins:
            for p in all_plugins(self.catalog, self.settings):
                if p.get("required") or p["id"] in (optional_plugins or []):
                    self.install_plugin(p)
        if workflows:
            self.install_workflows()
        if models:
            for m in self.model_list():
                self.download_model(m)
        self.log("✔ Setup finished. Restart ComfyUI to load new nodes.")

    # ----------------------------------------------------------- self update
    def tool_update_available(self) -> Optional[str]:
        latest = self.catalog.get("installer", {}).get("version")
        return latest if latest and version_tuple(latest) > version_tuple(VERSION) else None

    def self_update(self) -> Path:
        url = self.catalog["installer"]["script_url"]
        new = http_text(url, timeout=60)
        if "APP_NAME" not in new or "class Installer" not in new:
            raise RuntimeError("downloaded file does not look like StandRig Setup")
        if getattr(sys, "frozen", False):
            me = tool_dir() / "app" / "standrig_setup.py"   # the .exe runs this newer copy on next start
            me.parent.mkdir(parents=True, exist_ok=True)
        else:
            me = Path(__file__).resolve()
        if me.exists():
            me.with_suffix(".py.bak").write_text(me.read_text(encoding="utf-8"), encoding="utf-8")
        me.write_text(new, encoding="utf-8")
        for name in ("catalog.json",):
            try:
                (me.parent / name).write_text(http_text(url.rsplit("/", 1)[0] + "/" + name), encoding="utf-8")
            except Exception:
                pass
        return me


# ========================================================================= GUI

def run_gui() -> None:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    settings = Settings.load()
    logq: "queue.Queue[tuple]" = queue.Queue()
    log_file = tool_dir() / "setup.log"

    def log(msg: str) -> None:
        logq.put(("log", msg))
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(time.strftime("%H:%M:%S ") + msg + "\n")
        except Exception:
            pass

    def progress(label: str, frac: float, text: str) -> None:
        logq.put(("progress", label, frac, text))

    root = tk.Tk()
    root.title(f"{APP_NAME} {VERSION}")
    root.geometry("1060x860")
    root.minsize(880, 680)
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=8)
    style.configure("Head.TLabel", font=("Segoe UI", 13, "bold"))
    style.configure("Warn.TLabel", foreground="#b00020")
    style.configure("Ok.TLabel", foreground="#1b7f3b")

    state = {"env": ComfyEnv(), "catalog": {}, "inst": None, "busy": False}

    # ---------------------------------------------------------------- header
    head = ttk.Frame(root, padding=(12, 10, 12, 4))
    head.pack(fill="x")
    ttk.Label(head, text="StandRig Setup for ComfyUI", style="Head.TLabel").grid(row=0, column=0, columnspan=4, sticky="w")
    ttk.Label(head, text="ComfyUI folder:").grid(row=1, column=0, sticky="w", pady=(6, 0))
    base_var = tk.StringVar()
    base_entry = ttk.Entry(head, textvariable=base_var)
    base_entry.grid(row=1, column=1, sticky="we", padx=6, pady=(6, 0))
    head.columnconfigure(1, weight=1)
    info_var = tk.StringVar(value="Detecting…")
    info_lbl = ttk.Label(head, textvariable=info_var, wraplength=980, justify="left")
    info_lbl.grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
    warn_var = tk.StringVar()
    ttk.Label(head, textvariable=warn_var, style="Warn.TLabel", wraplength=980, justify="left").grid(
        row=4, column=0, columnspan=4, sticky="w")
    inst_row = ttk.Frame(head)
    inst_lbl = ttk.Label(inst_row, text="ComfyUI install:")
    inst_lbl.pack(side="left")
    inst_var = tk.StringVar()
    inst_box = ttk.Combobox(inst_row, textvariable=inst_var, state="readonly", width=60)
    inst_box.pack(side="left", padx=6)

    def on_instance(_e=None):
        env = state["env"]
        match = next((i for i in env.instances if f"{i['name']}  ({i['installPath']})" == inst_var.get()), None)
        if match:
            settings["instance_id"] = match["id"]
            settings.save()
            refresh_env()

    inst_box.bind("<<ComboboxSelected>>", on_instance)

    def browse():
        d = filedialog.askdirectory(title="Select your ComfyUI folder (contains models and custom_nodes)")
        if d:
            settings["base_path"] = d
            settings.save()
            refresh_env()

    def auto_detect():
        settings.pop("base_path", None)
        settings.save()
        refresh_env()

    ttk.Button(head, text="Browse…", command=browse).grid(row=1, column=2, pady=(6, 0))
    ttk.Button(head, text="Auto-detect", command=auto_detect).grid(row=1, column=3, padx=(6, 0), pady=(6, 0))

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=12, pady=6)

    # --------------------------------------------------------- bottom: log
    bottom = ttk.Frame(root, padding=(12, 0, 12, 10))
    bottom.pack(side="bottom", fill="both", before=nb)
    prog_var = tk.DoubleVar()
    prog = ttk.Progressbar(bottom, variable=prog_var, maximum=1.0)
    prog.pack(fill="x")
    status_var = tk.StringVar(value="Ready")
    srow = ttk.Frame(bottom)
    srow.pack(fill="x")
    ttk.Label(srow, textvariable=status_var).pack(side="left", pady=2)
    cancel_btn = ttk.Button(srow, text="Cancel", state="disabled",
                            command=lambda: state["inst"] and state["inst"].cancel.set())
    cancel_btn.pack(side="right")
    ttk.Button(srow, text="Open log", command=lambda: open_path(log_file)).pack(side="right", padx=6)
    logbox = tk.Text(bottom, height=10, wrap="word", font=("Consolas", 9), state="disabled")
    logbox.pack(fill="both", expand=True)

    def open_path(p: Path):
        p = Path(p)
        if not p.exists():
            messagebox.showinfo(APP_NAME, f"{p} does not exist yet")
            return
        if IS_WIN:
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(p)])

    # -------------------------------------------------------------- worker
    def run_task(title: str, fn: Callable[[Installer], None], after: Optional[Callable] = None,
                 needs_closed: bool = False):
        if state["busy"]:
            messagebox.showinfo(APP_NAME, "Another task is running.")
            return
        env = state["env"]
        if not env.base:
            messagebox.showerror(APP_NAME, "Choose your ComfyUI folder first.")
            return
        if needs_closed and comfy_running():
            if not messagebox.askyesno(APP_NAME, "ComfyUI seems to be running.\n\nInstalling Python packages while it "
                                                  "runs can fail on Windows (files in use). Close ComfyUI first.\n\n"
                                                  "Continue anyway?"):
                return
        inst = Installer(env, state["catalog"], settings, log, progress)
        state.update(inst=inst, busy=True)
        cancel_btn.configure(state="normal")
        status_var.set(title + "…")
        log(f"── {title} ──")

        def work():
            ok = True
            try:
                fn(inst)
            except Cancelled:
                ok = False
                log("Cancelled. Downloads resume where they stopped next time.")
            except Exception as e:
                ok = False
                log(f"ERROR: {e}")
                log(traceback.format_exc().strip().splitlines()[-1])
            logq.put(("done", title, ok, after))

        threading.Thread(target=work, daemon=True).start()

    def pump():
        try:
            while True:
                item = logq.get_nowait()
                if item[0] == "log":
                    logbox.configure(state="normal")
                    logbox.insert("end", item[1] + "\n")
                    logbox.see("end")
                    logbox.configure(state="disabled")
                elif item[0] == "progress":
                    _, label, frac, text = item
                    if frac < 0:
                        prog.configure(mode="indeterminate")
                        prog.step(0.02)
                    else:
                        prog.configure(mode="determinate")
                        prog_var.set(frac)
                    status_var.set(f"{label}: {text}")
                elif item[0] == "catalog":
                    state["catalog"] = item[1]
                    refresh_env()
                    latest = Installer(state["env"], item[1], settings).tool_update_available()
                    if latest:
                        log(f"A new version of this tool is available ({latest}): Settings → Check for tool update.")
                elif item[0] == "done":
                    _, title, ok, after = item
                    state["busy"] = False
                    cancel_btn.configure(state="disabled")
                    prog.configure(mode="determinate")
                    prog_var.set(1.0 if ok else 0.0)
                    status_var.set(f"{title}: {'done' if ok else 'stopped'}")
                    refresh_tables()
                    if after:
                        after()
        except queue.Empty:
            pass
        root.after(100, pump)

    # ------------------------------------------------------- tab: quick setup
    t_quick = ttk.Frame(nb, padding=14)
    nb.add(t_quick, text="  Quick setup  ")
    ttk.Label(t_quick, text="Install everything the StandRig workflows need, in one click.",
              font=("Segoe UI", 11)).pack(anchor="w")
    steps = ttk.LabelFrame(t_quick, text="Steps", padding=10)
    steps.pack(fill="x", pady=10)
    q_plugins = tk.BooleanVar(value=True)
    q_workflows = tk.BooleanVar(value=True)
    q_models = tk.BooleanVar(value=True)
    ttk.Checkbutton(steps, text="Custom nodes: StandRig Bridge + ComfyUI-See-through (and their Python packages)",
                    variable=q_plugins).pack(anchor="w")
    ttk.Checkbutton(steps, text="Workflows → ComfyUI Workflows sidebar (folder 'StandRig')",
                    variable=q_workflows).pack(anchor="w")
    ttk.Checkbutton(steps, text="Models (Qwen-Image-Edit-2511, text encoder, VAE, LoRAs, SAM3, BiRefNet)",
                    variable=q_models).pack(anchor="w")
    var_row = ttk.Frame(steps)
    var_row.pack(anchor="w", pady=(8, 0))
    ttk.Label(var_row, text="Qwen model weights:").pack(side="left")
    variant_var = tk.StringVar()
    variant_box = ttk.Combobox(var_row, textvariable=variant_var, state="readonly", width=52)
    variant_box.pack(side="left", padx=6)

    def on_variant(_e=None):
        mv = state["catalog"].get("model_variants", {}).get("qwen_edit")
        if not mv:
            return
        opt = next((o for o in mv["options"] if o["label"] == variant_var.get()), None)
        if opt:
            settings["qwen_variant"] = opt["id"]
            settings.save()
            refresh_tables()

    variant_box.bind("<<ComboboxSelected>>", on_variant)
    ttk.Label(t_quick, text="Downloads are large (tens of GB) and resume if interrupted. Close ComfyUI before "
                            "installing so Python packages can be updated. Restart ComfyUI afterwards.",
              wraplength=900, justify="left").pack(anchor="w", pady=(0, 10))
    btns = ttk.Frame(t_quick)
    btns.pack(anchor="w")
    ttk.Button(btns, text="Install / update everything", style="Big.TButton",
               command=lambda: run_task("Quick setup", lambda i: i.quick_setup(q_plugins.get(), q_workflows.get(),
                                                                                   q_models.get()),
                                        needs_closed=q_plugins.get())).pack(side="left")
    ttk.Button(btns, text="Open ComfyUI folder",
               command=lambda: state["env"].base and open_path(state["env"].base)).pack(side="left", padx=8)
    ttk.Button(btns, text="Open workflows folder",
               command=lambda: state["env"].base and open_path(
                   state["env"].base / state["catalog"].get("workflows", {}).get("dest", "user/default/workflows"))
               ).pack(side="left")
    howto = ttk.LabelFrame(t_quick, text="After setup", padding=10)
    howto.pack(fill="x", pady=12)
    ttk.Label(howto, justify="left", wraplength=900, text=(
        "1. Restart ComfyUI Desktop.\n"
        "2. Press W (Workflows sidebar) → StandRig → standrig_full_pipeline.\n"
        "3. Choose your character picture in 'Load character picture' and press Run.\n"
        "4. Results: <ComfyUI folder>\\output\\standrig\\<project>\\ (character.psd + references\\).")).pack(anchor="w")
    ttk.Button(howto, text="Project page / guide", command=lambda: webbrowser.open(
        "https://github.com/tsukiyomi-naoko/StandRiggerWorkflow#readme")).pack(anchor="w", pady=(6, 0))

    # --------------------------------------------------------- tab: workflows
    t_wf = ttk.Frame(nb, padding=10)
    nb.add(t_wf, text="  Workflows  ")
    feat_var = tk.StringVar()
    ttk.Label(t_wf, textvariable=feat_var, justify="left", wraplength=980).pack(anchor="w", pady=(0, 8))
    wcols = ("name", "status", "description")
    wtree = ttk.Treeview(t_wf, columns=wcols, show="headings", selectmode="browse", height=6)
    for c, w in zip(wcols, (230, 230, 560)):
        wtree.heading(c, text=c.capitalize())
        wtree.column(c, width=w, anchor="w")
    wtree.pack(fill="x")
    wdetail = tk.StringVar(value="Select a workflow to see what it makes and what it needs.")
    ttk.Label(t_wf, textvariable=wdetail, justify="left", wraplength=980).pack(anchor="w", pady=8)

    def on_wf_select(_e=None):
        sel = wtree.selection()
        w = next((x for x in state["catalog"].get("workflow_list", []) if x["file"] in sel), None)
        if not w or not state["env"].base:
            return
        st = Installer(state["env"], state["catalog"], settings).workflow_status(w)
        lines = [f"{w['name']}  ({w['file']})", w.get("description", ""), f"Produces: {w.get('outputs', '-')}"]
        if st["missing_plugins"]:
            lines.append("Missing plugins: " + ", ".join(st["missing_plugins"]))
        if st["missing_models"]:
            lines.append("Missing models: " + ", ".join(st["missing_models"]))
        if not (st["missing_plugins"] or st["missing_models"]):
            lines.append("Everything it needs is installed." if st["installed"] else
                         "Plugins and models are ready; install the workflow files.")
        wdetail.set("\n".join(lines))

    wtree.bind("<<TreeviewSelect>>", on_wf_select)
    wrow = ttk.Frame(t_wf)
    wrow.pack(anchor="w")
    ttk.Button(wrow, text="Install / update workflows", command=lambda: run_task(
        "Installing workflows", lambda i: i.install_workflows())).pack(side="left")
    ttk.Button(wrow, text="Install what the selected workflow needs", command=lambda: install_for_selected()
               ).pack(side="left", padx=6)
    ttk.Button(wrow, text="Open workflows folder", command=lambda: state["env"].base and open_path(
        Installer(state["env"], state["catalog"], settings).workflows_dest())).pack(side="left")
    ttk.Label(t_wf, wraplength=980, justify="left", text=(
        "In ComfyUI: press W (Workflows sidebar) → StandRig, open a workflow, choose your picture in "
        "'Load character picture' and press Run. Results: output\\standrig\\<project>\\.")).pack(anchor="w", pady=8)

    def install_for_selected():
        sel = wtree.selection()
        w = next((x for x in state["catalog"].get("workflow_list", []) if x["file"] in sel), None)
        if not w:
            messagebox.showinfo(APP_NAME, "Select a workflow first.")
            return

        def fn(i: Installer):
            plugins = {p["id"]: p for p in all_plugins(state["catalog"], settings)}
            for pid in w.get("plugins", []):
                if pid in plugins and not i.plugin_dir(plugins[pid]).exists():
                    i.install_plugin(plugins[pid])
            i.install_workflows()
            wanted = set(w.get("models", []))
            for m in i.model_list():
                if (m["id"] in wanted or m.get("variant_of") in wanted) and i.find_model(m) is None:
                    i.download_model(m)
            i.log(f"✔ {w['name']} is ready. Restart ComfyUI if plugins were installed.")
        run_task(f"Setting up {w['name']}", fn, needs_closed=True)

    # ----------------------------------------------------------- tab: plugins
    t_plug = ttk.Frame(nb, padding=10)
    nb.add(t_plug, text="  Plugins  ")
    ttk.Label(t_plug, wraplength=980, justify="left", text=(
        "Everything in your custom_nodes folder, however it was installed, plus the catalog's plugins. "
        "Disabled = renamed to '<name>.disabled' (ComfyUI skips those; ComfyUI-Manager's .disabled folder is "
        "shown too). Restart ComfyUI after changes.")).pack(anchor="w", pady=(0, 6))
    pf = ttk.Frame(t_plug)
    pf.pack(fill="both", expand=True)
    cols = ("name", "status", "folder", "description")
    ptree = ttk.Treeview(pf, columns=cols, show="headings", selectmode="extended", height=12)
    for c, w in zip(cols, (210, 260, 220, 360)):
        ptree.heading(c, text=c.capitalize())
        ptree.column(c, width=w, anchor="w")
    psb = ttk.Scrollbar(pf, orient="vertical", command=ptree.yview)
    ptree.configure(yscrollcommand=psb.set)
    ptree.pack(side="left", fill="both", expand=True)
    psb.pack(side="right", fill="y")
    prow = ttk.Frame(t_plug)
    prow.pack(fill="x", pady=6)
    plugin_rows: Dict[str, dict] = {}

    def selected_entries() -> List[dict]:
        return [plugin_rows[i] for i in ptree.selection() if i in plugin_rows]

    def need_selection() -> Optional[List[dict]]:
        sel = selected_entries()
        if not sel:
            messagebox.showinfo(APP_NAME, "Select one or more plugins in the list.")
            return None
        return sel

    def plug_check_updates():
        def fn(i: Installer):
            res = {}
            for e in i.scan_plugins():
                if e.get("installed") and e["kind"] == "git":
                    n = i.git_behind(e)
                    res[e["id"]] = (f"Update available ({n} commits)" if n else "Up to date") if n is not None \
                        else "git not available"
                elif e.get("repo"):
                    st = i.plugin_status(e, check_remote=True)["state"]
                    res[e["id"]] = st
            state["remote_status"] = res
            i.log("Update check finished.")
        run_task("Checking for plugin updates", fn)

    def plug_install():
        sel = need_selection()
        if sel:
            run_task("Installing / updating plugins", lambda i: [
                i.update_plugin_entry(e) if e.get("installed") else i.install_plugin(e) for e in sel],
                needs_closed=True)

    def plug_enable(flag: bool):
        sel = need_selection()
        if sel:
            run_task(("Enabling" if flag else "Disabling") + " plugins",
                     lambda i: [i.set_enabled(e, flag) for e in sel if e.get("installed")])

    def plug_reqs():
        sel = need_selection()
        if sel:
            run_task("Reinstalling Python packages", lambda i: [
                i.install_requirements_dir(Path(e["path"]), e["name"], force_log=True)
                for e in sel if e.get("installed") and e["kind"] != "file"], needs_closed=True)

    def plug_remove():
        sel = [e for e in (need_selection() or []) if e.get("installed")]
        if sel and messagebox.askyesno(APP_NAME, "Remove " + ", ".join(e["name"] for e in sel) +
                                       "?\nA copy is kept in the standrig_backups folder."):
            run_task("Removing plugins", lambda i: [i.remove_entry(e) for e in sel])

    def plug_open():
        sel = selected_entries()
        if sel and sel[0].get("path"):
            open_path(Path(sel[0]["path"]))
        elif state["env"].base:
            open_path(state["env"].custom_nodes)

    for text, cmd in (("Check for updates", plug_check_updates), ("Install / update", plug_install),
                      ("Enable", lambda: plug_enable(True)), ("Disable", lambda: plug_enable(False)),
                      ("Reinstall Python packages", plug_reqs), ("Remove", plug_remove),
                      ("Open folder", plug_open), ("Rescan", lambda: refresh_tables())):
        ttk.Button(prow, text=text, command=cmd).pack(side="left", padx=(0, 6))
    addrow = ttk.LabelFrame(t_plug, text="Add any custom node from GitHub", padding=8)
    addrow.pack(fill="x")
    url_var = tk.StringVar()
    ttk.Entry(addrow, textvariable=url_var).pack(side="left", fill="x", expand=True)

    def add_custom():
        p = parse_github_url(url_var.get())
        if not p:
            messagebox.showerror(APP_NAME, "Paste a GitHub URL like https://github.com/owner/repo")
            return
        lst = [x for x in settings.get("custom_plugins", []) if x["id"] != p["id"]] + [p]
        settings["custom_plugins"] = lst
        settings.save()
        url_var.set("")
        refresh_tables()
        log(f"Added {p['repo']} to the plugin list; select it and press Install / update.")

    def forget_custom():
        ids = set(ptree.selection())
        settings["custom_plugins"] = [x for x in settings.get("custom_plugins", []) if x["id"] not in ids]
        settings.save()
        refresh_tables()

    ttk.Button(addrow, text="Add to list", command=add_custom).pack(side="left", padx=6)
    ttk.Button(addrow, text="Forget selected custom", command=forget_custom).pack(side="left")

    # ------------------------------------------------------------ tab: models
    t_mod = ttk.Frame(nb, padding=10)
    nb.add(t_mod, text="  Models  ")
    mcols = ("name", "file", "folder", "size", "status")
    mtree = ttk.Treeview(t_mod, columns=mcols, show="headings", selectmode="extended", height=10)
    for c, w in zip(mcols, (250, 360, 130, 90, 120)):
        mtree.heading(c, text=c.capitalize())
        mtree.column(c, width=w, anchor="w")
    mtree.pack(fill="both", expand=True)
    mrow = ttk.Frame(t_mod)
    mrow.pack(fill="x", pady=6)
    sizes: Dict[str, Optional[int]] = {}

    def models_by_ids(ids) -> List[dict]:
        inst = Installer(state["env"], state["catalog"], settings, log, progress)
        return [m for m in inst.model_list() if m["id"] in ids]

    def mod_sizes():
        def fn(i: Installer):
            for m in i.model_list():
                sizes[m["id"]] = remote_size(m["url"], settings)
            total = sum(v for v in sizes.values() if v)
            log(f"Total download size: {fmt_size(total)}")
        run_task("Checking model sizes", fn)

    def mod_download(only_missing: bool):
        if only_missing:
            inst = Installer(state["env"], state["catalog"], settings, log, progress)
            sel = [m for m in inst.model_list() if inst.find_model(m) is None]
        else:
            sel = models_by_ids(set(mtree.selection()))
        if not sel:
            messagebox.showinfo(APP_NAME, "Nothing to download.")
            return
        run_task("Downloading models", lambda i: [i.download_model(m) for m in sel])

    ttk.Button(mrow, text="Check sizes", command=mod_sizes).pack(side="left")
    ttk.Button(mrow, text="Download selected", command=lambda: mod_download(False)).pack(side="left", padx=6)
    ttk.Button(mrow, text="Download all missing", command=lambda: mod_download(True)).pack(side="left")
    ttk.Button(mrow, text="Open models folder",
               command=lambda: state["env"].base and open_path(state["env"].models)).pack(side="left", padx=6)
    ttk.Label(t_mod, wraplength=900, justify="left", text=(
        "New files go to the 'Models download to' folder shown above (Comfy Desktop's DOWNLOADS folder). A model that "
        "already exists in any of your model folders counts as installed and is not downloaded again. Interrupted "
        "downloads are kept as .part files and resume. See-through downloads its own models the first time it runs.")).pack(anchor="w")

    # ----------------------------------------------------- tab: model library
    t_lib = ttk.Frame(nb, padding=10)
    nb.add(t_lib, text="  Model library  ")
    lib_top = ttk.Frame(t_lib)
    lib_top.pack(fill="x")
    lib_sum = tk.StringVar(value="Press Rescan to list every model file in your model folders.")
    ttk.Label(lib_top, textvariable=lib_sum).pack(side="left")
    filt_var = tk.StringVar()
    fe = ttk.Entry(lib_top, textvariable=filt_var, width=28)
    fe.pack(side="right")
    ttk.Label(lib_top, text="Filter:").pack(side="right", padx=4)
    lf = ttk.Frame(t_lib)
    lf.pack(fill="both", expand=True, pady=6)
    lcols = ("name", "type", "size", "modified", "root")
    ltree = ttk.Treeview(lf, columns=lcols, show="headings", selectmode="extended", height=11)
    for c, w in zip(lcols, (340, 140, 90, 130, 330)):
        ltree.heading(c, text=c.capitalize(), command=lambda c=c: sort_lib(c))
        ltree.column(c, width=w, anchor="w")
    lsb = ttk.Scrollbar(lf, orient="vertical", command=ltree.yview)
    ltree.configure(yscrollcommand=lsb.set)
    ltree.pack(side="left", fill="both", expand=True)
    lsb.pack(side="right", fill="y")
    lib_rows: List[dict] = []
    lib_sort = {"key": "type", "rev": False}

    def fill_lib():
        q = filt_var.get().strip().lower()
        ltree.delete(*ltree.get_children())
        key = lib_sort["key"]
        rows = sorted(lib_rows, key=lambda r: (r["size"] if key == "size" else r["mtime"] if key == "modified"
                                               else str(r.get(key, "")).lower()), reverse=lib_sort["rev"])
        shown = tot = 0
        for r in rows:
            if q and q not in r["name"].lower() and q not in r["type"].lower():
                continue
            shown += 1
            tot += r["size"]
            ltree.insert("", "end", iid=r["path"], values=(
                r["name"] + ("  (partial)" if r["partial"] else ""), r["type"], fmt_size(r["size"]),
                time.strftime("%Y-%m-%d", time.localtime(r["mtime"])), r["root"]))
        all_tot = sum(r["size"] for r in lib_rows)
        lib_sum.set(f"{len(lib_rows)} files, {fmt_size(all_tot)} total" +
                    (f"  ·  showing {shown} ({fmt_size(tot)})" if q else ""))

    def sort_lib(col):
        lib_sort["rev"] = not lib_sort["rev"] if lib_sort["key"] == col else col in ("size", "modified")
        lib_sort["key"] = col
        fill_lib()

    filt_var.trace_add("write", lambda *_: fill_lib())

    def on_tab(_e=None):
        try:
            if nb.select() == str(t_lib) and not lib_rows and not state["busy"] and state["env"].base:
                lib_rescan()
        except Exception:
            pass

    nb.bind("<<NotebookTabChanged>>", on_tab)

    def lib_rescan():
        def fn(i: Installer):
            rows = i.scan_models()
            lib_rows[:] = rows
            i.log(f"Model library: {len(rows)} files, {fmt_size(sum(r['size'] for r in rows))} in "
                  f"{len(i.env.all_model_dirs())} folders")
        run_task("Scanning model folders", fn, after=fill_lib)

    def lib_selected() -> List[str]:
        sel = list(ltree.selection())
        if not sel:
            messagebox.showinfo(APP_NAME, "Select one or more model files.")
        return sel

    def lib_delete():
        sel = lib_selected()
        if not sel:
            return
        size = sum(r["size"] for r in lib_rows if r["path"] in sel)
        where = "to the Recycle Bin" if IS_WIN else "permanently"
        if messagebox.askyesno(APP_NAME, f"Delete {len(sel)} file(s) ({fmt_size(size)}) {where}?\n\n" +
                               "\n".join(Path(x).name for x in sel[:12]) + ("\n…" if len(sel) > 12 else "")):
            run_task("Deleting models", lambda i: [i.delete_model(x) for x in sel], after=lib_rescan)

    def pick_root(title: str) -> Optional[Path]:
        roots = [str(d) for d in state["env"].all_model_dirs()]
        win = tk.Toplevel(root)
        win.title(title)
        win.transient(root)
        ttk.Label(win, text="Destination models folder (the type subfolder is kept):", padding=10).pack(anchor="w")
        var = tk.StringVar(value=roots[0] if roots else "")
        cb = ttk.Combobox(win, textvariable=var, values=roots, width=80)
        cb.pack(padx=10, fill="x")
        res: Dict[str, Optional[str]] = {"v": None}

        def browse_dest():
            d = filedialog.askdirectory(parent=win)
            if d:
                var.set(d)

        def ok():
            res["v"] = var.get().strip() or None
            win.destroy()

        row = ttk.Frame(win, padding=10)
        row.pack(fill="x")
        ttk.Button(row, text="Browse…", command=browse_dest).pack(side="left")
        ttk.Button(row, text="Cancel", command=win.destroy).pack(side="right")
        ttk.Button(row, text="Move here", command=ok).pack(side="right", padx=6)
        win.grab_set()
        root.wait_window(win)
        return Path(res["v"]) if res["v"] else None

    def lib_move():
        sel = lib_selected()
        if not sel:
            return
        dest = pick_root("Move models")
        if dest:
            run_task("Moving models", lambda i: [i.move_model(x, dest) for x in sel], after=lib_rescan)

    def lib_open():
        sel = list(ltree.selection())
        open_path(Path(sel[0]).parent if sel else state["env"].models)

    lrow = ttk.Frame(t_lib)
    lrow.pack(fill="x")
    for text, cmd in (("Rescan", lib_rescan), ("Delete", lib_delete), ("Move to…", lib_move),
                      ("Open folder", lib_open)):
        ttk.Button(lrow, text=text, command=cmd).pack(side="left", padx=(0, 6))

    dl = ttk.LabelFrame(t_lib, text="Download any model (Hugging Face page or direct link)", padding=8)
    dl.pack(fill="x", pady=(8, 0))
    dl_url, dl_type, dl_root, dl_name = tk.StringVar(), tk.StringVar(value="loras"), tk.StringVar(), tk.StringVar()
    ttk.Label(dl, text="URL").grid(row=0, column=0, sticky="w")
    ttk.Entry(dl, textvariable=dl_url).grid(row=0, column=1, columnspan=5, sticky="we", padx=6, pady=2)
    ttk.Label(dl, text="Type folder").grid(row=1, column=0, sticky="w")
    ttk.Combobox(dl, textvariable=dl_type, values=MODEL_TYPES, width=20).grid(row=1, column=1, sticky="w", padx=6)
    ttk.Label(dl, text="Into").grid(row=1, column=2, sticky="e")
    dl_root_box = ttk.Combobox(dl, textvariable=dl_root, width=55)
    dl_root_box.grid(row=1, column=3, sticky="we", padx=6)
    ttk.Label(dl, text="File name (optional)").grid(row=1, column=4, sticky="e")
    ttk.Entry(dl, textvariable=dl_name, width=26).grid(row=1, column=5, sticky="we", padx=6)
    dl.columnconfigure(3, weight=1)

    def dl_go():
        url = dl_url.get().strip()
        if not re.match(r"^https?://", url):
            messagebox.showerror(APP_NAME, "Paste a full https:// link to the model file.")
            return
        dest = Path(dl_root.get().strip() or state["env"].models)
        tf, fn_ = dl_type.get().strip() or "checkpoints", dl_name.get()
        run_task("Downloading model", lambda i: i.download_url(url, tf, dest, fn_), after=lib_rescan)

    ttk.Button(dl, text="Download", command=dl_go).grid(row=0, column=6, rowspan=2, padx=(6, 0))

    # ---------------------------------------------------------- tab: settings
    t_set = ttk.Frame(nb, padding=14)
    nb.add(t_set, text="  Settings  ")
    form = ttk.Frame(t_set)
    form.pack(fill="x")
    entries = {}
    for r, (key, label, secret) in enumerate([
            ("hf_token", "Hugging Face token (optional, faster/gated downloads)", True),
            ("github_token", "GitHub token (optional, avoids API rate limits)", True),
            ("desktop_app", "ComfyUI Desktop app folder (optional override)", False),
            ("catalog_url", "Catalog URL (advanced)", False)]):
        ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", pady=3)
        v = tk.StringVar(value=settings.get(key, ""))
        e = ttk.Entry(form, textvariable=v, show="•" if secret else "", width=70)
        e.grid(row=r, column=1, sticky="we", padx=8)
        entries[key] = v
    form.columnconfigure(1, weight=1)

    def save_settings():
        for k, v in entries.items():
            val = v.get().strip()
            if val:
                settings[k] = val
            else:
                settings.pop(k, None)
        settings.save()
        log("Settings saved.")
        refresh_env()

    ttk.Button(t_set, text="Save settings", command=save_settings).pack(anchor="w", pady=8)
    t_fold = ttk.Frame(nb, padding=14)
    nb.insert(t_set, t_fold, text="  Folders  ")
    ttk.Label(t_fold, wraplength=980, justify="left", text=(
        "Point the tool at any folder layout. Leave a box empty to use what was detected (shown in grey). "
        "Saved on this PC, so changes apply immediately without a new version of the tool.")).pack(anchor="w")
    fold = ttk.LabelFrame(t_fold, text="Folders", padding=10)
    fold.pack(fill="x", pady=8)
    fvars: Dict[str, tk.StringVar] = {}
    fhints: Dict[str, tk.StringVar] = {}
    ov0 = settings.get("folders") or {}
    for r, (key, label) in enumerate(FOLDER_KEYS.items()):
        ttk.Label(fold, text=label).grid(row=2 * r, column=0, sticky="w", pady=(4, 0))
        v = tk.StringVar(value=ov0.get(key, ""))
        ttk.Entry(fold, textvariable=v).grid(row=2 * r, column=1, sticky="we", padx=6, pady=(4, 0))
        is_file = key in ("python", "uv")

        def browse_f(v=v, is_file=is_file):
            d = filedialog.askopenfilename() if is_file else filedialog.askdirectory()
            if d:
                v.set(d)

        ttk.Button(fold, text="Browse…", command=browse_f).grid(row=2 * r, column=2, pady=(4, 0))
        h = tk.StringVar()
        ttk.Label(fold, textvariable=h, foreground="#666").grid(row=2 * r + 1, column=1, sticky="w", padx=6)
        fvars[key], fhints[key] = v, h
    ttk.Label(fold, text="Extra model folders (one per line)").grid(row=20, column=0, sticky="nw", pady=(6, 0))
    extra_txt = tk.Text(fold, height=3, width=60, font=("Consolas", 9))
    extra_txt.grid(row=20, column=1, sticky="we", padx=6, pady=(6, 0))
    extra_txt.insert("1.0", "\n".join(ov0.get("extra_model_dirs") or []))

    def add_extra():
        d = filedialog.askdirectory()
        if d:
            cur = extra_txt.get("1.0", "end").strip()
            extra_txt.delete("1.0", "end")
            extra_txt.insert("1.0", (cur + "\n" if cur else "") + d)

    ttk.Button(fold, text="Add…", command=add_extra).grid(row=20, column=2, sticky="n", pady=(6, 0))
    fold.columnconfigure(1, weight=1)

    def save_folders():
        ov = {k: v.get().strip() for k, v in fvars.items() if v.get().strip()}
        extras = [l.strip() for l in extra_txt.get("1.0", "end").splitlines() if l.strip()]
        if extras:
            ov["extra_model_dirs"] = extras
        settings["folders"] = ov
        settings.save()
        log("Folders saved." + (" Using: " + ", ".join(ov) if ov else " Using auto-detection."))
        refresh_env()

    def reset_folders():
        for v in fvars.values():
            v.set("")
        extra_txt.delete("1.0", "end")
        save_folders()

    frow = ttk.Frame(fold)
    frow.grid(row=21, column=1, sticky="w", padx=6, pady=(8, 0))
    ttk.Button(frow, text="Save folders", command=save_folders).pack(side="left")
    ttk.Button(frow, text="Reset to auto-detect", command=reset_folders).pack(side="left", padx=6)
    state["folder_hints"] = fhints
    upd = ttk.LabelFrame(t_set, text="This tool", padding=10)
    upd.pack(fill="x", pady=8)
    tool_var = tk.StringVar(value=f"Version {VERSION}")
    ttk.Label(upd, textvariable=tool_var).pack(anchor="w")

    def do_self_update():
        inst = Installer(state["env"], state["catalog"], settings, log, progress)
        latest = inst.tool_update_available()
        if not latest:
            messagebox.showinfo(APP_NAME, f"You have the latest version ({VERSION}).")
            return
        try:
            target = inst.self_update()
            messagebox.showinfo(APP_NAME, f"Updated to {latest}. The tool will now restart.")
            if getattr(sys, "frozen", False):
                subprocess.Popen([sys.executable], close_fds=True)
                root.destroy()
                sys.exit(0)
            os.execv(sys.executable, [sys.executable, str(target)])
        except Exception as e:
            messagebox.showerror(APP_NAME, f"Update failed: {e}")

    ttk.Button(upd, text="Check for tool update", command=do_self_update).pack(anchor="w", pady=(6, 0))
    ttk.Button(upd, text="Reload catalog", command=lambda: load_catalog_async()).pack(anchor="w", pady=(6, 0))

    # --------------------------------------------------------------- refresh
    def refresh_tables():
        cat = state["catalog"]
        env = state["env"]
        if not cat:
            return
        inst = Installer(env, cat, settings, log, progress)
        remote = state.get("remote_status", {})
        sel = ptree.selection()
        ptree.delete(*ptree.get_children())
        plugin_rows.clear()
        if env.base:
            for e in inst.scan_plugins():
                plugin_rows[e["id"]] = e
                st = remote.get(e["id"]) or inst.describe_plugin(e)
                if remote.get(e["id"]) and e.get("installed") and not e["enabled"]:
                    st = "Disabled · " + st
                ptree.insert("", "end", iid=e["id"], values=(
                    e["name"] + ("  ★" if e.get("required") else ""), st,
                    Path(e["path"]).name if e.get("path") else e["folder"], e.get("description", "")))
        for x in sel:
            if ptree.exists(x):
                ptree.selection_add(x)
        roots = [str(d) for d in env.all_model_dirs()] if env.base else []
        dl_root_box["values"] = roots
        if roots and not dl_root.get():
            dl_root.set(roots[0])
        auto = detect(Settings({k: v for k, v in settings.items() if k != "folders"})) if env.base else None
        hints = state.get("folder_hints") or {}
        if auto is not None:
            wf_auto = (auto.base / cat.get("workflows", {}).get("dest", "")) if auto.base else None
            for key, val in (("custom_nodes", auto.custom_nodes if auto.base else None),
                             ("models_download", auto.models if auto.base else None), ("workflows", wf_auto),
                             ("python", auto.python), ("uv", auto.uv)):
                if key in hints:
                    hints[key].set(f"detected: {val}" if val else "detected: (not found)")
        wsel = wtree.selection()
        wtree.delete(*wtree.get_children())
        for w in cat.get("workflow_list", []):
            st = inst.workflow_status(w)["state"] if env.base else "-"
            wtree.insert("", "end", iid=w["file"], values=(w["name"], st, w.get("description", "")))
        for x in wsel:
            if wtree.exists(x):
                wtree.selection_add(x)
        feats = cat.get("features", [])
        feat_var.set(("What these workflows do:\n" + "\n".join("  • " + f for f in feats)) if feats else "")
        msel = mtree.selection()
        mtree.delete(*mtree.get_children())
        for m in inst.model_list():
            mtree.insert("", "end", iid=m["id"], values=(m["name"], m["file"], m["dir"],
                                                          fmt_size(sizes.get(m["id"])) if m["id"] in sizes else "…",
                                                          inst.model_state(m) if env.base else "-"))
        for s in msel:
            if mtree.exists(s):
                mtree.selection_add(s)
        mv = cat.get("model_variants", {}).get("qwen_edit")
        if mv:
            variant_box["values"] = [o["label"] for o in mv["options"]]
            cur = settings.get("qwen_variant") or mv.get("default")
            variant_var.set(next((o["label"] for o in mv["options"] if o["id"] == cur), mv["options"][0]["label"]))
        latest = cat.get("installer", {}).get("version")
        if latest and version_tuple(latest) > version_tuple(VERSION):
            tool_var.set(f"Version {VERSION}. Update available: {latest}")

    def refresh_env():
        env = detect(settings)
        state["env"] = env
        state.pop("remote_status", None)
        base_var.set(str(env.base) if env.base else "")
        parts = []
        if env.base:
            parts.append({"desktop2": "Comfy Desktop", "desktop": "ComfyUI Desktop (legacy)", "portable": "Portable",
                          "manual": "Manual install", "custom": "Custom folders",
                          "unknown": "ComfyUI folder"}.get(env.kind, env.kind) +
                         (" (folders customised)" if env.overridden else "") +
                         (f" · install '{env.instance_name}'" if env.instance_name else ""))
            parts.append(f"ComfyUI core {env.core_version}" if env.core else "core: not found")
            parts.append("Python: found" if env.python else "Python: not found")
            if env.uv:
                parts.append("uv: found")
        if env.base:
            parts.append(f"Custom nodes → {env.custom_nodes}")
            parts.append(f"Models download to → {env.models}")
        info_var.set("\n".join(["   •   ".join(parts[:-2]), *parts[-2:]]) if env.base else "; ".join(env.notes))
        if len(env.instances) > 1:
            labels = [f"{i['name']}  ({i['installPath']})" for i in env.instances]
            inst_box["values"] = labels
            cur = next((f"{i['name']}  ({i['installPath']})" for i in env.instances if i["id"] == env.instance_id), labels[0])
            inst_var.set(cur)
            inst_row.grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))
        else:
            inst_row.grid_remove()
        warns = []
        if env.base and not env.python:
            warns.append("ComfyUI's Python environment was not found (start ComfyUI once to finish its setup), "
                         "so Python packages for plugins cannot be installed.")
        missing = core_missing(env, state["catalog"]) if state["catalog"] else []
        if missing:
            warns.append(state["catalog"].get("comfyui", {}).get("update_hint", "Update ComfyUI.") +
                         f"  (missing: {', '.join(Path(m).name for m in missing)})")
        warn_var.set("\n".join(warns))
        refresh_tables()

    def load_catalog_async():
        def work():
            try:
                cat = load_catalog(settings, log)
                logq.put(("catalog", cat))
            except Exception as e:
                log(f"ERROR loading catalog: {e}")
        threading.Thread(target=work, daemon=True).start()

    log(f"{APP_NAME} {VERSION}. Log file: {log_file}")
    refresh_env()
    load_catalog_async()
    root.after(100, pump)
    root.mainloop()


# ========================================================================= CLI

def run_cli(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=f"{APP_NAME} (command line)")
    ap.add_argument("--base", help="ComfyUI folder (default: auto-detect)")
    ap.add_argument("--no-plugins", action="store_true")
    ap.add_argument("--no-workflows", action="store_true")
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--no-requirements", action="store_true", help="skip pip installs")
    ap.add_argument("--qwen", choices=["int8", "bf16"])
    ap.add_argument("--instance", help="Comfy Desktop install id or name")
    ap.add_argument("--status", action="store_true", help="only print what is installed")
    a = ap.parse_args(argv)
    s = Settings.load()
    if a.base:
        s["base_path"] = a.base
    if a.qwen:
        s["qwen_variant"] = a.qwen
    cat = load_catalog(s, print)
    env = detect(s)
    if a.instance and env.instances:
        pick = next((i for i in env.instances if a.instance in (i["id"], i["name"])), None)
        if pick:
            s["instance_id"] = pick["id"]
            env = detect(s)
    print(f"ComfyUI: {env.base} ({env.kind}{', ' + str(env.instance_name) if env.instance_name else ''}), "
          f"core {env.core_version}, python {env.python}, uv {env.uv}")
    print(f"  custom nodes -> {env.custom_nodes}\n  models download -> {env.models}\n  model folders: "
          + "; ".join(str(d) for d in env.all_model_dirs()))
    for m in core_missing(env, cat):
        print("  missing core file:", m)
    if not env.base:
        print("ComfyUI folder not found; pass --base")
        return 2
    inst = Installer(env, cat, s, print, lambda l, f, t: None)
    if a.status:
        for p in all_plugins(cat, s):
            print(f"  plugin {p['name']}: {inst.plugin_status(p, check_remote=True)['state']}")
        for m in inst.model_list():
            print(f"  model  {m['file']}: {inst.model_state(m)}")
        return 0
    if a.no_requirements:
        inst.install_requirements = lambda p: print(f"[{p['name']}] (requirements skipped)")  # type: ignore
    inst.quick_setup(not a.no_plugins, not a.no_workflows, not a.no_models)
    return 0


def _run_newer_script_if_any() -> bool:
    """For the .exe: if Settings → Check for tool update saved a newer script, run that instead."""
    if not getattr(sys, "frozen", False) or os.environ.get("STANDRIG_NO_REDIRECT"):
        return False
    script = tool_dir() / "app" / "standrig_setup.py"
    if not script.exists():
        return False
    m = re.search(r'^VERSION = "([^"]+)"', script.read_text(encoding="utf-8", errors="replace"), re.M)
    if not m or version_tuple(m.group(1)) <= version_tuple(VERSION):
        return False
    os.environ["STANDRIG_NO_REDIRECT"] = "1"
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit:
        pass
    return True


if __name__ == "__main__":
    if _run_newer_script_if_any():
        sys.exit(0)
    if len(sys.argv) > 1:
        sys.exit(run_cli(sys.argv[1:]))
    run_gui()
