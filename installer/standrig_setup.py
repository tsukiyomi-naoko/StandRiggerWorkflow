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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

APP_NAME = "StandRig Setup"
VERSION = "1.0.0"
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

    @property
    def custom_nodes(self) -> Path:
        return self.base / "custom_nodes"

    @property
    def models(self) -> Path:
        return self.base / "models"


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


def detect(settings: Settings) -> ComfyEnv:
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
        d = self.env.base / "standrig_backups"
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
        if self.env.uv and self.env.kind == "desktop":
            return [str(self.env.uv), "pip", *args, "--python", str(py)]
        return [str(py), "-m", "pip", *args]

    def install_requirements(self, p: dict) -> None:
        req = self.plugin_dir(p) / "requirements.txt"
        if not req.exists():
            return
        lines = [l.strip() for l in req.read_text(encoding="utf-8", errors="replace").splitlines()]
        if not any(l and not l.startswith("#") for l in lines):
            return
        cmd = self.pip_command(["install", "-r", str(req)])
        self.log(f"[{p['name']}] installing Python requirements…")
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
        dest = self.env.base / wf["dest"]
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

    def model_state(self, m: dict) -> str:
        p = self.model_path(m)
        if p.exists():
            return "Installed"
        part = p.with_name(p.name + ".part")
        if part.exists():
            return f"Partial ({fmt_size(part.stat().st_size)})"
        return "Missing"

    def download_model(self, m: dict) -> None:
        dest = self.model_path(m)
        if dest.exists():
            self.log(f"[{m['name']}] already present")
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
        me = Path(__file__).resolve()
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
        row=3, column=0, columnspan=4, sticky="w")

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

    # ----------------------------------------------------------- tab: plugins
    t_plug = ttk.Frame(nb, padding=10)
    nb.add(t_plug, text="  Plugins  ")
    cols = ("name", "status", "required", "description")
    ptree = ttk.Treeview(t_plug, columns=cols, show="headings", selectmode="extended", height=10)
    for c, w in zip(cols, (220, 250, 80, 480)):
        ptree.heading(c, text=c.capitalize())
        ptree.column(c, width=w, anchor="w")
    ptree.pack(fill="both", expand=True)
    prow = ttk.Frame(t_plug)
    prow.pack(fill="x", pady=6)

    def selected_plugins() -> List[dict]:
        ids = set(ptree.selection())
        return [p for p in all_plugins(state["catalog"], settings) if p["id"] in ids]

    def plug_check_updates():
        def fn(i: Installer):
            res = {p["id"]: i.plugin_status(p, check_remote=True)["state"]
                   for p in all_plugins(state["catalog"], settings)}
            state["remote_status"] = res
        run_task("Checking for plugin updates", fn)

    def plug_install():
        sel = selected_plugins()
        if not sel:
            messagebox.showinfo(APP_NAME, "Select one or more plugins in the list.")
            return
        run_task("Installing plugins", lambda i: [i.install_plugin(p) for p in sel], needs_closed=True)

    def plug_remove():
        sel = selected_plugins()
        if sel and messagebox.askyesno(APP_NAME, "Remove " + ", ".join(p["name"] for p in sel) +
                                       "?\nA copy is kept in standrig_backups."):
            run_task("Removing plugins", lambda i: [i.remove_plugin(p) for p in sel])

    def plug_open():
        sel = selected_plugins()
        if sel:
            open_path(state["env"].custom_nodes / sel[0]["folder"])
        elif state["env"].base:
            open_path(state["env"].custom_nodes)

    ttk.Button(prow, text="Check for updates", command=plug_check_updates).pack(side="left")
    ttk.Button(prow, text="Install / update selected", command=plug_install).pack(side="left", padx=6)
    ttk.Button(prow, text="Remove selected", command=plug_remove).pack(side="left")
    ttk.Button(prow, text="Open folder", command=plug_open).pack(side="left", padx=6)
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
        log(f"Added {p['repo']} to the plugin list; select it and press Install.")

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
            sel = [m for m in inst.model_list() if not inst.model_path(m).exists()]
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
        "Files go to <ComfyUI folder>\\models\\<folder>. Interrupted downloads are kept as .part files and resume. "
        "See-through downloads its own models from Hugging Face the first time its nodes run.")).pack(anchor="w")

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
        if getattr(sys, "frozen", False):
            webbrowser.open(state["catalog"]["installer"].get("release_page", ""))
            return
        try:
            inst.self_update()
            messagebox.showinfo(APP_NAME, f"Updated to {latest}. The tool will now restart.")
            os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())])
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
        for p in all_plugins(cat, settings):
            st = remote.get(p["id"]) or (inst.plugin_status(p)["state"] if env.base else "-")
            ptree.insert("", "end", iid=p["id"], values=(p["name"], st, "yes" if p.get("required") else "optional",
                                                          p.get("description", "")))
        for s in sel:
            if ptree.exists(s):
                ptree.selection_add(s)
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
            parts.append(f"Type: {env.kind}")
            parts.append(f"ComfyUI core {env.core_version}" if env.core else "core: not found")
            parts.append(f"Python: {env.python}" if env.python else "Python: not found")
            if env.uv:
                parts.append("uv: found")
        info_var.set("   •   ".join(parts) if parts else "; ".join(env.notes))
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
    ap.add_argument("--status", action="store_true", help="only print what is installed")
    a = ap.parse_args(argv)
    s = Settings.load()
    if a.base:
        s["base_path"] = a.base
    if a.qwen:
        s["qwen_variant"] = a.qwen
    cat = load_catalog(s, print)
    env = detect(s)
    print(f"ComfyUI: {env.base} ({env.kind}), core {env.core_version}, python {env.python}, uv {env.uv}")
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


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(run_cli(sys.argv[1:]))
    run_gui()
