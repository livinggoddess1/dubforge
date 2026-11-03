"""DubForge in the browser: start dubs, watch progress, play results, edit and re-voice lines.

    dubforge web -c config.yaml            # http://127.0.0.1:8765 on this Mac
    dubforge web -c config.yaml --share    # plus a private link for friends anywhere

Each dub runs as an ordinary `dubforge run` in a background process, one at a time (the GPU can
only do one well), so the browser can be closed and reopened and the pipeline behaves exactly as
it does in Terminal. Jobs and uploads live in the data folder (default ./web_data).
"""


import csv
import json
import os
import queue
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from dubforge import audio, pipeline
from dubforge.config import load_config

STATIC = Path(__file__).with_name("static")
VIDEO_EXT = {".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v", ".ts"}
SUB_EXT = {".srt", ".ass", ".ssa", ".vtt"}
NOISE = re.compile(r"init param, map:|Warning, miss key in ckpt|^\s*\d+%\||it/s\]|s/it\]")
STAGE = re.compile(r"^\[(\d+)/(\d+)\] (\w+)")


def _safe_name(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\- ]+", "_", name).strip(" .") or "video"
    return name[:120]


# ---- running one job (on this Mac, or on a cloud GPU) -----------------------------------------

def build_command(job: dict, config_path: str | None) -> list[str]:
    cmd = [sys.executable, "-m", "dubforge", "run", job["video"]]
    if config_path:
        cmd += ["-c", config_path]
    if job.get("clip"):
        cmd += ["--clip", f"{job['clip'][0]}:{job['clip'][1]}"]
    for key, flag in (("style", "--style"), ("tts", "--tts"), ("subs", "--subs"),
                      ("timing", "--timing"), ("from_stage", "--from")):
        if job.get(key):
            cmd += [flag, str(job[key])]
    if job.get("audio_stream") is not None:
        cmd += ["--audio-stream", str(job["audio_stream"])]
    return cmd


def read_log(path, limit: int = 200) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return [l for l in lines if not NOISE.search(l)][-limit:]


def last_error(log_path) -> str:
    lines = [l for l in read_log(log_path, 400) if l.strip()]
    for i in range(len(lines) - 1, -1, -1):  # the last exception or SystemExit message
        if re.match(r"^\w*(Error|Exception|Exit)\b|^[\w.]+(Error|Exception): ", lines[i]):
            return lines[i]
    return lines[-1] if lines else "failed"


def context_for(job: dict, config_path: str | None, cwd: Path) -> pipeline.Context:
    here = os.getcwd()
    try:  # relative paths in config.yaml (work_root, show_bible…) are relative to where DubForge runs
        os.chdir(cwd)
        return pipeline.Context(job["video"], load_config(config_path),
                                clip=tuple(job["clip"]) if job.get("clip") else None)
    finally:
        os.chdir(here)


def execute(job: dict, config_path: str | None, cwd: Path, on_start=None) -> dict:
    """Run one dub as `dubforge run …` with its output in the job's log, then make the browser
    previews. Returns the fields to record on the job. Used on the Mac and on cloud GPUs alike."""
    import dubforge

    src = str(Path(dubforge.__file__).resolve().parents[1])  # so the run finds this copy of DubForge
    env = dict(os.environ, PYTHONUNBUFFERED="1",
               PYTHONPATH=os.pathsep.join(p for p in (src, os.environ.get("PYTHONPATH")) if p))
    cmd = build_command(job, config_path)
    Path(job["log"]).parent.mkdir(parents=True, exist_ok=True)
    with open(job["log"], "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL)
        if on_start:
            on_start(proc)
        rc = proc.wait()
    if rc != 0:
        return {"status": "failed", "returncode": rc, "finished": time.time(), "error": last_error(job["log"])}
    result = {"status": "done", "returncode": 0}
    try:
        make_previews(context_for(job, config_path, cwd))
    except Exception as e:
        result["preview_error"] = f"{type(e).__name__}: {e}"
    result["finished"] = time.time()
    return result


def write_status(data_dir: Path, job_id: str, fields: dict) -> None:
    """A cloud run reports back through a small file next to its log (the web side reads it)."""
    d = Path(data_dir) / "status"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{job_id}.tmp"
    tmp.write_text(json.dumps(fields))
    tmp.replace(d / f"{job_id}.json")


FINAL = ("done", "failed", "cancelled")


# ---- jobs ------------------------------------------------------------------------------------

class JobManager:
    """The list of dubs, persisted to jobs.json.

    On a Mac, dubs run one at a time in a background process here. In the cloud (`cloud` given),
    each dub is handed to a GPU worker; this side only records jobs and reads their progress from
    the shared storage the worker writes to.
    """

    def __init__(self, data_dir: Path, config_path: str | None, cwd: Path, cloud=None):
        self.dir, self.config_path, self.cwd, self.cloud = data_dir, config_path, cwd, cloud
        (self.dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.dir / "uploads").mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "jobs.json"
        self.lock = threading.RLock()
        self.io_lock = threading.Lock()  # cloud storage can't refresh while a file is being written
        self.jobs: dict[str, dict] = json.loads(self.file.read_text()) if self.file.exists() else {}
        self.q: queue.Queue[str] = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self.current: str | None = None
        self._synced = 0.0
        if cloud is None:
            for j in sorted(self.jobs.values(), key=lambda j: j["created"]):
                if j["status"] in ("running", "previewing"):
                    j.update(status="failed", error="DubForge was stopped while this was running. Start it again.")
                elif j["status"] == "queued":
                    self.q.put(j["id"])
            self._save()
            threading.Thread(target=self._worker, daemon=True).start()

    # persistence
    def _save(self) -> None:
        with self.lock:
            tmp = self.file.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.jobs, indent=1))
            tmp.replace(self.file)
        self.persist()

    def persist(self) -> None:
        """Make what this side wrote visible to cloud workers (no-op on a Mac)."""
        if self.cloud:
            with self.io_lock:
                self.cloud.commit()

    def sync(self, force: bool = False) -> None:
        """Cloud: pick up progress written by GPU workers (at most every 2 s)."""
        if not self.cloud or (not force and time.time() - self._synced < 2):
            return
        if self.io_lock.acquire(blocking=False):
            try:
                self.cloud.reload()
            except Exception:
                pass  # a file is open right now; the next poll will catch up
            finally:
                self.io_lock.release()
        self._synced = time.time()
        changed = False
        with self.lock:
            for j in self.jobs.values():
                if j["status"] in FINAL:
                    continue
                f = self.dir / "status" / f"{j['id']}.json"
                if f.exists():
                    try:
                        fields = json.loads(f.read_text())
                    except ValueError:
                        continue
                    if any(j.get(k) != v for k, v in fields.items()):
                        j.update(fields)
                        changed = True
        if changed:
            self._save()

    def cfg(self) -> dict:
        here = os.getcwd()
        try:
            os.chdir(self.cwd)
            return load_config(self.config_path)
        finally:
            os.chdir(here)

    def context(self, job: dict) -> pipeline.Context:
        return context_for(job, self.config_path, self.cwd)

    def work_dir(self, job: dict) -> Path:
        return self.context(job).work

    # queueing
    def add(self, opts: dict) -> dict:
        job = {"id": uuid.uuid4().hex[:10], "created": time.time(), "status": "queued", "stage": None,
               "stage_n": 0, "started": None, "finished": None, "error": None, **opts}
        job["log"] = str(self.dir / "logs" / f"{job['id']}.log")
        with self.lock:
            self.jobs[job["id"]] = job
            self._save()
        if self.cloud:
            try:
                job["call_id"] = self.cloud.spawn(job)
            except Exception as e:
                job.update(status="failed", error=f"Couldn't start a cloud GPU: {type(e).__name__}: {e}",
                           finished=time.time())
            self._save()
        else:
            self.q.put(job["id"])
        return job

    def busy_with(self, work: Path) -> bool:
        with self.lock:
            active = [j for j in self.jobs.values() if j["status"] not in FINAL]
        return any(self.work_dir(j) == work for j in active)

    def cancel(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            if self.cloud and job["status"] not in FINAL:
                try:
                    self.cloud.cancel(job.get("call_id"))
                finally:
                    job.update(status="cancelled", finished=time.time())
            elif job["status"] == "queued":
                job.update(status="cancelled", finished=time.time())
            elif job["status"] == "running" and self.proc and self.current == job_id:
                job["cancelling"] = True
                self.proc.send_signal(signal.SIGINT)
            self._save()

    def delete(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job and job["status"] not in FINAL:
                raise RuntimeError("Cancel it first.")
            self.jobs.pop(job_id, None)
            self._save()

    def command(self, job: dict) -> list[str]:
        return build_command(job, self.config_path)

    def _worker(self) -> None:
        while True:
            job_id = self.q.get()
            with self.lock:
                job = self.jobs.get(job_id)
                if not job or job["status"] != "queued":
                    continue
                job.update(status="running", started=time.time())
                self.current = job_id
                self._save()
            try:
                result = execute(job, self.config_path, self.cwd, on_start=lambda p: setattr(self, "proc", p))
                if job.pop("cancelling", False):
                    result = {"status": "cancelled", "finished": time.time()}
            except Exception as e:  # never let one bad job stop the queue
                result = {"status": "failed", "error": f"{type(e).__name__}: {e}", "finished": time.time()}
            finally:
                self.current, self.proc = None, None
            with self.lock:
                job.update(result)
                self._save()

    # progress
    def log_lines(self, job: dict, limit: int = 200) -> list[str]:
        return read_log(job["log"], limit)

    def progress(self, job: dict) -> tuple[int, int, str | None]:
        n, total, name = 0, len(pipeline.STAGES), None
        for line in self.log_lines(job, limit=100000):
            m = STAGE.match(line)
            if m:
                n, total, name = int(m.group(1)), int(m.group(2)), m.group(3)
        return n, total, name

    def last_error(self, job: dict) -> str:
        return last_error(job["log"])


def run_cloud_job(job: dict, config_path: str, data_dir: Path, cwd: Path, commit=lambda: None) -> dict:
    """What a cloud GPU worker does with one job: run it, report progress through shared storage
    (log + status file, committed every few seconds so the page can follow along), report the result."""
    stop = threading.Event()

    def keep_committing():
        while not stop.wait(8):
            try:
                commit()
            except Exception:
                pass
    write_status(data_dir, job["id"], {"status": "running", "started": time.time()})
    commit()
    t = threading.Thread(target=keep_committing, daemon=True)
    t.start()
    try:
        result = execute(job, config_path, cwd)
    except Exception as e:
        result = {"status": "failed", "error": f"{type(e).__name__}: {e}", "finished": time.time()}
    finally:
        stop.set()
        t.join(timeout=10)
    write_status(data_dir, job["id"], result)
    commit()
    return result


# ---- browser previews -----------------------------------------------------------------------

def _has_encoder(name: str) -> bool:
    try:
        out = audio.run_cmd(["ffmpeg", "-hide_banner", "-encoders"])
    except Exception:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if len(line.split()) > 1)


def make_previews(ctx) -> dict:
    """Browsers can't play .mkv (and many anime releases use 10-bit video they can't decode either),
    so make an English and a Japanese .mp4 plus WebVTT subtitles for the player."""
    out = Path(ctx.meta.get("output", ""))
    if not out.exists():
        raise FileNotFoundError("no dubbed file to preview")
    web = ctx.p("web", "x").parent
    en, ja, vtt = web / "preview_en.mp4", web / "preview_ja.mp4", web / "subs.vtt"
    if _has_encoder("h264_videotoolbox"):  # the Mac's hardware encoder: much faster
        vcodec = ["-c:v", "h264_videotoolbox", "-b:v", "5M"]
    else:
        vcodec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    audio.run_cmd(["ffmpeg", "-y", "-v", "error", "-i", out, "-map", "0:v:0", "-map", "0:a:0", *vcodec,
                   "-pix_fmt", "yuv420p", "-vf", "scale='min(1920,iw)':-2", "-c:a", "aac", "-b:a", "160k",
                   "-movflags", "+faststart", en])
    audio.run_cmd(["ffmpeg", "-y", "-v", "error", "-i", en, "-i", out, "-map", "0:v:0", "-map", "1:a:1",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", ja])
    srt = ctx.work / "output" / "dub_script.en.srt"
    if srt.exists():
        audio.run_cmd(["ffmpeg", "-y", "-v", "error", "-i", srt, vtt])
    return {"en": en, "ja": ja, "vtt": vtt}


# ---- the app ---------------------------------------------------------------------------------

FORWARDED = ("cf-connecting-ip", "x-forwarded-for", "x-real-ip", "forwarded")
NO_ACCESS = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DubForge</title><body style="font:16px/1.5 -apple-system,sans-serif;max-width:32rem;margin:15vh auto;padding:0 16px">
<h1 style="font-size:20px">This link isn't valid</h1>
<p>Ask the person running DubForge for the current link. It changes each time they start sharing.</p></body>"""


def create_app(config_path: str | None = None, data_dir: str | Path = "web_data", share_key: str | None = None,
               cwd: str | Path | None = None, cloud=None):
    """cloud: when DubForge runs in the cloud, an object with spawn(job) -> id, cancel(id), commit()
    and reload() (see deploy/modal_app.py), plus an `info` dict shown at the top of the page."""
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    cwd = Path(cwd or os.getcwd()).resolve()
    data = Path(data_dir).expanduser()
    data = (data if data.is_absolute() else cwd / data).resolve()
    if config_path:
        config_path = str((Path(config_path).expanduser() if Path(config_path).is_absolute()
                           else cwd / config_path).resolve())
    jobs = JobManager(data, config_path, cwd, cloud=cloud)
    # private history: each browser sees only the dubs it started (set for the YC/demo site)
    private_history = os.environ.get("DUBFORGE_PRIVATE_HISTORY") == "1"
    app = FastAPI(title="DubForge", docs_url=None, redoc_url=None)
    app.state.jobs = jobs
    # filled in by serve(): the friend link while sharing, and how to stop everything
    app.state.share = {"enabled": bool(share_key), "status": "off", "url": None, "error": None, "changed": 0}
    app.state.quit = None

    def local(request: Request) -> bool:
        """Is this request from someone sitting at this Mac? A tunnel (for friends elsewhere) also
        connects from 127.0.0.1, so anything carrying forwarding headers counts as remote."""
        if cloud is not None:  # in the cloud nobody is "at this Mac": everyone needs the private link
            return False
        host = request.client.host if request.client else ""
        return host in ("127.0.0.1", "::1", "testclient") and not any(h in request.headers for h in FORWARDED)

    @app.middleware("http")
    async def private_link(request: Request, call_next):
        """Friends get in with the private link (…/?key=…); there's nothing to type. The key is then
        kept in a cookie. Without sharing switched on, only this Mac can use DubForge at all."""
        visitor = request.cookies.get("dubforge_visitor") or ""
        new_visitor = not visitor
        if new_visitor:
            visitor = secrets.token_urlsafe(12)
        request.state.visitor = visitor
        if local(request):
            return await call_next(request)
        given = request.query_params.get("key") or request.cookies.get("dubforge_key") or ""
        if not share_key or not secrets.compare_digest(given, share_key):
            return HTMLResponse(NO_ACCESS, status_code=403)
        # The key stays in the address (no redirect): an iPhone Home Screen app doesn't share Safari's
        # cookies, so it needs the link itself to get back in.
        resp = await call_next(request)
        if new_visitor:
            https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
            resp.set_cookie("dubforge_visitor", visitor, max_age=60 * 60 * 24 * 365, httponly=True,
                            samesite="lax", secure=https)
        if request.query_params.get("key"):
            https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
            resp.set_cookie("dubforge_key", share_key, max_age=60 * 60 * 24 * 60, httponly=True,
                            samesite="lax", secure=https)
        return resp

    def get_job(job_id: str) -> dict:
        job = jobs.jobs.get(job_id)
        if not job:
            raise HTTPException(404, "No such job")
        return job

    def public(job: dict) -> dict:
        n, total, name = jobs.progress(job)
        out = {k: job.get(k) for k in ("id", "name", "status", "created", "started", "finished", "error",
                                        "clip", "style", "tts", "subs", "from_stage", "kind", "preview_error")}
        out.update(stage_n=n, stage_total=total, stage=name)
        return out

    # --- pages & status
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/manifest.webmanifest")
    def manifest(request: Request):
        """Lets DubForge be added to an iPhone's Home Screen as an app. A friend's app starts at
        their private link, since Home Screen apps don't share Safari's cookies."""
        given = request.query_params.get("key") or request.cookies.get("dubforge_key")
        start = f"/?key={share_key}" if share_key and not local(request) and given == share_key else "/"
        return JSONResponse({
            "name": "DubForge", "short_name": "DubForge", "start_url": start, "scope": "/",
            "display": "standalone", "background_color": "#141312", "theme_color": "#141312",
            "icons": [{"src": "/static/icon-180.png", "sizes": "180x180", "type": "image/png"},
                      {"src": "/static/icon-512.png", "sizes": "512x512", "type": "image/png"}],
        }, media_type="application/manifest+json")

    @app.get("/api/share")
    def share_status(request: Request):
        """The friend link, for showing (and as a QR code) on this Mac's page only."""
        if not local(request):
            raise HTTPException(403, "Only on the Mac running DubForge")
        st = dict(app.state.share)
        st["link"] = f"{st['url']}/?key={share_key}" if st.get("url") and share_key else None
        return st

    @app.post("/api/quit")
    def quit_app(request: Request):
        if not local(request):
            raise HTTPException(403, "Only on the Mac running DubForge")
        if jobs.current:
            jobs.cancel(jobs.current)
        if app.state.quit:
            threading.Timer(0.5, app.state.quit).start()
        return {"ok": True}

    @app.get("/api/status")
    def status(request: Request):
        cfg = jobs.cfg()
        if cloud is not None:
            return {"local": False, "config": None, "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
                    "style": cfg["adapt"]["style"], **getattr(cloud, "info", {})}
        from dubforge.backends.tts import find_indextts
        from dubforge.stages.separate import _separator_cli
        it = find_indextts(cfg["tts"]["indextts2"])
        return {"local": local(request), "config": config_path,
                "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY")) or cfg["adapt"]["backend"] != "anthropic",
                "voice": f"IndexTTS {it['version']}" if it else "Chatterbox",
                "separation": "RoFormer" if _separator_cli() else "Demucs",
                "style": cfg["adapt"]["style"]}

    def visitor_of(request: Request) -> str:
        return getattr(request.state, "visitor", "") or ""

    @app.get("/api/jobs")
    def list_jobs(request: Request):
        jobs.sync()
        mine = list(jobs.jobs.values())
        if private_history:
            me = visitor_of(request)
            mine = [j for j in mine if me and j.get("owner") == me]
        return sorted((public(j) for j in mine), key=lambda j: -j["created"])

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str):
        jobs.sync()
        job = get_job(job_id)
        info = public(job)
        info["log"] = jobs.log_lines(job, 300)
        info["command"] = jobs.command(job)
        work = jobs.work_dir(job)
        summary = work / "eval" / "summary.json"
        info["summary"] = json.loads(summary.read_text()) if summary.exists() else None
        info["previews"] = {k: (work / "web" / f).exists() for k, f in
                            (("en", "preview_en.mp4"), ("ja", "preview_ja.mp4"), ("vtt", "subs.vtt"))}
        ctx = jobs.context(job)
        info["output"] = bool(ctx.meta.get("output") and Path(ctx.meta["output"]).exists())
        info["has_script"] = bool(ctx.segments)
        info["busy"] = jobs.busy_with(work)
        return info

    # --- starting jobs
    # Uploads arrive in pieces: Cloudflare's tunnel refuses any single request over 100 MB, and a
    # dropped piece can simply be sent again instead of restarting a 1 GB upload.
    uploads: dict[str, dict] = {}

    @app.post("/api/uploads")
    def upload_start(body: dict = Body(...)):
        name = _safe_name(str(body.get("name") or "upload"))
        ext = Path(name).suffix.lower()
        if ext not in VIDEO_EXT | SUB_EXT:
            raise HTTPException(400, f"Not a video or subtitle file: {name}")
        try:
            size = int(body.get("size"))
        except (TypeError, ValueError):
            raise HTTPException(400, "Missing file size")
        uid = uuid.uuid4().hex[:12]
        folder = data / "uploads" / uid
        folder.mkdir(parents=True)
        uploads[uid] = {"dest": folder / name, "part": folder / (name + ".part"), "size": size, "name": name,
                        "kind": "subs" if ext in SUB_EXT else "video"}
        uploads[uid]["part"].touch()
        return {"id": uid}

    @app.put("/api/uploads/{uid}")
    async def upload_piece(uid: str, request: Request, offset: int = 0):
        u = uploads.get(uid)
        if not u:
            raise HTTPException(404, "Upload not found; start it again")
        have = u["part"].stat().st_size
        if offset > have:
            raise HTTPException(409, f"Expected offset {have}")
        with jobs.io_lock, open(u["part"], "r+b") as f:  # a re-sent piece overwrites itself
            f.truncate(offset)
            f.seek(offset)
            async for chunk in request.stream():
                f.write(chunk)
                if f.tell() > u["size"]:
                    raise HTTPException(400, "More data than the file size")
        return {"received": u["part"].stat().st_size}

    @app.post("/api/uploads/{uid}/finish")
    def upload_finish(uid: str):
        u = uploads.get(uid)
        if not u:
            raise HTTPException(404, "Upload not found; start it again")
        got = u["part"].stat().st_size
        if got != u["size"]:
            raise HTTPException(400, f"Upload incomplete ({got} of {u['size']} bytes)")
        u["part"].replace(u["dest"])
        uploads.pop(uid)
        jobs.persist()
        return {"path": str(u["dest"]), "name": u["name"], "kind": u["kind"]}

    @app.post("/api/jobs")
    def start(request: Request, body: dict = Body(...)):
        video = Path(str(body.get("video", ""))).expanduser()
        if not str(video).startswith(str(data / "uploads")) and not local(request):
            raise HTTPException(403, "Only uploads can be dubbed from another computer.")
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXT:
            raise HTTPException(400, f"Video not found: {video}")
        clip = None
        if body.get("clip"):
            try:
                clip = [float(body["clip"][0]), float(body["clip"][1])]
            except (TypeError, ValueError, IndexError):
                raise HTTPException(400, "Clip must be [start seconds, length seconds]")
            if clip[0] < 0 or clip[1] <= 0:
                raise HTTPException(400, "Clip start must be ≥ 0 and length > 0")
        opts = {"video": str(video.resolve()), "name": body.get("name") or video.name, "clip": clip, "kind": "dub",
                "style": body.get("style") or None, "tts": body.get("tts") or None,
                "subs": body.get("subs") or None, "timing": body.get("timing") or None,
                "audio_stream": body.get("audio_stream"), "from_stage": body.get("from_stage") or None,
                "owner": visitor_of(request)}
        for key, allowed in (("style", {"verbatim", "faithful", "natural"}),
                             ("tts", {"auto", "indextts2", "chatterbox", "say", "tone"}),
                             ("timing", {"natural", "fit"}), ("from_stage", set(pipeline.STAGES))):
            if opts[key] and opts[key] not in allowed:
                raise HTTPException(400, f"Bad {key}: {opts[key]}")
        if opts["subs"] and opts["subs"] not in ("auto", "embedded", "ocr"):
            p = Path(opts["subs"]).expanduser()
            if not p.is_file() or p.suffix.lower() not in SUB_EXT:
                raise HTTPException(400, f"Subtitle file not found: {p}")
            opts["subs"] = str(p.resolve())
        return public(jobs.add(opts))

    @app.post("/api/jobs/{job_id}/revoice")
    def revoice(request: Request, job_id: str, body: dict = Body(default={})):
        job = get_job(job_id)
        if jobs.busy_with(jobs.work_dir(job)):
            raise HTTPException(409, "This video is still being processed.")
        stage = body.get("from_stage") or "synthesize"
        if stage not in pipeline.STAGES:
            raise HTTPException(400, f"Bad stage {stage}")
        opts = {k: job.get(k) for k in ("video", "name", "clip", "style", "tts", "subs", "timing", "audio_stream",
                                        "owner")}
        for k in ("tts", "style"):
            if body.get(k):
                opts[k] = body[k]
        return public(jobs.add({**opts, "from_stage": stage, "kind": "revoice" if stage == "synthesize" else "redo"}))

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        get_job(job_id)
        jobs.cancel(job_id)
        return {"ok": True}

    @app.delete("/api/jobs/{job_id}")
    def delete(job_id: str):
        get_job(job_id)
        try:
            jobs.delete(job_id)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    # --- the script
    def ratings_for(work: Path) -> dict:
        f = work / "eval" / "ratings.csv"
        if not f.exists():
            return {}
        with open(f, newline="", encoding="utf-8") as fh:
            return {(r.get("id"), r.get("take")): r for r in csv.DictReader(fh)}

    @app.get("/api/jobs/{job_id}/lines")
    def lines(job_id: str):
        jobs.sync()
        job = get_job(job_id)
        ctx = jobs.context(job)
        names = ctx.cfg["speakers"]["names"]
        rated = ratings_for(ctx.work)
        out = []
        for s in sorted(ctx.segments, key=lambda s: s.start):
            take = s.fit.get("hash") or ""
            r = rated.get((str(s.id), take), {})
            out.append({"id": s.id, "start": s.start, "end": s.end, "speaker": s.speaker,
                        "name": names.get(s.speaker, s.speaker), "ja": s.text_src, "hint": s.hint,
                        "en": s.text_dub, "delivery": s.delivery, "locked": s.locked, "nonverbal": s.nonverbal,
                        "emotion": (s.emotion or {}).get("label"), "intensity": (s.emotion or {}).get("intensity"),
                        "described": (s.emotion or {}).get("described"), "fit": s.fit.get("status"),
                        "has_original": bool(s.ref_audio and ctx.abs(s.ref_audio).exists()),
                        "has_dub": bool(s.dub_audio and ctx.abs(s.dub_audio).exists()), "take": take,
                        "rating": {k: r.get(k, "") for k in ("same_feeling_1to5", "natural_english_1to5",
                                                             "joke_or_intent_landed_y/n", "notes")}})
        return out

    @app.patch("/api/jobs/{job_id}/lines/{seg_id}")
    def edit_line(job_id: str, seg_id: int, body: dict = Body(...)):
        job = get_job(job_id)
        ctx = jobs.context(job)
        if jobs.busy_with(ctx.work):
            raise HTTPException(409, "Wait for the current run on this video to finish.")
        seg = next((s for s in ctx.segments if s.id == seg_id), None)
        if seg is None:
            raise HTTPException(404, "No such line")
        changed = False
        for key, attr in (("en", "text_dub"), ("delivery", "delivery")):
            if key in body and body[key] is not None:
                val = str(body[key]).strip() or None
                if val != getattr(seg, attr):
                    setattr(seg, attr, val)
                    changed = True
        if "locked" in body:
            seg.locked = bool(body["locked"])
        elif changed:
            seg.locked = True  # a hand edit is kept on later runs
        ctx.save()
        jobs.persist()
        return {"ok": True, "locked": seg.locked}

    @app.put("/api/jobs/{job_id}/ratings/{seg_id}")
    def rate(job_id: str, seg_id: int, body: dict = Body(...)):
        from dubforge.stages.evaluate import RATING_COLUMNS

        job = get_job(job_id)
        ctx = jobs.context(job)
        if jobs.busy_with(ctx.work):  # the run rewrites ratings.csv at the end; don't race it
            raise HTTPException(409, "Wait for the current run on this video to finish, then rate.")
        seg = next((s for s in ctx.segments if s.id == seg_id), None)
        if seg is None or not seg.fit.get("hash"):
            raise HTTPException(404, "This line has no dub to rate yet")
        f = ctx.p("eval", "ratings.csv")
        rows = []
        if f.exists():
            with open(f, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        take = seg.fit["hash"]
        row = next((r for r in rows if r.get("id") == str(seg_id) and r.get("take") == take), None)
        if row is None:
            names = ctx.cfg["speakers"]["names"]
            row = {"id": str(seg_id), "take": take, "speaker": names.get(seg.speaker, seg.speaker),
                   "ja": seg.text_src, "en": seg.text_dub, "dub_file": ""}
            rows.append(row)
        for k in ("same_feeling_1to5", "natural_english_1to5", "joke_or_intent_landed_y/n", "notes"):
            if k in body:
                row[k] = str(body[k] or "")
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=RATING_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in RATING_COLUMNS})
        jobs.persist()
        return {"ok": True}

    # --- media
    @app.get("/api/jobs/{job_id}/audio/{kind}/{seg_id}")
    def line_audio(job_id: str, kind: str, seg_id: int):
        ctx = jobs.context(get_job(job_id))
        seg = next((s for s in ctx.segments if s.id == seg_id), None)
        rel = None if seg is None else seg.ref_audio if kind == "original" else seg.dub_audio if kind == "dub" else None
        path = ctx.abs(rel) if rel else None
        if not path or not path.exists():
            raise HTTPException(404, "No audio")
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.get("/api/jobs/{job_id}/file/{what}")
    def media(job_id: str, what: str):
        ctx = jobs.context(get_job(job_id))
        files = {"en": (ctx.work / "web" / "preview_en.mp4", "video/mp4", None),
                 "ja": (ctx.work / "web" / "preview_ja.mp4", "video/mp4", None),
                 "vtt": (ctx.work / "web" / "subs.vtt", "text/vtt", None),
                 "srt": (ctx.work / "output" / "dub_script.en.srt", "application/x-subrip", "dub_script.en.srt")}
        if ctx.meta.get("output"):
            out = Path(ctx.meta["output"])
            files["mkv"] = (out, "video/x-matroska", out.name)
            # iPhones can't play .mkv: the English .mp4 is the one to save to Photos/Files
            files["mp4"] = (ctx.work / "web" / "preview_en.mp4", "video/mp4", out.stem + ".mp4")
        if what not in files or not files[what][0].exists():
            raise HTTPException(404, "Not available")
        path, mime, download = files[what]
        return FileResponse(path, media_type=mime, filename=download,
                            headers={"Cache-Control": "no-store"} if what not in ("mkv", "mp4") else None)

    @app.exception_handler(Exception)
    async def errors(request: Request, exc: Exception):
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _share_key(data_dir: Path, new: bool) -> str:
    f = data_dir / "share_key"
    if new or not f.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        f.write_text(secrets.token_urlsafe(24))
        f.chmod(0o600)
    return f.read_text().strip()


def _start_tunnel(port: int) -> tuple[subprocess.Popen, str]:
    """A free Cloudflare quick tunnel: a public https address that forwards to this Mac. No account,
    no router settings. The address changes every time it starts."""
    import shutil

    exe = shutil.which("cloudflared")
    if not exe:
        raise SystemExit("Sharing with friends elsewhere uses Cloudflare's free tunnel. Install it once:\n"
                         "  brew install cloudflared\n  then run this again.")
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    found: dict[str, str] = {}
    pattern = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")

    def read():
        for line in proc.stdout:
            m = pattern.search(line)
            if m and "url" not in found:
                found["url"] = m.group(0)
    threading.Thread(target=read, daemon=True).start()
    deadline = time.time() + 60
    while "url" not in found and time.time() < deadline and proc.poll() is None:
        time.sleep(0.2)
    if "url" not in found:
        proc.terminate()
        raise SystemExit("Couldn't open the Cloudflare tunnel (check your internet connection and try again).")
    return proc, found["url"]


def _already_running(port: int) -> bool:
    import socket

    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def serve(config_path: str | None, port: int, data_dir: str, open_browser: bool, share: bool = False,
          new_link: bool = False) -> None:
    import atexit
    import shutil
    import webbrowser

    import uvicorn

    url = f"http://127.0.0.1:{port}"
    if _already_running(port):  # double-clicked again: just show the page that's already running
        print(f"DubForge is already running at {url}")
        if open_browser:
            webbrowser.open(url)
        return

    data = Path(data_dir).expanduser()
    data = data if data.is_absolute() else Path.cwd() / data
    key = _share_key(data, new_link) if share else None
    app = create_app(config_path, data_dir, share_key=key)
    state = app.state.share
    stopping = threading.Event()
    tunnel_box: dict[str, subprocess.Popen] = {}

    def stop_tunnel():
        stopping.set()
        if tunnel_box.get("proc") and tunnel_box["proc"].poll() is None:
            tunnel_box["proc"].terminate()
    atexit.register(stop_tunnel)

    def keep_tunnel_open():
        """Open the Cloudflare tunnel and reopen it if it drops (the address changes when it does,
        and the page on this Mac shows the new link)."""
        time.sleep(1.0)  # let the server start first
        while not stopping.is_set():
            state.update(status="starting", error=None)
            try:
                proc, public = _start_tunnel(port)
            except SystemExit as e:
                state.update(status="failed", error=str(e), url=None)
                print(f"\n  {e}\n  Trying again in 30 seconds…")
                stopping.wait(30)
                continue
            tunnel_box["proc"] = proc
            if state.get("url") and state["url"] != public:
                state["changed"] = time.time()
            state.update(status="on", url=public)
            print(f"\n  Link for your friends (anywhere, no password):\n\n    {public}/?key={key}\n")
            print("  (It also shows on the DubForge page on this Mac, with a QR code for your iPhone.)")
            proc.wait()
            if not stopping.is_set():
                state.update(status="reconnecting")
                print("  The tunnel dropped; reopening it (the link will change)…")
                stopping.wait(3)

    def quit_everything():
        stop_tunnel()
        os.kill(os.getpid(), signal.SIGINT)  # uvicorn shuts down cleanly on Ctrl+C
    app.state.quit = quit_everything

    print(f"DubForge is running at {url}")
    if share:
        threading.Thread(target=keep_tunnel_open, daemon=True).start()
        if shutil.which("caffeinate"):  # a sleeping Mac would take the site down
            awake = subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
            atexit.register(awake.terminate)
            print("  Keeping this Mac awake while sharing (it can still lock the screen).")
    print("  Leave this running while it's in use. Quit from the page, or press Control + C here.")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", proxy_headers=False)
