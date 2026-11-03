"""Audio helpers: ffmpeg wrappers, loading/saving, loudness, silence trimming, time-stretching."""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


class FFmpegError(RuntimeError):
    pass


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"Command failed: {' '.join(map(str, cmd))}\n{proc.stderr[-2000:]}")
    return proc.stdout


def require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise FFmpegError("ffmpeg/ffprobe not found. On a Mac: brew install ffmpeg")


def probe(path: str | Path) -> dict:
    out = run_cmd(["ffprobe", "-v", "error", "-print_format", "json",
                   "-show_format", "-show_streams", path])
    return json.loads(out)


def duration_of(path: str | Path) -> float:
    return float(probe(path)["format"]["duration"])


LANG_TAGS = {"ja": {"ja", "jpn", "jp", "japanese"}, "en": {"en", "eng", "english"}}


def lang_matches(stream: dict, lang: str) -> bool:
    tag = ((stream.get("tags") or {}).get("language") or "").lower()
    return tag in LANG_TAGS.get(lang, {lang})


def pick_audio_stream(video, lang: str, choice="auto", log=print) -> int:
    """Index (among audio streams) of the original-language track. Dual-audio releases often put
    the English dub first, so the first track isn't safe to assume."""
    streams = [s for s in probe(video)["streams"] if s.get("codec_type") == "audio"]
    if not streams:
        raise FFmpegError("This video has no audio track.")
    if choice not in (None, "auto"):
        idx = int(choice)
        if idx >= len(streams):
            raise FFmpegError(f"Audio stream {idx} not found (video has {len(streams)}).")
        return idx
    for i, s in enumerate(streams):
        if lang_matches(s, lang):
            return i
    if len(streams) > 1:
        tags = [((s.get("tags") or {}).get("language") or "?") for s in streams]
        log(f"warning: {len(streams)} audio tracks ({', '.join(tags)}) and none tagged '{lang}'; using the "
            "first. If that's the wrong language, set extract.audio_stream to the right index.")
    return 0


def extract_audio(src, dst, sr=44100, channels=2, start=None, duration=None, stream: int = 0) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start}"]
    cmd += ["-i", src]
    if duration is not None:
        cmd += ["-t", f"{duration}"]
    cmd += ["-map", f"0:a:{stream}", "-vn", "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_f32le", dst]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    run_cmd(cmd)


@functools.lru_cache(maxsize=None)
def has_ffmpeg_filter(name: str) -> bool:
    try:
        out = run_cmd(["ffmpeg", "-hide_banner", "-filters"])
    except Exception:
        return False
    return any(line.split()[1:2] == [name] for line in out.splitlines() if len(line.split()) > 1)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or len(x) == 0:
        return x
    try:
        import soxr
        return soxr.resample(x, sr_in, sr_out).astype(np.float32)
    except ImportError:
        pass
    try:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g, axis=0).astype(np.float32)
    except ImportError:
        n_out = int(round(len(x) * sr_out / sr_in))
        t_in = np.linspace(0, 1, len(x), endpoint=False)
        t_out = np.linspace(0, 1, n_out, endpoint=False)
        if x.ndim == 1:
            return np.interp(t_out, t_in, x).astype(np.float32)
        return np.stack([np.interp(t_out, t_in, x[:, c]) for c in range(x.shape[1])], 1).astype(np.float32)


def load(path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    path = Path(path)
    if path.suffix.lower() not in {".wav", ".flac", ".ogg", ".aiff", ".aif"}:
        tmp = path.with_suffix(".tmp_load.wav")
        run_cmd(["ffmpeg", "-y", "-v", "error", "-i", path, "-c:a", "pcm_f32le", tmp])
        x, file_sr = sf.read(tmp, dtype="float32", always_2d=False)
        tmp.unlink(missing_ok=True)
    else:
        x, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if mono and x.ndim == 2:
        x = x.mean(axis=1)
    if sr and sr != file_sr:
        x = resample(x, file_sr, sr)
        file_sr = sr
    return x.astype(np.float32), file_sr


def save(path, x: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr, subtype="FLOAT")


def slice_audio(x: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    a = max(0, int(start * sr))
    b = min(len(x), int(end * sr))
    return x[a:b].copy() if b > a else x[:0].copy()


def rms_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return -120.0
    if x.ndim == 2:
        x = x.mean(axis=1)
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9))


def active_rms_db(x: np.ndarray, sr: int, frame: float = 0.03, floor_db: float = -50.0) -> float:
    """RMS over frames that actually contain speech (ignores the silent gaps)."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = max(1, int(frame * sr))
    frames = len(x) // n
    if frames == 0:
        return rms_db(x)
    f = x[: frames * n].reshape(frames, n)
    db = 20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-9)
    active = f[db > max(floor_db, db.max() - 35)]
    return rms_db(active.ravel()) if len(active) else rms_db(x)


def trim_silence(x: np.ndarray, sr: int, top_db: float = 40.0, pad: float = 0.04) -> np.ndarray:
    if len(x) == 0:
        return x
    n = max(1, int(0.01 * sr))
    frames = len(x) // n
    if frames < 2:
        return x
    env = np.sqrt((x[: frames * n].reshape(frames, n) ** 2).mean(axis=1))
    db = 20 * np.log10(env + 1e-9)
    keep = np.where(db > db.max() - top_db)[0]
    if len(keep) == 0:
        return x
    a = max(0, keep[0] * n - int(pad * sr))
    b = min(len(x), (keep[-1] + 1) * n + int(pad * sr))
    return x[a:b]


def fade(x: np.ndarray, sr: int, ms: float = 10.0) -> np.ndarray:
    n = min(len(x) // 2, int(sr * ms / 1000))
    if n > 0:
        ramp = np.linspace(0, 1, n, dtype=np.float32)
        x = x.copy()
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def time_stretch(src, dst, tempo: float, sr: int = 44100) -> None:
    """Pitch-preserving tempo change (tempo > 1 = faster/shorter). Uses rubberband if ffmpeg has it."""
    filters = []
    if abs(tempo - 1.0) > 1e-3:
        if has_ffmpeg_filter("rubberband"):
            filters.append(f"rubberband=tempo={tempo:.4f}:pitchq=quality")
        else:
            t = tempo
            while t > 2.0:
                filters.append("atempo=2.0")
                t /= 2.0
            while t < 0.5:
                filters.append("atempo=0.5")
                t /= 0.5
            filters.append(f"atempo={t:.4f}")
    filters.append(f"aresample={sr}")
    run_cmd(["ffmpeg", "-y", "-v", "error", "-i", src, "-af", ",".join(filters),
             "-ac", "1", "-c:a", "pcm_f32le", dst])


def pitch_shift(src, dst, semitones: float, sr: int = 44100) -> None:
    """Shift pitch without changing speed. Rubberband keeps formants (the voice's character) intact."""
    if has_ffmpeg_filter("rubberband"):
        ratio = 2 ** (semitones / 12)
        run_cmd(["ffmpeg", "-y", "-v", "error", "-i", src, "-af",
                 f"rubberband=pitch={ratio:.5f}:formant=preserved:pitchq=quality,aresample={sr}",
                 "-ac", "1", "-c:a", "pcm_f32le", dst])
        return
    import librosa

    x, in_sr = load(src, mono=True)
    save(dst, librosa.effects.pitch_shift(x, sr=in_sr, n_steps=semitones).astype(np.float32), in_sr)


def median_pitch_st(path) -> float | None:
    """Median pitch of a recording in semitones above 55 Hz (same scale as prosody.f0_median_st)."""
    import math

    import librosa

    x, sr = load(path, mono=True)
    y = resample(x, sr, 16000)
    if len(y) < 4000:
        return None
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=1000, sr=16000, frame_length=1024)
    f0v = f0[~np.isnan(f0)]
    return 12 * math.log2(float(np.median(f0v)) / 55.0) if len(f0v) >= 5 else None


def soft_limit(x: np.ndarray, ceiling: float = 0.97, sr: int = 44100, window_ms: float = 10.0) -> np.ndarray:
    """Peak limiter: turns the level down only around samples that would go over `ceiling`.

    (The old version ran the whole track through tanh whenever one sample was too loud, which
    coloured every line in the episode.) Gain = the needed reduction, spread over a short window
    on both sides of each peak so there are no clicks; everything else passes through untouched."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    level = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    if float(level.max()) <= ceiling:
        return x
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    win = max(1, int(sr * window_ms / 1000))
    need = np.minimum(1.0, ceiling / np.maximum(level, 1e-9))
    # min over 2*win, then average over win: the smoothed gain never exceeds what any sample needs
    gain = uniform_filter1d(minimum_filter1d(need, size=2 * win + 1, mode="nearest"), size=win, mode="nearest")
    gain = np.minimum(gain, need).astype(np.float32)
    y = x * (gain[:, None] if x.ndim == 2 else gain)
    return np.clip(y, -ceiling, ceiling).astype(np.float32)


def match_tone(x: np.ndarray, ref: np.ndarray, sr: int, max_db: float = 9.0,
               lo_hz: float = 80.0, hi_hz: float = 12000.0) -> np.ndarray:
    """Give `x` the tonal colour of `ref`: its long-term spectrum, smoothed to about a third of an
    octave so it captures the recording (phone, door, distance, mic, room EQ), not the words.

    The dub is recorded "in the booth" while the original line may be muffled, thin or boomy; a
    Japanese viewer hears the scene's sound, so the English line should too. The overall level is
    left unchanged (loudness is matched separately) and corrections are limited to ±max_db."""
    from scipy.signal import fftconvolve, firwin2, welch

    x = np.asarray(x, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if ref.ndim == 2:
        ref = ref.mean(axis=1)
    if len(x) < sr // 4 or len(ref) < sr // 2:
        return x
    nper = 2048 if sr >= 32000 else 1024
    f, p_ref = welch(ref, sr, nperseg=min(nper, len(ref)))
    _, p_x = welch(x if x.ndim == 1 else x.mean(axis=1), sr, nperseg=min(nper, len(x)))
    # third-octave bands: average power in each, then the dB difference
    edges = lo_hz * 2 ** (np.arange(0, np.log2(hi_hz / lo_hz) + 1e-9, 1 / 3))
    centers, diff = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        band = (f >= a) & (f < b)
        if band.sum() == 0:
            continue
        centers.append(np.sqrt(a * b))
        diff.append(10 * np.log10((p_ref[band].mean() + 1e-12) / (p_x[band].mean() + 1e-12)))
    if len(centers) < 4:
        return x
    diff = np.array(diff)
    centers = np.array(centers)
    speech = (centers >= 300) & (centers <= 3000)
    diff -= diff[speech].mean() if speech.any() else diff.mean()   # tone only, not level
    diff = np.clip(diff, -max_db, max_db)
    grid = np.concatenate([[0.0], centers, [sr / 2]])
    gains_db = np.concatenate([[diff[0]], diff, [diff[-1]]])
    taps = firwin2(1025, grid, 10 ** (gains_db / 20), fs=sr)
    if x.ndim == 2:
        return np.stack([fftconvolve(x[:, c], taps, mode="same") for c in range(x.shape[1])], 1).astype(np.float32)
    return fftconvolve(x, taps, mode="same").astype(np.float32)
