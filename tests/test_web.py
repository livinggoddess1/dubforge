"""The browser interface, end to end with the lightweight test backends (no ML models)."""

import time
from pathlib import Path

import pytest
import yaml

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from dubforge.web.app import create_app  # noqa: E402
from tests.test_pipeline import _make_episode  # noqa: E402


def _config(tmp_path: Path) -> Path:
    cfg = {"work_root": str(tmp_path / "work"), "output_dir": str(tmp_path / "out"),
           "subtitles": {"path": str(tmp_path / "ja.srt"), "lang": "ja"}, "separate": {"backend": "none"},
           "transcribe": {"backend": "subs"}, "diarize": {"backend": "single"},
           "analyze": {"emotion_model": "none"}, "adapt": {"backend": "passthrough"},
           "tts": {"backend": "tone"}, "speakers": {"names": {"SPEAKER_00": "Haruka"}}}
    f = tmp_path / "config.yaml"
    f.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return f


def _wait(client, job_id, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.3)
    raise AssertionError(f"job {job_id} still {j['status']}")


@pytest.fixture()
def client(tmp_path):
    video = _make_episode(tmp_path)
    app = create_app(str(_config(tmp_path)), tmp_path / "web_data", cwd=tmp_path)
    with TestClient(app) as c:
        c.video = video
        yield c


def test_page_and_status(client):
    assert "DubForge" in client.get("/").text
    s = client.get("/api/status").json()
    assert s["local"] and s["voice"] in ("Chatterbox",) or s["voice"].startswith("IndexTTS")


def test_upload_dub_play_edit_rate_revoice(client):
    up = _upload(client, "Episode 01.mkv", client.video.read_bytes())
    assert up["kind"] == "video" and Path(up["path"]).exists()
    assert Path(up["path"]).read_bytes() == client.video.read_bytes()

    job = client.post("/api/jobs", json={"video": up["path"], "clip": [2, 10], "tts": "tone"}).json()
    done = _wait(client, job["id"])
    assert done["status"] == "done", done.get("error")
    assert done["stage_n"] == 10 and done["output"] and done["summary"]["lines"] >= 1
    assert all(done["previews"].values())

    # browser player: mp4 with range requests (Safari needs them), subtitles, downloads
    r = client.get(f"/api/jobs/{job['id']}/file/en", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    assert client.get(f"/api/jobs/{job['id']}/file/vtt").text.startswith("WEBVTT")
    assert client.get(f"/api/jobs/{job['id']}/file/mkv").status_code == 200
    mp4 = client.get(f"/api/jobs/{job['id']}/file/mp4")                         # for saving on an iPhone
    assert mp4.status_code == 200 and mp4.headers["content-type"] == "video/mp4"
    assert ".mp4" in mp4.headers["content-disposition"]

    lines = client.get(f"/api/jobs/{job['id']}/lines").json()
    spoken = [l for l in lines if not l["nonverbal"] and l["has_dub"]]
    assert spoken and spoken[0]["name"] == "Haruka"
    line = spoken[0]
    assert client.get(f"/api/jobs/{job['id']}/audio/original/{line['id']}").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}/audio/dub/{line['id']}").status_code == 200

    # edit → locked; rate; re-voice uses the edit
    r = client.patch(f"/api/jobs/{job['id']}/lines/{line['id']}", json={"en": "Edited by hand.", "delivery": "pleading"})
    assert r.json()["locked"] is True
    assert client.put(f"/api/jobs/{job['id']}/ratings/{line['id']}", json={"same_feeling_1to5": "4"}).status_code == 200
    again = client.get(f"/api/jobs/{job['id']}/lines").json()
    assert next(l for l in again if l["id"] == line["id"])["rating"]["same_feeling_1to5"] == "4"

    rv = client.post(f"/api/jobs/{job['id']}/revoice", json={}).json()
    assert rv["from_stage"] == "synthesize" and rv["kind"] == "revoice"
    done2 = _wait(client, rv["id"])
    assert done2["status"] == "done", done2.get("error")
    after = client.get(f"/api/jobs/{rv['id']}/lines").json()
    edited = next(l for l in after if l["id"] == line["id"])
    assert edited["en"] == "Edited by hand." and edited["locked"] and edited["take"] != line["take"]
    assert edited["rating"]["same_feeling_1to5"] == ""          # a new take starts unrated


def test_bad_requests(client):
    assert client.post("/api/jobs", json={"video": "/nope/x.mkv"}).status_code == 400
    assert client.post("/api/jobs", json={"video": str(client.video), "tts": "evil"}).status_code == 400
    assert client.post("/api/uploads", json={"name": "x.exe", "size": 2}).status_code == 400
    assert client.get("/api/jobs/nope").status_code == 404


def test_failed_job_reports_the_reason(client, tmp_path):
    bad = tmp_path / "broken.mkv"
    bad.write_bytes(b"not a video")
    job = client.post("/api/jobs", json={"video": str(bad)}).json()
    done = _wait(client, job["id"])
    assert done["status"] == "failed" and done["error"]


def _upload(client, name, data, piece=1000, headers=None):
    uid = client.post("/api/uploads", json={"name": name, "size": len(data)}, headers=headers).json()["id"]
    for off in range(0, len(data), piece):
        r = client.put(f"/api/uploads/{uid}?offset={off}", content=data[off:off + piece], headers=headers)
        assert r.status_code == 200, r.text
    return client.post(f"/api/uploads/{uid}/finish", headers=headers).json()


def test_upload_in_pieces_with_a_resent_piece(client):
    data = bytes(range(256)) * 20
    uid = client.post("/api/uploads", json={"name": "clip.mkv", "size": len(data)}).json()["id"]
    client.put(f"/api/uploads/{uid}?offset=0", content=data[:2000])
    client.put(f"/api/uploads/{uid}?offset=1000", content=data[1000:3000])      # piece sent again
    assert client.post(f"/api/uploads/{uid}/finish").status_code == 400          # not complete yet
    client.put(f"/api/uploads/{uid}?offset=3000", content=data[3000:])
    done = client.post(f"/api/uploads/{uid}/finish").json()
    assert Path(done["path"]).read_bytes() == data


TUNNEL = {"cf-connecting-ip": "203.0.113.7", "x-forwarded-proto": "https"}


def test_private_link_for_friends(tmp_path):
    video = _make_episode(tmp_path)
    app = create_app(str(_config(tmp_path)), tmp_path / "wd", share_key="k" * 32, cwd=tmp_path)
    with TestClient(app, follow_redirects=False) as c:
        # this Mac: always allowed, nothing to type
        assert c.get("/api/jobs").status_code == 200
        # someone over the tunnel without the link
        assert c.get("/", headers=TUNNEL).status_code == 403
        assert c.get("/api/jobs?key=wrong", headers=TUNNEL).status_code == 403
        # a friend opening the link: let in, key moved into a cookie
        r = c.get("/?key=" + "k" * 32, headers=TUNNEL)
        assert r.status_code == 200 and "dubforge_key" in r.headers["set-cookie"]
        # iPhone Home Screen: the app made from a friend's page starts at their private link
        m = c.get("/manifest.webmanifest?key=" + "k" * 32, headers=TUNNEL).json()
        assert m["start_url"] == "/?key=" + "k" * 32 and m["display"] == "standalone"
        assert c.get("/manifest.webmanifest").json()["start_url"] == "/"           # this Mac
        assert c.get("/static/icon-180.png", headers={**TUNNEL, "cookie": "dubforge_key=" + "k" * 32}).status_code == 200
        c.cookies.set("dubforge_key", "k" * 32)
        assert c.get("/api/jobs", headers=TUNNEL).status_code == 200
        assert c.get("/api/status", headers=TUNNEL).json()["local"] is False
        # friends can upload and dub, but not point DubForge at files already on this Mac
        assert c.post("/api/jobs", json={"video": str(video)}, headers=TUNNEL).status_code == 403
        up = _upload(c, "ep.mkv", video.read_bytes(), piece=50_000, headers=TUNNEL)
        assert c.post("/api/jobs", json={"video": up["path"], "tts": "tone"}, headers=TUNNEL).status_code == 200


def test_no_sharing_means_this_mac_only(tmp_path):
    _make_episode(tmp_path)
    app = create_app(str(_config(tmp_path)), tmp_path / "wd", cwd=tmp_path)
    with TestClient(app) as c:
        assert c.get("/api/jobs").status_code == 200
        assert c.get("/api/jobs", headers=TUNNEL).status_code == 403


def test_share_key_is_kept_until_renewed(tmp_path):
    from dubforge.web.app import _share_key

    k1 = _share_key(tmp_path, new=False)
    assert len(k1) >= 30 and _share_key(tmp_path, new=False) == k1
    assert _share_key(tmp_path, new=True) != k1


# ---- cloud mode (Modal), with a stand-in that runs the "GPU" job in a thread ---------------------

class FakeCloud:
    info = {"voice": "IndexTTS (cloud GPU)", "separation": "RoFormer", "where": "cloud"}

    def __init__(self, config, data_dir, cwd):
        self.config, self.data_dir, self.cwd = config, data_dir, cwd
        self.commits = self.reloads = 0
        self.cancelled = []

    def spawn(self, job):
        import threading
        from dubforge.web.app import run_cloud_job
        threading.Thread(target=run_cloud_job, args=(dict(job), self.config, self.data_dir, self.cwd),
                         kwargs={"commit": self.commit}, daemon=True).start()
        return "fc-" + job["id"]

    def cancel(self, call_id):
        self.cancelled.append(call_id)

    def commit(self):
        self.commits += 1

    def reload(self):
        self.reloads += 1


def test_cloud_mode(tmp_path):
    video = _make_episode(tmp_path)
    cfg = _config(tmp_path)
    cloud = FakeCloud(str(cfg), tmp_path / "wd", tmp_path)
    key = "c" * 32
    app = create_app(str(cfg), tmp_path / "wd", share_key=key, cwd=tmp_path, cloud=cloud)
    with TestClient(app) as c:
        # nobody is "at the Mac" in the cloud: even a direct request needs the link
        assert c.get("/api/jobs").status_code == 403
        c.cookies.set("dubforge_key", key)
        s = c.get("/api/status").json()
        assert s["voice"] == "IndexTTS (cloud GPU)" and s["local"] is False
        assert c.post("/api/jobs", json={"video": str(video)}).status_code == 403   # no server paths
        up = _upload(c, "Episode 01.mkv", video.read_bytes(), piece=200_000)
        job = c.post("/api/jobs", json={"video": up["path"], "clip": [2, 10], "tts": "tone"}).json()
        done = _wait(c, job["id"])
        assert done["status"] == "done", done.get("error")
        assert all(done["previews"].values()) and cloud.reloads > 0 and cloud.commits > 0
        lines = c.get(f"/api/jobs/{job['id']}/lines").json()
        assert any(l["has_dub"] for l in lines)
        # cancelling a cloud run stops its GPU
        job2 = c.post("/api/jobs", json={"video": up["path"], "clip": [2, 10], "tts": "tone"}).json()
        c.post(f"/api/jobs/{job2['id']}/cancel")
        assert cloud.cancelled == ["fc-" + job2["id"]]
        assert c.get(f"/api/jobs/{job2['id']}").json()["status"] == "cancelled"
