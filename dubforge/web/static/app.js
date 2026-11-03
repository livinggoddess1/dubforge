"use strict";
// DubForge web UI. Plain JavaScript, no build step: the server is the source of truth and this
// page polls it. Text from subtitles and the model is always inserted as text, never as HTML.

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, props = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null) n.append(kid instanceof Node ? kid : document.createTextNode(kid));
  return n;
};

// replaceChildren() would print "null" for skipped optional items
const setKids = (node, ...kids) => node.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    ...opts,
    body: opts.body && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
  });
  if (!res.ok) {
    let msg = `${res.status}`;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), 4500);
}

const fmtTime = (s) => {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return (h ? `${h}:${String(m).padStart(2, "0")}` : `${m}`) + `:${String(sec).padStart(2, "0")}`;
};
const parseTime = (v) => {
  const parts = String(v).trim().split(":").map(Number);
  if (parts.some(isNaN)) return NaN;
  return parts.reduce((acc, p) => acc * 60 + p, 0);
};
const ago = (t) => {
  const d = Date.now() / 1000 - t;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return new Date(t * 1000).toLocaleDateString();
};
const STAGE_WORDS = {
  extract: "Pulling out the audio and subtitles", separate: "Separating voices from music and effects",
  transcribe: "Transcribing the Japanese", diarize: "Working out who's speaking",
  analyze: "Measuring how each line is performed", adapt: "Writing the English script",
  synthesize: "Performing the English lines", mix: "Mixing onto the original soundtrack",
  mux: "Writing the video file", evaluate: "Scoring how well the dub follows the original",
};
const STATUS_WORDS = { queued: "Waiting", running: "Working", previewing: "Preparing the player",
  done: "Done", failed: "Failed", cancelled: "Cancelled" };

// ---- state -----------------------------------------------------------------------------------
const state = { status: null, video: null, subsPath: null, selected: null, detailTimer: null,
  lines: [], edited: new Set(), lastDetail: null };

// ---- top bar ---------------------------------------------------------------------------------
async function loadStatus() {
  try {
    const s = await api("/api/status");
    state.status = s;
    const chips = $("#chips");
    setKids(chips,
      s.where === "cloud" ? el("span", { class: "chip" }, el("b", { text: "☁︎ Cloud" }), " runs while your Mac is off") : null,
      el("span", { class: "chip" }, "Voice ", el("b", { text: s.voice })),
      el("span", { class: "chip" }, "Separation ", el("b", { text: s.separation })),
      el("span", { class: "chip" }, "Script ", el("b", { text: s.style })),
      s.anthropic_key ? null : el("span", { class: "chip warn", text: "No ANTHROPIC_API_KEY: lines without subtitles can't be translated" }),
    );
    $("#path-row").classList.toggle("hidden", !s.local);
  } catch (e) { toast(`Can't reach DubForge: ${e.message}`); }
}

// ---- new dub ---------------------------------------------------------------------------------
function setVideo(path, name) {
  state.video = path;
  const c = $("#chosen");
  c.textContent = `🎬 ${name}`;
  c.classList.remove("hidden");
  $("#start").disabled = false;
}

// Uploads go in 32 MB pieces: the tunnel friends use refuses anything over 100 MB at once, and a
// piece that fails on a hotel Wi-Fi is just sent again instead of restarting the whole file.
const PIECE = 32 * 1024 * 1024;

function sendPiece(url, blob, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded);
    xhr.onload = () => (xhr.status < 300 ? resolve() : reject(new Error(xhr.responseText || xhr.status)));
    xhr.onerror = () => reject(new Error("connection dropped"));
    xhr.send(blob);
  });
}

async function upload(file, onProgress) {
  const { id } = await api("/api/uploads", { method: "POST", body: { name: file.name, size: file.size } });
  for (let offset = 0; offset < file.size || offset === 0; offset += PIECE) {
    const blob = file.slice(offset, offset + PIECE);
    for (let attempt = 1; ; attempt++) {
      try {
        await sendPiece(`/api/uploads/${id}?offset=${offset}`, blob, (n) => onProgress((offset + n) / Math.max(file.size, 1)));
        break;
      } catch (e) {
        if (attempt >= 5) throw e;
        await new Promise((r) => setTimeout(r, attempt * 1500));
      }
    }
    if (file.size === 0) break;
  }
  return api(`/api/uploads/${id}/finish`, { method: "POST" });
}

async function uploadVideo(file) {
  const bar = $("#upload-progress");
  bar.classList.remove("hidden");
  $("#start").disabled = true;
  $("#start-msg").textContent = `Uploading ${file.name}…`;
  try {
    const r = await upload(file, (p) => { bar.firstElementChild.style.width = `${(p * 100).toFixed(1)}%`; });
    if (r.kind !== "video") throw new Error("That's a subtitle file; choose it under Options → Subtitles.");
    setVideo(r.path, r.name);
    $("#path").value = "";
    $("#start-msg").textContent = "";
  } catch (e) {
    $("#start-msg").textContent = `Upload failed: ${e.message}`;
  } finally {
    setTimeout(() => bar.classList.add("hidden"), 600);
  }
}

function wireNewDub() {
  const drop = $("#drop"), file = $("#file");
  drop.addEventListener("click", () => file.click());
  drop.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && file.click());
  file.addEventListener("change", () => file.files[0] && uploadVideo(file.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && uploadVideo(e.dataTransfer.files[0]));
  $("#path").addEventListener("input", (e) => {
    const v = e.target.value.trim().replace(/^['"]|['"]$/g, "");
    if (v) setVideo(v, v.split("/").pop());
  });
  $("#use-clip").addEventListener("change", (e) => $("#clip-fields").classList.toggle("hidden", !e.target.checked));
  $("#subs").addEventListener("change", (e) => {
    const isFile = e.target.value === "file";
    $("#subs-file").classList.toggle("hidden", !isFile);
    if (isFile) $("#subs-file").click();
  });
  $("#subs-file").addEventListener("change", async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      const r = await upload(f, () => {});
      state.subsPath = r.path;
      toast(`Subtitles ready: ${r.name}`);
    } catch (err) { toast(`Subtitle upload failed: ${err.message}`); }
  });
  $("#start").addEventListener("click", startJob);
}

async function startJob() {
  const msg = $("#start-msg");
  const body = { video: state.video, tts: $("#tts").value || null, style: $("#style").value || null };
  const subs = $("#subs").value;
  if (subs === "file") {
    if (!state.subsPath) { msg.textContent = "Choose a subtitle file first."; return; }
    body.subs = state.subsPath;
  } else if (subs) body.subs = subs;
  const stream = $("#audio-stream").value;
  if (stream !== "") body.audio_stream = Number(stream);
  if ($("#use-clip").checked) {
    const start = parseTime($("#clip-start").value), len = Number($("#clip-len").value);
    if (isNaN(start) || !(len > 0)) { msg.textContent = "Clip start should look like 5:00 and length like 120."; return; }
    body.clip = [start, len];
  }
  $("#start").disabled = true;
  try {
    const job = await api("/api/jobs", { method: "POST", body });
    msg.textContent = "";
    await refreshJobs();
    select(job.id);
  } catch (e) {
    msg.textContent = e.message;
  } finally {
    $("#start").disabled = !state.video;
  }
}

// ---- job list --------------------------------------------------------------------------------
async function refreshJobs() {
  let jobs;
  try { jobs = await api("/api/jobs"); } catch { return; }
  const ul = $("#jobs");
  if (!jobs.length) { ul.replaceChildren(el("li", { class: "empty", text: "Nothing yet." })); return; }
  ul.replaceChildren(...jobs.map((j) => {
    const clip = j.clip ? ` · ${fmtTime(j.clip[0])}–${fmtTime(j.clip[0] + j.clip[1])}` : "";
    const what = j.kind === "revoice" ? "re-voice" : j.kind === "redo" ? `redo from ${j.from_stage}` : "";
    const status = j.status === "running" && j.stage ? `${j.stage_n}/${j.stage_total} ${j.stage}` : STATUS_WORDS[j.status] || j.status;
    return el("li", { class: `job${state.selected === j.id ? " sel" : ""}`, onclick: () => select(j.id) },
      el("div", { class: "name", text: j.name }),
      el("div", { class: "meta" }, el("span", { class: `dot ${j.status}` }), status,
        el("span", { text: [what, clip.replace(/^ · /, ""), ago(j.created)].filter(Boolean).join(" · ") })));
  }));
}

// ---- job detail ------------------------------------------------------------------------------
function select(id) {
  state.selected = id;
  state.lines = [];
  state.edited.clear();
  state.lastDetail = null;
  const d = $("#detail");
  d.replaceChildren($("#detail-tpl").content.cloneNode(true));
  wireDetail();
  refreshJobs();
  loadDetail(true);
  if (window.matchMedia("(max-width: 900px)").matches) d.scrollIntoView({ behavior: "smooth" });
}

function wireDetail() {
  const d = $("#detail");
  d.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => switchLang(b.dataset.lang)));
  $(".subs-toggle input", d).addEventListener("change", (e) => {
    const t = $(".player", d).textTracks[0];
    if (t) t.mode = e.target.checked ? "showing" : "hidden";
  });
  $(".search", d).addEventListener("input", renderLines);
  $("select.who", d).addEventListener("change", renderLines);
  $(".revoice", d).addEventListener("click", () => revoice("synthesize"));
}

function switchLang(lang) {
  const d = $("#detail"), v = $(".player", d);
  const t = v.currentTime, playing = !v.paused;
  v.src = `/api/jobs/${state.selected}/file/${lang}?t=${Date.now()}`;
  v.addEventListener("loadedmetadata", () => { v.currentTime = t; if (playing) v.play(); }, { once: true });
  d.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.lang === lang));
}

async function loadDetail(first = false) {
  clearTimeout(state.detailTimer);
  const id = state.selected;
  let j;
  try { j = await api(`/api/jobs/${id}`); } catch (e) { toast(e.message); return; }
  if (id !== state.selected) return;
  const d = $("#detail"), prev = state.lastDetail;
  state.lastDetail = j;

  $(".title", d).textContent = j.name;
  const bits = [STATUS_WORDS[j.status] || j.status];
  if (j.clip) bits.push(`${fmtTime(j.clip[0])} – ${fmtTime(j.clip[0] + j.clip[1])}`);
  if (j.kind === "revoice") bits.push("re-voice"); else if (j.kind === "redo") bits.push(`redo from ${j.from_stage}`);
  if (j.tts) bits.push(`voice: ${j.tts}`);
  if (j.finished && j.started) bits.push(`took ${fmtTime(j.finished - j.started)}`);
  $(".sub", d).textContent = bits.join(" · ");

  const actions = $(".actions", d);
  const active = ["queued", "running", "previewing"].includes(j.status);
  setKids(actions,
    active && j.status !== "previewing" ? el("button", { class: "ghost", text: "Cancel", onclick: async () => {
      try { await api(`/api/jobs/${id}/cancel`, { method: "POST" }); loadDetail(); } catch (e) { toast(e.message); } } }) : null,
    !active && j.has_script ? el("button", { class: "ghost", text: "Re-voice all", title: "Perform every line again (unchanged lines are reused)", onclick: () => revoice("synthesize") }) : null,
    !active && j.has_script ? el("button", { class: "ghost", text: "Rewrite English", title: "Write the English script again (locked lines are kept), then re-voice", onclick: () => revoice("adapt") }) : null,
    !active ? el("button", { class: "ghost", text: "Remove", onclick: async () => {
      if (!confirm("Remove this from the list? (Files on disk are kept.)")) return;
      try { await api(`/api/jobs/${id}`, { method: "DELETE" }); state.selected = null; refreshJobs();
        $("#detail").replaceChildren(el("div", { class: "placeholder" }, el("p", { text: "Removed." }))); } catch (e) { toast(e.message); } } }) : null,
  );

  // progress
  const pw = $(".progress-wrap", d);
  pw.classList.toggle("hidden", !active);
  if (active) {
    const pct = j.status === "previewing" ? 98 : j.status === "queued" ? 0 : Math.max(3, ((j.stage_n - 0.5) / j.stage_total) * 100);
    $(".bar > div", pw).style.width = `${pct}%`;
    $(".stage-text", pw).textContent = j.status === "queued" ? "Waiting for the dub ahead of it to finish…"
      : j.status === "previewing" ? "Making a version your browser can play…"
      : j.stage ? `Step ${j.stage_n} of ${j.stage_total}: ${STAGE_WORDS[j.stage] || j.stage}…` : "Starting…";
  }
  const err = $(".error", d);
  err.classList.toggle("hidden", !(j.status === "failed" && j.error));
  if (j.status === "failed") err.textContent = `${j.error}\n\nThe full log is below. Fix the cause, then use Re-voice / Rewrite, or start a new dub.`;

  // player
  const pv = j.previews || {};
  const pwrap = $(".player-wrap", d);
  pwrap.classList.toggle("hidden", !pv.en);
  if (pv.en && (!prev || !prev.previews || !prev.previews.en || prev.status !== j.status)) {
    const v = $(".player", d);
    v.replaceChildren();
    v.src = `/api/jobs/${id}/file/en?t=${Date.now()}`;
    if (pv.vtt) {
      v.append(el("track", { kind: "subtitles", srclang: "en", label: "Dub script", src: `/api/jobs/${id}/file/vtt?t=${Date.now()}`, default: true }));
    }
    $('.seg button[data-lang="ja"]', d).disabled = !pv.ja;
    d.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.lang === "en"));
    $(".dl.mkv", d).href = `/api/jobs/${id}/file/mkv`;
    $(".dl.mp4", d).href = `/api/jobs/${id}/file/mp4`;
    $(".dl.srt", d).href = `/api/jobs/${id}/file/srt`;
  }
  $(".dl.mkv", d).classList.toggle("hidden", !j.output);
  $(".dl.mp4", d).classList.toggle("hidden", !(j.output && pv.en));
  if (j.preview_error && !pv.en) toast(`Couldn't make a browser preview (${j.preview_error}); the .mkv download still works.`);

  // metrics
  const m = $(".metrics", d), s = j.summary;
  m.classList.toggle("hidden", !s);
  if (s) {
    const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
    const num = (v) => (v == null ? "–" : v.toFixed(2));
    const tiles = [
      [pct(s.emotion_label_agreement), "Lines with the same main emotion"],
      [num(s.emotion_cos_mean), "Emotion similarity (0–1)"],
      [num(s.energy_tracking), "Gets loud where the original does"],
      [num(s.melody_tracking), "Rises and falls where the original does"],
      [num(s.pitch_tracking), "High and low where the original is"],
      [s.duration_ratio_median == null ? "–" : `${s.duration_ratio_median.toFixed(2)}×`, "Length vs. the Japanese"],
    ];
    m.replaceChildren(...tiles.filter(([v]) => v !== "–").map(([v, k]) => el("div", { class: "metric" }, el("div", { class: "v", text: v }), el("div", { class: "k", text: k }))));
  }

  // script (load once per finished state; keep edits while the user types)
  const showScript = j.has_script && !["running", "queued"].includes(j.status);
  $(".script", d).classList.toggle("hidden", !showScript);
  if (showScript && (first || !prev || prev.status !== j.status)) await loadLines();

  // log
  const pre = $(".log pre", d), atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 30;
  pre.textContent = j.log.join("\n");
  if (atBottom) pre.scrollTop = pre.scrollHeight;
  if (j.status === "failed") $(".log", d).open = true;

  if (active) state.detailTimer = setTimeout(loadDetail, 2000);
  if (prev && prev.status !== j.status) refreshJobs();
}

async function revoice(stage) {
  try {
    const job = await api(`/api/jobs/${state.selected}/revoice`, { method: "POST", body: { from_stage: stage } });
    toast(stage === "synthesize" ? "Re-voicing: unchanged lines are reused, so this is quicker." : "Rewriting the English, then re-voicing.");
    await refreshJobs();
    select(job.id);
  } catch (e) { toast(e.message); }
}

// ---- script editor ---------------------------------------------------------------------------
async function loadLines() {
  try { state.lines = await api(`/api/jobs/${state.selected}/lines`); } catch (e) { toast(e.message); return; }
  const who = $("#detail select.who"), current = who.value;
  const names = [...new Set(state.lines.map((l) => l.name))].sort();
  who.replaceChildren(el("option", { value: "", text: "Everyone" }), ...names.map((n) => el("option", { value: n, text: n })));
  who.value = names.includes(current) ? current : "";
  renderLines();
}

let audioPlayer = null, audioBtn = null;
function playLine(btn, kind, lineId) {
  if (audioPlayer) { audioPlayer.pause(); audioBtn && audioBtn.classList.remove("playing"); }
  if (audioBtn === btn && audioPlayer && !audioPlayer.ended && btn.dataset.playing === "1") {
    btn.dataset.playing = "0"; audioBtn = null; return;
  }
  audioPlayer = new Audio(`/api/jobs/${state.selected}/audio/${kind}/${lineId}?t=${Date.now()}`);
  audioBtn = btn;
  btn.dataset.playing = "1";
  btn.classList.add("playing");
  audioPlayer.addEventListener("ended", () => { btn.classList.remove("playing"); btn.dataset.playing = "0"; });
  audioPlayer.play().catch(() => { btn.classList.remove("playing"); toast("Couldn't play that line."); });
}

const saveTimers = {};
function queueSave(line, field, value, row) {
  clearTimeout(saveTimers[`${line.id}:${field}`]);
  saveTimers[`${line.id}:${field}`] = setTimeout(async () => {
    try {
      const r = await api(`/api/jobs/${state.selected}/lines/${line.id}`, { method: "PATCH", body: { [field]: value } });
      line[field] = value;
      line.locked = r.locked;
      state.edited.add(line.id);
      row.classList.add("edited");
      $(".lock input", row).checked = r.locked;
      $("#detail .revoice").classList.remove("hidden");
    } catch (e) { toast(`Not saved: ${e.message}`); }
  }, 500);
}

async function saveRating(line, key, value) {
  try {
    await api(`/api/jobs/${state.selected}/ratings/${line.id}`, { method: "PUT", body: { [key]: value } });
    line.rating[key] = value;
  } catch (e) { toast(`Rating not saved: ${e.message}`); }
}

function renderLines() {
  const d = $("#detail"), box = $(".lines", d);
  const q = $(".search", d).value.trim().toLowerCase(), who = $("select.who", d).value;
  const shown = state.lines.filter((l) => (!who || l.name === who) &&
    (!q || [l.ja, l.en, l.hint, l.delivery].some((t) => (t || "").toLowerCase().includes(q))));
  const opts = (vals, cur) => vals.map((v) => el("option", { value: v, text: v || "–", selected: String(cur) === v }));
  box.replaceChildren(...shown.map((l) => {
    const row = el("div", { class: `line${state.edited.has(l.id) ? " edited" : ""}` });
    const intensity = l.intensity == null ? null : Math.round(l.intensity * 100);
    const described = l.described ? Object.entries(l.described).filter(([, v]) => v !== "normal").map(([k, v]) => `${v} ${k === "melody" ? "melody" : k}`).join(", ") : "";
    const left = el("div", {},
      el("div", { class: "when", text: `${fmtTime(l.start)} · ${(l.end - l.start).toFixed(1)}s` }),
      el("div", { class: "who", text: l.name }));
    const playRow = el("div", { class: "play" },
      l.has_original ? el("button", { text: "▶ Original", onclick: (e) => playLine(e.currentTarget, "original", l.id) }) : null,
      l.has_dub ? el("button", { text: l.nonverbal ? "▶ Kept original" : "▶ Dub", onclick: (e) => playLine(e.currentTarget, "dub", l.id) }) : null,
      l.fit && !["natural", "ok", "original"].includes(l.fit) ? el("span", { class: `tag ${l.fit}`, text: l.fit === "overlong" ? "runs long" : l.fit }) : null);
    const right = el("div", {},
      el("div", { class: "ja", text: l.ja }),
      l.hint ? el("div", { class: "subline", text: `Subtitle: ${l.hint}` }) : null,
      el("div", {},
        l.emotion ? el("span", { class: "emo", title: `intensity ${intensity ?? "?"}%` }, l.emotion,
          intensity == null ? null : el("i", { style: `--w:${intensity}%` })) : null,
        described ? el("span", { class: "emo", text: described }) : null,
        l.nonverbal ? el("span", { class: "emo", text: "original sound kept" }) : null),
      playRow);
    if (!l.nonverbal) {
      const en = el("textarea", { rows: 2, "aria-label": "English line" });
      en.value = l.en || "";
      en.addEventListener("input", () => queueSave(l, "en", en.value, row));
      const delivery = el("input", { placeholder: "delivery, e.g. terrified, pleading", "aria-label": "Delivery note" });
      delivery.value = l.delivery || "";
      delivery.addEventListener("input", () => queueSave(l, "delivery", delivery.value, row));
      const lock = el("input", { type: "checkbox" });
      lock.checked = !!l.locked;
      lock.addEventListener("change", async () => {
        try { await api(`/api/jobs/${state.selected}/lines/${l.id}`, { method: "PATCH", body: { locked: lock.checked } }); l.locked = lock.checked; }
        catch (e) { toast(e.message); lock.checked = !lock.checked; }
      });
      right.append(el("div", { class: "edit" }, en,
        el("div", {}, delivery, el("label", { class: "lock", title: "Locked lines are kept when the script is rewritten" }, lock, "🔒 keep my version"))));
      if (l.has_dub) {
        const r = l.rating || {};
        const feel = el("select", { onchange: (e) => saveRating(l, "same_feeling_1to5", e.target.value) }, opts(["", "1", "2", "3", "4", "5"], r.same_feeling_1to5));
        const nat = el("select", { onchange: (e) => saveRating(l, "natural_english_1to5", e.target.value) }, opts(["", "1", "2", "3", "4", "5"], r.natural_english_1to5));
        const landed = el("select", { onchange: (e) => saveRating(l, "joke_or_intent_landed_y/n", e.target.value) }, opts(["", "y", "n"], r["joke_or_intent_landed_y/n"]));
        const notes = el("input", { class: "notes", placeholder: "notes" });
        notes.value = r.notes || "";
        notes.addEventListener("change", () => saveRating(l, "notes", notes.value));
        right.append(el("div", { class: "rate" }, "Same feeling", feel, "Natural English", nat, "Landed", landed, notes));
      }
    }
    row.append(left, right);
    return row;
  }));
  if (!shown.length) box.replaceChildren(el("p", { class: "hint", text: "No lines match." }));
}

// ---- start -----------------------------------------------------------------------------------
// A friend's private link carries ?key=…; pass it on to the Home Screen manifest so an iPhone app
// made from this page opens at the same link (Home Screen apps don't share Safari's cookies).
(() => {
  const key = new URLSearchParams(location.search).get("key");
  if (key) $("#manifest").href = `/manifest.webmanifest?key=${encodeURIComponent(key)}`;
})();
wireNewDub();
loadStatus();
refreshJobs();
setInterval(refreshJobs, 4000);
