# DubForge

An expressive Japanese → English dubbing pipeline that runs on an Apple Silicon Mac.

It doesn't just translate and read lines aloud. It **measures how each original line was performed** (emotion, loudness, pitch, melody, pace, relative to that character's normal voice), hands that to a script-adaptation model along with mouth-flap timing, and then has a voice model **perform** the English line in the character's voice with the original's emotional shape. Output is a normal `.mkv` you can play in IINA/VLC with an English dub track, the original Japanese track, and subtitles.

Every stage writes plain files into a work folder. You can stop anywhere, look, edit the script by hand, swap any model, and resume. That's the point: layer 1 is a working machine built from the best open parts; later layers replace parts with your own.

---

## How it works

```
 episode.mkv
     │
 1 extract ──── audio/original.wav (+ subtitles from the file or a .srt/.ass)
 2 separate ─── RoFormer (or Demucs) → audio/vocals.wav  +  audio/background.wav (music & effects)
 3 transcribe ─ Whisper (MLX, on the M4 GPU) → timed Japanese lines; official subs attached as hints
 4 diarize ──── pyannote → who speaks each line
 5 analyze ──── emotion2vec + pitch/energy/pace → per-line performance, z-scored per character
 6 adapt ────── LLM rewrites for English: intent, jokes, character voice, syllable budget, delivery note
 7 synthesize ─ IndexTTS (emotion from the actor's own audio) or Chatterbox voices each character
                → time-fit (pitch-preserving stretch; too-long lines get rewritten shorter, then re-voiced)
 8 mix ──────── dub lines on the original background, loudness- and tone-matched to the original line
 9 mux ──────── dubbed/<name>.dub.en.mkv   (English dub default, Japanese kept, dub-script subs)
10 evaluate ─── how well did the dub track the original's emotion? + files for rating by ear
```

**The core idea in code.** Each original line becomes numbers in `script.json`:

```json
"emotion": {
  "label": "angry", "scores": {"angry": 0.71, "neutral": 0.18, ...},
  "z": {"rms_db": 1.9, "f0_median_st": 0.8, "f0_range_st": 1.4, "rate": 0.6},
  "described": {"energy": "loud", "pitch": "high", "melody": "very animated", "pace": "normal"},
  "intensity": 0.86
}
```

`z` means "compared to how *this character* normally sounds", so a quiet character's outburst and a loud character's normal line aren't confused. Those numbers steer three things: the LLM's word choice and delivery note, the voice model's emotion controls, and the evaluation at the end, which checks whether the dub rises and falls in the same places as the original.

---

## Setup on your M4 Mac

You need [Homebrew](https://brew.sh). Use **Python 3.11**; the ML libraries here are most reliable on it.

```bash
brew install ffmpeg python@3.11
cd dubforge
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# Everything, held to the torch version all the models agree on (constraints.txt).
# Always pass -c constraints.txt when installing ML packages into this venv; without it pip
# can silently upgrade torch, which breaks both Chatterbox and pyannote.
pip install -c constraints.txt -e ".[mac]"

dubforge doctor          # shows what's installed and what's missing
```

**Keys** (put these in `~/.zshrc` or export them in the terminal):

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # console.anthropic.com — for script adaptation
export HF_TOKEN=hf_...                # huggingface.co/settings/tokens — for pyannote
```

For pyannote, also open these two pages on Hugging Face while signed in and accept their terms: `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.

Models download automatically the first time each stage runs. Expect several GB in total.

### Don't want to pay for an API?

Run the adapter locally. Either install [Ollama](https://ollama.com), run `ollama pull qwen3:14b`, and set `adapt.backend: ollama`, `adapt.model: qwen3:14b`; or `pip install mlx-lm` and use `adapt.backend: mlx_lm`. A 14B model fits comfortably with 16 GB+ of memory. Expect noticeably weaker joke and character work than a frontier model; this stage is where model quality shows the most.

---

## Using it in your browser

```bash
dubforge web -c config.yaml
```

This opens DubForge in your browser (http://127.0.0.1:8765). Drop in a video, pick the part you want, and press **Start dubbing**. You can watch each step as it runs, play the result with the English dub or the original Japanese, and download the `.mkv`. Under **Script**, every line has buttons to hear the original and the dub; edit the English or the delivery note, rate it, and press **Re-voice edited lines**. Leave the Terminal window open while you use it; press Control + C there to stop.

**Friends anywhere (no password):** install Cloudflare's free tunnel once, then start with `--share`:

```bash
brew install cloudflared
dubforge web -c config.yaml --share
```

It prints a private link; send it to your friends. They click it and they're in, from any country, with nothing to type. Anyone without the link sees nothing, and only you (at this Mac) can dub files that are already on your Mac; friends upload theirs. Your Mac stays awake while sharing. The link stops working when you press Control + C; the next `--share` reuses it with a new web address, so send the new link. `--new-link` also changes the secret part, for when a link has been passed on too far.

**On an iPhone:** open the link in Safari, tap the Share button, then **Add to Home Screen**. DubForge gets its own icon and opens full screen like an app (a friend's app keeps their private link). Upload straight from Photos or Files, watch the dub in the page, and use **Save video (.mp4)** to keep it on the phone (iPhones can't play .mkv). The dubbing itself still happens on your Mac, so it has to be on and sharing.

Everything runs on your Mac and your API keys: friends' dubs use your Claude credit and queue behind yours (one at a time). Uploads and jobs are kept in `web_data/`.

---

## Running in the cloud (works while your Mac is off)

For friends to use DubForge any time, put it on [Modal](https://modal.com): the page runs on a small cloud machine that sleeps when nobody's using it, and each dub runs on a rented Nvidia GPU that starts when someone presses **Start dubbing** and shuts off when it's done. You pay only while a dub runs; Modal's free plan includes monthly credit.

```bash
cd ~/Downloads/dubforge
source .venv/bin/activate
bash deploy/deploy.sh
```

The first time, it opens a browser to sign in to Modal, asks for your Claude and Hugging Face keys if they aren't in `~/.zshrc`, builds the cloud image (15–30 minutes) and prints the link. The link is permanent: run `deploy.sh` again after updating DubForge and it stays the same. To make a new link (the old one stops working), delete `deploy/.share_key` and deploy again.

Differences from the Mac version: Whisper runs as faster-whisper on the GPU, and burned-in subtitles are read with Tesseract (less accurate than Apple's text recognition; subtitle files and subtitle tracks are unaffected). Settings live in `deploy/cloud_config.yaml`; choose the GPU with `DUBFORGE_GPU=L4 bash deploy/deploy.sh` (cheaper, slower) or `L40S` (faster). At most 2 dubs run at once (`DUBFORGE_MAX_PARALLEL`), which caps spending. Usage and costs: https://modal.com/apps.

---

## First run: do a 2-minute clip

```bash
cp config.example.yaml config.yaml        # edit it
dubforge run ~/Anime/ep01.mkv -c config.yaml --clip 300:120
```

`--clip 300:120` means start at 5:00, do 120 seconds. Iterate on clips; do whole episodes once you like the settings.

The result lands in `dubbed/`. Open it in IINA or VLC; the English dub is the default audio track and you can flip to Japanese to compare.

### The editing loop (where quality comes from)

```bash
dubforge run ep01.mkv -c config.yaml --clip 300:120 --to adapt   # stop after the script is written
dubforge show ep01.mkv -c config.yaml --clip 300:120             # read it: JP, emotion, EN, delivery
```

Open `dub_work/<name>/script.json`, fix any line's `text_dub` or `delivery`, and set `"locked": true` on lines you edit so re-runs keep your version. Then:

```bash
dubforge run ep01.mkv -c config.yaml --clip 300:120 --from synthesize
```

Other useful moves:

- Name the characters: after the first run, `dubforge show` lists `SPEAKER_00`, `SPEAKER_01`, … Put real names in `speakers.names`, then `--from adapt`. The LLM writes much better knowing who is talking.
- Write a show bible (`show_bible.example.md`): character personalities, catchphrases, running jokes. Biggest single quality lever.
- Re-voice only: `--from synthesize` reuses everything before it. Unchanged lines are cached and not re-rendered.
- `--only mix` / `--only mux` to rebuild just the audio or the file.

---

## Choosing how close the English stays to the subtitles

`adapt.style` in `config.yaml` (or `--style` on the command line):

| Style | What the English dub says |
|---|---|
| `faithful` | The subtitle wording, changed only to be speakable and to fit timing. Without subtitles, a close, careful translation of the Japanese: no localized jokes or swapped idioms. |
| `verbatim` (default) | The subtitle text word for word. Speech with no subtitle still gets dubbed, with a `faithful` translation. Expect more stretched lines. |
| `natural` | A free English adaptation that rewrites lines to sound like native English speech, the way most commercial dubs do. |

To match a particular subtitle translation, DubForge needs it as text: either subtitles inside the video file (`subtitles.path: embedded`) or a separate `.srt`/`.ass` file (`--subs path/to/file.srt`). With several subtitle tracks, `auto` picks the one in `subtitles.lang` and skips "Signs & Songs" tracks; set `subtitles.stream` to an index to choose yourself.

**Dual-audio files.** DubForge uses the audio track tagged Japanese. If the tracks aren't tagged, it uses the first one and warns you; pick the right one with `--audio-stream N` (0 = first track).

**Subtitles burned into the picture (hardsubs).** With `subtitles.path: auto` and no subtitle text in the file, DubForge first samples a couple of dozen frames; it reads the picture only if text is on screen in at least 25% of them (`subtitles.ocr.auto_min_coverage`), so a raw episode's credits and signs aren't mistaken for dialogue. Force it with `--subs ocr`. DubForge can read them off the video with your Mac's built-in text recognition (`pip install ocrmac`, included in `.[mac]`):

```bash
dubforge run ep01.mp4 -c config.yaml --clip 600:120 --subs ocr --style verbatim
```

It saves what it read to `dub_work/<name>/subs/ocr.srt`. Text recognition can misread a word now and then, so for an exact match open that file, fix any mistakes, and rerun with `--from extract`; your edited file is used from then on (delete it to read the video again). To read a whole episode into one file you can proofread once and reuse for every clip: `dubforge ocr-subs ep01.mp4` writes `ep01.ocr.srt` next to the video; then pass `--subs ep01.ocr.srt`. Settings live under `subtitles.ocr` (`fps`, `crop_bottom` if your subtitles sit higher on screen).

---

## Getting the most realistic result

The goal is what a Japanese viewer hears, in English: the actor's performance, in the scene's sound. Three optional installs get closest:

1. **IndexTTS** carries the emotion straight from the original actor's audio for each line: `bash scripts/install_indextts.sh`.
2. **RoFormer separation** pulls voices out of the music and effects much more cleanly than Demucs, so less of the Japanese performance is left under the English and the emotion reference is cleaner: `pip install -c constraints.txt audio-separator`. `separate.backend: auto` uses it when installed and falls back to Demucs if it fails.
3. **Tone matching** is on by default (`mix.match_tone`): each English line takes the original line's tonal colour, so a phone call, a voice through a door or a shout down a hall sounds like it's in the scene rather than in a booth.

Run `dubforge doctor` to see what `auto` will use. Written stutters ("P-Please") are spoken as the plain word with the actor's measured emotion; compared by ear, every spelled-out stutter sounded slow and staged. The subtitles keep the stutter.

---

## Choosing the voice model

| | **Chatterbox** (used when IndexTTS isn't installed) | **IndexTTS 2 / 2.5** (used automatically when installed) |
|---|---|---|
| Emotion control | One "exaggeration" knob, driven by measured intensity | Emotion from a separate audio clip, an 8-value vector, or a text description |
| Who it hears | One reference clip (`tts.reference`) | Speaker clip for *voice* + original line for *emotion* |
| Setup | `pip install` | `bash scripts/install_indextts.sh` (~5 GB weights) |
| Why use it | Easy, stable, fast to iterate | The closest open match to your idea: keep the voice, carry the performance across |

Chatterbox: `tts.reference: auto` clones from the line itself when it's 2s+ long (carries more of that moment's emotion) and from the character's 12-second voice print otherwise. `cfg_weight: 0` keeps a Japanese accent from bleeding into the English.

IndexTTS2: three ways to set emotion via `tts.indextts2.emotion_source`. Comparing them is the central experiment of this project:

- `audio`: the original Japanese line's audio drives the emotion directly
- `vector`: the measured emotion2vec probabilities become an explicit 8-number emotion vector (the "mathematical" route)
- `text`: the LLM's delivery note ("smug, drawn-out tease") is interpreted by the model

**Installing IndexTTS (recommended).** It needs its own environment (different torch than Chatterbox), but you never switch environments yourself: one script installs it, and `tts.backend: auto` (the default) uses it whenever it's installed.

```bash
bash scripts/install_indextts.sh
```

This installs IndexTTS 2.5 (Bilibili's cross-language emotion-transfer release) into `~/index-tts` when that version is available, otherwise IndexTTS 2; pass `2` to force version 2. `dubforge doctor` shows which one `auto` will use. The voice comes from the character's voice print (`refs/speakers/`), the emotion from the original actor's line. On a Mac it runs on PyTorch's MPS backend and is noticeably slower than Chatterbox.

Community MLX ports of IndexTTS2 for Apple Silicon exist (`mlx-indextts`, `index-tts-2.5-mlx`) and are faster than the official PyTorch-on-Mac path. Wiring one in as a backend is a good first "layer 2" project; see `dubforge/backends/tts.py` for how small a backend is.

**Sounding like the same person.** After each English line is spoken, `tts.voice_match` makes it sound like the original actor switched languages: `convert` runs Chatterbox's voice conversion into the character's voice print (`dub_work/<name>/refs/speakers/`), so the timbre stays identical across lines; `pitch` shifts the line to the average pitch the actor used on that line, keeping the voice's character (formants) intact. The voice print is built from the speaker's cleanest lines, so correct speaker detection matters: if two characters got merged into one `SPEAKER_xx`, their voices blend. Both are off by default: in testing the plain Chatterbox clone sounded more natural, while conversion added a slightly synthetic quality and pitch shifting added lag and static. Try them with `--convert` / `--pitch`.

**Quick sanity check with no voice model at all:** `--tts say` uses the macOS system voice. It sounds robotic but proves timing and mixing end to end.

---

## Reading the evaluation

After each run, `dub_work/<name>/eval/` contains:

- `summary.json`: overall numbers
  - `emotion_cos_mean`: how similar the dub's emotion embedding is to the original's (higher is better)
  - `emotion_label_agreement`: fraction of lines whose top emotion matches
  - `energy_tracking`, `melody_tracking`, `pitch_tracking`: correlation between the original's and the dub's per-character z-scores. **1.0 means the dub gets louder or more animated exactly where the original does.** This is the "did we preserve the shape of the performance" score.
  - `timing_status`: how many lines fit as-is, were stretched, or are still too long
- `report.csv`: line-by-line numbers, open in Numbers/Excel
- `pairs/`: `0042_a_original.wav` next to `0042_b_dub.wav` for every line
- `ratings.csv`: a sheet to rate lines by ear (same feeling 1–5, natural English 1–5, did the joke land). Every rendered version of a line (a *take*) gets its own row and its own file (`pairs/0042_b_dub_<take>.wav`), so re-rendering adds rows and never changes audio you've already rated.

**The ratings are the most valuable output of this whole project.** Once you've rated a few hundred lines, you can test your hypothesis directly: do the numbers above predict your ratings? If they do, you've shown that expressive equivalence across languages is measurable. If they don't, the gap shows you what the math is missing.

---

## Testing against other dubbing tools

To find out whether your dub beats ElevenLabs, HeyGen or Rask, compare them blind on the same scenes. Make the other dub of the same episode, then:

```bash
dubforge abtest make dubbed/ep01.dub.en.mkv ~/Downloads/ep01_elevenlabs.mp4 \
    --clips 300:20,345:20,410:20,500:20,610:20,700:20,820:20,905:20,1010:20,1130:20 \
    --name-b ElevenLabs -o abtest/elevenlabs
```

Each clip becomes `clip_NN_X.wav` and `clip_NN_Y.wav`, loudness-matched, with X and Y shuffled per clip. Fill in `sheet.csv` without opening `key.json`, then:

```bash
dubforge abtest score abtest/elevenlabs
```

Pick scenes with different emotions, include at least one joke, and rate 10 or more clips. If your dub is a `--clip` run, pass its start with `--offset-a` (e.g. `--offset-a 300`). See `TESTING.md` for the full checklist.

---

## Project layout

```
dubforge/
  cli.py            dubforge run | show | ocr-subs | abtest | doctor
  abtest.py         blind A/B listening test against another dub
  pipeline.py       stage order, work folder, resume logic
  config.py         every setting and its default
  models.py         Segment: one line, carried through all stages
  prosody.py        the measurable-expression layer (pitch, energy, pace, z-scores, intensity)
  fit.py            timing: slots, stretch limits
  audio.py          ffmpeg, loudness, trimming, time-stretch
  subtitles.py      .srt/.ass/.vtt reading, embedded-track extraction
  stages/           one file per stage (extract … evaluate)
  backends/
    llm.py          anthropic | ollama | mlx_lm | passthrough
    tts.py          chatterbox | indextts2 | say | tone
    emotion.py      emotion2vec wrapper
tests/              offline tests (no models needed): python -m pytest tests -q
```

To add a backend, write a class with the same method as its neighbors and add it to the `BACKENDS` dict in that file. Nothing else needs to change.

---

## Roadmap: the layers

**Layer 1: this.** Open models wired together, with measurement built in. Goal: watchable dubs and a feel for where quality breaks.

**Layer 2: better parts and your own data.**
- Swap in the MLX IndexTTS2 port for speed; try a Japanese-specialized ASR (kotoba-whisper) if transcription errors hurt.
- Use the `ratings.csv` data to tune the intensity formula in `prosody.py` and the exaggeration mapping in `tts.py`, so the numbers predict what *you* hear as right.
- Per-line A/B: render each line with two settings and keep the one you rate higher. That builds a preference dataset.

**Layer 3: your own models.**
- Train a small model that maps a Japanese line's prosody contour (frame-level pitch/energy, not just averages) to an English target contour. Your rated pairs are its training data. This is the "learned mapping between languages" from earlier: not copying the pitch, but learning what the equivalent English shape is.
- Fine-tune a TTS model on licensed or self-recorded bilingual performances so it learns English delivery that matches anime acting conventions.
- Lip-sync: score each English line against mouth movement in the video, and feed that back into the adapter's budget.

---

## A note on use

Cloning a voice actor's voice is fine for your own private viewing of media you have. Don't distribute dubs made with cloned voices: that's both a copyright issue and a voice-rights issue that actors are actively fighting over. If you ever want to share or build something public, use `tts.voices` to point characters at voices from people who've agreed, and license the source material. Chatterbox also watermarks its output inaudibly.
