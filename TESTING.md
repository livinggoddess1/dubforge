# Testing checklist

Work through these in order on your Mac. Each step only needs the ones before it.

## 1. Offline tests (2 minutes, no models)

```bash
cd dubforge
source .venv/bin/activate
pip install -e . pytest
python -m pytest tests -q
```

Expect `48 passed`. They need `ffmpeg` but no ML models.

## 2. Check your install

```bash
dubforge doctor
```

Everything under "Stages" should be ✓ except the optional ones. torch and torchaudio should both be 2.6.0.

## 3. Plumbing check on a real episode (no voice model)

```bash
dubforge run ~/Anime/ep01.mkv -c config.yaml --clip 300:120 --tts say
```

The voice will sound robotic; that's expected. What to check:

- [ ] The log says `subtitles: auto → …` with the track you expected. On a dual-audio file, no warning about untagged audio tracks (if there is one, add `--audio-stream N`).
- [ ] `dubforge show ep01.mkv -c config.yaml --clip 300:120`: short words like はい / なに？ / まって have English lines; only gasps, sighs and laughs say `original performance`.
- [ ] With `adapt.backend: anthropic`, the adapt stage writes lines (it used to fail on the `temperature` setting).
- [ ] With `--style verbatim`, a line spoken without a subtitle still shows up in `show` with an English line.
- [ ] The timing summary at the end of synthesize shows few or no `overlong` lines (they are now rewritten shorter).
- [ ] `dubbed/…mkv` opens in IINA with the English dub, the Japanese track and the subtitles.

## 4. Real voices

```bash
dubforge run ~/Anime/ep01.mkv -c config.yaml --clip 300:120 --from synthesize
```

- [ ] Loud scenes don't sound fuzzy or crunchy across the board (the old limiter distorted the whole mix).
- [ ] Open `dub_work/<name>/eval/ratings.csv`: each row has a `take` and a `dub_file`. Rate a few lines.
- [ ] Re-run with a setting changed (e.g. `--tts indextts2`, or switch `tts.indextts2.emotion_source`). The lines are re-voiced, and ratings.csv gains new rows while your earlier ratings stay next to their own audio.

## 5. Head-to-head with ElevenLabs / HeyGen / Rask

1. Dub the **same full episode** with DubForge and with the other tool. Use the same source video for both.
2. Pick 10+ moments, about 20 seconds each, in episode time (seconds). Mix calm talk, a shouting match, a sad scene, a joke, and a scene with three or more characters.
3. Make the blind test:

   ```bash
   dubforge abtest make dubbed/ep01.dub.en.mkv ~/Downloads/ep01_elevenlabs.mp4 \
       --clips 300:20,345:20,410:20,500:20,610:20,700:20,820:20,905:20,1010:20,1130:20 \
       --name-b ElevenLabs -o abtest/elevenlabs
   ```

4. Listen to each `clip_NN_X.wav` / `clip_NN_Y.wav` pair and fill in `abtest/elevenlabs/sheet.csv`. Don't open `key.json`.
5. `dubforge abtest score abtest/elevenlabs`

Repeat with a new `-o` folder for HeyGen and Rask. Their exports usually have one audio track; if one has several, pick it with `--stream-b N`.

If you lose on a clip, note why in the `notes` column (wrong emotion, unnatural English, voice doesn't fit, timing). Those notes show what to work on next.
