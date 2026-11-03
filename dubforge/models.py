"""The Segment: one spoken line, carried through every stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text_src: str                       # original-language line (Japanese)
    speaker: str = "SPEAKER_00"
    hint: str | None = None             # official subtitle text for this line, if any
    sub_name: str | None = None         # character name from ASS subtitles, if any
    words: list = field(default_factory=list)

    # analyze
    ref_audio: str | None = None        # original actor's isolated line (relative to work dir)
    prosody: dict = field(default_factory=dict)
    emotion: dict = field(default_factory=dict)
    nonverbal: bool = False

    # adapt
    text_dub: str | None = None         # English line to perform
    delivery: str | None = None         # short actor direction, e.g. "smug, teasing"
    intensity: float | None = None      # 0..1
    syllable_budget: int | None = None
    locked: bool = False                # set true after hand-editing so re-runs keep your text

    # synthesize / fit
    dub_audio: str | None = None
    voice_match: dict = field(default_factory=dict)  # what was done to sound like the original actor
    fit: dict = field(default_factory=dict)

    # evaluate
    scores: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
