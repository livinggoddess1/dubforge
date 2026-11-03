"""LLM backends for script adaptation. Each exposes complete(system, user) -> str."""

from __future__ import annotations

import json
import re
import urllib.request


class AnthropicLLM:
    def __init__(self, cfg):
        import anthropic

        # The SDK finds credentials itself (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an
        # `ant auth login` profile), so don't insist on one particular environment variable.
        self.client, self.cfg = anthropic.Anthropic(), cfg

    def complete(self, system: str, user: str) -> str:
        # No temperature: current Claude models reject sampling parameters with a 400 error.
        import anthropic

        try:
            msg = self.client.messages.create(model=self.cfg["model"], max_tokens=self.cfg["max_tokens"],
                                              system=system, messages=[{"role": "user", "content": user}])
        except anthropic.AuthenticationError as e:
            raise SystemExit("Claude API rejected the credentials. Set ANTHROPIC_API_KEY "
                             "(console.anthropic.com) or choose another adapt.backend.") from e
        if msg.stop_reason == "refusal":
            raise RuntimeError("the model declined this batch (stop_reason=refusal)")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("reply was cut off at adapt.max_tokens; raise it or lower adapt.batch_size")
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OllamaLLM:
    def __init__(self, cfg):
        self.cfg = cfg

    def complete(self, system: str, user: str) -> str:
        body = json.dumps({"model": self.cfg["model"], "stream": False,
                           "options": {"temperature": self.cfg["temperature"], "num_ctx": 16384},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(self.cfg["ollama_url"].rstrip("/") + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["message"]["content"]


class MlxLLM:
    def __init__(self, cfg):
        from mlx_lm import load

        self.cfg = cfg
        self.model, self.tok = load(cfg["model"])

    def complete(self, system: str, user: str) -> str:
        from mlx_lm import generate

        prompt = self.tok.apply_chat_template([{"role": "system", "content": system},
                                               {"role": "user", "content": user}],
                                              add_generation_prompt=True, tokenize=False)
        return generate(self.model, self.tok, prompt=prompt, max_tokens=self.cfg["max_tokens"])


class PassthroughLLM:
    """No model: uses the subtitle hint (or the source text). For testing the plumbing."""

    def __init__(self, cfg):
        self.cfg = cfg

    def complete(self, system: str, user: str) -> str:
        lines = json.loads(user.split("LINES_JSON:", 1)[1].strip())
        return json.dumps([{"id": l["id"], "en": l.get("current_en") or l.get("subtitle_hint") or l["ja"],
                            "delivery": l.get("measured", {}).get("emotion", "neutral"),
                            "intensity": l.get("measured", {}).get("intensity", 0.5)} for l in lines])


BACKENDS = {"anthropic": AnthropicLLM, "ollama": OllamaLLM, "mlx_lm": MlxLLM, "passthrough": PassthroughLLM}


def get_llm(cfg: dict):
    b = cfg["backend"]
    if b not in BACKENDS:
        raise SystemExit(f"Unknown adapt.backend {b!r}. Options: {', '.join(BACKENDS)}")
    return BACKENDS[b](cfg)


def parse_json_array(text: str) -> list:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    a, b = text.find("["), text.rfind("]")
    if a < 0 or b <= a:
        raise ValueError("No JSON array in model output")
    return json.loads(text[a:b + 1])
