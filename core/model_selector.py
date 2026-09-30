"""Runtime model detection:  Qwen (preferred)  ->  Llama (fallback)  ->  setup message."""
from __future__ import annotations

from dataclasses import dataclass

SETUP_INSTRUCTIONS = (
    "No Qwen or Llama model found in Ollama.\n"
    "Install one (the app never downloads models silently):\n"
    "  ollama pull qwen2.5:3b     (preferred, ~2 GB)\n"
    "  ollama pull llama3.2       (fallback, ~2 GB)\n"
    "Then click 'Refresh models'."
)

_SKIP_WORDS = ("embed",)                # embedding-only models cannot chat
_LOW_PRIORITY_WORDS = ("coder", "vl", "math", "guard")  # specialised variants: use only if nothing better


@dataclass
class ModelChoice:
    model: str | None
    family: str | None      # "qwen" | "llama" | None
    message: str

    @property
    def ok(self) -> bool:
        return self.model is not None


def _candidates(installed: list[str], family: str) -> list[str]:
    found = [
        name for name in installed
        if family in name.lower() and not any(w in name.lower() for w in _SKIP_WORDS)
    ]
    # general chat models first, specialised variants last; alphabetical keeps it deterministic
    return sorted(found, key=lambda n: (any(w in n.lower() for w in _LOW_PRIORITY_WORDS), n))


def resolve_model(installed: list[str]) -> ModelChoice:
    qwen = _candidates(installed, "qwen")
    if qwen:
        return ModelChoice(qwen[0], "qwen", f"Using Qwen model: {qwen[0]}")
    llama = _candidates(installed, "llama")
    if llama:
        return ModelChoice(llama[0], "llama", f"Qwen not found - falling back to Llama model: {llama[0]}")
    return ModelChoice(None, None, SETUP_INSTRUCTIONS)


def family_of(model_name: str | None) -> str | None:
    lowered = (model_name or "").lower()
    if "qwen" in lowered:
        return "qwen"
    if "llama" in lowered:
        return "llama"
    return None
