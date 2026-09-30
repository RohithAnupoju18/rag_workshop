"""Talks to the local Ollama server over HTTP (http://localhost:11434). No API key."""
from __future__ import annotations

import shutil

import requests

OLLAMA_URL = "http://localhost:11434"


class OllamaError(Exception):
    pass


def is_installed() -> bool:
    return shutil.which("ollama") is not None


def is_running(base_url: str = OLLAMA_URL, timeout: float = 2.0) -> bool:
    try:
        return requests.get(f"{base_url}/api/tags", timeout=timeout).ok
    except requests.RequestException:
        return False


def list_models(base_url: str = OLLAMA_URL, timeout: float = 3.0) -> list[str]:
    """Names of locally installed models, e.g. ['qwen2.5:3b', 'llama3.2:latest']."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    except requests.RequestException as exc:
        raise OllamaError("Ollama is not reachable. Start it with 'ollama serve' (or open the Ollama app).") from exc


def chat(
    model: str,
    messages: list[dict],
    temperature: float = 0.2,
    base_url: str = OLLAMA_URL,
    timeout: float = 180.0,
) -> str:
    payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}}
    try:
        response = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        if response.status_code == 404:
            raise OllamaError(f"Model '{model}' is not installed. Run: ollama pull {model}")
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except requests.Timeout as exc:
        raise OllamaError("The model took too long to answer. Try a smaller model or a smaller Top-K.") from exc
    except requests.ConnectionError as exc:
        raise OllamaError("Lost connection to Ollama. Is it still running?") from exc
    except (requests.RequestException, KeyError, ValueError) as exc:
        raise OllamaError(f"Ollama returned an unexpected response ({exc.__class__.__name__}).") from exc
