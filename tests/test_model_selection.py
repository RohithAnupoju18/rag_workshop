from core.model_selector import SETUP_INSTRUCTIONS, family_of, resolve_model


def test_qwen_preferred_when_both_installed():
    choice = resolve_model(["llama3.2:latest", "qwen2.5:3b"])
    assert choice.family == "qwen" and choice.model == "qwen2.5:3b"


def test_llama_fallback():
    choice = resolve_model(["llama3.1:8b", "mistral:7b"])
    assert choice.family == "llama" and "falling back" in choice.message


def test_no_model_shows_setup_message():
    choice = resolve_model(["mistral:7b", "nomic-embed-text"])
    assert not choice.ok and choice.message == SETUP_INSTRUCTIONS and "ollama pull" in choice.message


def test_empty_list():
    assert not resolve_model([]).ok


def test_qwen3_variants_and_embedding_models_skipped():
    choice = resolve_model(["qwen3-embedding:0.6b", "qwen2.5-coder:7b", "qwen3:4b"])
    assert choice.model == "qwen3:4b"  # general chat model beats coder; embedding model never chosen
    assert resolve_model(["qwen3-embedding:0.6b"]).ok is False


def test_case_insensitive_and_family_of():
    assert resolve_model(["Qwen2.5:7B"]).family == "qwen"
    assert family_of("llama3.2:latest") == "llama" and family_of(None) is None
