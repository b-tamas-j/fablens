import pytest

from fablens import llm


def test_missing_model_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "load_dotenv", lambda *args: None)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(llm.ConfigError, match="LLM_MODEL is not set"):
        llm.chat_models()
