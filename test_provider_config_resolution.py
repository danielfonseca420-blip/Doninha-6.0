from pipeline import _resolve_provider_settings


def test_resolve_provider_settings_prefers_explicit_model_for_cloud_provider():
    config = {
        "generation": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "ollama_model": "doninha8:latest",
        }
    }

    resolved = _resolve_provider_settings(config, section="generation")

    assert resolved["provider"] == "openai"
    assert resolved["model"] == "gpt-4o-mini"
    assert resolved["resolved_model"] == "gpt-4o-mini"


def test_resolve_provider_settings_falls_back_to_ollama_model_only_when_needed():
    config = {
        "finalization": {
            "provider": "ollama",
            "ollama_model": "doninha8:latest",
        }
    }

    resolved = _resolve_provider_settings(config, section="finalization")

    assert resolved["provider"] == "ollama"
    assert resolved["resolved_model"] == "doninha8:latest"
