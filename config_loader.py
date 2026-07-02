"""
Carregamento de configurao centralizada.
==========================================
L config.yaml (ou variveis de ambiente) e expe um nico dicionrio
para pipeline, API e agentes.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict

# Diretrio raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _normalize_agent_flag(value: Any) -> bool:
    """Normaliza a flag de uso do agente.

    A intenção do projeto é que o agente seja usado automaticamente.
    Mesmo quando o valor configurado vier como False, este helper
    garante que a execução continue com o agente ativo.
    """
    if value is False:
        return True
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "false", "off", "no", "disabled", "none", "null", ""}:
            return True
        return lowered not in {"", "0"} and bool(lowered)
    return bool(value)


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Carrega a configurao a partir de config.yaml.
    Se o arquivo no existir ou estiver incompleto, usa defaults e env.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    config: Dict[str, Any] = {
        "knowledge_base": {
            "path": "",
            "chroma_path": os.getenv("VECTOR_DB_PATH", "meu_vector_db"),
            "default_kb": "",
            "domain_specific_kbs": {},
        },
        "l3": {
            "model_path": "truth_scoring_model.pt",
            "backbone": "cross-encoder/nli-MiniLM2-L6-H768",
        },
        "l4": {
            "russell_concepts_path": "l4_russell_concepts.json",
        },
        "l4_chain_verification": {
            "provider": os.getenv("L4_COVE_PROVIDER", "template"),
            "model": os.getenv("L4_COVE_MODEL", ""),
            "base_url": os.getenv("L4_COVE_BASE_URL", ""),
            "api_key": os.getenv("L4_COVE_API_KEY", ""),
            "ollama_model": os.getenv("L4_COVE_OLLAMA_MODEL", "doninha8:latest"),
            "ollama_host": os.getenv("L4_COVE_OLLAMA_HOST", "http://localhost:11434"),
            "custom_lm_path": "",
        },
        "generation": {
            "provider": os.getenv("GENERATION_PROVIDER", "ollama"),
            "model": os.getenv("GENERATION_MODEL", ""),
            "base_url": os.getenv("GENERATION_BASE_URL", ""),
            "api_key": os.getenv("GENERATION_API_KEY", ""),
            "custom_lm_path": "",
            "ollama_model": os.getenv("OLLAMA_MODEL", "doninha8:latest"),
            "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        },
        "finalization": {
            "provider": os.getenv("FINALIZATION_PROVIDER", "ollama"),
            "model": os.getenv("FINALIZATION_MODEL", ""),
            "base_url": os.getenv("FINALIZATION_BASE_URL", ""),
            "api_key": os.getenv("FINALIZATION_API_KEY", ""),
            "custom_lm_path": "",
            "ollama_model": os.getenv("FINALIZATION_OLLAMA_MODEL", "doninha8:latest"),
            "ollama_host": os.getenv("FINALIZATION_OLLAMA_HOST", "http://localhost:11434"),
        },
        "l7": {
            "provider": os.getenv("L7_PROVIDER", "ollama"),
            "model": os.getenv("L7_MODEL", ""),
            "base_url": os.getenv("L7_BASE_URL", ""),
            "api_key": os.getenv("L7_API_KEY", ""),
            "custom_lm_path": "",
            "ollama_host": os.getenv("L7_OLLAMA_HOST", "http://localhost:11434"),
        },
        "agent": {
            "use_agent": _normalize_agent_flag(os.getenv("USE_AGENT", "true")),
            "vector_db_path": os.getenv("VECTOR_DB_PATH", "meu_vector_db"),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "api": {
            "host": os.getenv("API_HOST", "0.0.0.0"),
            "port": int(os.getenv("API_PORT", "8000")),
        },
        "chat": {
            "max_turns_in_context": 10,
        },
    }

    if path.exists():
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                _deep_merge(config, loaded)
        except Exception:
            pass

    _apply_env_overrides(config)

    if "agent" in config and isinstance(config["agent"], dict):
        config["agent"]["use_agent"] = _normalize_agent_flag(config["agent"].get("use_agent", True))

    # Resolve paths relativos ao projeto
    for key in ["model_path", "russell_concepts_path", "custom_lm_path", "path", "default_kb"]:
        for section in ["l3", "l4", "l4_chain_verification", "generation", "finalization", "l7", "knowledge_base"]:
            if section in config and key in config[section]:
                val = config[section][key]
                if val and not Path(val).is_absolute():
                    config[section][key] = str(PROJECT_ROOT / val)

    if "knowledge_base" in config and isinstance(config["knowledge_base"].get("domain_specific_kbs"), dict):
        for name, value in config["knowledge_base"]["domain_specific_kbs"].items():
            if value and not Path(value).is_absolute():
                config["knowledge_base"]["domain_specific_kbs"][name] = str(PROJECT_ROOT / value)

    return config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Mescla override em base recursivamente."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    """Aplica overrides de ambiente com precedncia sobre o YAML."""
    env_map = {
        "l4_chain_verification": {
            "provider": "L4_COVE_PROVIDER",
            "model": "L4_COVE_MODEL",
            "base_url": "L4_COVE_BASE_URL",
            "api_key": "L4_COVE_API_KEY",
            "ollama_model": "L4_COVE_OLLAMA_MODEL",
            "ollama_host": "L4_COVE_OLLAMA_HOST",
        },
        "generation": {
            "provider": "GENERATION_PROVIDER",
            "model": "GENERATION_MODEL",
            "base_url": "GENERATION_BASE_URL",
            "api_key": "GENERATION_API_KEY",
            "custom_lm_path": "GENERATION_CUSTOM_LM_PATH",
            "ollama_model": "OLLAMA_MODEL",
            "ollama_host": "OLLAMA_HOST",
        },
        "finalization": {
            "provider": "FINALIZATION_PROVIDER",
            "model": "FINALIZATION_MODEL",
            "base_url": "FINALIZATION_BASE_URL",
            "api_key": "FINALIZATION_API_KEY",
            "custom_lm_path": "FINALIZATION_CUSTOM_LM_PATH",
            "ollama_model": "FINALIZATION_OLLAMA_MODEL",
            "ollama_host": "FINALIZATION_OLLAMA_HOST",
        },
        "l7": {
            "provider": "L7_PROVIDER",
            "model": "L7_MODEL",
            "base_url": "L7_BASE_URL",
            "api_key": "L7_API_KEY",
            "custom_lm_path": "L7_CUSTOM_LM_PATH",
            "ollama_host": "L7_OLLAMA_HOST",
        },
        "knowledge_base": {
            "path": "VECTOR_DB_PATH",
            "chroma_path": "VECTOR_DB_PATH",
        },
        "agent": {
            "use_agent": "USE_AGENT",
            "vector_db_path": "VECTOR_DB_PATH",
        },
        "api": {
            "host": "API_HOST",
            "port": "API_PORT",
        },
    }

    for section, mapping in env_map.items():
        if section not in config or not isinstance(config[section], dict):
            continue
        for key, env_var in mapping.items():
            env_value = os.getenv(env_var)
            if env_value is None:
                continue
            if key == "port" and env_value.isdigit():
                config[section][key] = int(env_value)
            elif key == "use_agent" and env_value.lower() in {"1", "true", "yes", "on"}:
                config[section][key] = True
            elif key == "use_agent":
                config[section][key] = False
            else:
                config[section][key] = env_value
