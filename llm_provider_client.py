from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

__all__ = [
    "SUPPORTED_PROVIDERS",
    "DEFAULT_MODELS",
    "normalize_provider",
    "generate_text",
    "generate_provider_text",
]

SUPPORTED_PROVIDERS = {
    "ollama",
    "openai",
    "anthropic",
    "gemini",
    "grok",
    "groq",
    "meta",
    "template",
    "custom_lm",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "gemini": "gemini-1.5-flash",
    "grok": "grok-2-latest",
    "groq": "llama-3.1-8b-instant",
    "meta": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def normalize_provider(provider: Optional[str]) -> str:
    return (provider or "").strip().lower()


def _provider_is_remote(provider: str) -> bool:
    return provider in {"openai", "anthropic", "gemini", "grok", "groq", "meta"}


def _get_model(provider: str, model: Optional[str]) -> str:
    return model or DEFAULT_MODELS.get(provider, "")


def _extract_text_from_openai_like(data: Dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if isinstance(text, str):
                            parts.append(text)
                if parts:
                    return "\n".join(parts)
            if isinstance(content, str):
                return content
    return ""


def _extract_text_from_anthropic(data: Dict[str, Any]) -> str:
    content = data.get("content") or []
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return ""


def _extract_text_from_gemini(data: Dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        texts = []
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
        if texts:
            return "\n".join(texts)
    return ""


def _extract_text_from_response(provider: str, data: Dict[str, Any]) -> str:
    if provider == "anthropic":
        return _extract_text_from_anthropic(data)
    if provider == "gemini":
        return _extract_text_from_gemini(data)
    return _extract_text_from_openai_like(data)


def _request_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int = 90,
) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not installed")
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def generate_text(
    provider: Optional[str],
    prompt: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    timeout: int = 90,
    reasoning_level: Optional[str] = None,
    enable_search: Optional[bool] = None,
) -> str:
    """
    Gera texto usando providers remotos ou locais.
    Retorna uma string vazia em caso de falha.
    """
    provider_name = normalize_provider(provider)
    if provider_name not in SUPPORTED_PROVIDERS:
        logger.warning(f"Provider desconhecido: {provider_name}")
        return ""

    if provider_name == "template":
        return ""

    if provider_name == "ollama":
        try:
            import ollama
        except Exception:
            logger.warning("Ollama SDK não está disponível")
            return ""

        model_name = model or os.getenv("OLLAMA_MODEL", "doninha8:latest")
        host = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if host:
            os.environ["OLLAMA_HOST"] = host
            response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            think=reasoning_level if reasoning_level is not None else "high",
            search_web=enable_search if enable_search is not None else False,
            options={
                "temperature": temperature,
                "num_ctx": 8192,
            },
        )
        if isinstance(response, dict):
            content = response.get("message", {}).get("content", "")
            return content.strip() if isinstance(content, str) else ""
        return str(response).strip() if response else ""

    if provider_name == "custom_lm":
        return ""

    if not _provider_is_remote(provider_name):
        return ""

    api_key = api_key or os.getenv(f"{provider_name.upper()}_API_KEY")
    if not api_key and provider_name != "gemini":
        logger.warning(f"API key não configurada para provider {provider_name}")
        return ""

    if provider_name == "openai":
        endpoint = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        payload = {
            "model": _get_model(provider_name, model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = _request_json(endpoint, headers=headers, payload=payload, timeout=timeout)
        return _extract_text_from_response(provider_name, data)

    if provider_name == "grok":
        endpoint = base_url or os.getenv(
            "GROK_BASE_URL",
            "https://api.x.ai/v1/chat/completions",
        )
        payload = {
            "model": _get_model(provider_name, model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = _request_json(endpoint, headers=headers, payload=payload, timeout=timeout)
        return _extract_text_from_response(provider_name, data)

    if provider_name == "groq":
        endpoint = base_url or os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1/chat/completions",
        )
        payload = {
            "model": _get_model(provider_name, model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = _request_json(endpoint, headers=headers, payload=payload, timeout=timeout)
        return _extract_text_from_response(provider_name, data)

    if provider_name == "meta":
        endpoint = base_url or os.getenv(
            "META_BASE_URL",
            "https://api.llama-api.com/v1/chat/completions",
        )
        payload = {
            "model": _get_model(provider_name, model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = _request_json(endpoint, headers=headers, payload=payload, timeout=timeout)
        return _extract_text_from_response(provider_name, data)

    if provider_name == "anthropic":
        endpoint = base_url or os.getenv(
            "ANTHROPIC_BASE_URL",
            "https://api.anthropic.com/v1/messages",
        )
        payload = {
            "model": _get_model(provider_name, model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
        }
        data = _request_json(endpoint, headers=headers, payload=payload, timeout=timeout)
        return _extract_text_from_response(provider_name, data)

    if provider_name == "gemini":
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("API key não configurada para Gemini")
            return ""
        model_name = _get_model(provider_name, model)
        endpoint = base_url or os.getenv(
            "GEMINI_BASE_URL",
            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
        )
        if "{model_name}" in endpoint:
            endpoint = endpoint.format(model_name=model_name)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        params = {"key": api_key}
        if requests is None:
            raise RuntimeError("requests is not installed")
        response = requests.post(endpoint, params=params, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return _extract_text_from_response(provider_name, data)

    return ""


def generate_provider_text(
    provider: Optional[str],
    prompt: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    timeout: int = 90,
    reasoning_level: Optional[str] = None,
    enable_search: Optional[bool] = None,
) -> str:
    """
    Wrapper público para geração via provider.

    Esta função existe para permitir uso direto, sem depender de aliases
    internos ou do nome da função original.
    """
    return generate_text(
        provider=provider,
        prompt=prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        reasoning_level=reasoning_level,
        enable_search=enable_search,
    )
