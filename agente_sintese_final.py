"""
Agente de sntese final L1L7.
==============================
Gera um texto final fluido e coeso a partir do raciocnio acumulado
nas camadas L1 a L6, usando a engine L7 j disponvel no projeto.

Uso rpido:
    python agente_sintese_final.py --prompt "Explique..."
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from l7_final_text import FinalTextEngine


DEFAULT_PROVIDER = os.getenv("FINAL_TEXT_PROVIDER", "ollama")
DEFAULT_MODEL = os.getenv("FINAL_TEXT_MODEL", "doninha8:latest")


def synthesize_final_text(
    prompt: str,
    l1_summary: str = "",
    l2_summary: str = "",
    l3_summary: str = "",
    l4_response: str = "",
    l5_text: str = "",
    l6_text: str = "",
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    cot_context: str = "",
) -> str:
    """Gera o texto final de alta qualidade usando a camada L7."""
    engine = FinalTextEngine(config={"l7": {"model": model}})
    return engine.finalize_text(
        prompt=prompt,
        l1_summary=l1_summary,
        l2_summary=l2_summary,
        l3_summary=l3_summary,
        l4_response=l4_response,
        l5_text=l5_text,
        l6_text=l6_text,
        provider=provider,
        model=model,
        cot_context=cot_context,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def load_json_file(path: str) -> Dict[str, Any]:
    """Carrega um arquivo JSON com o raciocnio acumulado."""
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agente de sntese final L1L7")
    parser.add_argument("--prompt", required=True, help="Pergunta ou prompt original para sintetizar.")
    parser.add_argument("--l1", default="", help="Resumo da camada L1 em texto livre.")
    parser.add_argument("--l2", default="", help="Resumo da camada L2 em texto livre.")
    parser.add_argument("--l3", default="", help="Resumo da camada L3 em texto livre.")
    parser.add_argument("--l4", default="", help="Resposta da camada L4 em texto livre.")
    parser.add_argument("--l5", default="", help="Texto de gerao da camada L5.")
    parser.add_argument("--l6", default="", help="Texto refinado da camada L6.")
    parser.add_argument("--json", default="", help="Caminho para JSON com os campos l1..l6.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Provider para L7 (ollama, template, custom_lm).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Modelo Ollama para a sntese final.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura da gerao final.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Mximo de tokens para a sntese final.")
    parser.add_argument("--cot", default="", help="Contexto Chain of Thought auditvel em texto/markdown.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    data: Dict[str, Any] = {}
    if args.json:
        data = load_json_file(args.json)

    final_text = synthesize_final_text(
        prompt=args.prompt,
        l1_summary=data.get("l1_summary", data.get("l1", args.l1)),
        l2_summary=data.get("l2_summary", data.get("l2", args.l2)),
        l3_summary=data.get("l3_summary", data.get("l3", args.l3)),
        l4_response=data.get("l4_response", data.get("l4", args.l4)),
        l5_text=data.get("l5_text", data.get("l5", args.l5)),
        l6_text=data.get("l6_text", data.get("l6", args.l6)),
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cot_context=data.get("cot_context", data.get("cot_markdown", args.cot)),
    )

    print(final_text)


if __name__ == "__main__":
    main()
