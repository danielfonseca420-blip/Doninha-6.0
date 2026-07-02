#!/usr/bin/env python3
"""
SCRIPT DE CONSOLIDAÇÃO AUTOMÁTICA V2
====================================
Consolidação corrigida que preserva estrutura de docstrings.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Set

# Arquivos a INCLUIR (ordem topológica)
FILES_TO_INCLUDE = [
    # Nível 0: Sem dependências internas
    "layer_titles.py",
    "config_loader.py",
    "chat_session.py",
    "paraconsistent_rules.py",
    "custom_tokenizer.py",
    "custom_lm_model.py",
    "neural_truth_model.py",
    "knowledge_base.py",
    "l4_russell_equivalence.py",
    "l4_chain_verification.py",
    "corpus_utils.py",
    
    # Nível 1: Dependências de nível 0
    "l1_concept_table.py",
    "l5_generation.py",
    "rag_hybrid_context_injection.py",
    
    # Nível 2
    "l2_kantian_judgments.py",
    "l3_paraconsistent.py",
    "metrics.py",
    
    # Nível 3
    "l4_synthesis.py",
    "l1_l2_rag_integration.py",
    "syllogism_module.py",
    
    # Nível 4
    "l6_final_response.py",
    "l7_final_text.py",
    
    # Nível 5
    "agente_sintese_final.py",
    "pipeline.py",
    
    # Nível 6
    "api.py",
]

INTERNAL_MODULES = {
    "layer_titles",
    "config_loader",
    "chat_session",
    "paraconsistent_rules",
    "custom_tokenizer",
    "custom_lm_model",
    "neural_truth_model",
    "knowledge_base",
    "l4_russell_equivalence",
    "l4_chain_verification",
    "corpus_utils",
    "l1_concept_table",
    "l5_generation",
    "rag_hybrid_context_injection",
    "l2_kantian_judgments",
    "l3_paraconsistent",
    "metrics",
    "l4_synthesis",
    "l1_l2_rag_integration",
    "syllogism_module",
    "l6_final_response",
    "l7_final_text",
    "agente_sintese_final",
    "pipeline",
    "api",
    "agente_busca_web",
}

PROJECT_ROOT = Path(__file__).resolve().parent


def read_file(path: Path) -> str:
    """Lê arquivo com tratamento de encoding."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def is_internal_import(line: str) -> bool:
    """Verifica se é um import interno a remover."""
    # Remove imports relativos
    if re.match(r'^from \. import', line) or re.match(r'^from \.\. import', line):
        return True
    
    # Checa imports de módulos internos
    match = re.match(r'(?:from\s+([\w.]+)|import\s+([\w.]+))', line.strip())
    if match:
        module = match.group(1) or match.group(2)
        module_base = module.split('.')[0]
        return module_base in INTERNAL_MODULES
    
    return False


def clean_file_content(content: str) -> str:
    """Remove imports internos, mantém a estrutura correta."""
    lines = content.split("\n")
    result = []
    in_docstring = False
    docstring_char = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Detecta docstrings
        for quote_type in ['"""', "'''"]:
            if quote_type in stripped:
                count = stripped.count(quote_type)
                if count % 2 == 1:  # Toggle docstring state
                    in_docstring = not in_docstring
                    docstring_char = quote_type if in_docstring else None
        
        # Se está em docstring, mantém
        if in_docstring:
            result.append(line)
        # Se é import, filtra
        elif stripped.startswith("import ") or stripped.startswith("from "):
            if not is_internal_import(stripped):
                result.append(line)
        # Outro conteúdo
        else:
            result.append(line)
        
        i += 1
    
    return "\n".join(result)


def extract_imports(content: str) -> Set[str]:
    """Extrai todos os imports externos únicos."""
    imports = set()
    
    for line in content.split("\n"):
        stripped = line.strip()
        if (stripped.startswith("import ") or stripped.startswith("from ")) and stripped:
            if not is_internal_import(stripped):
                imports.add(stripped)
    
    return imports


def consolidate() -> str:
    """Consolida todos os arquivos."""
    output = []
    all_imports = set()
    
    # Cabeçalho
    output.append("""# =============================================================================
# DONINHA IA -- Middleware Epistemologico Standalone
# Gerado automaticamente a partir de 25 modulos fonte
# Pipeline: L1 (Aristotelian) -> L2 (Kantian/BERT) -> L3 (Paraconsistent/QUPC)
#           -> L4 (Russellian CoV) -> L5 (Generation) -> L6 (Refinement)
#           -> L7 (Epistemic Synthesis) -> API REST & Agentes
#
# Dependencias externas: instale via `pip install -r requirements.txt`
# =============================================================================

from __future__ import annotations
""")
    
    # Coleta imports
    for filename in FILES_TO_INCLUDE:
        path = PROJECT_ROOT / filename
        if path.exists():
            content = read_file(path)
            all_imports.update(extract_imports(content))
    
    # Organiza imports
    stdlib_patterns = {
        "os", "sys", "re", "json", "time", "math", "logging", "argparse",
        "uuid", "dataclasses", "typing", "collections", "pathlib", "enum",
        "warnings", "traceback", "functools", "itertools", "copy", "pickle"
    }
    
    stdlib = []
    thirdparty = []
    
    for imp in sorted(all_imports):
        matched = False
        for stdlib_name in stdlib_patterns:
            if f" {stdlib_name}" in imp or f"import {stdlib_name}" == imp.strip():
                stdlib.append(imp)
                matched = True
                break
        if not matched:
            thirdparty.append(imp)
    
    if stdlib:
        output.append("\n# Stdlib")
        output.extend(stdlib)
    
    if thirdparty:
        output.append("\n# Third-party")
        output.extend(thirdparty)
    
    output.append("\n\n")
    
    # Processa cada arquivo
    for filename in FILES_TO_INCLUDE:
        path = PROJECT_ROOT / filename
        if not path.exists():
            print(f"Aviso: {filename} nao encontrado")
            continue
        
        content = read_file(path)
        cleaned = clean_file_content(content)
        
        # Remove blocos de imports do começo
        lines = cleaned.split("\n")
        body_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not (stripped.startswith(("import ", "from ", "'''", '"""')) or stripped.startswith("#") and "ORIGEM" not in stripped):
                body_start = i
                break
        
        body = "\n".join(lines[body_start:]).strip()
        
        if body:
            output.append("\n" + "#" * 80)
            output.append(f"# ORIGEM: {filename}")
            output.append("#" * 80 + "\n")
            output.append(body)
    
    # Entrada
    output.append("\n\n" + "#" * 80)
    output.append("# PONTO DE ENTRADA")
    output.append("#" * 80)
    output.append("""

if __name__ == "__main__":
    print("Doninha IA - Middleware Epistemologico Standalone")
    print("Versao: 1.0")
    print("Use: python doninha_standalone.py")
""")
    
    return "\n".join(output)


if __name__ == "__main__":
    print("Consolidando arquivos...")
    result = consolidate()
    
    output_path = PROJECT_ROOT / "doninha_standalone.py"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)
    
    print(f"OK: {output_path}")
    print(f"Tamanho: {len(result)} bytes")
    print(f"Linhas: {len(result.split(chr(10)))}")
