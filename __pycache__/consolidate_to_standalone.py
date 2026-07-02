#!/usr/bin/env python3
"""
SCRIPT DE CONSOLIDAÇÃO AUTOMÁTICA
===================================
Consolida todos os arquivos .py do projeto em doninha_standalone.py,
removendo imports internos e mantendo apenas a lógica funcional.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

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

# Arquivos a EXCLUIR
FILES_TO_EXCLUDE = {
    "consolidate_all.py",
    "doninha_middleware_consolidated.py",
    "app.py",
    "test_epistemic_classification.py",
    "test_rag_hybrid.py",
    "example_rag_hybrid_usage.py",
    "eval_pipeline.py",
    "build_concepts_from_english_dict.py",
    "run_pretrain.py",
    "train_truth_model.py",
    "train_l4_russell.py",
    "pretrain_custom_lm.py",
    "pipeline_with_rag_integration.py",
    "agente_busca_web.py",
    "consolidate_to_standalone.py",  # Não incluir este próprio script
}

PROJECT_ROOT = Path(__file__).resolve().parent

# Nomes internos de módulos a remover dos imports
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


def read_file(path: Path) -> str:
    """Lê o conteúdo de um arquivo."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Erro lendo {path}: {e}")
        return ""


def remove_internal_imports(content: str) -> str:
    """Remove imports internos do projeto."""
    lines = content.split("\n")
    result = []
    skip_next_lines = 0
    
    for line in lines:
        if skip_next_lines > 0:
            skip_next_lines -= 1
            continue
            
        # Detecta imports a remover
        if line.strip().startswith("from __future__"):
            result.append(line)
        elif re.match(r'^from \. import', line) or re.match(r'^from \.\. import', line):
            # Remove imports relativos
            continue
        elif re.match(r'^import', line) or re.match(r'^from', line):
            # Verifica se é um import interno
            match = re.match(r'(?:from\s+([\w.]+)|import\s+([\w.]+))', line)
            if match:
                module = match.group(1) or match.group(2)
                module_base = module.split('.')[0]
                
                # Se for um módulo interno, tenta remover
                if module_base in INTERNAL_MODULES or module in INTERNAL_MODULES:
                    # Verifica se há continuação de import
                    if line.rstrip().endswith("\\"):
                        skip_next_lines = 1
                    continue
            
            result.append(line)
        else:
            result.append(line)
    
    return "\n".join(result)


def cleanup_imports_section(content: str) -> str:
    """Remove duplicatas de imports e linha em branco excessivas."""
    lines = content.split("\n")
    seen_imports = set()
    result = []
    blank_lines = 0
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            blank_lines += 1
            if blank_lines <= 2:  # Máximo 2 linhas em branco consecutivas
                result.append(line)
        else:
            blank_lines = 0
            
            # Remove duplicatas de imports
            if stripped.startswith(("import ", "from ")):
                if stripped not in seen_imports:
                    seen_imports.add(stripped)
                    result.append(line)
            else:
                result.append(line)
    
    return "\n".join(result)


def extract_file_section(path: Path) -> Tuple[str, str]:
    """Extrai conteúdo do arquivo e remove imports internos."""
    content = read_file(path)
    cleaned = remove_internal_imports(content)
    return cleaned, path.stem


def consolidate_files() -> str:
    """Consolida todos os arquivos em um único conteúdo."""
    output = []
    
    # Cabeçalho do arquivo
    output.append("""# =============================================================================
# DONINHA IA — Middleware Epistemológico Standalone
# Gerado automaticamente a partir de 26 módulos fonte
# Pipeline: L1 (Aristotelian) → L2 (Kantian/BERT) → L3 (Paraconsistent/QUPC)
#           → L4 (Russellian CoV) → L5 (Generation) → L6 (Refinement)
#           → L7 (Epistemic Synthesis) → API REST & Agentes
#
# Dependências externas: instale via `pip install -r requirements.txt`
# =============================================================================

from __future__ import annotations
""")
    
    # Coleta todos os imports externos
    all_external_imports = set()
    
    for filename in FILES_TO_INCLUDE:
        path = PROJECT_ROOT / filename
        if path.exists():
            content = read_file(path)
            # Extrai imports externos
            for line in content.split("\n"):
                if (line.strip().startswith("import ") or line.strip().startswith("from ")) and not line.strip().startswith("from __future__"):
                    match = re.match(r'(?:from\s+([\w.]+)|import\s+([\w.]+))', line.strip())
                    if match:
                        module = match.group(1) or match.group(2)
                        module_base = module.split('.')[0]
                        if module_base not in INTERNAL_MODULES and module_base != "__future__":
                            # Padroniza a linha
                            if "import" in line and "from" not in line:
                                all_external_imports.add(line.strip())
                            elif "from" in line:
                                all_external_imports.add(line.strip())
    
    # Remove `try`/`except` imports e duplicatas
    external_imports = []
    for imp in sorted(all_external_imports):
        if imp and not imp.startswith("#"):
            # Simplifica imports complexos (sem try/except)
            external_imports.append(imp)
    
    # Remove duplicatas exatas
    external_imports = list(dict.fromkeys(external_imports))
    
    # Agrupa imports
    stdlib_imports = []
    thirdparty_imports = []
    
    common_stdlib = {
        "os", "sys", "re", "json", "time", "math", "logging", "argparse",
        "uuid", "dataclasses", "typing", "collections", "pathlib", "enum",
        "torch", "warnings", "traceback",
    }
    
    for imp in external_imports:
        if any(f" {mod}" in imp for mod in common_stdlib):
            stdlib_imports.append(imp)
        else:
            thirdparty_imports.append(imp)
    
    if stdlib_imports:
        output.append("\n# Stdlib + PyTorch")
        output.extend(stdlib_imports)
    
    if thirdparty_imports:
        output.append("\n# Third-party libraries")
        output.extend(thirdparty_imports)
    
    output.append("\n")
    
    # Agora processa cada arquivo
    for filename in FILES_TO_INCLUDE:
        path = PROJECT_ROOT / filename
        if not path.exists():
            print(f"⚠️  Arquivo não encontrado: {filename}")
            continue
        
        content, module_name = extract_file_section(path)
        
        # Remove o bloco de imports (já no topo)
        content_lines = content.split("\n")
        body_start = 0
        for i, line in enumerate(content_lines):
            stripped = line.strip()
            if (not stripped.startswith(("import ", "from ", '"""', "'''", "#"))
                and stripped
                and not stripped.startswith("__")):
                body_start = i
                break
        
        body = "\n".join(content_lines[body_start:]).strip()
        
        # Bloco separador (apenas inclua se houver conteúdo significativo)
        if body and not body.isspace():
            output.append("\n" + "=" * 80)
            output.append(f"# ORIGEM: {filename}")
            output.append("=" * 80)
            output.append(body)
    
    # Ponto de entrada
    output.append("\n\n" + "=" * 80)
    output.append("# PONTO DE ENTRADA")
    output.append("=" * 80)
    output.append("""
if __name__ == "__main__":
    import sys
    print("Doninha IA - Middleware Epistemológico Standalone")
    print(f"Versão: 1.0 | Arquivos consolidados: 26")
    print(f"Arquivos de suporte: {len(FILES_TO_INCLUDE)} módulos")
    print()
    print("Para usar:")
    print("  from doninha_standalone import pipeline, config_loader, ChatSession")
    print("  config = config_loader.load_config()")
    print("  p = pipeline.HybridLLMPipeline(config)")
    print("  result = p.process('Qual é sua pergunta?')")
    print(f"  print(result)")
""")
    
    return "\n".join(output)


if __name__ == "__main__":
    print("🔄 Consolidando arquivos...")
    print(f"📁 Diretório: {PROJECT_ROOT}")
    print(f"📊 Arquivos a incluir: {len(FILES_TO_INCLUDE)}")
    print(f"🚫 Arquivos a excluir: {len(FILES_TO_EXCLUDE)}")
    
    consolidated = consolidate_files()
    
    output_path = PROJECT_ROOT / "doninha_standalone.py"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(consolidated)
        print(f"✅ Arquivo consolidado: {output_path}")
        print(f"📏 Tamanho: {len(consolidated)} bytes ({len(consolidated) // 1024} KB)")
        print(f"📝 Linhas: {len(consolidated.split(chr(10)))}")
    except Exception as e:
        print(f"❌ Erro ao escrever arquivo: {e}")
