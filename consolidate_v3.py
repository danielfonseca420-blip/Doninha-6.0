#!/usr/bin/env python3
"""
CONSOLIDACAO ROBUSTA - Versao 3
"""

import os
import re
from pathlib import Path
from typing import List, Set

FILES_TO_INCLUDE = [
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
    "l1_concept_table.py",
    "l5_generation.py",
    "rag_hybrid_context_injection.py",
    "l2_kantian_judgments.py",
    "l3_paraconsistent.py",
    "metrics.py",
    "l4_synthesis.py",
    "l1_l2_rag_integration.py",
    "syllogism_module.py",
    "l6_final_response.py",
    "l7_final_text.py",
    "agente_sintese_final.py",
    "pipeline.py",
    "api.py",
]

INTERNAL = {"layer_titles", "config_loader", "chat_session", "paraconsistent_rules", "custom_tokenizer", "custom_lm_model", "neural_truth_model", "knowledge_base", "l4_russell_equivalence", "l4_chain_verification", "corpus_utils", "l1_concept_table", "l5_generation", "rag_hybrid_context_injection", "l2_kantian_judgments", "l3_paraconsistent", "metrics", "l4_synthesis", "l1_l2_rag_integration", "syllogism_module", "l6_final_response", "l7_final_text", "agente_sintese_final", "pipeline", "api", "agente_busca_web"}

ROOT = Path(__file__).parent


def is_internal(line):
    for mod in INTERNAL:
        if f"import {mod}" in line or f"from {mod}" in line or f"from .{mod}" in line:
            return True
    return False


def clean_file(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        result = []
        skip = False
        
        for line in lines:
            s = line.strip()
            
            # Pula imports internos
            if (s.startswith('import ') or s.startswith('from ')) and is_internal(s):
                skip = True
                if not s.endswith('\\'):
                    skip = False
                continue
            
            if skip and not s.endswith('\\'):
                skip = False
                continue
            
            result.append(line)
        
        return ''.join(result)
    except:
        return ""


def main():
    imports = set()
    code_sections = []
    
    # Coleta imports
    for fname in FILES_TO_INCLUDE:
        fpath = ROOT / fname
        if fpath.exists():
            content = clean_file(fpath)
            for line in content.split('\n'):
                s = line.strip()
                if (s.startswith('import ') or s.startswith('from ')) and not is_internal(s) and s:
                    imports.add(s)
    
    # Gera saida
    output = ['# Doninha IA Standalone\nfrom __future__ import annotations\n']
    
    # Imports
    stdlib = sorted([i for i in imports if any(x in i for x in ['os', 'sys', 're', 'json', 'time', 'math', 'typing'])])
    other = sorted([i for i in imports if i not in stdlib])
    
    if stdlib:
        output.extend(stdlib)
    if other:
        output.append('')
        output.extend(other)
    
    output.append('\n')
    
    # Codigo
    for fname in FILES_TO_INCLUDE:
        fpath = ROOT / fname
        if fpath.exists():
            content = clean_file(fpath)
            lines = content.split('\n')
            
            # Pula imports do inicio
            start = 0
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().startswith(('import ', 'from ', '#', '"""', "'''", 'from __future__')):
                    start = i
                    break
            
            body = '\n'.join(lines[start:]).strip()
            
            if body:
                output.append(f'\n# ========== {fname} ==========\n')
                output.append(body)
    
    # Escreve
    outpath = ROOT / 'doninha_standalone.py'
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output))
    
    print(f'OK: {outpath}')


if __name__ == '__main__':
    main()
