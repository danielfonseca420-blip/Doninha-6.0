#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCRIPT DE CONSOLIDAÇÃO AUTOMÁTICA: Gera doninha_standalone.py
Consolida 28 módulos Python em 1 arquivo, mantendo toda a lógica intacta
"""

import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent

# Lista de arquivos em ordem topológica (ordem CRÍTICA)
FILES_TO_CONSOLIDATE = [
    # FASE 1: Fundação (sem dependências internas)
    'layer_titles.py',
    'config_loader.py', 
    'chat_session.py',
    'knowledge_base.py',
    'corpus_utils.py',
    'paraconsistent_rules.py',
    'l4_russell_equivalence.py',
    'neural_truth_model.py',
    'custom_tokenizer.py',
    'custom_lm_model.py',
    'metrics.py',
    
    # FASE 2: Camadas L1-L3
    'l1_concept_table.py',
    'l2_kantian_judgments.py',
    'l3_paraconsistent.py',
    
    # FASE 3: Lógica + L4
    'syllogism_module.py',
    'l4_chain_verification.py',
    'l4_synthesis.py',
    
    # FASE 4: Geradores L5-L7
    'l5_generation.py',
    'l6_final_response.py',
    'l7_final_text.py',
    
    # FASE 5: RAG
    'rag_hybrid_context_injection.py',
    'l1_l2_rag_integration.py',
    
    # FASE 6: Pipelines + Agentes
    'pipeline.py',
    'pipeline_with_rag_integration.py',
    'agente_busca_web.py',
    'agente_sintese_final.py',
    'build_concepts_from_english_dict.py',
    
    # FASE 7: APIs
    'api.py',
    'app.py',
]

def extract_docstring(content):
    """Extrai docstring do arquivo"""
    match = re.match(r'^\s*"""(.+?)"""', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.match(r"^\s*'''(.+?)'''", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def remove_imports_internal(content, exclude_patterns=None):
    """Remove imports internos do projeto"""
    if exclude_patterns is None:
        exclude_patterns = [
            'from l', 'import l',  # imports de camadas
            'from agente_', 'import agente_',  # agentes
            'from pipeline',
            'from knowledge_base import',
            'from config_loader import',
            'from neural_truth_model import',
            'from custom_lm_model import',
            'from custom_tokenizer import',
            'from l4_russell_equivalence import',
            'from l4_chain_verification import',
            'from l4_synthesis import',
            'from l5_generation import',
            'from l6_final_response import',
            'from l7_final_text import',
            'from rag_hybrid_context_injection import',
            'from l1_l2_rag_integration import',
            'from chat_session import',
            'from layer_titles import',
            'from metrics import',
            'from paraconsistent_rules import',
            'from corpus_utils import',
            'from build_concepts',
            'from . import',
            'from .. import',
        ]
    
    lines = content.split('\n')
    filtered = []
    for line in lines:
        # Verifica se é um import a ser removido
        is_internal = any(pat.lower() in line.lower() for pat in exclude_patterns)
        
        # Mas mantém imports dentro de try/except (fallbacks)
        if is_internal and 'try:' not in ''.join(lines[max(0, lines.index(line)-3):lines.index(line)]):
            # Verifica se há try/except abaixo
            try:
                idx = lines.index(line)
                if idx > 0 and 'try:' in lines[idx - 1]:
                    filtered.append(line)
                    continue
            except:
                pass
            
            # Remove a linha
            continue
        
        filtered.append(line)
    
    return '\n'.join(filtered)

def consolidate_all():
    """Consolida todos os arquivos"""
    
    header = '''# =============================================================================
# DONINHA IA — Middleware Epistemológico Standalone
# Gerado automaticamente a partir de 28 módulos fonte
# Pipeline: L1 (Aristotelian) → L2 (Kantian/BERT) → L3 (Paraconsistent/QUPC)
#           → L4 (Russellian CoV) → L5 (Generation) → L6 (Refinement)
#           → L7 (Epistemic Synthesis)
# Dependências externas: instale via `pip install -r requirements.txt`
# =============================================================================

"""
DONINHA IA — Middleware Neuro-Simbólico Híbrido Consolidado
============================================================

Este é um arquivo Python ÚNICO e AUTOSSUFICIENTE contendo:
  • 28 módulos principais consolid ados em um único arquivo
  • Todas as 7 camadas de processamento (L1–L7)
  • RAG híbrido integrado
  • APIs FastAPI e Chainlit
  • Agentes especializados

Uso:
    python doninha_standalone.py --help
    python doninha_standalone.py --prompt "Sua pergunta aqui"
    python doninha_standalone.py --repl
    
Para usar a API:
    python doninha_standalone.py --api

Para usar Chainlit:
    python doninha_standalone.py --chainlit

Referência de módulos de origem inclusos (veja comentários === ORIGEM: ...)
"""

# =============================================================================
# IMPORTS EXTERNOS — Bibliotecas third-party e stdlib
# =============================================================================

from __future__ import annotations

# Standard library
import os
import sys
import re
import time
import json
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import uuid
from enum import Enum

# Third-party with fallbacks
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    nn = None
    Dataset = object

try:
    from transformers import AutoModel, AutoTokenizer, pipeline
except ImportError:
    AutoModel = None
    AutoTokenizer = None
    pipeline = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    import sentencepiece as spm
except ImportError:
    spm = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import ollama
except ImportError:
    ollama = None

try:
    from langchain import LLMChain, PromptTemplate
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.llms import Ollama
except ImportError:
    LLMChain = None
    PromptTemplate = None
    Chroma = None
    HuggingFaceEmbeddings = None
    AgentExecutor = None
    create_react_agent = None
    Tool = None
    DuckDuckGoSearchRun = None
    Ollama = None

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    FastAPI = None
    HTTPException = None
    BaseModel = object

try:
    import chainlit as cl
except ImportError:
    cl = None

'''
    
    # Consolidar arquivos
    all_content = []
    
    for filename in FILES_TO_CONSOLIDATE:
        filepath = ROOT / filename
        if not filepath.exists():
            print(f"⚠ Arquivo não encontrado: {filename}")
            continue
        
        content = filepath.read_text(encoding='utf-8-sig')  # Remove BOM se presente
        
        # Extrai docstring e remover do código
        docstring = extract_docstring(content)
        if docstring:
            # Remove o docstring do topo
            content = re.sub(r'^\s*"""(.+?)"""\s*\n', '', content, flags=re.DOTALL)
            content = re.sub(r"^\s*'''(.+?)'''\s*\n", '', content, flags=re.DOTALL)
        
        # Remove imports internos
        content = remove_imports_internal(content)
        
        # Remove linhas em branco excessivas
        lines = content.split('\n')
        cleaned = []
        prev_blank = False
        for line in lines:
            is_blank = not line.strip()
            if is_blank and prev_blank:
                continue
            cleaned.append(line)
            prev_blank = is_blank
        content = '\n'.join(cleaned)
        
        # Adiciona bloco de origem
        origin_block = f'''
# {"─" * 77}
# ORIGEM: {filename}
# {"─" * 77}

'''
        all_content.append(origin_block + content.strip() + '\n')
    
    # Monta o arquivo final
    final_content = header + '\n'.join(all_content)
    
    # Escreve o arquivo (sem BOM)
    output_path = ROOT / 'doninha_standalone.py'
    output_path.write_text(final_content, encoding='utf-8')
    
    print(f"✅ Arquivo consolidado criado: {output_path}")
    print(f"   Tamanho: {len(final_content)} caracteres")
    print(f"   Arquivos consolidados: {len(FILES_TO_CONSOLIDATE)}")

if __name__ == '__main__':
    consolidate_all()
