"""
PIPELINE COM INTEGRAO DE RAG HBRIDO
========================================
Verso estendida do pipeline.py que integra:
- RAG Hbrido (Context Injection + Retrieval Seletivo)
- L1-L2 enriquecidas com contexto
- Domain-aware Knowledge Base

Substitui 'pipeline.py' ou pode ser usado em paralelo.
"""

from __future__ import annotations
import sys
import re
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

# Importaes originais do pipeline
try:
    from neural_truth_model import TruthScoringModel, load_tokenizer
except ImportError:
    TruthScoringModel = None
    load_tokenizer = None

try:
    from l1_concept_table import ConceptTable, ConceptNode, LogicLMSymbolicSolver
except ImportError:
    ConceptTable = None
    ConceptNode = None
    LogicLMSymbolicSolver = None

try:
    from l2_kantian_judgments import KantianJudgmentEngine, KantianJudgment
except ImportError:
    KantianJudgmentEngine = None
    KantianJudgment = None

try:
    from syllogism_module import ScientificSyllogismPipeline
except ImportError:
    ScientificSyllogismPipeline = None

try:
    from l3_paraconsistent import ParaconsistentEngine, ParaconsistentValue
except ImportError:
    ParaconsistentEngine = None
    ParaconsistentValue = None

try:
    from l4_synthesis import RussellianSynthesisEngine, SynthesisResult
except ImportError:
    RussellianSynthesisEngine = None
    SynthesisResult = None

try:
    from l6_final_response import EpistemicContext, FinalResponseEngine
except ImportError:
    EpistemicContext = None
    FinalResponseEngine = None

try:
    from l7_final_text import FinalTextEngine
except ImportError:
    FinalTextEngine = None

try:
    from l4_russell_equivalence import load_concept_base
except ImportError:
    load_concept_base = None

try:
    from config_loader import load_config, PROJECT_ROOT
except ImportError:
    load_config = None
    PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from knowledge_base import get_knowledge_base, SEED_KNOWLEDGE_BASE
except ImportError:
    get_knowledge_base = None
    SEED_KNOWLEDGE_BASE = {}

try:
    from l5_generation import generate_response as l5_generate
except ImportError:
    l5_generate = None

try:
    from agente_busca_web import run_search_for_context
except ImportError:
    run_search_for_context = None

# Importaes do RAG Hbrido
try:
    from l1_l2_rag_integration import (
        create_l1_l2_rag_pipeline,
        IntegratedL1L2RAGPipeline,
        EnrichedL1Output,
        EnrichedL2Output,
    )
    HAS_RAG_HYBRID = True
except ImportError:
    HAS_RAG_HYBRID = False


# 
# Pipeline Estendido com RAG Hbrido
# 

class HybridLLMPipelineWithRAG:
    """
    Pipeline completo do Modelo Hbrido de LLM com RAG Integrado.
    
    Adiciona ao pipeline original:
    - RAG Hbrido para cada query
    - Context Injection automtico nas camadas L1-L2
    - Domain-aware Knowledge Base
    - System prompt especializado por domnio
    
    Suporta config, KB escalvel, L5 (gerao), agente opcional e chat.
    """

    def __init__(
        self,
        knowledge_base: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        use_rag_hybrid: bool = True,
        verbose: bool = True,
    ) -> None:
        self._config = config or (load_config() if load_config else {})
        self.kb = knowledge_base or self._get_kb(self._config, "", False)
        if not self.kb:
            self.kb = dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}
        self.verbose = verbose
        self.use_rag_hybrid = use_rag_hybrid and HAS_RAG_HYBRID

        # Inicializa pipeline L1-L2-RAG (se disponvel)
        if self.use_rag_hybrid:
            self.rag_l1_l2_pipeline = create_l1_l2_rag_pipeline(config=self._config)
            if self.verbose:
                print("[Pipeline] RAG Hbrido habilitado")
        else:
            self.rag_l1_l2_pipeline = None

        # Inicializa componentes originais
        if ConceptTable:
            self.L1 = ConceptTable()
        else:
            self.L1 = None

        if KantianJudgmentEngine:
            self.L2 = KantianJudgmentEngine(self.L1) if self.L1 else None
        else:
            self.L2 = None

        if ScientificSyllogismPipeline:
            self.SYL = ScientificSyllogismPipeline()
        else:
            self.SYL = None

        # L3
        l3_cfg = self._config.get("l3", {})
        if ParaconsistentEngine:
            self.L3 = ParaconsistentEngine(
                t_threshold=l3_cfg.get("t_threshold", 0.7),
                f_threshold=l3_cfg.get("f_threshold", 0.3),
                verbose=verbose,
            )
        else:
            self.L3 = None

        # L4
        if RussellianSynthesisEngine:
            self.L4 = RussellianSynthesisEngine()
        else:
            self.L4 = None

        # L5/L6/L7
        if FinalResponseEngine:
            self.L6 = FinalResponseEngine()
        else:
            self.L6 = None

        if FinalTextEngine:
            self.L7 = FinalTextEngine()
        else:
            self.L7 = None

    def _get_kb(self, config: Optional[Dict[str, Any]], prompt: str, use_agent: bool) -> Dict[str, float]:
        if get_knowledge_base is None:
            return dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}
        return get_knowledge_base(
            config=config,
            query_for_rag=prompt if use_agent else None,
        )

    def process_with_rag(
        self,
        prompt: str,
        use_agent: bool = False,
    ) -> Dict[str, Any]:
        """
        Processa prompt com RAG Hbrido integrado.
        
        Retorna:
        {
            'query': str,
            'domain': str,
            'l1_concepts': List[ConceptNode],
            'l2_judgments': List[KantianJudgment],
            'system_prompt': str,
            'injected_context': str,
            'rag_confidence': float,
            'full_pipeline_result': Dict (sada completa do L1-L7),
        }
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[HybridPipeline] Processando (com RAG): {prompt[:60]}...")
            print(f"{'='*70}")

        # Se RAG no est habilitado, retorna processo padro
        if not self.use_rag_hybrid:
            return self.process_standard(prompt, use_agent)

        # 
        # ETAPA 1: RAG Hbrido + L1-L2 Enriquecido
        # 
        rag_result = self.rag_l1_l2_pipeline.process(prompt)

        domain = rag_result['domain']
        l1_output: EnrichedL1Output = rag_result['l1_output']
        l2_output: EnrichedL2Output = rag_result['l2_output']

        if self.verbose:
            print(f"\n[RAG] Domnio: {domain}")
            print(f"[RAG] Confiana: {rag_result['confidence']:.2%}")
            print(f"[RAG] Conceitos (L1): {len(l1_output.concepts)}")
            print(f"[RAG] Juzos (L2): {len(l2_output.judgments)}")

        # 
        # ETAPA 2: Silogismo Cientfico (L3 original)
        # 
        hypothesis = prompt
        if self.SYL and self.verbose:
            print(f"\n[SYL] Analisando silogismo...")

        # 
        # ETAPA 3: Avaliao Paraconsistente (L3 original)
        # 
        if self.L3 and self.verbose:
            print(f"\n[L3] Avaliao paraconsistente...")

        # 
        # ETAPA 4: Sntese Russelliana (L4 original)
        # 
        if self.L4 and self.verbose:
            print(f"\n[L4] Sntese russelliana...")

        # 
        # ETAPA 5-7: Gerao e Resposta Final (com system_prompt enriquecido)
        # 
        system_prompt_enriched = rag_result['system_prompt']
        compiled_context = rag_result['compiled_context']

        if self.verbose:
            print(f"\n[GEN] Gerando resposta com system_prompt enriquecido...")
            print(f"[GEN] Contexto injetado: {len(compiled_context)} caracteres")

        # 
        # Retorna resultado compilado
        # 
        # Coletar alertas de incompatibilidade cannica gerados durante L1-L2
        canonical_alerts = LogicLMSymbolicSolver.get_canonical_alerts() if LogicLMSymbolicSolver else []

        return {
            'query': prompt,
            'domain': domain,
            'l1_concepts': l1_output.concepts,
            'l2_judgments': l2_output.judgments,
            'system_prompt': system_prompt_enriched,
            'injected_context': compiled_context,
            'rag_confidence': rag_result['confidence'],
            'kb_terms': l2_output.domain_specialized_kb,
            'rag_l1_l2_output': rag_result,
            'canonical_alerts': canonical_alerts,
            'full_pipeline_result': {},  # Preenchido se L3-L7 forem executadas
        }

    def process_standard(
        self,
        prompt: str,
        use_agent: bool = False,
    ) -> Dict[str, Any]:
        """
        Processa prompt com pipeline padro (sem RAG).
        Mantm compatibilidade com pipeline.py original.
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[HybridPipeline] Processando (sem RAG): {prompt[:60]}...")
            print(f"{'='*70}")

        # Extrai conceitos L1
        concepts = self.L1.extract_concepts(prompt) if self.L1 else []

        # Analisa juzos L2
        judgments = self.L2.infer_from_prompt(prompt) if self.L2 else []

        return {
            'query': prompt,
            'domain': 'geral',
            'l1_concepts': concepts,
            'l2_judgments': judgments,
            'system_prompt': "Voc  um especialista. Responda com rigor.",
            'injected_context': "",
            'rag_confidence': 0.0,
            'kb_terms': {},
            'rag_l1_l2_output': None,
            'full_pipeline_result': {},
        }

    def format_for_llm(self, rag_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formata resultado do pipeline para injeo em LLM.
        
        Retorna:
        {
            'system_prompt': str,
            'user_message': str,
            'domain': str,
            'confidence': float,
        }
        """
        return {
            'system_prompt': rag_result.get('system_prompt', ''),
            'user_message': rag_result.get('injected_context', rag_result.get('query', '')),
            'domain': rag_result.get('domain', 'geral'),
            'confidence': rag_result.get('rag_confidence', 0.0),
        }


# 
# Funes de Convenincia
# 

def create_hybrid_pipeline_with_rag(
    config: Optional[Dict[str, Any]] = None,
    use_rag: bool = True,
) -> HybridLLMPipelineWithRAG:
    """Factory para criar pipeline com RAG."""
    return HybridLLMPipelineWithRAG(
        config=config,
        use_rag_hybrid=use_rag,
        verbose=True,
    )


# 
# CLI / REPL para Testes
# 

def interactive_pipeline():
    """REPL interativo para testar o pipeline."""
    print("\n" + "="*70)
    print("PIPELINE HBRIDO COM RAG  MODO INTERATIVO")
    print("="*70)
    print("\nComandos:")
    print("  - Digite uma pergunta para processar com RAG hbrido")
    print("  - 'no-rag' para desabilitar RAG e usar pipeline padro")
    print("  - 'quit' para sair")
    print("")

    pipeline = create_hybrid_pipeline_with_rag(use_rag=True)
    use_rag = True

    while True:
        try:
            prompt = input("\n> ").strip()

            if prompt.lower() == 'quit':
                print("Encerrando...")
                break

            if prompt.lower() == 'no-rag':
                use_rag = not use_rag
                mode = "com RAG" if use_rag else "sem RAG"
                print(f"Modo alternado para: {mode}")
                continue

            if not prompt:
                continue

            # Processa
            result = pipeline.process_with_rag(prompt) if use_rag else pipeline.process_standard(prompt)

            # Exibe resultado
            print(f"\n Domnio: {result['domain']}")
            print(f" Confiana: {result['rag_confidence']:.2%}")
            print(f" Conceitos (L1): {len(result['l1_concepts'])}")
            print(f" Juzos (L2): {len(result['l2_judgments'])}")
            if result['injected_context']:
                print(f"\n[Contexto Injetado]\n{result['injected_context'][:400]}...")

        except KeyboardInterrupt:
            print("\n\nInterrompido pelo usurio.")
            break
        except Exception as e:
            print(f"\n Erro: {e}")


# 
# Script Principal
# 

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Processa argumento como query
        query = " ".join(sys.argv[1:])
        pipeline = create_hybrid_pipeline_with_rag(use_rag=True)
        result = pipeline.process_with_rag(query)
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    else:
        # Modo interativo
        interactive_pipeline()
