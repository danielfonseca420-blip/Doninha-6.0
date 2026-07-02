"""
INTEGRAO L1-L2 COM RAG HBRIDO
=================================
Estende as camadas L1 (Conceitos) e L2 (Juzos) para trabalhar com:
- RAG Hbrido (Context Injection + Retrieval Seletivo)
- Domain-Aware Knowledge Base
- Injeo de contexto nas tabelas de conceitos e juzos

Workflow:
1. RAG processa query e detecta domnio
2. Contexto injetado enriquece L1 (ConceptTable)
3. L2 refina juzos usando KB especializado do domnio
4. Sistema retorna tabelas enriquecidas
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

try:
    from l1_concept_table import ConceptTable, ConceptNode
except ImportError:
    ConceptTable = None  # type: ignore
    ConceptNode = None   # type: ignore

try:
    from l2_kantian_judgments import KantianJudgmentEngine, KantianJudgment, EpistemicClassification
except ImportError:
    KantianJudgmentEngine = None  # type: ignore
    KantianJudgment = None         # type: ignore
    EpistemicClassification = None # type: ignore

try:
    from rag_hybrid_context_injection import (
        HybridRAGContextInjectionEngine,
        RAGContext,
        RetrievalStrategy,
        DomainContext,
    )
except ImportError:
    HybridRAGContextInjectionEngine = None  # type: ignore
    RAGContext = None                        # type: ignore
    RetrievalStrategy = None                 # type: ignore
    DomainContext = None                     # type: ignore


@dataclass
class EnrichedL1Output:
    """Sada enriquecida da camada L1 com contexto RAG."""
    concepts: List[ConceptNode] = field(default_factory=list)
    domain: str = "geral"
    kb_terms: Dict[str, float] = field(default_factory=dict)
    injected_docs: int = 0
    domain_confidence: float = 0.0
    system_prompt: str = ""
    rag_context_summary: str = ""


@dataclass
class EnrichedL2Output:
    """Sada enriquecida da camada L2 com contexto RAG."""
    judgments: List[KantianJudgment] = field(default_factory=list)
    domain: str = "geral"
    top_judgment: Optional[KantianJudgment] = None
    domain_specialized_kb: Dict[str, float] = field(default_factory=dict)
    epistemic_evidence: Dict[str, float] = field(default_factory=dict)
    rag_impact_score: float = 0.0


class L1RAGEnricher:
    """
    Enriquece a Camada L1 (Tbua de Conceitos) com contexto RAG.
    
    Workflow:
    1. Recebe query + L1 ConceptTable
    2. Processa com RAG hbrido
    3. Enriquece conceitos com conhecimento injetado
    4. Retorna L1 expandido
    """

    def __init__(
        self,
        concept_table: Optional[Any] = None,
        rag_engine: Optional[HybridRAGContextInjectionEngine] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        if ConceptTable is None:
            raise RuntimeError("l1_concept_table no pde ser importado")

        self.concept_table = concept_table or ConceptTable()
        self.rag_engine = rag_engine or HybridRAGContextInjectionEngine(config=config)
        self.config = config or {}
        self.verbose = verbose

    def extract_and_enrich(
        self,
        query: str,
        auto_detect_domain: bool = True,
    ) -> EnrichedL1Output:
        """
        Extrai conceitos do query e enriquece com contexto RAG.
        
        Etapas:
        1. RAG: Detecta domnio e recupera documentos
        2. Extrai conceitos padro (L1 original)
        3. Enriquece conceitos com KB injetado
        4. Retorna output enriquecido
        """
        # Etapa 1: Processa com RAG
        rag_context = self.rag_engine.process(
            query=query,
            auto_detect_domain=auto_detect_domain,
            strategy=RetrievalStrategy.HYBRID,
        )

        domain = rag_context.domain
        injected_kb = rag_context.injected_knowledge
        retrieved_docs = rag_context.retrieved_documents

        if self.verbose:
            print(f"\n[L1-RAG] Domnio: {domain}")
            print(f"[L1-RAG] Docs injetados: {sum(1 for d in retrieved_docs if d.is_injected)}")
            print(f"[L1-RAG] Docs recuperados: {sum(1 for d in retrieved_docs if not d.is_injected)}")

        # Etapa 2: Extrai conceitos originais (L1)
        concepts = self.concept_table.extract_concepts(query, domain=domain)

        # Etapa 3: Enriquece conceitos com KB injetado
        enriched_concepts = self._enrich_concepts_with_kb(
            concepts=concepts,
            injected_kb=injected_kb,
            domain=domain,
            rag_context=rag_context,
        )

        # Etapa 4: Compila output
        return EnrichedL1Output(
            concepts=enriched_concepts,
            domain=domain,
            kb_terms=injected_kb,
            injected_docs=sum(1 for d in retrieved_docs if d.is_injected),
            domain_confidence=rag_context.confidence_score,
            system_prompt=self.rag_engine.domains[domain].system_prompt,
            rag_context_summary=self._summarize_context(retrieved_docs),
        )

    def _enrich_concepts_with_kb(
        self,
        concepts: List[Any],
        injected_kb: Dict[str, float],
        domain: str,
        rag_context: RAGContext,
    ) -> List[Any]:
        """
        Enriquece cada conceito com informaes do KB injetado e documentos recuperados.
        Adiciona: domnio, contexto de aplicao, fonte cannica.
        """
        if not concepts:
            return concepts

        enriched = []
        for concept in concepts:
            if not isinstance(concept, ConceptNode):
                enriched.append(concept)
                continue

            # Clone do conceito para enriquecer
            enriched_concept = self.concept_table._clone_node(concept)

            # Atribui domnio
            enriched_concept.domain = domain

            # Busca evidncia no KB
            kb_score = injected_kb.get(concept.term.lower(), 0.0)
            if kb_score > 0:
                enriched_concept.definition += f"\n[KB-{domain}: {kb_score:.2f}]"

            # Busca fonte nos documentos
            for doc in rag_context.retrieved_documents:
                if concept.term.lower() in doc.content.lower():
                    enriched_concept.canonical_source = f"{doc.source}"
                    enriched_concept.application_context = doc.truncate(200)
                    break

            enriched.append(enriched_concept)

        return enriched

    def _summarize_context(self, docs: List[Any]) -> str:
        """Cria um resumo textual do contexto recuperado."""
        if not docs:
            return "Nenhum contexto recuperado."

        lines = []
        injected = sum(1 for d in docs if getattr(d, "is_injected", False))
        retrieved = len(docs) - injected

        lines.append(f"Contexto RAG: {injected} injetado(s), {retrieved} recuperado(s)")
        for i, doc in enumerate(docs[:3]):
            source = getattr(doc, "source", "desconhecido")
            score = getattr(doc, "relevance_score", 0.0)
            lines.append(f"  {i+1}. [{source}] score={score:.2f}")

        return "; ".join(lines)


class L2RAGEnricher:
    """
    Enriquece a Camada L2 (Juzos Kantianos) com contexto RAG.
    
    Workflow:
    1. Recebe query + L2 KantianJudgmentEngine
    2. Processa com RAG hbrido (domnio especializado)
    3. Refina juzos usando KB especializado
    4. Retorna L2 expandido com classificao epistemolgica aprimorada
    """

    def __init__(
        self,
        concept_table: Optional[Any] = None,
        judgment_engine: Optional[Any] = None,
        rag_engine: Optional[HybridRAGContextInjectionEngine] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        if KantianJudgmentEngine is None:
            raise RuntimeError("l2_kantian_judgments no pde ser importado")

        self.concept_table = concept_table
        self.judgment_engine = judgment_engine or KantianJudgmentEngine(concept_table)
        self.rag_engine = rag_engine or HybridRAGContextInjectionEngine(config=config)
        self.config = config or {}
        self.verbose = verbose

    def analyze_and_enrich(
        self,
        query: str,
        concepts: Optional[List[Any]] = None,
        auto_detect_domain: bool = True,
    ) -> EnrichedL2Output:
        """
        Analisa query com L2 e enriquece juzos com contexto RAG.
        
        Etapas:
        1. RAG: Recupera contexto especializado do domnio
        2. L2: Gera juzos kantianos (original)
        3. Enriquece: Refina modalidades e evidncias usando KB
        4. Retorna L2 enriquecido
        """
        # Etapa 1: RAG especializado
        rag_context = self.rag_engine.process(
            query=query,
            concepts=[c.term if isinstance(c, ConceptNode) else str(c) for c in (concepts or [])] if concepts else None,
            auto_detect_domain=auto_detect_domain,
            strategy=RetrievalStrategy.HYBRID,
        )

        domain = rag_context.domain
        domain_kb = rag_context.injected_knowledge

        if self.verbose:
            print(f"\n[L2-RAG] Domnio: {domain}")
            print(f"[L2-RAG] KB Terms: {len(domain_kb)}")

        # Etapa 2: Anlise L2 padro
        judgments = self.judgment_engine.infer_from_prompt(query)

        # Etapa 3: Enriquecimento baseado em RAG
        enriched_judgments = self._enrich_judgments_with_kb(
            judgments=judgments,
            domain_kb=domain_kb,
            domain=domain,
            rag_context=rag_context,
        )

        # Calcula top judgment
        top_judgment = max(enriched_judgments, key=lambda j: j.prioridade) if enriched_judgments else None

        # Computa epistemic evidence
        epistemic_evidence = self._compute_epistemic_evidence(enriched_judgments, domain_kb)

        # Computa RAG impact
        rag_impact = self._compute_rag_impact(enriched_judgments, rag_context)

        return EnrichedL2Output(
            judgments=enriched_judgments,
            domain=domain,
            top_judgment=top_judgment,
            domain_specialized_kb=domain_kb,
            epistemic_evidence=epistemic_evidence,
            rag_impact_score=rag_impact,
        )

    def _enrich_judgments_with_kb(
        self,
        judgments: List[Any],
        domain_kb: Dict[str, float],
        domain: str,
        rag_context: RAGContext,
    ) -> List[Any]:
        """
        Enriquece juzos com informao do KB e documentos.
        - Atualiza prioridade baseado em KB relevncia
        - Refina classificao epistemolgica
        - Adiciona evidncia de suporte
        """
        if not judgments:
            return judgments

        enriched = []
        for judgment in judgments:
            if not isinstance(judgment, KantianJudgment):
                enriched.append(judgment)
                continue

            # Extrai termos da proposio
            prop_terms = [w.lower() for w in judgment.proposicao.split() if len(w) > 3]

            # Calcula boost de prioridade baseado em KB
            kb_boost = 0.0
            for term in prop_terms:
                if term in domain_kb:
                    kb_boost += domain_kb[term] * 0.1

            # Refina prioridade
            judgment.prioridade = min(1.0, judgment.prioridade + kb_boost)

            # Refina classificao epistemolgica
            judgment.epistemic_classification = self._refine_epistemic_classification(
                judgment.epistemic_classification,
                domain_kb,
                prop_terms,
            )

            enriched.append(judgment)

        return enriched

    def _refine_epistemic_classification(
        self,
        current_classification: Any,
        domain_kb: Dict[str, float],
        terms: List[str],
    ) -> Any:
        """
        Refina a classificao epistemolgica usando informao do KB.
        """
        if not EpistemicClassification:
            return current_classification

        # Calcula scores baseado em KB
        kb_truth_score = sum(domain_kb.get(t, 0.0) for t in terms) / max(len(terms), 1)
        kb_indeterminacy = 1.0 - kb_truth_score if kb_truth_score < 0.5 else 0.0

        # Cria nova classificao refinada
        refined = EpistemicClassification(
            truth=min(1.0, current_classification.truth + kb_truth_score * 0.2),
            indeterminacy=max(0.0, current_classification.indeterminacy - kb_indeterminacy * 0.1),
            falsity=max(0.0, current_classification.falsity - kb_truth_score * 0.1),
        )

        return refined

    def _compute_epistemic_evidence(
        self,
        judgments: List[Any],
        domain_kb: Dict[str, float],
    ) -> Dict[str, float]:
        """Computa evidncia epistemolgica para cada dimenso."""
        evidence = {
            "truth_evidence": 0.0,
            "indeterminacy_evidence": 0.0,
            "falsity_evidence": 0.0,
            "clarity_evidence": 0.0,
        }

        if not judgments or not domain_kb:
            return evidence

        kb_values = list(domain_kb.values())
        if not kb_values:
            return evidence

        avg_kb_value = sum(kb_values) / len(kb_values)

        evidence["truth_evidence"] = avg_kb_value
        evidence["indeterminacy_evidence"] = 1.0 - avg_kb_value
        evidence["clarity_evidence"] = 1.0 - abs(avg_kb_value - 0.5)

        return evidence

    def _compute_rag_impact(self, judgments: List[Any], rag_context: RAGContext) -> float:
        """Computa o impacto do RAG na qualidade dos juzos."""
        if not judgments:
            return 0.0

        # Baseado na confiana do contexto e qualidade dos juzos
        base_confidence = rag_context.confidence_score
        judgment_quality = sum(
            min(1.0, j.prioridade if hasattr(j, "prioridade") else 0.0)
            for j in judgments
        ) / len(judgments)

        return base_confidence * judgment_quality


class IntegratedL1L2RAGPipeline:
    """
    Pipeline completo integrado de L1 + L2 com RAG Hbrido.
    
    Workflow completo:
    1. Recebe query
    2. RAG Hbrido processa (domain detection + context retrieval)
    3. L1 extrai e enriquece conceitos
    4. L2 analisa e enriquece juzos
    5. Retorna output estruturado com ambas as camadas
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ):
        self.config = config or {}
        self.verbose = verbose

        # Inicializa componentes
        self.concept_table = ConceptTable() if ConceptTable else None
        self.rag_engine = HybridRAGContextInjectionEngine(config=config, verbose=verbose)
        self.l1_enricher = L1RAGEnricher(
            concept_table=self.concept_table,
            rag_engine=self.rag_engine,
            config=config,
            verbose=verbose,
        )
        self.l2_enricher = L2RAGEnricher(
            concept_table=self.concept_table,
            rag_engine=self.rag_engine,
            config=config,
            verbose=verbose,
        )

    def process(
        self,
        query: str,
        auto_detect_domain: bool = True,
    ) -> Dict[str, Any]:
        """
        Processa query atravs do pipeline completo L1-L2-RAG.
        
        Retorna:
        {
            'query': str,
            'domain': str,
            'l1_output': EnrichedL1Output,
            'l2_output': EnrichedL2Output,
            'compiled_context': str,  # Contexto injetado para LLM
            'system_prompt': str,
            'confidence': float,
        }
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[L1-L2-RAG PIPELINE] Processando query: {query[:60]}...")
            print(f"{'='*70}")

        # L1: Extrao e enriquecimento de conceitos
        l1_output = self.l1_enricher.extract_and_enrich(
            query=query,
            auto_detect_domain=auto_detect_domain,
        )

        if self.verbose:
            print(f"\n[L1] Conceitos extrados: {len(l1_output.concepts)}")
            print(f"[L1] Domnio: {l1_output.domain}")

        # L2: Anlise e enriquecimento de juzos
        l2_output = self.l2_enricher.analyze_and_enrich(
            query=query,
            concepts=l1_output.concepts,
            auto_detect_domain=False,  # J detectado por L1-RAG
        )

        if self.verbose:
            print(f"\n[L2] Juzos gerados: {len(l2_output.judgments)}")
            if l2_output.top_judgment:
                print(f"[L2] Top judgment: {str(l2_output.top_judgment)[:100]}...")

        # Compila sada final
        return {
            "query": query,
            "domain": l1_output.domain,
            "l1_output": l1_output,
            "l2_output": l2_output,
            "compiled_context": self._compile_final_context(l1_output, l2_output),
            "system_prompt": l1_output.system_prompt,
            "confidence": max(l1_output.domain_confidence, l2_output.rag_impact_score),
            "rag_context_summary": l1_output.rag_context_summary,
        }

    def _compile_final_context(self, l1_output: EnrichedL1Output, l2_output: EnrichedL2Output) -> str:
        """Compila o contexto final para injeo em LLM."""
        lines = [
            "## Contexto Estruturado (L1-L2-RAG)",
            "",
            f"**Domnio Detectado**: {l1_output.domain}",
            f"**Confiana**: {max(l1_output.domain_confidence, l2_output.rag_impact_score):.2%}",
            "",
            "### Camada L1 (Conceitos)",
            f"Conceitos extrados: {len(l1_output.concepts)}",
        ]

        for concept in l1_output.concepts[:5]:
            concept_name = concept.term if hasattr(concept, "term") else str(concept)
            lines.append(f"  - {concept_name}")

        lines.extend([
            "",
            "### Camada L2 (Juzos Kantianos)",
            f"Juzos gerados: {len(l2_output.judgments)}",
        ])

        if l2_output.top_judgment:
            lines.append(f"  **Top Judgment**: {str(l2_output.top_judgment)[:150]}...")

        lines.extend([
            "",
            "### Knowledge Base (Injetado)",
            f"Termos-chave: {len(l2_output.domain_specialized_kb)}",
        ])

        for term, score in list(l2_output.domain_specialized_kb.items())[:5]:
            lines.append(f"  - {term}: {score:.2f}")

        lines.extend([
            "",
            "---",
            "Use o contexto acima para formular uma resposta rigorosa e bem fundamentada.",
            "",
        ])

        return "\n".join(lines)


# 
# Funes de Convenincia
# 

def create_l1_l2_rag_pipeline(
    config: Optional[Dict[str, Any]] = None,
) -> IntegratedL1L2RAGPipeline:
    """Factory para criar pipeline integrado."""
    return IntegratedL1L2RAGPipeline(config=config, verbose=True)


def process_with_l1_l2_rag(query: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Funo de convenincia para processar query com pipeline L1-L2-RAG.
    
    Exemplo:
        result = process_with_l1_l2_rag("O que  conhecimento?")
        print(result["compiled_context"])
    """
    pipeline = create_l1_l2_rag_pipeline(config=config)
    return pipeline.process(query)
