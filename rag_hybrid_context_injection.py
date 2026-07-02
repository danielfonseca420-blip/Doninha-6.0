"""
RAG HBRIDO COM CONTEXT INJECTION
===================================
Camada de Retrieval-Augmented Generation (RAG) que trabalha de forma conjunta
com as camadas L1 e L2, usando um protocolo hbrido de:
  1. Context Injection (stuffing direto)  injeta contexto pr-selecionado
  2. Retrieval Seletivo por Domnios  busca documentos relevantes dinamicamente

A soluo  HIBRIDA: injeo direta + retrieval seletivo baseado em domnios.
Integrao com KB especializado (knowledge_base.py) e ChromaDB.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import re
from enum import Enum

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from knowledge_base import get_domain_knowledge_base, load_kb_from_file, merge_kb
except ImportError:
    get_domain_knowledge_base = None
    load_kb_from_file = None
    merge_kb = None


# 
# Enums e Estruturas de Dados
# 

class RetrievalStrategy(Enum):
    """Estratgia de retrieval seletivo."""
    DIRECT_INJECTION = "direct_injection"          # Apenas contexto injetado
    SEMANTIC_RETRIEVAL = "semantic_retrieval"      # Busca semntica em ChromaDB
    HYBRID = "hybrid"                               # Injeo + Retrieval seletivo
    DOMAIN_AWARE = "domain_aware"                   # Retrieval baseado em domnio


@dataclass
class DomainContext:
    """Contexto especializado de um domnio."""
    domain_name: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    kb_path: str = ""                               # Caminho para KB do domnio
    chroma_collection: str = ""                     # Nome da coleo no ChromaDB
    system_prompt: str = ""                         # System prompt especializado
    injection_weight: float = 0.8                   # Peso da injeo direta [0,1]
    retrieval_weight: float = 0.2                   # Peso do retrieval [0,1]
    max_injected_docs: int = 3                      # Mx de docs injetados
    max_retrieved_docs: int = 5                     # Mx de docs recuperados


@dataclass
class RetrievedDocument:
    """Um documento recuperado do knowledge base."""
    content: str
    source: str = ""
    domain: str = ""
    relevance_score: float = 1.0
    is_injected: bool = False                       # Se vem de injeo direta
    metadata: Dict[str, Any] = field(default_factory=dict)

    def truncate(self, max_length: int = 500) -> str:
        """Trunca o contedo para no poluir o contexto."""
        if len(self.content) > max_length:
            return self.content[:max_length].rstrip() + "..."
        return self.content


@dataclass
class RAGContext:
    """Contexto hbrido compilado para injeo no prompt."""
    query: str
    domain: str = "geral"
    retrieved_documents: List[RetrievedDocument] = field(default_factory=list)
    injected_knowledge: Dict[str, float] = field(default_factory=dict)
    compiled_context: str = ""
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    confidence_score: float = 0.0


# 
# Sistema de Domnios Pr-configurados
# 

DEFAULT_DOMAINS: Dict[str, DomainContext] = {
    "filosofia": DomainContext(
        domain_name="filosofia",
        description="Filosofia, epistemologia, lgica clssica",
        keywords=["conhecimento", "verdade", "ser", "essncia", "substncia", "silogismo"],
        kb_path="data/kb_filosofia.json",
        chroma_collection="filosofia_corpus",
        system_prompt="""Voc  um especialista rigoroso em filosofia com acesso a uma base de conhecimento 
especializada em epistemologia, lgica e metafsica. Responda sempre usando o contexto fornecido quando 
relevante. Seja preciso, cite fontes filosficas e mantenha o rigor conceitual.""",
        injection_weight=0.8,
        retrieval_weight=0.2,
    ),
    "lgica": DomainContext(
        domain_name="lgica",
        description="Lgica formal, lgica paraconsistente, teoria de modelos",
        keywords=["proposio", "predicado", "quantificador", "inferncia", "validade", "contradio"],
        kb_path="data/kb_logica.json",
        chroma_collection="logica_corpus",
        system_prompt="""Voc  um especialista em lgica formal e paraconsistncia. Responda sempre 
usando o contexto fornecido quando relevante. Mantenha a preciso tcnica, use notao apropriada e 
cite definies formais quando necessrio.""",
        injection_weight=0.75,
        retrieval_weight=0.25,
    ),
    "epistemologia": DomainContext(
        domain_name="epistemologia",
        description="Epistemologia, teoria do conhecimento, justificao epistmica",
        keywords=["justificao", "crena", "conhecimento", "evidncia", "confiabilismo"],
        kb_path="data/kb_epistemologia.json",
        chroma_collection="epistemologia_corpus",
        system_prompt="""Voc  um especialista rigoroso em epistemologia com acesso a uma base de 
conhecimento especializada. Responda sempre usando o contexto fornecido quando relevante. Cite teorias 
epistemolgicas estabelecidas e seja preciso na caracterizao de conceitos.""",
        injection_weight=0.8,
        retrieval_weight=0.2,
    ),
    "geral": DomainContext(
        domain_name="geral",
        description="Conhecimento geral e enciclopdico",
        keywords=[],
        kb_path="data/kb.json",
        chroma_collection="general_corpus",
        system_prompt="""Voc  um especialista rigoroso com acesso a uma base de conhecimento especializada.
Responda sempre usando o contexto fornecido quando relevante. Seja preciso e cite fontes quando possvel.""",
        injection_weight=0.7,
        retrieval_weight=0.3,
    ),
}


# 
# Motor RAG Hbrido com Context Injection
# 

class HybridRAGContextInjectionEngine:
    """
    Motor principal de RAG hbrido que combina:
    - Context Injection (injeo direta de KB/documentos pr-selecionados)
    - Semantic Retrieval (busca em ChromaDB por similaridade)
    - Domain-Aware Selection (seleo baseada em domnio)
    
    A estratgia HYBRID usa injeo como contexto de base + retrieval seletivo.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chroma_path: str = "chromadb",
        verbose: bool = True,
    ):
        self.config = config or {}
        self.embedding_model = embedding_model
        self.chroma_path = Path(chroma_path)
        self.verbose = verbose
        self.domains = dict(DEFAULT_DOMAINS)
        self.chroma_stores: Dict[str, Any] = {}  # Cache de lojas ChromaDB
        self._initialize_chroma()

    def _initialize_chroma(self) -> None:
        """Inicializa conexes com ChromaDB para cada domnio."""
        if not HAS_CHROMA:
            if self.verbose:
                print("[RAG] ChromaDB no disponvel, usando apenas injeo direta.")
            return

        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            for domain_name in self.domains:
                chroma_dir = self.chroma_path / domain_name
                if chroma_dir.exists() and chroma_dir.is_dir():
                    try:
                        store = Chroma(
                            persist_directory=str(chroma_dir),
                            embedding_function=embeddings,
                            collection_name=self.domains[domain_name].chroma_collection,
                        )
                        self.chroma_stores[domain_name] = store
                        if self.verbose:
                            print(f"[RAG] ChromaDB carregado para domnio '{domain_name}'")
                    except Exception as e:
                        if self.verbose:
                            print(f"[RAG] Erro ao carregar ChromaDB para '{domain_name}': {e}")
        except Exception as e:
            if self.verbose:
                print(f"[RAG] Erro ao inicializar ChromaDB: {e}")

    def register_domain(self, domain: DomainContext) -> None:
        """Registra um novo domnio."""
        self.domains[domain.domain_name] = domain

    def detect_domain(self, query: str, concepts: Optional[List[str]] = None) -> Tuple[str, float]:
        """
        Detecta qual domnio  mais relevante para a query usando keywords matching.
        Retorna (domain_name, confidence_score).
        """
        query_lower = query.lower()
        scores = {}

        for domain_name, domain_ctx in self.domains.items():
            score = 0.0
            if domain_ctx.keywords:
                for kw in domain_ctx.keywords:
                    if kw.lower() in query_lower:
                        score += 1.0
            if concepts:
                for concept in concepts:
                    if concept.lower() in query_lower:
                        score += 0.5

            scores[domain_name] = score

        # Normaliza scores
        max_score = max(scores.values()) if scores else 0.0
        if max_score > 0:
            best_domain = max(scores, key=scores.get)
            confidence = scores[best_domain] / (max_score + 1)
        else:
            best_domain = "geral"
            confidence = 0.1

        return best_domain, confidence

    def get_injected_knowledge(
        self,
        domain: str,
        query: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Recupera conhecimento para injeo direta do KB do domnio.
        Usa get_domain_knowledge_base se disponvel.
        """
        if not get_domain_knowledge_base:
            return {}

        try:
            kb = get_domain_knowledge_base(
                domain=domain,
                config=self.config,
                query_for_rag=query,
            )
            return kb
        except Exception as e:
            if self.verbose:
                print(f"[RAG] Erro ao recuperar KB do domnio '{domain}': {e}")
            return {}

    def retrieve_documents(
        self,
        query: str,
        domain: str = "geral",
        k: int = 5,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    ) -> List[RetrievedDocument]:
        """
        Recupera documentos relevantes usando a estratgia especificada.
        
        Strategies:
        - DIRECT_INJECTION: Sem retrieval, apenas contexto injetado
        - SEMANTIC_RETRIEVAL: Apenas busca em ChromaDB
        - HYBRID: Injeo + Retrieval seletivo
        - DOMAIN_AWARE: Retrieval especfico do domnio
        """
        results: List[RetrievedDocument] = []

        if strategy == RetrievalStrategy.DIRECT_INJECTION:
            # Apenas contexto injetado, sem retrieval dinmico
            return results

        domain_ctx = self.domains.get(domain, self.domains["geral"])
        max_injected = domain_ctx.max_injected_docs
        max_retrieved = domain_ctx.max_retrieved_docs

        # 
        # Estratgia HYBRID: Injeo + Retrieval seletivo
        # 
        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.DOMAIN_AWARE):
            # Etapa 1: Contexto injetado (KB direto)
            injected_kb = self.get_injected_knowledge(domain, query)
            if injected_kb:
                # Seleciona top-k termos por relevncia
                sorted_terms = sorted(injected_kb.items(), key=lambda x: x[1], reverse=True)
                for i, (term, score) in enumerate(sorted_terms[:max_injected]):
                    results.append(
                        RetrievedDocument(
                            content=f"Termo: {term}",
                            source=f"KB-{domain}",
                            domain=domain,
                            relevance_score=float(score),
                            is_injected=True,
                            metadata={"type": "kb_term", "weight": score},
                        )
                    )

        # Etapa 2: Retrieval semntico (ChromaDB)
        if strategy in (RetrievalStrategy.SEMANTIC_RETRIEVAL, RetrievalStrategy.HYBRID):
            if domain in self.chroma_stores:
                try:
                    chroma = self.chroma_stores[domain]
                    docs = chroma.similarity_search(query, k=max_retrieved)
                    for doc in docs:
                        # Extrai score se disponvel
                        score = getattr(doc, "metadata", {}).get("score", 0.8)
                        results.append(
                            RetrievedDocument(
                                content=doc.page_content if hasattr(doc, "page_content") else str(doc),
                                source=f"ChromaDB-{domain}",
                                domain=domain,
                                relevance_score=float(score),
                                is_injected=False,
                                metadata=getattr(doc, "metadata", {}),
                            )
                        )
                except Exception as e:
                    if self.verbose:
                        print(f"[RAG] Erro ao recuperar de ChromaDB-{domain}: {e}")

        # Ordena por relevncia
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:k]

    def compile_context(
        self,
        query: str,
        retrieved_docs: List[RetrievedDocument],
        injected_kb: Optional[Dict[str, float]] = None,
        domain: str = "geral",
        include_system_prompt: bool = True,
    ) -> RAGContext:
        """
        Compila o contexto final para injeo no prompt.
        Combina documentos recuperados, KB injetado e system prompt.
        """
        domain_ctx = self.domains.get(domain, self.domains["geral"])
        lines = []

        # 
        # Parte 1: System Prompt especializado
        # 
        if include_system_prompt and domain_ctx.system_prompt:
            lines.append("## Instrues do Sistema")
            lines.append(domain_ctx.system_prompt)
            lines.append("")

        # 
        # Parte 2: Documentos Injetados (Context Injection)
        # 
        injected_docs = [d for d in retrieved_docs if d.is_injected]
        if injected_docs:
            lines.append("## Contexto Base Injetado (Domnio)")
            for doc in injected_docs:
                lines.append(f"- **{doc.source}** [{doc.relevance_score:.2f}]: {doc.truncate()}")
            lines.append("")

        # 
        # Parte 3: Documentos Recuperados (Semantic Retrieval)
        # 
        retrieved_only = [d for d in retrieved_docs if not d.is_injected]
        if retrieved_only:
            lines.append("## Contexto Recuperado (ChromaDB)")
            for doc in retrieved_only:
                lines.append(f"- **{doc.source}**: {doc.truncate()}")
            lines.append("")

        # 
        # Parte 4: Knowledge Base Terms (se fornecido)
        # 
        if injected_kb:
            lines.append("## Termos-Chave do Knowledge Base")
            sorted_terms = sorted(injected_kb.items(), key=lambda x: x[1], reverse=True)[:10]
            for term, score in sorted_terms:
                lines.append(f"- {term}: {score:.2f}")
            lines.append("")

        # 
        # Parte 5: Instruo de Resposta
        # 
        lines.append("## Pergunta do Usurio")
        lines.append(f"{query}")
        lines.append("")
        lines.append("---")
        lines.append("Baseando-se no contexto injetado e recuperado acima, elabore uma resposta rigorosa.")
        lines.append("")

        compiled = "\n".join(lines)

        # Calcula confidence score
        conf = 0.0
        if injected_docs:
            conf += domain_ctx.injection_weight * (sum(d.relevance_score for d in injected_docs) / len(injected_docs))
        if retrieved_only:
            conf += domain_ctx.retrieval_weight * (sum(d.relevance_score for d in retrieved_only) / len(retrieved_only))
        conf = min(1.0, conf)

        return RAGContext(
            query=query,
            domain=domain,
            retrieved_documents=retrieved_docs,
            injected_knowledge=injected_kb or {},
            compiled_context=compiled,
            strategy=RetrievalStrategy.HYBRID,
            confidence_score=conf,
        )

    def process(
        self,
        query: str,
        concepts: Optional[List[str]] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        k: int = 8,
        auto_detect_domain: bool = True,
    ) -> RAGContext:
        """
        Pipeline completo de RAG hbrido.
        
        Etapas:
        1. Detecta domnio (se auto_detect_domain=True)
        2. Recupera documentos (injeo + retrieval)
        3. Compila contexto final
        4. Retorna RAGContext pronto para injeo
        """
        # Detecta domnio
        if auto_detect_domain:
            domain, conf = self.detect_domain(query, concepts)
            if self.verbose:
                print(f"[RAG] Domnio detectado: {domain} (confiana: {conf:.2f})")
        else:
            domain = "geral"

        # Recupera conhecimento injetado
        injected_kb = self.get_injected_knowledge(domain, query)

        # Recupera documentos
        retrieved_docs = self.retrieve_documents(
            query=query,
            domain=domain,
            k=k,
            strategy=strategy,
        )

        # Compila contexto
        rag_context = self.compile_context(
            query=query,
            retrieved_docs=retrieved_docs,
            injected_kb=injected_kb,
            domain=domain,
            include_system_prompt=True,
        )

        return rag_context

    def format_for_l1_l2(self, rag_context: RAGContext) -> Dict[str, Any]:
        """
        Formata o contexto RAG para consumo pelas camadas L1 (Conceitos) e L2 (Juzos).
        Retorna um dicionrio com:
        - domain: domnio detectado
        - injected_context: string do contexto injetado
        - kb_terms: dicionrio termo -> score
        - system_prompt: system prompt especializado
        - documents: lista de documentos
        """
        domain_ctx = self.domains.get(rag_context.domain, self.domains["geral"])

        return {
            "domain": rag_context.domain,
            "injected_context": rag_context.compiled_context,
            "kb_terms": rag_context.injected_knowledge,
            "system_prompt": domain_ctx.system_prompt,
            "documents": [
                {
                    "content": doc.truncate(1000),
                    "source": doc.source,
                    "relevance": doc.relevance_score,
                    "is_injected": doc.is_injected,
                }
                for doc in rag_context.retrieved_documents
            ],
            "confidence": rag_context.confidence_score,
        }


# 
# Funes Auxiliares de Alto Nvel
# 

def create_hybrid_rag_engine(
    config: Optional[Dict[str, Any]] = None,
    chroma_path: str = "chromadb",
) -> HybridRAGContextInjectionEngine:
    """Factory para criar uma instncia do motor RAG."""
    return HybridRAGContextInjectionEngine(config=config, chroma_path=chroma_path)


def process_query_with_rag(
    query: str,
    concepts: Optional[List[str]] = None,
    domain: Optional[str] = None,
    auto_detect: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> RAGContext:
    """
    Funo de convenincia para processar uma query com RAG hbrido.
    
    Exemplo:
        rag_ctx = process_query_with_rag("O que  conhecimento?", domain="epistemologia")
        print(rag_ctx.compiled_context)
    """
    engine = create_hybrid_rag_engine(config=config)
    return engine.process(
        query=query,
        concepts=concepts,
        auto_detect_domain=auto_detect,
        strategy=RetrievalStrategy.HYBRID,
    )
