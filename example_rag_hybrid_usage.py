"""
EXEMPLO DE USO: RAG Hbrido com L1-L2
======================================
Demonstra como usar o sistema completo:
- RAG Hbrido (Context Injection + Retrieval Seletivo)
- Integrao com L1 (Conceitos) e L2 (Juzos Kantianos)
- Domain-aware Knowledge Base
- Context Injection com system_prompt especializado
"""

from pathlib import Path
import json
from typing import Optional, Dict, Any

# 
# 1. EXEMPLO BSICO: Apenas RAG Hbrido
# 

def example_1_basic_rag():
    """
    Exemplo 1: Usa RAG hbrido para processar uma query.
    Demonstra: Context Injection + Retrieval Seletivo.
    """
    print("\n" + "="*70)
    print("EXEMPLO 1: RAG Hbrido Bsico")
    print("="*70)

    from rag_hybrid_context_injection import (
        HybridRAGContextInjectionEngine,
        RetrievalStrategy,
    )

    # Cria motor RAG
    rag_engine = HybridRAGContextInjectionEngine(verbose=True)

    # Query de teste
    query = "O que  verdade em lgica paraconsistente?"

    # Processa com RAG hbrido
    rag_context = rag_engine.process(
        query=query,
        strategy=RetrievalStrategy.HYBRID,
        auto_detect_domain=True,
    )

    print(f"\n Domnio detectado: {rag_context.domain}")
    print(f" Confiana: {rag_context.confidence_score:.2%}")
    print(f" Documentos recuperados: {len(rag_context.retrieved_documents)}")
    print(f"\n--- Contexto Compilado (para injeo no LLM) ---")
    print(rag_context.compiled_context[:500] + "...")

    return rag_context


# 
# 2. EXEMPLO: RAG + L1-L2 Pipeline Completo
# 

def example_2_l1_l2_rag_pipeline():
    """
    Exemplo 2: Pipeline completo L1-L2-RAG.
    Demonstra: Conceitos + Juzos + RAG Hbrido integrados.
    """
    print("\n" + "="*70)
    print("EXEMPLO 2: Pipeline Completo L1-L2-RAG")
    print("="*70)

    from l1_l2_rag_integration import create_l1_l2_rag_pipeline

    # Cria pipeline
    pipeline = create_l1_l2_rag_pipeline()

    # Query de teste
    query = "Qual  a definio epistemolgica de conhecimento justificado?"

    # Processa
    result = pipeline.process(query)

    print(f"\n Domnio: {result['domain']}")
    print(f" Confiana: {result['confidence']:.2%}")
    print(f" Conceitos (L1): {len(result['l1_output'].concepts)}")
    print(f" Juzos (L2): {len(result['l2_output'].judgments)}")

    print(f"\n--- L1 (Conceitos) ---")
    for concept in result['l1_output'].concepts[:3]:
        term = concept.term if hasattr(concept, "term") else str(concept)
        print(f"   {term}")

    print(f"\n--- L2 (Top Judgment) ---")
    if result['l2_output'].top_judgment:
        print(f"  {str(result['l2_output'].top_judgment)[:200]}...")

    print(f"\n--- Contexto Compilado (para injeo) ---")
    print(result['compiled_context'][:600] + "...")

    return result


# 
# 3. EXEMPLO: Domain Detection e Context Injection Seletiva
# 

def example_3_domain_specific_injection():
    """
    Exemplo 3: Detecta domnio automaticamente e usa system_prompt especializado.
    Demonstra: Domain-aware context injection.
    """
    print("\n" + "="*70)
    print("EXEMPLO 3: Domain-Specific Context Injection")
    print("="*70)

    from rag_hybrid_context_injection import HybridRAGContextInjectionEngine

    rag_engine = HybridRAGContextInjectionEngine(verbose=True)

    # Queries de diferentes domnios
    queries = [
        ("Aristteles define a substncia como categoria fundamental", "filosofia"),
        ("Na lgica paraconsistente,  possvel ter P e P simultaneamente?", "lgica"),
        ("Como a justificao interna diferencia-se da justificao externa?", "epistemologia"),
    ]

    for query, expected_domain in queries:
        print(f"\n--- Query: {query[:60]}... ---")

        rag_context = rag_engine.process(
            query=query,
            auto_detect_domain=True,
        )

        detected = rag_context.domain
        match = "" if detected == expected_domain else ""
        print(f"{match} Domnio detectado: {detected} (esperado: {expected_domain})")

        # Mostra system_prompt especializado
        domain_ctx = rag_engine.domains[detected]
        print(f"\nSystem Prompt (domnio {detected}):")
        print(f"  {domain_ctx.system_prompt[:150]}...")


# 
# 4. EXEMPLO: Retrieval Strategy Comparativo
# 

def example_4_strategy_comparison():
    """
    Exemplo 4: Compara diferentes estratgias de retrieval.
    Demonstra: Injeo direta vs Semantic Retrieval vs Hybrid.
    """
    print("\n" + "="*70)
    print("EXEMPLO 4: Estratgias de Retrieval Comparativas")
    print("="*70)

    from rag_hybrid_context_injection import (
        HybridRAGContextInjectionEngine,
        RetrievalStrategy,
    )

    rag_engine = HybridRAGContextInjectionEngine(verbose=False)
    query = "O que  uma proposio numa lgica no-clssica?"

    strategies = [
        RetrievalStrategy.DIRECT_INJECTION,
        RetrievalStrategy.SEMANTIC_RETRIEVAL,
        RetrievalStrategy.HYBRID,
    ]

    for strategy in strategies:
        rag_context = rag_engine.process(
            query=query,
            strategy=strategy,
            auto_detect_domain=True,
        )

        injected = sum(1 for d in rag_context.retrieved_documents if d.is_injected)
        retrieved = len(rag_context.retrieved_documents) - injected

        print(f"\n--- Estratgia: {strategy.value} ---")
        print(f"  Injetados: {injected}")
        print(f"  Recuperados: {retrieved}")
        print(f"  Confiana: {rag_context.confidence_score:.2%}")
        print(f"  Contexto (chars): {len(rag_context.compiled_context)}")


# 
# 5. EXEMPLO AVANADO: Formatao para LLM com System Prompt Customizado
# 

def example_5_llm_formatted_output():
    """
    Exemplo 5: Formata sada para injeo direta em LLM local.
    Demonstra: Context Injection com system_prompt customizado.
    """
    print("\n" + "="*70)
    print("EXEMPLO 5: Formatao para Injeo em LLM")
    print("="*70)

    from l1_l2_rag_integration import create_l1_l2_rag_pipeline

    pipeline = create_l1_l2_rag_pipeline()

    query = "Explique a diferena entre conhecimento e opinio justificada."
    result = pipeline.process(query)

    # System prompt customizado (fornecido pelo usurio)
    system_prompt_custom = """Voc  um especialista rigoroso com acesso a uma base de conhecimento especializada.
Responda sempre usando o contexto fornecido quando relevante. Seja preciso e cite fontes quando possvel."""

    # Monta mensagens para LLM
    messages = [
        {
            "role": "system",
            "content": system_prompt_custom,
        },
        {
            "role": "user",
            "content": result["compiled_context"],
        },
    ]

    print("\n--- Mensagens formatadas para LLM (JSON) ---")
    print(json.dumps(messages, indent=2, ensure_ascii=False)[:800] + "...")

    print("\n--- Como usar com Ollama local ---")
    print("""
    import ollama

    response = ollama.chat(
        model="doninha8:latest",
        messages=[{"role": "user", "content": result["compiled_context"]}],
        stream=False,
        options={
            "temperature": 0.3,
            "num_ctx": 8192,
        },
    )
    if isinstance(response, dict):
        print(response.get("message", {}).get("content", ""))
    else:
        print(response)
    """)

    return messages


# 
# 6. EXEMPLO: Criando domnios customizados
# 

def example_6_custom_domains():
    """
    Exemplo 6: Cria e registra domnios customizados.
    Demonstra: Extensibilidade do sistema.
    """
    print("\n" + "="*70)
    print("EXEMPLO 6: Domnios Customizados")
    print("="*70)

    from rag_hybrid_context_injection import (
        HybridRAGContextInjectionEngine,
        DomainContext,
    )

    rag_engine = HybridRAGContextInjectionEngine(verbose=True)

    # Define novo domnio customizado
    domain_direito = DomainContext(
        domain_name="direito",
        description="Direito civil, constitucional e penal",
        keywords=["lei", "cdigo", "artigo", "direito", "obrigao", "contrato"],
        kb_path="data/kb_direito.json",
        chroma_collection="direito_corpus",
        system_prompt="""Voc  um especialista em direito com rigorosa base legal.
Cite sempre artigos, precedentes e legislao pertinente. Mantenha preciso tcnica e referencias s leis.""",
        injection_weight=0.85,
        retrieval_weight=0.15,
    )

    # Registra domnio
    rag_engine.register_domain(domain_direito)

    print(f"\n Domnio 'direito' registrado")
    print(f"  Keywords: {', '.join(domain_direito.keywords[:3])}...")
    print(f"  System Prompt: {domain_direito.system_prompt[:100]}...")

    # Testa com query de direito
    query = "Qual  o prazo para prescrio de dbitos fiscais?"
    detected_domain, conf = rag_engine.detect_domain(query)
    print(f"\n Query sobre direito detectado como: {detected_domain}")


# 
# 7. PIPELINE COMPLETO DE PONTA A PONTA
# 

def example_7_end_to_end_pipeline():
    """
    Exemplo 7: Pipeline completo de ponta a ponta.
    Demonstra: Fluxo completo desde query at resposta estruturada.
    """
    print("\n" + "="*70)
    print("EXEMPLO 7: Pipeline de Ponta a Ponta")
    print("="*70)

    from l1_l2_rag_integration import create_l1_l2_rag_pipeline

    # Cria pipeline
    pipeline = create_l1_l2_rag_pipeline()

    # Query
    query = "Como Kant define o juzo analtico?"

    print(f"\n[1] Input: {query}")

    # Etapa 1: Processamento completo
    result = pipeline.process(query)

    print(f"\n[2] Domain Detection")
    print(f"     Domain: {result['domain']}")
    print(f"     Confidence: {result['confidence']:.2%}")

    print(f"\n[3] L1 (Conceitos) - Extrao e Enriquecimento")
    print(f"     Conceitos extrados: {len(result['l1_output'].concepts)}")
    for i, c in enumerate(result['l1_output'].concepts[:3], 1):
        print(f"      {i}. {c.term if hasattr(c, 'term') else str(c)}")

    print(f"\n[4] L2 (Juzos Kantianos) - Anlise e Enriquecimento")
    print(f"     Juzos gerados: {len(result['l2_output'].judgments)}")
    if result['l2_output'].top_judgment:
        judgment_str = str(result['l2_output'].top_judgment)
        print(f"     Top Judgment: {judgment_str[:120]}...")

    print(f"\n[5] Context Injection")
    print(f"     System Prompt: {result['system_prompt'][:100]}...")
    print(f"     Contexto compilado: {len(result['compiled_context'])} caracteres")

    print(f"\n[6] Sada Final (pronta para LLM)")
    print(f"     RAG Context Summary: {result['rag_context_summary']}")

    return result


# 
# MAIN: Executa todos os exemplos
# 

def main():
    """Executa todos os exemplos."""
    print("\n" + "="*70)
    print("DEMONSTRAO: RAG Hbrido com Context Injection (L1-L2)")
    print("="*70)

    try:
        # Exemplo 1: RAG bsico
        example_1_basic_rag()
    except Exception as e:
        print(f"\n Exemplo 1 falhou: {e}")

    try:
        # Exemplo 2: L1-L2-RAG pipeline
        example_2_l1_l2_rag_pipeline()
    except Exception as e:
        print(f"\n Exemplo 2 falhou: {e}")

    try:
        # Exemplo 3: Domain-specific injection
        example_3_domain_specific_injection()
    except Exception as e:
        print(f"\n Exemplo 3 falhou: {e}")

    try:
        # Exemplo 4: Strategy comparison
        example_4_strategy_comparison()
    except Exception as e:
        print(f"\n Exemplo 4 falhou: {e}")

    try:
        # Exemplo 5: LLM formatted output
        example_5_llm_formatted_output()
    except Exception as e:
        print(f"\n Exemplo 5 falhou: {e}")

    try:
        # Exemplo 6: Custom domains
        example_6_custom_domains()
    except Exception as e:
        print(f"\n Exemplo 6 falhou: {e}")

    try:
        # Exemplo 7: End-to-end
        example_7_end_to_end_pipeline()
    except Exception as e:
        print(f"\n Exemplo 7 falhou: {e}")

    print("\n" + "="*70)
    print(" Demonstrao concluda!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
