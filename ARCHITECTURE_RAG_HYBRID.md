"""
ARQUITETURA — RAG HÍBRIDO COM L1-L2
====================================
Diagrama e fluxo de dados do sistema completo.
"""

ARCHITECTURE_DIAGRAM = """

┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLUXO DE ENTRADA (QUERY)                          │
│                                                                              │
│  Usuário: "O que é conhecimento justificado em epistemologia?"              │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│           1. RAG HYBRID CONTEXT INJECTION ENGINE (rag_hybrid...py)          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1A. DETECÇÃO DE DOMÍNIO (Auto-detect)                              │    │
│  │     Keywords matching: "conhecimento", "justificado"               │    │
│  │     └─→ Domínio detectado: "epistemologia" (90% confiança)        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1B. CONTEXT INJECTION (Injeção Direta)                             │    │
│  │     Carrega: data/kb_epistemologia.json                            │    │
│  │     └─→ Termos injetados: {"justificação": 0.9, ...}              │    │
│  │         [Estratégia: Peso 80% da confiança]                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 1C. SEMANTIC RETRIEVAL (Busca em ChromaDB)                         │    │
│  │     Query embedding → ChromaDB/epistemologia/                      │    │
│  │     └─→ Docs recuperados (top-5):                                  │    │
│  │         • "A justificação interna..." (score: 0.85)                │    │
│  │         • "Confiabilismo epistêmico..." (score: 0.78)              │    │
│  │         [Estratégia: Peso 20% da confiança]                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Output: RAGContext {                                                       │
│    domain: "epistemologia",                                                 │
│    retrieved_documents: [...],          # Injetados + Recuperados           │
│    compiled_context: "## Contexto Base...",                                 │
│    confidence_score: 0.88                                                   │
│  }                                                                           │
│                                                                              │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│    2. L1-L2-RAG INTEGRATION (l1_l2_rag_integration.py)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────┐          ┌────────────────────────┐             │
│  │   L1RAGEnricher        │          │   L2RAGEnricher        │             │
│  │  (Conceitos)           │          │  (Juízos Kantianos)    │             │
│  └────────┬───────────────┘          └────────┬───────────────┘             │
│           │                                   │                             │
│  Input:   │ RAGContext +            Input:   │ RAGContext +                │
│           │ query                            │ l1_concepts                 │
│           │                                  │                             │
│  ┌────────▼───────────────┐          ┌──────▼──────────────────┐           │
│  │ extract_concepts()     │          │ analyze_and_enrich()   │           │
│  │ + KB injection         │          │ + domain KB refinement │           │
│  │                        │          │                        │           │
│  │ ConceptNode[] com:     │          │ KantianJudgment[] com: │           │
│  │ • domain               │          │ • refined prioridade   │           │
│  │ • kb_score             │          │ • refined epistemic    │           │
│  │ • source               │          │   classification       │           │
│  └────────┬───────────────┘          └──────┬──────────────────┘           │
│           │                                 │                              │
│  Output:  │ EnrichedL1Output        Output: │ EnrichedL2Output            │
│           │                                 │                              │
│           └────────────┬────────────────────┘                              │
│                        │                                                   │
└────────────────────────┼───────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. COMPILAÇÃO DE CONTEXTO FINAL (Context Injection para LLM)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Resultado compilado_context:                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ ## Instruções do Sistema                                            │    │
│  │ Você é um especialista rigoroso em epistemologia...                │    │
│  │                                                                     │    │
│  │ ## Contexto Base Injetado (Domínio)                               │    │
│  │ - KB-epistemologia [0.90]: Termo: justificação                   │    │
│  │ - KB-epistemologia [0.85]: Termo: crença                         │    │
│  │                                                                     │    │
│  │ ## Contexto Recuperado (ChromaDB)                                 │    │
│  │ - ChromaDB-epistemologia: A justificação interna diferencia-se... │    │
│  │ - ChromaDB-epistemologia: Confiabilismo epistêmico define...      │    │
│  │                                                                     │    │
│  │ ## Termos-Chave do Knowledge Base                                 │    │
│  │ - justificação: 0.90                                               │    │
│  │ - crença: 0.85                                                     │    │
│  │ - conhecimento: 0.92                                               │    │
│  │                                                                     │    │
│  │ ## Pergunta do Usuário                                            │    │
│  │ O que é conhecimento justificado em epistemologia?                │    │
│  │                                                                     │    │
│  │ Baseando-se no contexto acima, elabore uma resposta rigorosa.     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│           4. INJEÇÃO EM LLM (OpenAI, Claude, Ollama, etc)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  messages = [                                                              │
│    {                                                                        │
│      "role": "system",                                                     │
│      "content": system_prompt  # Especializado por domínio                │
│    },                                                                       │
│    {                                                                        │
│      "role": "user",                                                       │
│      "content": compiled_context  # Context injection (injeção)            │
│    }                                                                        │
│  ]                                                                          │
│                                                                              │
│  response = client.chat.completions.create(                               │
│    model="gpt-4",                                                          │
│    messages=messages,                                                      │
│    temperature=0.3,  # Mais determinístico                                │
│    max_tokens=1024                                                        │
│  )                                                                          │
│                                                                              │
│  ✓ LLM gera resposta estruturada e fundamentada                           │
│                                                                              │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      5. RESPOSTA FINAL (ao usuário)                         │
│                                                                              │
│  "Conhecimento justificado, na tradição epistemológica analítica, é       │
│   compreendido como aquilo que satisfaz as condições de: (1) ser uma      │
│   crença; (2) ser verdadeiro; e (3) estar adequadamente justificado. A    │
│   justificação pode ser interna (dependente do sujeito) ou externa...     │
│   [Resposta fundamentada com contexto enriquecido]"                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
FLUXO DE DADOS DETALHADO
================================================================================

Query
  ↓
[1] RAG Hybrid Engine
  ├─→ Detect Domain
  ├─→ Load KB (context injection)
  ├─→ Retrieve from ChromaDB (semantic retrieval)
  ├─→ Compile RAGContext
  └─→ Output: RAGContext
      
[2] L1 Enricher
  ├─→ Extract Concepts (original)
  ├─→ Enrich with KB injection
  ├─→ Add domain & source info
  └─→ Output: EnrichedL1Output
      
[3] L2 Enricher
  ├─→ Infer Judgments (original)
  ├─→ Refine with domain KB
  ├─→ Update epistemic classification
  └─→ Output: EnrichedL2Output
      
[4] Context Compiler
  ├─→ Add System Prompt (domain-specific)
  ├─→ Add Injected Context (from L1-L2)
  ├─→ Add Retrieved Documents
  └─→ Output: compiled_context (string)
      
[5] LLM Input Formatter
  ├─→ system_prompt (from domain)
  ├─→ user_message (compiled_context)
  └─→ Output: messages (for LLM API)
      
[6] LLM Processing
  └─→ Generate Response with injected context

================================================================================
COMPARAÇÃO: COM vs SEM RAG HÍBRIDO
================================================================================

SEM RAG (Padrão):
─────────────────
Query → L1 → L2 → [Sem contexto especializado] → LLM
Resultado: Genérico, sem fundamentação em KB

COM RAG HÍBRIDO:
────────────────
Query → [Domain Detection] → [KB Injection + ChromaDB Retrieval] 
  → Enriched L1 → Enriched L2 → [Context Injection Automático]
  → LLM com System Prompt Especializado
Resultado: Fundamentado, domain-aware, com citações de KB

Benefícios:
• Context Injection: Garante contexto relevante
• Domain Detection: Seleciona KB especializado
• L1-L2 Enriquecimento: Conceitos e juízos melhorados
• System Prompt: Personalizado por domínio
• Hybrid: Combina velocidade (injeção) + flexibilidade (retrieval)

================================================================================
CONFIGURAÇÃO EXEMPLAR
================================================================================

Domínio: Epistemologia

rag_hybrid:
  domains:
    epistemologia:
      injection_weight: 0.8    # 80% do score vem de injeção
      retrieval_weight: 0.2    # 20% do score vem de retrieval
      max_injected_docs: 3     # Máximo 3 documentos injetados
      max_retrieved_docs: 5    # Máximo 5 documentos recuperados
      system_prompt: |
        Você é um especialista rigoroso em epistemologia com acesso
        a uma base de conhecimento especializada. Responda sempre
        usando o contexto fornecido quando relevante. Cite teorias
        epistemológicas estabelecidas e seja preciso na caracterização
        de conceitos.

Resultado:
  • Queries sobre epistemologia ativam automáticamente
    - KB especializado (data/kb_epistemologia.json)
    - ChromaDB collection "epistemologia_corpus"
    - System prompt especializado
  • Resposta é fundamentada e precisa no domínio

================================================================================
INTEGRAÇÃO COM PIPELINE.PY EXISTENTE
================================================================================

Passo 1: Importe RAG
─────────────────────
  from l1_l2_rag_integration import create_l1_l2_rag_pipeline

Passo 2: Inicialize na classe
──────────────────────────────
  class HybridLLMPipeline:
    def __init__(self, use_rag=True):
      if use_rag:
        self.rag_pipeline = create_l1_l2_rag_pipeline()

Passo 3: Use no process()
─────────────────────────
  def process(self, prompt):
    if self.use_rag:
      result = self.rag_pipeline.process(prompt)
      return result  # Já tem L1, L2, contexto compilado
    else:
      # Fallback para método original

Passo 4: Integre com L3-L7
─────────────────────────
  result['rag_l1_l2_output'] → alimenta L3 (Paraconsistent)
  result['compiled_context'] → injetar em LLM (L5-L6-L7)

================================================================================
PRÓXIMOS PASSOS
================================================================================

1. Teste inicial:
   python example_rag_hybrid_usage.py

2. Execute testes unitários:
   python test_rag_hybrid.py

3. Use no modo interativo:
   python pipeline_with_rag_integration.py

4. Integre com seu LLM:
   - OpenAI (veja exemplo no README)
   - Claude (veja exemplo no README)
   - Ollama local (veja exemplo no README)

5. Customize domínios:
   - Edite config_rag.yaml
   - Ou adicione via code (ver exemplo)

6. Monitore performance:
   - confidence_score (0.0-1.0)
   - domain_accuracy
   - retrieval_quality

================================================================================
"""

QUICK_REFERENCE = """

QUICK REFERENCE
===============

Importações principais:
  from rag_hybrid_context_injection import HybridRAGContextInjectionEngine
  from l1_l2_rag_integration import create_l1_l2_rag_pipeline
  from pipeline_with_rag_integration import create_hybrid_pipeline_with_rag

Uso básico (RAG sozinho):
  engine = HybridRAGContextInjectionEngine()
  rag_ctx = engine.process("sua query")
  print(rag_ctx.compiled_context)

Uso recomendado (RAG + L1-L2):
  pipeline = create_l1_l2_rag_pipeline()
  result = pipeline.process("sua query")
  print(result['compiled_context'])

Integração com pipeline.py:
  from pipeline_with_rag_integration import create_hybrid_pipeline_with_rag
  pipeline = create_hybrid_pipeline_with_rag(use_rag=True)
  result = pipeline.process_with_rag("sua query")

Injeção em LLM:
  formatted = pipeline.format_for_llm(result)
  # formatted['system_prompt']
  # formatted['user_message']

Domínios pré-configurados:
  • filosofia (Aristóteles, Kant, Hegel, etc.)
  • lógica (proposições, predicados, silogismo)
  • epistemologia (justificação, crença, conhecimento)
  • geral (fallback)

Criar novo domínio:
  from rag_hybrid_context_injection import DomainContext
  domain = DomainContext(
    domain_name="meu_dominio",
    keywords=["palavra1", "palavra2"],
    system_prompt="Você é um especialista em..."
  )
  engine.register_domain(domain)

================================================================================
"""

if __name__ == "__main__":
    print(ARCHITECTURE_DIAGRAM)
    print("\n\n")
    print(QUICK_REFERENCE)
