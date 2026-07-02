"""
RAG HÍBRIDO COM CONTEXT INJECTION — DOCUMENTAÇÃO COMPLETA
===========================================================

Sistema de Retrieval-Augmented Generation (RAG) híbrido que trabalha em conjunto
com as camadas L1 (Conceitos) e L2 (Juízos Kantianos), usando:

  ✓ Context Injection (injeção direta de KB)
  ✓ Retrieval Seletivo (busca semântica em ChromaDB)
  ✓ Domain-Aware Knowledge Base (KB especializado por domínio)
  ✓ System Prompt Especializado (customizado por domínio)

================================================================================
ÍNDICE
================================================================================

1. VISÃO GERAL
2. ARQUIVOS CRIADOS
3. COMO USAR
4. EXEMPLOS DE CÓDIGO
5. INTEGRAÇÃO COM PIPELINE EXISTENTE
6. CONFIGURAÇÃO
7. API DE REFERÊNCIA
8. TROUBLESHOOTING

================================================================================
1. VISÃO GERAL
================================================================================

O sistema RAG Híbrido combina duas estratégias:

A. CONTEXT INJECTION (Injeção Direta)
   - Carrega Knowledge Base pré-compilado do domínio
   - Injeta contexto diretamente no prompt
   - Peso configurável (padrão: 70-80% da confiança)
   
   Benefício: Contexto garantido, rápido, determinístico

B. SEMANTIC RETRIEVAL (Retrieval Seletivo)
   - Busca por similaridade em ChromaDB
   - Recupera documentos relevantes dinamicamente
   - Peso configurável (padrão: 20-30% da confiança)
   
   Benefício: Flexibilidade, adaptação a queries não-vistas

INTEGRAÇÃO COM L1-L2:
   - L1 (ConceptTable): Conceitos enriquecidos com KB injetado
   - L2 (KantianJudgments): Juízos refinados com contexto especializado
   - System Prompt: Customizado por domínio detectado

================================================================================
2. ARQUIVOS CRIADOS
================================================================================

a) CORE DO RAG HÍBRIDO:
   └─ rag_hybrid_context_injection.py
      - HybridRAGContextInjectionEngine (motor principal)
      - DomainContext (definição de domínios)
      - RetrievalStrategy (estratégias de retrieval)
      - RAGContext (contexto compilado)

b) INTEGRAÇÃO L1-L2:
   └─ l1_l2_rag_integration.py
      - L1RAGEnricher (enriquece conceitos)
      - L2RAGEnricher (enriquece juízos)
      - IntegratedL1L2RAGPipeline (pipeline completo)

c) PIPELINE INTEGRADO:
   └─ pipeline_with_rag_integration.py
      - HybridLLMPipelineWithRAG (estende pipeline.py original)
      - Modo interativo (REPL para testes)

d) EXEMPLOS:
   └─ example_rag_hybrid_usage.py
      - 7 exemplos práticos completos
      - Testes de cada componente
      - Demonstração de injeção em LLM

e) CONFIGURAÇÃO:
   └─ config_rag.yaml
      - Configuração de todos os parâmetros
      - Domínios pré-definidos
      - System prompts especializados

================================================================================
3. COMO USAR
================================================================================

OPÇÃO A: RAG Simples (sem L1-L2)
─────────────────────────────────────

  from rag_hybrid_context_injection import HybridRAGContextInjectionEngine
  
  # Cria motor
  engine = HybridRAGContextInjectionEngine(verbose=True)
  
  # Processa query
  rag_context = engine.process(
      query="O que é conhecimento?",
      auto_detect_domain=True,
      strategy="hybrid"  # direct_injection | semantic_retrieval | hybrid
  )
  
  # Acessa contexto compilado
  print(rag_context.compiled_context)  # Pronto para injetar em LLM


OPÇÃO B: RAG + L1-L2 Enriquecidos (recomendado)
────────────────────────────────────────────────

  from l1_l2_rag_integration import create_l1_l2_rag_pipeline
  
  # Cria pipeline
  pipeline = create_l1_l2_rag_pipeline()
  
  # Processa query
  result = pipeline.process("O que é conhecimento?")
  
  # Acessa saídas estruturadas
  print(f"Domínio: {result['domain']}")
  print(f"L1 Conceitos: {len(result['l1_output'].concepts)}")
  print(f"L2 Juízos: {len(result['l2_output'].judgments)}")
  print(f"Contexto: {result['compiled_context']}")


OPÇÃO C: Integração com Pipeline Existente
─────────────────────────────────────────────

  from pipeline_with_rag_integration import create_hybrid_pipeline_with_rag
  
  # Cria pipeline estendido
  pipeline = create_hybrid_pipeline_with_rag(use_rag=True)
  
  # Processa com RAG (novo método)
  result = pipeline.process_with_rag("Sua pergunta aqui")
  
  # Formata para LLM
  formatted = pipeline.format_for_llm(result)
  
  # Envia para OpenAI / Claude / outro LLM
  # formatted['system_prompt']
  # formatted['user_message']


OPÇÃO D: Modo Interativo (REPL)
──────────────────────────────────

  python pipeline_with_rag_integration.py
  
  > O que é verdade?
  ✓ Domínio: filosofia
  ✓ Confiança: 85%
  ...
  
  > no-rag       # Alterna para modo sem RAG
  > quit         # Sai


================================================================================
4. EXEMPLOS DE CÓDIGO
================================================================================

EXEMPLO 1: Detecção de Domínio
──────────────────────────────

  from rag_hybrid_context_injection import HybridRAGContextInjectionEngine
  
  engine = HybridRAGContextInjectionEngine()
  
  queries = [
      "Aristóteles define a substância como categoria fundamental",
      "Na lógica paraconsistente, é possível ter P e ¬P simultaneamente?",
      "Como a justificação interna diferencia-se da justificação externa?"
  ]
  
  for query in queries:
      domain, confidence = engine.detect_domain(query)
      print(f"Query: {query[:50]}...")
      print(f"  → Domínio: {domain} ({confidence:.0%})\n")
  
  # Output:
  # Query: Aristóteles define a substância como...
  #   → Domínio: filosofia (92%)
  # 
  # Query: Na lógica paraconsistente, é possível...
  #   → Domínio: lógica (88%)
  # 
  # Query: Como a justificação interna diferencia-se...
  #   → Domínio: epistemologia (90%)


EXEMPLO 2: Injeção de Contexto em LLM (OpenAI)
────────────────────────────────────────────────

  from l1_l2_rag_integration import create_l1_l2_rag_pipeline
  from openai import OpenAI
  
  # Processa com RAG
  pipeline = create_l1_l2_rag_pipeline()
  result = pipeline.process("O que é conhecimento justificado?")
  
  # Prepara para OpenAI
  client = OpenAI(api_key="sk-...")
  
  response = client.chat.completions.create(
      model="gpt-4",
      messages=[
          {
              "role": "system",
              "content": result["system_prompt"]
          },
          {
              "role": "user",
              "content": result["compiled_context"]
          }
      ],
      temperature=0.3,
      max_tokens=1024
  )
  
  print(response.choices[0].message.content)


EXEMPLO 3: Uso com LLMs Locais (Ollama)
───────────────────────────────────────

  from l1_l2_rag_integration import create_l1_l2_rag_pipeline
  import requests
  import json
  
  # Processa com RAG
  pipeline = create_l1_l2_rag_pipeline()
  result = pipeline.process("Explique a dialética hegeliana")
  
  # Envia para Ollama
  response = requests.post(
      'http://localhost:11434/api/chat',
      json={
          'model': 'mistral',  # ou outro modelo disponível
          'messages': [
              {'role': 'system', 'content': result['system_prompt']},
              {'role': 'user', 'content': result['compiled_context']}
          ],
          'stream': False
      }
  )
  
  answer = response.json()['message']['content']
  print(answer)


EXEMPLO 4: Comparação de Estratégias
──────────────────────────────────────

  from rag_hybrid_context_injection import HybridRAGContextInjectionEngine, RetrievalStrategy
  
  engine = HybridRAGContextInjectionEngine(verbose=False)
  query = "O que é uma proposição?"
  
  strategies = [
      RetrievalStrategy.DIRECT_INJECTION,
      RetrievalStrategy.SEMANTIC_RETRIEVAL,
      RetrievalStrategy.HYBRID,
  ]
  
  for strategy in strategies:
      rag_ctx = engine.process(query, strategy=strategy)
      
      injected = sum(1 for d in rag_ctx.retrieved_documents if d.is_injected)
      retrieved = len(rag_ctx.retrieved_documents) - injected
      
      print(f"Estratégia: {strategy.value}")
      print(f"  Injetados: {injected}")
      print(f"  Recuperados: {retrieved}")
      print(f"  Confiança: {rag_ctx.confidence_score:.2%}\n")


EXEMPLO 5: Domínio Customizado
───────────────────────────────

  from rag_hybrid_context_injection import HybridRAGContextInjectionEngine, DomainContext
  
  engine = HybridRAGContextInjectionEngine()
  
  # Define novo domínio
  domain_direito = DomainContext(
      domain_name="direito",
      description="Direito civil, constitucional e penal",
      keywords=["lei", "código", "artigo", "direito", "obrigação", "contrato"],
      kb_path="data/kb_direito.json",
      system_prompt="""Você é um especialista em direito com rigorosa base legal.
      Cite sempre artigos, precedentes e legislação pertinente.""",
      injection_weight=0.85,
      retrieval_weight=0.15
  )
  
  # Registra
  engine.register_domain(domain_direito)
  
  # Usa
  result = engine.process("Qual é o prazo para prescrição de débitos fiscais?")
  print(result.domain)  # 'direito'


================================================================================
5. INTEGRAÇÃO COM PIPELINE EXISTENTE
================================================================================

Para integrar ao pipeline.py EXISTENTE, faça:

PASSO 1: Adicione as importações ao topo do pipeline.py
──────────────────────────────────────────────────────

  try:
      from l1_l2_rag_integration import create_l1_l2_rag_pipeline
      HAS_RAG_HYBRID = True
  except ImportError:
      HAS_RAG_HYBRID = False


PASSO 2: Modifique o __init__ da classe HybridLLMPipeline
───────────────────────────────────────────────────────

  def __init__(self, knowledge_base=None, config=None, verbose=True, use_rag=True):
      # ... código original ...
      
      self.use_rag = use_rag and HAS_RAG_HYBRID
      if self.use_rag:
          self.rag_l1_l2_pipeline = create_l1_l2_rag_pipeline(config=config)


PASSO 3: Crie novo método process_with_rag
──────────────────────────────────────────

  def process_with_rag(self, prompt: str):
      """Processa prompt com RAG híbrido."""
      if not self.use_rag:
          return self.process(prompt)  # Fallback para original
      
      # Processa com RAG
      rag_result = self.rag_l1_l2_pipeline.process(prompt)
      
      # Integra com pipeline original
      self.kb = {**self.kb, **rag_result['l2_output'].domain_specialized_kb}
      
      return rag_result


PASSO 4: Use no código
──────────────────────

  pipeline = HybridLLMPipeline(use_rag=True)
  result = pipeline.process_with_rag("Sua pergunta")


================================================================================
6. CONFIGURAÇÃO
================================================================================

O arquivo config_rag.yaml define:

RAG_HYBRID → Engine
  - embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  - chroma_path: "chromadb"
  - default_strategy: "hybrid"

RAG_HYBRID → Domínios
  - filosofia
    - keywords: [conhecimento, verdade, ...]
    - kb_path: data/kb_filosofia.json
    - system_prompt: "Você é um especialista rigoroso..."
  
  - lógica
  - epistemologia
  - geral (default)

RAG_HYBRID → Retrieval
  - max_documents_total: 8
  - min_relevance_score: 0.3
  - context_truncation_length: 500

RAG_HYBRID → L1-L2 Integration
  - enabled: true
  - auto_detect_domain: true
  - enrich_concepts: true
  - enrich_judgments: true


Para CUSTOMIZAR:

1. Crie seu próprio config_meu_rag.yaml
2. Carregue em seu código:
   
   from config_loader import load_config
   config = load_config("config_meu_rag.yaml")
   engine = HybridRAGContextInjectionEngine(config=config)


================================================================================
7. API DE REFERÊNCIA
================================================================================

HybridRAGContextInjectionEngine
───────────────────────────────

  Atributos:
    - config: Dict[str, Any]
    - embedding_model: str
    - chroma_path: Path
    - domains: Dict[str, DomainContext]
    - chroma_stores: Dict[str, Any]
    - verbose: bool

  Métodos:
    - __init__(config, embedding_model, chroma_path, verbose)
    - process(query, concepts, strategy, k, auto_detect_domain) → RAGContext
    - detect_domain(query, concepts) → Tuple[str, float]
    - retrieve_documents(query, domain, k, strategy) → List[RetrievedDocument]
    - get_injected_knowledge(domain, query) → Dict[str, float]
    - compile_context(...) → RAGContext
    - register_domain(domain: DomainContext) → None
    - format_for_l1_l2(rag_context) → Dict[str, Any]


IntegratedL1L2RAGPipeline
──────────────────────────

  Métodos:
    - process(query, auto_detect_domain) → Dict[str, Any]
    
  Retorna:
    {
      'query': str,
      'domain': str,
      'l1_output': EnrichedL1Output,
      'l2_output': EnrichedL2Output,
      'compiled_context': str,
      'system_prompt': str,
      'confidence': float,
    }


RAGContext (Dataclass)
─────────────────────

  Atributos:
    - query: str
    - domain: str
    - retrieved_documents: List[RetrievedDocument]
    - injected_knowledge: Dict[str, float]
    - compiled_context: str
    - strategy: RetrievalStrategy
    - confidence_score: float


DomainContext (Dataclass)
──────────────────────────

  Atributos:
    - domain_name: str
    - description: str
    - keywords: List[str]
    - kb_path: str
    - chroma_collection: str
    - system_prompt: str
    - injection_weight: float
    - retrieval_weight: float
    - max_injected_docs: int
    - max_retrieved_docs: int


================================================================================
8. TROUBLESHOOTING
================================================================================

PROBLEMA: ChromaDB não carregado
─────────────────────────────────

  Solução:
  1. Verifique se chromadb está instalado: pip install chromadb
  2. Verifique se o diretório chromadb/{domínio} existe
  3. Use strategy=RetrievalStrategy.DIRECT_INJECTION para testar sem ChromaDB


PROBLEMA: Domínio não detectado corretamente
──────────────────────────────────────────────

  Solução:
  1. Adicione keywords mais específicas ao domínio
  2. Use detect_domain(query, concepts=None) com conceitos
  3. Especifique domain explicitamente se necessário


PROBLEMA: Contexto muito longo (trunca respostas do LLM)
──────────────────────────────────────────────────────────

  Solução:
  1. Reduza max_documents_total em config_rag.yaml
  2. Reduza context_truncation_length para cada documento
  3. Use RetrievalStrategy.DIRECT_INJECTION com max_injected_docs reduzido


PROBLEMA: Confiança muito baixa
───────────────────────────────

  Solução:
  1. Ajuste injection_weight e retrieval_weight no domínio
  2. Aumente min_relevance_score se documentos ruins forem incluídos
  3. Enriqueça o KB do domínio com mais termos


PROBLEMA: Importação de módulos falhando
──────────────────────────────────────────

  Solução:
  1. Verifique se todos os arquivos estão no diretório correto
  2. Execute: pip install -r requirements.txt (se houver)
  3. Adicione print de debug nas importações para localizar erro


================================================================================
SUPORTE E PRÓXIMOS PASSOS
================================================================================

1. Execute os exemplos: python example_rag_hybrid_usage.py

2. Teste o modo interativo: python pipeline_with_rag_integration.py

3. Crie seus próprios domínios customizados

4. Integre com seu LLM preferido (OpenAI, Claude, Ollama, etc)

5. Monitore métricas: confidence_score, domain_accuracy, retrieval_quality

================================================================================
"""

# Se executado como script, mostra esta documentação
if __name__ == "__main__":
    print(__doc__)
