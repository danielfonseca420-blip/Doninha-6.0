"""
RESUMO EXECUTIVO E GUIA DE SETUP
=================================
RAG Híbrido com Context Injection para L1-L2
"""

EXECUTIVE_SUMMARY = """

════════════════════════════════════════════════════════════════════════════════
                    RAG HÍBRIDO COM CONTEXT INJECTION
                   Resumo Executivo e Guia de Implementação
════════════════════════════════════════════════════════════════════════════════

O QUE FOI IMPLEMENTADO?
=======================

Um sistema completo de Retrieval-Augmented Generation (RAG) híbrido que integra:

✓ CONTEXT INJECTION (Injeção Direta de Contexto)
  • Carrega Knowledge Base especializado por domínio
  • Injeta diretamente no contexto do usuário
  • Determinístico, rápido, confiável

✓ SEMANTIC RETRIEVAL (Busca Semântica Seletiva)
  • Busca por similaridade em ChromaDB
  • Recupera documentos dinâmicos
  • Flexível, adaptável a queries não-vistas

✓ DOMAIN-AWARE INTEGRATION
  • Auto-detecta domínio da query
  • Seleciona KB especializado
  • System prompt customizado por domínio

✓ L1-L2 ENRICHMENT
  • Enriquece ConceptTable (L1) com KB injetado
  • Enriquece KantianJudgments (L2) com contexto
  • Refina classificação epistemológica

RESULTADO PRÁTICO:
==================

Query genérica → Resposta fundamentada, estruturada, com citações de KB

Exemplo:
  Input:  "O que é conhecimento?"
  Output: [Contexto enriquecido com L1-L2 + KB + System Prompt]
          → LLM gera: "Conhecimento justificado é... [fundamentado em KB]"

ARQUIVOS CRIADOS (5 arquivos principais):
===========================================

1. rag_hybrid_context_injection.py (500+ linhas)
   └─ Motor principal do RAG híbrido

2. l1_l2_rag_integration.py (600+ linhas)
   └─ Integração com L1-L2 + Pipeline

3. pipeline_with_rag_integration.py (400+ linhas)
   └─ Extensão compatível com pipeline.py existente

4. example_rag_hybrid_usage.py (500+ linhas)
   └─ 7 exemplos completos de uso

5. config_rag.yaml (150+ linhas)
   └─ Configuração centralizada

ARQUIVOS ADICIONAIS:
====================

• test_rag_hybrid.py (300+ linhas) — Suite de testes
• README_RAG_HIBRIDO.md — Documentação completa em português
• ARCHITECTURE_RAG_HYBRID.md — Diagrama e fluxo de dados
• SETUP_GUIDE.md — Este arquivo

════════════════════════════════════════════════════════════════════════════════
                            GUIA DE SETUP
════════════════════════════════════════════════════════════════════════════════

PASSO 1: Verificar Dependências
════════════════════════════════

Verificar o que já está instalado:
  pip list | grep -E "langchain|chroma|transformers|torch"

Versões recomendadas:
  • langchain >= 0.0.300
  • chromadb >= 0.3.21
  • transformers >= 4.30
  • torch >= 2.0 (ou CPU se sem GPU)
  • sentence-transformers >= 2.2


PASSO 2: Instalar Dependências Necessárias
═════════════════════════════════════════════

Execute:
  pip install chromadb
  pip install langchain
  pip install langchain-community
  pip install sentence-transformers
  pip install torch  # ou: pip install torch --index-url https://download.pytorch.org/whl/cpu

Versão simplificada (requirements.txt):
  Adicione ao seu requirements.txt:
  
  # RAG Híbrido
  chromadb>=0.3.21
  langchain>=0.0.300
  langchain-community>=0.0.50
  sentence-transformers>=2.2
  torch>=2.0


PASSO 3: Copiar Arquivos para o Projeto
═════════════════════════════════════════

Os seguintes arquivos já foram criados em: d:\Desktop\IA Doninha\

✓ rag_hybrid_context_injection.py
✓ l1_l2_rag_integration.py
✓ pipeline_with_rag_integration.py
✓ example_rag_hybrid_usage.py
✓ config_rag.yaml
✓ test_rag_hybrid.py

Nenhuma ação necessária (já estão no diretório).


PASSO 4: Preparar Knowledge Base
═════════════════════════════════

O sistema espera KBs em:
  data/kb_filosofia.json
  data/kb_logica.json
  data/kb_epistemologia.json
  data/kb.json (fallback)

Se não existirem, o sistema usa SEED_KNOWLEDGE_BASE (fallback automático).

Opcional: Criar seus próprios KBs

  Formato esperado:
  {
    "termo1": 0.9,
    "termo2": 0.8,
    ...
  }

  Exemplo para epistemologia (data/kb_epistemologia.json):
  {
    "justificação": 0.95,
    "crença": 0.92,
    "conhecimento": 0.94,
    "verdade": 0.93,
    "evidência": 0.90,
    "confiabilismo": 0.88,
    "internalismo": 0.87,
    "externalismo": 0.87
  }


PASSO 5: Preparar ChromaDB (Opcional)
═════════════════════════════════════

ChromaDB é OPCIONAL. Sistema funciona sem:
  • Se ChromaDB não existir: usa apenas Context Injection
  • Se existir: combina Injection + Retrieval (HYBRID)

Para usar ChromaDB:
  1. Crie estrutura de diretórios:
     chromadb/
     ├─ filosofia/
     ├─ lógica/
     ├─ epistemologia/
     └─ general/

  2. (Opcional) Popule com documentos via code:
     
     from langchain_community.vectorstores import Chroma
     from langchain_community.embeddings import HuggingFaceEmbeddings
     
     docs = [...] # seus documentos
     embeddings = HuggingFaceEmbeddings()
     vector_store = Chroma.from_documents(
         docs, embeddings, 
         persist_directory="chromadb/filosofia"
     )


PASSO 6: Testar Instalação
═══════════════════════════

Teste básico:
  python -c "from rag_hybrid_context_injection import HybridRAGContextInjectionEngine; print('✓ RAG OK')"

Teste L1-L2:
  python -c "from l1_l2_rag_integration import create_l1_l2_rag_pipeline; print('✓ L1-L2 OK')"

Testes completos:
  python test_rag_hybrid.py


════════════════════════════════════════════════════════════════════════════════
                          COMO USAR AGORA
════════════════════════════════════════════════════════════════════════════════

USO IMEDIATO - OPÇÃO 1: Exemplos Práticos
═══════════════════════════════════════════

Execute todos os exemplos:
  python example_rag_hybrid_usage.py

Isso vai executar:
  • Exemplo 1: RAG básico
  • Exemplo 2: L1-L2-RAG pipeline
  • Exemplo 3: Domain detection
  • Exemplo 4: Strategy comparison
  • Exemplo 5: LLM formatting
  • Exemplo 6: Custom domains
  • Exemplo 7: End-to-end


USO IMEDIATO - OPÇÃO 2: Modo Interativo (REPL)
═══════════════════════════════════════════════

Execute:
  python pipeline_with_rag_integration.py

Você pode:
  > Digite uma pergunta
  > O sistema processa com RAG e mostra resultados
  > Digite 'no-rag' para desabilitar RAG
  > Digite 'quit' para sair


USO IMEDIATO - OPÇÃO 3: Em Seu Código
═══════════════════════════════════════

  from l1_l2_rag_integration import create_l1_l2_rag_pipeline
  
  # Cria pipeline
  pipeline = create_l1_l2_rag_pipeline()
  
  # Processa query
  result = pipeline.process("Sua pergunta aqui")
  
  # Acessa resultados
  print(f"Domínio: {result['domain']}")
  print(f"Contexto compilado:\n{result['compiled_context']}")


USO COM LLM EXTERNO
════════════════════

Com OpenAI:
  import openai
  
  pipeline = create_l1_l2_rag_pipeline()
  result = pipeline.process("Sua pergunta")
  
  response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
          {"role": "system", "content": result["system_prompt"]},
          {"role": "user", "content": result["compiled_context"]}
      ]
  )
  print(response["choices"][0]["message"]["content"])


Com Claude (Anthropic):
  import anthropic
  
  client = anthropic.Anthropic(api_key="...")
  pipeline = create_l1_l2_rag_pipeline()
  result = pipeline.process("Sua pergunta")
  
  response = client.messages.create(
      model="claude-3-opus-20240229",
      max_tokens=1024,
      system=result["system_prompt"],
      messages=[{"role": "user", "content": result["compiled_context"]}]
  )
  print(response.content[0].text)


Com Ollama (local):
  import requests
  
  pipeline = create_l1_l2_rag_pipeline()
  result = pipeline.process("Sua pergunta")
  
  response = requests.post(
      'http://localhost:11434/api/chat',
      json={
          'model': 'mistral',
          'messages': [
              {'role': 'system', 'content': result['system_prompt']},
              {'role': 'user', 'content': result['compiled_context']}
          ],
          'stream': False
      }
  )
  print(response.json()['message']['content'])


════════════════════════════════════════════════════════════════════════════════
                      INTEGRAÇÃO COM PIPELINE.PY
════════════════════════════════════════════════════════════════════════════════

Se você quer integrar ao seu pipeline.py existente:

Opção A: Usar nova classe compatível (RECOMENDADO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  from pipeline_with_rag_integration import create_hybrid_pipeline_with_rag
  
  # Cria pipeline com RAG
  pipeline = create_hybrid_pipeline_with_rag(use_rag=True)
  
  # Processa com RAG (novo método)
  result = pipeline.process_with_rag("sua query")
  
  # Processa sem RAG (método original)
  result = pipeline.process_standard("sua query")


Opção B: Modificar pipeline.py existente (AVANÇADO)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

No topo do seu pipeline.py:
  
  try:
      from l1_l2_rag_integration import create_l1_l2_rag_pipeline
      HAS_RAG_HYBRID = True
  except ImportError:
      HAS_RAG_HYBRID = False

No __init__ da classe HybridLLMPipeline:
  
  def __init__(self, ..., use_rag_hybrid=True):
      # ... código original ...
      self.use_rag = use_rag_hybrid and HAS_RAG_HYBRID
      if self.use_rag:
          self.rag_pipeline = create_l1_l2_rag_pipeline(config=config)

Novo método:
  
  def process_with_rag(self, prompt):
      if self.use_rag:
          return self.rag_pipeline.process(prompt)
      else:
          return self.process(prompt)  # fallback original


════════════════════════════════════════════════════════════════════════════════
                        CHECKLIST DE SETUP
════════════════════════════════════════════════════════════════════════════════

□ Dependências instaladas (pip install chromadb langchain...)
□ Arquivos copiados para o projeto
□ Knowledge Base em data/kb_*.json (ou fallback automático)
□ ChromaDB preparado (opcional)
□ Testes executados com sucesso (python test_rag_hybrid.py)
□ Exemplos rodaram (python example_rag_hybrid_usage.py)
□ Modo interativo testado (python pipeline_with_rag_integration.py)
□ Integração com seu LLM configurada
□ Documentação lida (README_RAG_HIBRIDO.md)


════════════════════════════════════════════════════════════════════════════════
                    PRÓXIMOS PASSOS RECOMENDADOS
════════════════════════════════════════════════════════════════════════════════

CURTO PRAZO:
  1. Execute os exemplos para familiarizar-se
  2. Teste com suas próprias queries
  3. Ajuste weights e parâmetros conforme necessário

MÉDIO PRAZO:
  1. Crie domínios customizados para seus casos de uso
  2. Popule ChromaDB com seus documentos
  3. Integre com seu LLM preferido
  4. Monitore performance e qualidade

LONGO PRAZO:
  1. Refine system prompts por domínio
  2. Implemente feedback loop
  3. Adicione novos domínios
  4. Optimize retrieval strategy


════════════════════════════════════════════════════════════════════════════════
                          TROUBLESHOOTING
════════════════════════════════════════════════════════════════════════════════

Problema: "ModuleNotFoundError: No module named 'chromadb'"
Solução: pip install chromadb

Problema: "ChromaDB não carregado..."
Solução: Esperado se chromadb/ não existir. Sistema usa fallback.

Problema: "Domínio não detectado corretamente"
Solução: Edite keywords em config_rag.yaml para seu domínio

Problema: "Contexto muito longo (trunca respostas)"
Solução: Reduza max_documents_total ou context_truncation_length

Para mais detalhes, veja: TROUBLESHOOTING em README_RAG_HIBRIDO.md


════════════════════════════════════════════════════════════════════════════════
                          SUPORTE E RECURSOS
════════════════════════════════════════════════════════════════════════════════

Documentação:
  • README_RAG_HIBRIDO.md — Documentação completa
  • ARCHITECTURE_RAG_HYBRID.md — Arquitetura e fluxo
  • example_rag_hybrid_usage.py — 7 exemplos práticos

Código-fonte:
  • rag_hybrid_context_injection.py — Motor principal
  • l1_l2_rag_integration.py — Integração L1-L2
  • pipeline_with_rag_integration.py — Integração pipeline

Testes:
  • test_rag_hybrid.py — Suite de testes

════════════════════════════════════════════════════════════════════════════════

✓ Sistema pronto para usar!

Qualquer dúvida, consulte a documentação ou execute os exemplos.

════════════════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(EXECUTIVE_SUMMARY)
