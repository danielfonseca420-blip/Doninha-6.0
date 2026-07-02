# Doninha IA - Hybrid Neuro-Symbolic Middleware

> Doninha IA transforms a base LLM into a neuro-symbolic reasoning middleware with epistemological layers, paraconsistent logic, hybrid RAG, verification, auditing, and final synthesis.

Doninha IA was created by Daniel Barros Fonseca. Use is permitted for private users only. For commercial or governmental use, contact the developer at `danielfonseca420@proton.me` for information on licensed use subject to royalty payments.

## Overview

Doninha is not simply an LLM caller. It is a layered middleware that receives a prompt, transforms that prompt into concepts, propositions, logical weights, and verified syntheses, and only then delivers a textual response.

The project's goal is to combine:

- statistical language generation;
- symbolic structure inspired by Aristotle, Kant, Russell, Popper, Hempel, and paraconsistent logic;
- context retrieval via local KB and RAG;
- auditing of confidence, contradiction, truth-value, and sources;
- auditable Chain of Thought per layer, exposed as a structured summary rather than raw internal reasoning.

## Main Flow L1–L7

```text
User prompt
  |
  v
L1 - Concept Table
  |
  v
L2 - Kantian Judgments and epistemic classification
  |
  v
Scientific syllogism + Hempel + Popper
  |
  v
L3 - Paraconsistent Logic + Weighted Dynamic Ensemble
  |
  v
L4 - Russellian Synthesis + Chain of Verification
  |
  v
L5 - Text generation with base provider
  |
  v
L6 - Final refinement
  |
  v
L7 - Definitive and auditable text
```

The main entry point is [pipeline.py](pipeline.py), through the `HybridLLMPipeline` class.

## Middleware Layers

### L1 - Concept Table

Main file: [l1_concept_table.py](l1_concept_table.py)

Main tools:

- `ConceptNode`: concept structure with definition, synonyms, antonyms, hyponyms, hyperonyms, domain, application context, and canonical source.
- `ConceptTable`: semantic table that extracts known concepts from the prompt.
- `SpacySemanticTermExtractor`: optional spaCy-based extractor for terms and phrases.
- `LogicLMSymbolicSolver`: symbolic enricher that evaluates context compatibility and generates canonical warnings.

Layer function:

- extract concepts from the prompt;
- identify semantic relationships;
- infer domain;
- enrich concepts with application context;
- prevent incompatible use of canonical concepts;
- prepare semantic material for L2.

### L2 - Kantian Judgments

Main file: [l2_kantian_judgments.py](l2_kantian_judgments.py)

Main tools:

- `KantianJudgment`: represents a refined proposition with quantity, quality, relation, modality, priority, and epistemic classification.
- `SyntaxProfile`: minimal syntactic profile of the prompt.
- `BERTAssertionClassifier`: optional classifier based on `transformers.pipeline` for T/I/F.
- `KantianJudgmentEngine`: engine that transforms concepts into prioritised judgments.

Layer function:

- convert concepts into propositions;
- classify propositions by the Kantian table of judgments;
- generate affirmative, negative, hypothetical, and apodictic variations;
- assign epistemological priority;
- supply hypotheses to the scientific filters and to L3.

### Intermediate Step - Syllogism, Hempel, and Popper

Main file: [syllogism_module.py](syllogism_module.py)

Main tools:

- `Syllogism`: premises-and-conclusion structure.
- `AristotelianSyllogismValidator`: validates syllogistic relations.
- `HempelFilter`: removes spurious hypotheses due to low semantic relevance.
- `PopperFalsifiability`: evaluates falsifiability.
- `ScientificSyllogismPipeline`: integrates the three routines before L3.

Step function:

- filter overly loose hypotheses;
- preserve more testable propositions;
- reduce noise before the paraconsistent computation.

### L3 - Paraconsistent Logic

Main files:

- [l3_paraconsistent.py](l3_paraconsistent.py)
- [paraconsistent_rules.py](paraconsistent_rules.py)
- [neural_truth_model.py](neural_truth_model.py)

Main tools:

- `ParaconsistentValue`: logical value with `mu`, `lambda`, certainty, contradiction, state, truth-value, and ensemble metadata.
- `ManyValuedRouter`: routes proposition pairs to real contradiction, statistical uncertainty, ambiguity, or unclassified.
- `ParaconsistentEngine`: computes fuzzy annotations and logical states.
- `ParaconsistentRules`: 12-state rules derived from `data/Fuzzy.txt`.
- `TruthScoringModel`: Transformer-based neural model for paraconsistent state and truth-value.
- `neural_annotations`: converts neural output to `mu/lambda`.

Layer function:

- compute `mu` as supporting evidence;
- compute `lambda` as opposing evidence;
- derive certainty `Gc = mu - lambda`;
- derive contradiction `Gct = mu + lambda - 1`;
- classify the logical state;
- combine heuristic and neural model via Weighted Dynamic Ensemble.

### Weighted Dynamic Ensemble in L3

L3 combines two sources:

- heuristic based on KB, L2 priority, and local contradictions;
- neural annotation from `TruthScoringModel`, when available.

The flow is:

1. compute `h_mu/h_lam` from the heuristic;
2. compute `n_mu/n_lam` from the neural model;
3. measure fuzzy agreement between the two sources;
4. set dynamic weights:
   - `heuristic_weight = 0.65 + 0.25 * agreement`
   - `neural_weight = 1.0 - heuristic_weight`
5. combine `mu/lambda`;
6. apply paraconsistent regularisation to `Gc/Gct`;
7. record `confidence`, `ensemble_agreement`, `neural_state`, and `neural_truth`.

When the neural model is unavailable, the layer continues operating in heuristic mode, with `heuristic_weight=1.0` and `neural_weight=0.0`.

### L4 - Russellian Synthesis + CoVe

Main files:

- [l4_synthesis.py](l4_synthesis.py)
- [l4_russell_equivalence.py](l4_russell_equivalence.py)
- [l4_chain_verification.py](l4_chain_verification.py)

Main tools:

- `SynthesisResult`: structured synthesis result.
- `RussellianSynthesisEngine`: combines L3 propositions with L2 priorities and KB.
- `RussellConceptBase`: conceptual base extracted from `data/russell.txt`.
- `score_proposition_by_concepts`: computes proposition-to-fact conceptual correspondence.
- `ChainOfVerificationAgent`: applies Chain of Verification to the result.

Layer function:

- select and weight the best hypothesis;
- compute synthesis based on truth-value, priority, and correspondence;
- verify the response via CoVe;
- produce a structured response before text generation.

### L5 - Text Generation

Main file: [l5_generation.py](l5_generation.py)

Main tools:

- `build_context_for_generation`: assembles the L1–L4 context for the provider.
- `generate_with_ollama_l5`: generates text with Ollama.
- `generate_with_custom_lm`: uses the local custom model.
- `generate_response`: routes generation to Ollama, remote provider, custom LM, or fallback template.

Layer function:

- transform the L4 synthesis into natural language;
- preserve epistemological context;
- use the configured provider without losing the local fallback.

### L6 - Final Refinement

Main file: [l6_final_response.py](l6_final_response.py)

Main tools:

- `EpistemicContext`: aggregated context with L3 states, paraconsistent routes, and BERT classifications.
- `FinalResponseEngine`: generates and rewrites the fluent response.

Layer function:

- improve clarity and cohesion;
- adjust tone to the confidence level;
- mention uncertainties and contradictions when necessary;
- preserve the epistemic data.

### L7 - Definitive Final Text

Main files:

- [l7_final_text.py](l7_final_text.py)
- [agente_sintese_final.py](agente_sintese_final.py)

Main tools:

- `FinalTextEngine`: builds the final prompt and calls provider/template.
- `synthesize_final_text`: reusable final synthesis agent via CLI or pipeline.

Layer function:

- integrate L1–L6 into a final text;
- classify audience (`layperson`, `technical`, `academic`);
- adjust tone according to confidence, contradiction, and state;
- use the auditable CoT trace as synthesis context.

## Auditable Chain of Thought

Main files:

- [cot_hierarchical.py](cot_hierarchical.py)
- [prompt_engineering.py](prompt_engineering.py)

Main tools:

- `CoTStep`: records layer, title, reasoning summary, key decisions, output, and duration.
- `HierarchicalCoTTrace`: aggregates steps L1–L7 and exports as `dict` or Markdown.
- `HierarchicalCoTOrchestrator`: wrapper to run the pipeline with trace return.
- `get_layer_prompt`: generates layer-specific prompts for L1–L7.

Function:

- generate a per-layer audit trail;
- record decisions and summaries without exposing raw internal reasoning;
- enable `return_cot=True` in the pipeline.

Usage:

```python
from pipeline import HybridLLMPipeline

pipeline = HybridLLMPipeline(verbose=False)
result = pipeline.process("What is knowledge?", return_cot=True)

print(result.response)
print(result.cot_markdown)
```

## LLM Providers

Main file: [llm_provider_client.py](llm_provider_client.py)

Supported providers:

- `ollama`
- `openai`
- `anthropic`
- `gemini`
- `grok`
- `groq`
- `meta`
- `template`
- `custom_lm`

Functions:

- normalise providers;
- set default models;
- generate text locally with Ollama;
- call remote APIs via `requests`;
- extract text from OpenAI-like, Anthropic, and Gemini formats;
- provide fallback when no provider is configured.

## RAG, KB, and Search

### Knowledge Base

Main file: [knowledge_base.py](knowledge_base.py)

Tools:

- `SEED_KNOWLEDGE_BASE`: minimal fallback base.
- `load_kb_from_file`: loads JSON, JSONL/NDJSON, or documents.
- `merge_kb`: merges bases.
- `enrich_kb_from_chroma`: enriches KB via ChromaDB.
- `get_domain_knowledge_base`: retrieves KB by domain.
- `get_knowledge_base`: general KB entry point.

Function:

- supply terms and weights to L1, L3, and L4;
- allow generic, domain-specific, and RAG-enriched KB.

### Hybrid RAG

Main files:

- [rag_hybrid_context_injection.py](rag_hybrid_context_injection.py)
- [l1_l2_rag_integration.py](l1_l2_rag_integration.py)
- [pipeline_with_rag_integration.py](pipeline_with_rag_integration.py)

Tools:

- `RetrievalStrategy`: `direct_injection`, `semantic_retrieval`, `hybrid`, `domain_aware`.
- `DomainContext`: configures domain, KB, Chroma, prompt, and weights.
- `RetrievedDocument`: retrieved document.
- `RAGContext`: compiled context.
- `HybridRAGContextInjectionEngine`: retrieval + injection engine.
- `IntegratedL1L2RAGPipeline`: integrates RAG with L1 and L2.
- `HybridLLMPipelineWithRAG`: alternative pipeline with embedded RAG.

Function:

- detect domain;
- inject context directly from KB;
- retrieve documents from ChromaDB;
- compile context for L1/L2 and the final response.

### Search Agent

Main file: [agente_busca_web.py](agente_busca_web.py)

Tools:

- `get_retriever_tool`: local ChromaDB search tool.
- `get_duckduckgo_tool`: DuckDuckGo tool when installed.
- `build_agent`: builds a ReAct agent via LangChain.
- `run_search_for_context`: returns textual context for the pipeline.

Function:

- search local and/or web context;
- enrich responses when L3 indicates uncertainty, indetermination, or heuristic fallback.

## Interfaces

### CLI

Main file: [pipeline.py](pipeline.py)

Commands:

```bash
python pipeline.py --demo
python pipeline.py --prompt "Explain paraconsistent logic in 5 lines"
python pipeline.py --repl
```

### REST API

Main file: [api.py](api.py)

Technologies:

- FastAPI
- Pydantic
- Uvicorn

Endpoints:

- `GET /health`: API status.
- `POST /process`: processes a single prompt.
- `POST /chat`: processes a message with session.
- `POST /agent`: runs only the search agent.

### Chainlit

Main file: [app.py](app.py)

Technologies:

- Chainlit
- Ollama

Function:

- simple chat interface;
- streaming via `ollama.chat`;
- basic in-memory history.

### Standalone

Main file: [doninha_standalone.py](doninha_standalone.py)

Function:

- consolidated file with most modules embedded;
- useful for distribution or execution without separating multiple files;
- includes support for the CoT trace L1–L7 and the Weighted Dynamic Ensemble.

## Training and Models

### TruthScoringModel

Files:

- [neural_truth_model.py](neural_truth_model.py)
- [train_truth_model.py](train_truth_model.py)

Tools:

- PyTorch
- Transformers (`AutoModel`, `AutoTokenizer`)
- Custom proposition dataset
- Rules from `data/Fuzzy.txt`

Function:

- train a classifier/regressor for paraconsistent state and truth-value;
- generate `truth_scoring_model.pt`;
- feed the neural L3.

### Custom Language Model

Files:

- [custom_tokenizer.py](custom_tokenizer.py)
- [custom_lm_model.py](custom_lm_model.py)
- [pretrain_custom_lm.py](pretrain_custom_lm.py)
- [run_pretrain.py](run_pretrain.py)

Tools:

- SentencePiece
- PyTorch
- Custom Transformer
- Local philosophical corpus

Function:

- train a custom tokenizer;
- train a small local LM;
- enable the `custom_lm` provider.

### Russellian Base

Files:

- [l4_russell_equivalence.py](l4_russell_equivalence.py)
- [train_l4_russell.py](train_l4_russell.py)

Function:

- extract equivalence/correspondence concepts from `data/russell.txt`;
- generate `l4_russell_concepts.json`;
- weight L4 by conceptual correspondence.

## Metrics and Evaluation

Main files:

- [metrics.py](metrics.py)
- [eval_pipeline.py](eval_pipeline.py)
- [test_epistemic_classification.py](test_epistemic_classification.py)
- [test_l7_tone_guidance.py](test_l7_tone_guidance.py)
- [test_provider_config_resolution.py](test_provider_config_resolution.py)
- [test_citation_behavior.py](test_citation_behavior.py)
- [test_rag_hybrid.py](test_rag_hybrid.py)

Tools:

- `coherence_l3`: measures coherence between truth, state, and contradiction.
- Simple BLEU.
- Simple ROUGE-L.
- Semantic similarity via SentenceTransformers.
- Test suites for L2, L7, providers, citation, and RAG.

## Configuration

Files:

- [config.yaml](config.yaml)
- [config_loader.py](config_loader.py)
- [config_rag.yaml](config_rag.yaml)

`config_loader.py` resolves:

- relative paths;
- providers;
- models;
- KB and Chroma;
- agent configuration;
- API;
- chat;
- L1 spaCy.

Example:

```yaml
generation:
  provider: "ollama"
  ollama_model: "doninha8:latest"
  ollama_host: "http://localhost:11434"

finalization:
  provider: "ollama"
  ollama_model: "doninha8:latest"

l7:
  provider: "ollama"
  model: "doninha8:latest"
```

Useful variables:

- `OLLAMA_MODEL`
- `OLLAMA_HOST`
- `GENERATION_PROVIDER`
- `FINALIZATION_PROVIDER`
- `L7_PROVIDER`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `GROK_API_KEY`
- `GROQ_API_KEY`
- `META_API_KEY`
- `VECTOR_DB_PATH`

## Doninha IA Middleware Tools

### Internal reasoning tools

| Tool | File | Function |
|---|---|---|
| `ConceptTable` | `l1_concept_table.py` | Extracts concepts and semantic relations. |
| `LogicLMSymbolicSolver` | `l1_concept_table.py` | Enriches concepts with context and validates canonical compatibility. |
| `KantianJudgmentEngine` | `l2_kantian_judgments.py` | Generates prioritised Kantian judgments. |
| `BERTAssertionClassifier` | `l2_kantian_judgments.py` | Classifies assertoric propositions as true, indeterminate, or false. |
| `ScientificSyllogismPipeline` | `syllogism_module.py` | Applies syllogism, Hempel filter, and Popperian falsifiability. |
| `ParaconsistentEngine` | `l3_paraconsistent.py` | Computes `mu/lambda`, state, truth, certainty, and contradiction. |
| `ManyValuedRouter` | `l3_paraconsistent.py` | Classifies proposition pairs as contradiction, uncertainty, or ambiguity. |
| `ParaconsistentRules` | `paraconsistent_rules.py` | Implements 12-state fuzzy rules. |
| `TruthScoringModel` | `neural_truth_model.py` | Neural model for paraconsistent scoring. |
| `RussellianSynthesisEngine` | `l4_synthesis.py` | Synthesises propositions by equivalence/correspondence. |
| `ChainOfVerificationAgent` | `l4_chain_verification.py` | Verifies and revises the response via CoVe. |
| `FinalResponseEngine` | `l6_final_response.py` | Refines the final response and builds the epistemic context. |
| `FinalTextEngine` | `l7_final_text.py` | Produces the definitive final text. |
| `HierarchicalCoTTrace` | `cot_hierarchical.py` | Records the auditable Chain of Thought for L1–L7. |
| `get_layer_prompt` | `prompt_engineering.py` | Generates layer-specific prompts. |

### Context, search, and RAG tools

| Tool | File | Function |
|---|---|---|
| `get_knowledge_base` | `knowledge_base.py` | Loads general or domain-specific KB. |
| `enrich_kb_from_chroma` | `knowledge_base.py` | Enriches KB with ChromaDB excerpts. |
| `HybridRAGContextInjectionEngine` | `rag_hybrid_context_injection.py` | Combines direct injection and semantic retrieval. |
| `IntegratedL1L2RAGPipeline` | `l1_l2_rag_integration.py` | Uses RAG to enrich L1/L2. |
| `run_search_for_context` | `agente_busca_web.py` | Runs local/web search and returns context. |
| `DuckDuckGoSearchRun` | `agente_busca_web.py` | Web search via DuckDuckGo when available. |
| `Chroma` | various | Local vector base for retrieval. |
| `HuggingFaceEmbeddings` | various | Embeddings for ChromaDB. |

### Generation tools

| Tool | File | Function |
|---|---|---|
| `generate_text` | `llm_provider_client.py` | Routes calls to local/remote providers. |
| Ollama | `llm_provider_client.py`, `l5_generation.py`, `l7_final_text.py` | Default local provider. |
| OpenAI | `llm_provider_client.py` | OpenAI-compatible remote provider. |
| Anthropic | `llm_provider_client.py` | Remote Claude provider. |
| Gemini | `llm_provider_client.py` | Remote Google Gemini provider. |
| Grok | `llm_provider_client.py` | Remote xAI provider. |
| Groq | `llm_provider_client.py` | Groq OpenAI-compatible remote provider. |
| Meta/Llama API | `llm_provider_client.py` | Remote provider for Llama models. |
| `custom_lm` | `custom_lm_model.py` | Trainable local custom model. |
| `template` | various | Fallback with no external call. |

### Training and data tools

| Tool | File | Function |
|---|---|---|
| `train_truth_model.py` | L3 training | Trains the `TruthScoringModel`. |
| `train_l4_russell.py` | L4 training | Generates the Russellian concept base. |
| `pretrain_custom_lm.py` | Custom LM | Trains a custom language model. |
| `custom_tokenizer.py` | tokenisation | Trains/loads SentencePiece. |
| `corpus_utils.py` | corpus | Reads and prepares corpus. |
| `build_concepts_from_english_dict.py` | concepts | Extracts concepts from an English dictionary. |
| `philosophy-corpus/encode_corpus.py` | corpus | Encodes corpus for training. |

### Interface and operation tools

| Tool | File | Function |
|---|---|---|
| `HybridLLMPipeline` | `pipeline.py` | Orchestrates L1–L7. |
| `HybridLLMPipelineWithRAG` | `pipeline_with_rag_integration.py` | Variant with embedded hybrid RAG. |
| FastAPI | `api.py` | HTTP API. |
| Chainlit | `app.py` | Local chat interface. |
| `ChatSession` | `chat_session.py` | Conversation history. |
| `doninha_standalone.py` | standalone | Consolidated distribution. |
| `consolidate_*` scripts | various | Generate consolidated files. |

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start Ollama:

```bash
ollama serve
ollama pull doninha8:latest
```

Run the pipeline:

```bash
python pipeline.py --prompt "Explain paraconsistent logic in 5 lines"
python pipeline.py --demo
python pipeline.py --repl
```

Run the API:

```bash
python api.py
```

Run Chainlit:

```bash
chainlit run app.py
```

Run hybrid RAG:

```bash
python pipeline_with_rag_integration.py
python example_rag_hybrid_usage.py
```

Train neural L3:

```bash
python train_truth_model.py
```

Train the L4 Russell base:

```bash
python train_l4_russell.py
```

Train the custom LM:

```bash
python run_pretrain.py
```

## How to run on Openweb-ui

Install open-webui normally. Select the model and set this configurations:

- Think: On
- Thinking level: High
- Max tokens: set it up to maximum possible

## Output Auditing

The pipeline appends audit blocks such as:

- `[AUDIT L4]`
- `[AUDIT L5]`
- `[AUDIT L6]`
- `[AUDIT L7]`

These may include:

- provider used;
- model used;
- truth-value;
- certainty;
- contradiction;
- logical state;
- local/canonical sources;
- L2 summary;
- L3 summary;
- L3 ensemble weights.

With `return_cot=True`, the result also includes:

- `result.cot_trace`
- `result.cot_markdown`

## File Structure

```text
pipeline.py                     Main L1–L7 orchestrator
api.py                          FastAPI API
app.py                          Chainlit/Ollama interface
l1_concept_table.py             L1 - concepts
l2_kantian_judgments.py         L2 - judgments
syllogism_module.py             Syllogism/Hempel/Popper
l3_paraconsistent.py            L3 - paraconsistent
paraconsistent_rules.py         Fuzzy rules
neural_truth_model.py           Neural truth scoring model
l4_synthesis.py                 L4 - synthesis
l4_chain_verification.py        CoVe
l4_russell_equivalence.py       Russellian base
l5_generation.py                L5 - generation
l6_final_response.py            L6 - refinement
l7_final_text.py                L7 - final text
prompt_engineering.py           Per-layer prompts
cot_hierarchical.py             Auditable CoT trace
knowledge_base.py               KB
rag_hybrid_context_injection.py Hybrid RAG
llm_provider_client.py          Providers
```

## Important Notes

- Doninha works even without a neural L3 model, using the paraconsistent heuristic.
- The `template` provider allows parts of the flow to run without an external LLM.
- The recommended local provider is Ollama.
- RAG depends on ChromaDB and embeddings when configured.
- The standalone file is large because it contains many consolidated modules.
- Files in `data/` and `philosophy-corpus/` are used as corpus, KB, and training sources.

## Reference

Daniel Fonseca — creator of Doninha AI, a middleware that transforms LLMs into a hybrid neuro-symbolic model with auditing tools.

The theoretical foundation is presented in the article "A True Epistemology for Artificial Intelligence".

All rights reserved to Daniel Barros Fonseca, subject to licensing for third-party use. Doninha IA technology is currently undergoing patent registration. Plagiarism or misappropriation of this proprietary technology will be subject to indemnification, damages, loss-of-profit recovery, and mandatory publication of a retraction letter for industrial espionage.
