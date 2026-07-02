# -*- coding: utf-8 -*-
"""
PIPELINE PRINCIPAL — Modelo Híbrido de LLM
===========================================
Orquestra as 10 etapas do fluxo completo:

  1. Recepção do prompt
  2. Extração de conceitos [L1]
  3. Refinamento por Juízos Kantianos [L2]
  4. Silogismo Científico + Hempel
  5. Falseabilidade de Popper
  6. Avaliação Paraconsistente [L3]
  7. Síntese por Equivalência [L4]
  8. Geração da Resposta [L5 — opcional]
  9. Resposta Final em Texto Fluída [L6]
 10. Texto Final Definitivo [L7]

Usa config_loader, knowledge_base (KB escalável + RAG opcional), l5_generation
e opcionalmente o agente de pesquisa para enriquecer contexto.
"""

from __future__ import annotations

# ── Encoding UTF-8 para Windows ──────────────────────────────────────────────
# Deve vir antes de qualquer import que possa emitir output.
import sys, io

def _fix_utf8_streams() -> None:
    """Força UTF-8 em stdout/stderr no Windows sem quebrar ambientes já corretos."""
    for attr in ("stdout", "stderr"):
        stream = getattr(sys, attr)
        # Se já é TextIOWrapper com UTF-8, não faz nada
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower().replace("-", "") == "utf8":
            continue
        try:
            buf = getattr(stream, "buffer", None)
            if buf is not None:
                setattr(sys, attr, io.TextIOWrapper(buf, encoding="utf-8", errors="replace", line_buffering=True))
        except Exception:
            pass  # Ambiente sem buffer (pytest, redirect) — ignora silenciosamente

_fix_utf8_streams()
# ─────────────────────────────────────────────────────────────────────────────
import re
import time
import os
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import torch
import traceback
from llm_provider_client import DEFAULT_MODELS, SUPPORTED_PROVIDERS, normalize_provider
from neural_truth_model import TruthScoringModel, load_tokenizer
from l1_concept_table import ConceptTable, ConceptNode, LogicLMSymbolicSolver
from l2_kantian_judgments import KantianJudgmentEngine, KantianJudgment
from syllogism_module import ScientificSyllogismPipeline
from l3_paraconsistent import ParaconsistentEngine, ParaconsistentValue
from l4_synthesis import RussellianSynthesisEngine, SynthesisResult
from l6_final_response import EpistemicContext, FinalResponseEngine
from l7_final_text import FinalTextEngine
from cot_hierarchical import HierarchicalCoTTrace
from layer_titles import LAYER_TITLES

try:
    from l4_russell_equivalence import load_concept_base
except Exception:
    load_concept_base = None  # type: ignore

try:
    from config_loader import load_config, PROJECT_ROOT
except Exception:
    load_config = None  # type: ignore
    PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from knowledge_base import get_knowledge_base, SEED_KNOWLEDGE_BASE
except Exception:
    get_knowledge_base = None  # type: ignore
    SEED_KNOWLEDGE_BASE = {}

try:
    from l5_generation import generate_response as l5_generate
except Exception:
    l5_generate = None  # type: ignore

try:
    from agente_busca_web import run_search_for_context
except Exception:
    run_search_for_context = None  # type: ignore

try:
    from agente_sintese_final import synthesize_final_text as synthesize_final_agent
except Exception:
    synthesize_final_agent = None  # type: ignore


def _get_kb(config: Optional[Dict[str, Any]], prompt: str, use_agent: bool) -> Dict[str, float]:
    if get_knowledge_base is None:
        return dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}
    return get_knowledge_base(
        config=config,
        query_for_rag=prompt if use_agent else None,
    )


def _ensure_l1_config(config: Dict[str, Any]) -> Dict[str, Any]:
    l1_cfg = config.setdefault("l1", {})
    if not isinstance(l1_cfg, dict):
        l1_cfg = {}
        config["l1"] = l1_cfg
    l1_cfg.setdefault("spacy_enabled", True)
    languages = l1_cfg.get("spacy_languages", ["pt", "en"])
    if not isinstance(languages, list):
        languages = ["pt", "en"]
    l1_cfg["spacy_languages"] = [lang for lang in languages if lang in {"pt", "en"}] or ["pt", "en"]
    return config


def _resolve_provider_settings(
    config: Optional[Dict[str, Any]],
    section: str,
    default_provider: str = "ollama",
) -> Dict[str, Any]:
    """Normaliza configuraes de provider e escolhe o modelo correto para a etapa."""
    section_cfg = (config or {}).get(section, {}) if isinstance(config, dict) else {}
    provider = normalize_provider(section_cfg.get("provider") or default_provider)
    if provider not in SUPPORTED_PROVIDERS:
        provider = normalize_provider(default_provider)

    explicit_model = (section_cfg.get("model") or "").strip()
    ollama_model = (section_cfg.get("ollama_model") or explicit_model or "doninha8:latest").strip()
    base_url = (section_cfg.get("base_url") or "").strip()
    api_key = (section_cfg.get("api_key") or "").strip()
    ollama_host = (section_cfg.get("ollama_host") or "http://localhost:11434").strip()

    if provider == "ollama":
        resolved_model = ollama_model
    elif provider in {"template", "custom_lm"}:
        resolved_model = explicit_model or ollama_model
    else:
        resolved_model = explicit_model or DEFAULT_MODELS.get(provider, ollama_model)

    return {
        "provider": provider,
        "model": explicit_model,
        "resolved_model": resolved_model,
        "base_url": base_url,
        "api_key": api_key,
        "ollama_model": ollama_model,
        "ollama_host": ollama_host,
    }


class HybridLLMPipeline:
    """
    Pipeline completo do Modelo Hbrido de LLM.
    Suporta config, KB escalvel, L5 (gerao), agente opcional e chat.
    """

    def __init__(
        self,
        knowledge_base: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
    ) -> None:
        self._config = _ensure_l1_config(config or (load_config() if load_config else {}))
        self.kb = knowledge_base or _get_kb(self._config, "", False)
        if not self.kb:
            self.kb = dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}
        self.verbose = verbose

        self.L1 = ConceptTable()
        self.L2 = KantianJudgmentEngine(self.L1)
        self.SYL = ScientificSyllogismPipeline()
        self.L3 = None
        try:
            from paraconsistent_engine import ParaconsistentEngine
            
            l3_cfg = self._config.get("l3", {}) if isinstance(self._config, dict) else {}
            
            self.L3 = ParaconsistentEngine(
                t_threshold=l3_cfg.get("t_threshold", 0.7),
                f_threshold=l3_cfg.get("f_threshold", 0.3),
                verbose=verbose
            )
            if self.verbose:
                print("✅ L3 (ParaconsistentEngine) carregado com sucesso!")
                
        except ModuleNotFoundError:
            print("⚠️ Módulo 'paraconsistent_engine' não encontrado.")
            print("   → Você precisa criar o arquivo ou ajustar o caminho.")
            self.L3 = None
            
        except TypeError as e:
            print(f"⚠️ Erro nos parâmetros do ParaconsistentEngine: {e}")
            try:
                self.L3 = ParaconsistentEngine(verbose=verbose)
                print("✅ L3 inicializado sem thresholds.")
            except Exception as e2:
                print(f"❌ Falha ao criar L3: {e2}")
                self.L3 = None
                
        except Exception as e:
            print(f"❌ Erro ao inicializar L3: {e}")
            self.L3 = None

        # L3
        l3_cfg = self._config.get("l3", {})
        model_path = l3_cfg.get("model_path", "truth_scoring_model.pt")
        backbone_name = l3_cfg.get("backbone", 'pytcross-encoder/nli-MiniLM2-L6-H768')
        if not Path(model_path).is_absolute():
            model_path = str(PROJECT_ROOT / model_path)
        neural_model = None
        neural_tokenizer = None
        if os.path.exists(model_path):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                neural_tokenizer = load_tokenizer(backbone_name)
                neural_model = TruthScoringModel(backbone_name=backbone_name)
                state = torch.load(model_path, map_location=device)
                neural_model.load_state_dict(state)
                neural_model.to(device)
                if self.verbose:
                    print(f"[L3] Modelo neural carregado de '{model_path}'")
                self.L3 = ParaconsistentEngine(neural_model=neural_model, neural_tokenizer=neural_tokenizer, device=device)
            except Exception as exc:
                print(f"[L3] Falha detectada: {exc}")
                traceback.print_exc()
                self.L3 = ParaconsistentEngine()

        # L4
        russell_base = None
        rpath = self._config.get("l4", {}).get("russell_concepts_path", "l4_russell_concepts.json")
        if not Path(rpath).is_absolute():
            rpath = str(PROJECT_ROOT / rpath)
        if load_concept_base and os.path.exists(rpath):
            try:
                russell_base = load_concept_base(rpath)
                if self.verbose:
                    print("[L4] Base russelliana carregada.")
            except Exception:
                pass
        if russell_base is None and load_concept_base:
            try:
                from l4_russell_equivalence import build_russell_concept_base
                russell_base = build_russell_concept_base()
            except Exception:
                pass
        self.L4 = RussellianSynthesisEngine(
            self.kb,
            russell_concept_base=russell_base,
            use_concept_based_weights=(russell_base is not None),
            verification_config=self._config.get("l4_chain_verification", {}),
        )
        self.L6 = FinalResponseEngine()
        self.L7 = FinalTextEngine(config=self._config)  # Passa config para suportar mltiplos providers

    def _build_many_valued_routes(self, pv_list: List[ParaconsistentValue]) -> List[Dict[str, Any]]:
        """
        Constrói a lista de rotas paraconsistentes de forma defensiva,
        lidando com qualquer formato retornado por route_contradictions:
          - tupla (left, right, route, confidence, explanation)   → formato esperado original
          - tupla (left, right)                                   → formato reduzido
          - ParaconsistentValue único                             → itens avulsos
          - qualquer outro iterável                               → ignorado com segurança
        """
        routes = []
        try:
            raw = self.L3.route_contradictions(pv_list)
        except Exception:
            return routes

        for item in raw:
            try:
                # Formato completo esperado: (left, right, route, confidence, explanation)
                if isinstance(item, (list, tuple)) and len(item) == 5:
                    left, right, route, confidence, explanation = item
                    routes.append({
                        "left": getattr(left, "proposition", str(left)),
                        "left_type": getattr(left, "proposition_kind", None) or "Desconhecido",
                        "right": getattr(right, "proposition", str(right)),
                        "right_type": getattr(right, "proposition_kind", None) or "Desconhecido",
                        "route": route,
                        "confidence": confidence,
                        "explanation": explanation,
                    })
                # Formato reduzido: (left, right)
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    left, right = item
                    routes.append({
                        "left": getattr(left, "proposition", str(left)),
                        "left_type": getattr(left, "proposition_kind", None) or "Desconhecido",
                        "right": getattr(right, "proposition", str(right)),
                        "right_type": getattr(right, "proposition_kind", None) or "Desconhecido",
                        "route": "contradição",
                        "confidence": getattr(left, "confidence", 0.0) or 0.0,
                        "explanation": "",
                    })
                # ParaconsistentValue único — usa os próprios atributos como rota
                elif hasattr(item, "proposition"):
                    routes.append({
                        "left": item.proposition,
                        "left_type": getattr(item, "proposition_kind", None) or "Desconhecido",
                        "right": "",
                        "right_type": "Desconhecido",
                        "route": getattr(item, "state", "desconhecido"),
                        "confidence": getattr(item, "confidence", 0.0) or 0.0,
                        "explanation": "",
                    })
                # Qualquer outro formato: ignora silenciosamente
            except Exception:
                continue

        return routes

    def _log(self, msg: str) -> None:
        """Imprime mensagem apenas quando verbose=True."""
        if self.verbose:
            print(msg)

    def _infer_domain(self, concepts: List[ConceptNode]) -> str:
        """Inferncia simples de domnio majoritrio a partir dos conceitos extrados."""
        if not concepts:
            return "geral"
        domain_counts = {}
        for concept in concepts:
            domain = concept.domain.lower().strip() if concept.domain else "geral"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return max(domain_counts, key=domain_counts.get)

    def _collect_canonical_sources(self, concepts: List[ConceptNode]) -> List[str]:
        sources = []
        seen = set()
        for concept in concepts:
            source = (concept.canonical_source or "").strip()
            if source and source not in seen:
                seen.add(source)
                sources.append(source)
        return sources

    def _summarize_judgments(self, judgments: List[KantianJudgment]) -> str:
        parts = []
        for idx, judgment in enumerate(judgments[:5], start=1):
            cls = getattr(judgment.epistemic_classification, "classification", "no_classificado")
            truth = getattr(judgment.epistemic_classification, "truth", 0.0)
            ind = getattr(judgment.epistemic_classification, "indeterminacy", 0.0)
            fals = getattr(judgment.epistemic_classification, "falsity", 0.0)
            parts.append(
                f"L2-{idx}: {judgment.proposicao[:120]} | pri={judgment.prioridade:.2f} | class={cls} | T/I/F={truth:.2f}/{ind:.2f}/{fals:.2f}"
            )
        return " ; ".join(parts) if parts else "nenhum juzo L2 disponvel"

    def _summarize_paraconsistent(self, pv_list: List[ParaconsistentValue]) -> str:
        parts = []
        for idx, pv in enumerate(pv_list[:5], start=1):
            ensemble_bits = []
            if getattr(pv, "confidence", None) is not None:
                ensemble_bits.append(f"conf={pv.confidence:.3f}")
            if getattr(pv, "heuristic_weight", None) is not None and getattr(pv, "neural_weight", None) is not None:
                ensemble_bits.append(f"h={pv.heuristic_weight:.3f}/n={pv.neural_weight:.3f}")
            if getattr(pv, "ensemble_agreement", None) is not None:
                ensemble_bits.append(f"agreement={pv.ensemble_agreement:.3f}")
            ensemble_summary = f" | ensemble {' '.join(ensemble_bits)}" if ensemble_bits else ""
            parts.append(
                f"L3-{idx}: ={pv.mu:.3f} ={pv.lam:.3f} state={pv.state} truth={pv.truth_value:.3f} certainty={pv.certainty:+.3f} contradiction={pv.contradiction:+.3f}{ensemble_summary}"
            )
        return " ; ".join(parts) if parts else "nenhuma avaliao L3 disponvel"

    def _build_citation_note(self, concepts: List[ConceptNode], agent_context: str) -> str:
        sources = self._collect_canonical_sources(concepts)
        if sources:
            return (
                "Fontes locais identificadas no contexto, mas nenhuma citao bibliogrfica externa foi confirmada para esta resposta; "
                "a exibio de referncias exige verificao direta da consulta ao documento."
            )
        if agent_context:
            return "Contexto externo detectado, mas nenhuma citao bibliogrfica foi confirmada para esta resposta."
        return "Nenhuma citao bibliogrfica foi confirmada para esta resposta; o resultado foi produzido com base interna e sem referncia externa verificada."

    # Casa apenas blocos [AUDIT ...] de UMA linha, no mesmo formato que nos mesmos
    # emitimos (ex.: "[AUDIT L4] truth=0.87 ..."). Isso permite remover, com
    # seguranca, um [AUDIT] que o LLM tenha fabricado dentro do proprio texto
    # gerado, sem arriscar apagar paragrafos legitimos de analise do modelo
    # (que normalmente nao comecam a linha com esse padrao exato).
    _FAKE_AUDIT_LINE = re.compile(r"^[ \t]*\[AUDIT[^\]\n]*\][^\n]*$", re.IGNORECASE | re.MULTILINE)

    @staticmethod
    def _real_hash(content: str, length: int = 16) -> str:
        """
        Calcula um SHA-256 REAL sobre 'content'. Nunca fabrica, estima ou
        randomiza um hash — se 'content' for vazio, faz hash da string vazia
        (resultado deterministico e 100% reproduzivel, jamais aleatorio).
        Para verificar manualmente: sha256(content.encode('utf-8')).hexdigest().
        """
        safe_content = content or ""
        digest = hashlib.sha256(safe_content.encode("utf-8", errors="replace")).hexdigest()
        return digest[:length] if length else digest

    def _strip_fake_audit_lines(self, text: str) -> str:
        """
        Remove linhas '[AUDIT ...]' que ja vieram embutidas no texto gerado
        pelo provider LLM (L5/L6/L7), ANTES de anexarmos o bloco [AUDIT] real
        calculado em Python. Sem isso, um timestamp/hash inventado pelo modelo
        pode conviver lado a lado com o timestamp/hash real — e o usuario nao
        tem como distinguir qual e qual.
        """
        if not text or "[AUDIT" not in text.upper():
            return text
        cleaned = self._FAKE_AUDIT_LINE.sub("", text)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)  # colapsa linhas em branco deixadas pela remocao
        return cleaned.strip()

    def _append_audit_block(
        self,
        text: str,
        label: str,
        details: str,
        pv_list: Optional[List[ParaconsistentValue]] = None,
        hash_source: Optional[str] = None,
        hash_sink: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Anexa bloco [AUDIT] raw com:
          - timestamp real do sistema (timezone-aware, ISO-8601 UTC) — nunca fabricado
          - label da camada
          - hash SHA-256 REAL calculado sobre o conteudo efetivo da camada
            (hash_source quando fornecido; caso contrario, sobre 'text')
          - detalhes livres
          - ensemble weights (h/n/agreement) das ParaconsistentValues, quando fornecidas

        Antes de anexar, remove qualquer '[AUDIT ...]' que ja existisse no texto
        recebido (tipicamente fabricado pelo LLM ao narrar sua propria auditoria),
        garantindo que exista apenas UM bloco [AUDIT label] por camada — o real.

        Se 'hash_sink' for fornecido, o hash calculado e registrado em
        hash_sink[label], permitindo montar depois uma cadeia de hashes real
        (hash chain) das camadas L1-L7 em vez de um valor inventado.
        """
        text = self._strip_fake_audit_lines(text)

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        content_for_hash = hash_source if hash_source is not None else text
        real_hash = self._real_hash(f"{label}|{ts}|{content_for_hash}|{details}")
        if hash_sink is not None:
            hash_sink[label] = real_hash

        ensemble_raw = ""
        if pv_list:
            parts = []
            for i, pv in enumerate(pv_list[:5], 1):
                h  = getattr(pv, "heuristic_weight",  None)
                n  = getattr(pv, "neural_weight",      None)
                ag = getattr(pv, "ensemble_agreement", None)
                cf = getattr(pv, "confidence",         None)
                bits = []
                if h  is not None: bits.append(f"h={h:.4f}")
                if n  is not None: bits.append(f"n={n:.4f}")
                if ag is not None: bits.append(f"agr={ag:.4f}")
                if cf is not None: bits.append(f"conf={cf:.4f}")
                if bits:
                    parts.append(f"pv{i}[{' '.join(bits)}]")
            if parts:
                ensemble_raw = " | ensemble_weights: " + " ".join(parts)

        block = f"\n\n[AUDIT {label} | ts={ts} | hash={real_hash}] {details}{ensemble_raw}"
        return (text + block).strip()

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000

    @staticmethod
    def _clip(text: str, limit: int = 420) -> str:
        text = " ".join((text or "").split())
        return text[:limit] + "..." if len(text) > limit else text

    def process(
        self,
        prompt: str,
        chat_session: Optional[Any] = None,
        use_agent: Optional[bool] = None,
        skip_l5: bool = False,
        skip_l6: bool = False,
        return_cot: bool = False,
    ) -> SynthesisResult:
        """Executa o pipeline e retorna SynthesisResult (com response j gerada por L5 se ativo)."""
        t0 = time.perf_counter()
        cot_trace = HierarchicalCoTTrace(prompt=prompt)
        layer_hashes: Dict[str, str] = {}
        pipeline_started_at = datetime.datetime.now(datetime.timezone.utc)
        self._log(f"[PIPELINE] Inicio real (UTC): {pipeline_started_at.isoformat(timespec='seconds')}")
        use_agent = use_agent if use_agent is not None else self._config.get("agent", {}).get("use_agent", True)
        if chat_session and hasattr(chat_session, "get_context_for_prompt"):
            prompt_for_kb = chat_session.get_context_for_prompt(prompt, self._config.get("chat", {}).get("max_turns_in_context", 10))
        else:
            prompt_for_kb = prompt

        # KB pode ser enriquecido por RAG (Chroma) quando use_agent
        if use_agent and get_knowledge_base:
            self.kb = _get_kb(self._config, prompt_for_kb, True)
            if not self.kb:
                self.kb = dict(SEED_KNOWLEDGE_BASE) if SEED_KNOWLEDGE_BASE else {}

        self._log("\n" + "=" * 60)
        self._log(f"  PROMPT: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        self._log("=" * 60)

        limit = RussellianSynthesisEngine.check_fundamental_limits(prompt)
        if limit:
            self._log(f"\n{limit}")

        self._log("\n[ETAPA 2] L1  Extrao de Conceitos")
        layer_start = time.perf_counter()
        concepts: List[ConceptNode] = self.L1.extract_concepts(prompt, llm_context=prompt_for_kb, domain="geral", config=self._config)
        domain = self._infer_domain(concepts)
        if domain != "geral":
            # Re-extrai com domnio especfico para enriquecer com KB do domnio
            concepts = self.L1.extract_concepts(prompt, llm_context=prompt_for_kb, domain=domain, config=self._config)
        concepts_summary = ""
        if self.verbose and concepts:
            for c in concepts:
                syns = ", ".join(c.synonyms[:2]) or ""
                self._log(f"   {c.term:15s} | sinnimos: {syns}")
            concepts_summary = "; ".join(f"{c.term}({', '.join(c.synonyms[:2])})" for c in concepts[:8])
        cot_trace.add_step(
            "L1",
            LAYER_TITLES["l1"],
            "Extracao semantica de conceitos, dominio e relacoes de aplicacao a partir do prompt e do contexto disponivel.",
            [
                f"{len(concepts)} conceitos identificados",
                f"Dominio inferido: {domain}",
            ],
            concepts_summary or "Nenhum conceito estruturado encontrado.",
            self._elapsed_ms(layer_start),
        )
        layer_hashes["L1"] = self._real_hash(f"L1|{concepts_summary}|dominio={domain}")

        self._log("\n[ETAPA 3] L2  Juzos Kantianos")
        layer_start = time.perf_counter()
        judgments: List[KantianJudgment] = self.L2.refine(prompt, concepts)
        top_judgments = ""
        if judgments:
            top_judgments = "\n".join(j.proposicao for j, _ in list(zip(judgments, [None] * 6))[:6])
        cot_trace.add_step(
            "L2",
            LAYER_TITLES["l2"],
            "Construcao de proposicoes epistemologicas, classificacao kantiana e priorizacao para as etapas formais.",
            [
                f"{len(judgments)} juizos gerados",
                f"Maior prioridade: {judgments[0].prioridade:.2f}" if judgments else "Sem juizos priorizados",
            ],
            self._clip(top_judgments, 520) or "Nenhum juizo gerado.",
            self._elapsed_ms(layer_start),
        )
        layer_hashes["L2"] = self._real_hash(f"L2|{top_judgments}")

        self._log("\n[ETAPAS 4+5] Silogismo + Hempel + Popper")
        prompt_terms = set(re.findall(r"[a-zA-Z]+", prompt.lower()))
        kb_scores = {j.proposicao[:30]: self.kb.get(j.proposicao.split()[0], 0.3) for j in judgments}
        filtered = self.SYL.run(judgments, prompt_terms, kb_scores)
        self._log(f"  {len(judgments)} hipteses  {len(filtered)} aps filtros")

        self._log("\n[ETAPA 6] L3  Lgica Paraconsistente + Classificao Epistemolgica L2")
        layer_start = time.perf_counter()
        props_with_priority = [(j.proposicao, score) for j, score in filtered]
        pv_list: List[ParaconsistentValue] = self.L3.evaluate(props_with_priority, self.kb)
        consistent = self.L3.check_global_consistency(pv_list)
        self._log(f"  Consistncia global: {'Sim' if consistent else 'No'}")
        l3_cot_summary = self._summarize_paraconsistent(pv_list)
        cot_trace.add_step(
            "L3",
            LAYER_TITLES["l3"],
            "Avaliacao paraconsistente com mu/lambda, estados logicos e pesos dinamicos do ensemble quando disponiveis.",
            [
                f"Consistencia global: {'sim' if consistent else 'nao'}",
                f"{len(set(pv.state for pv in pv_list))} estados logicos distintos" if pv_list else "Nenhum estado logico produzido",
            ],
            self._clip(l3_cot_summary, 620),
            self._elapsed_ms(layer_start),
        )
        layer_hashes["L3"] = self._real_hash(f"L3|consistente={consistent}|{l3_cot_summary}")

        epistemic_context = EpistemicContext(
            proposition_states=[
                {
                    "proposition": pv.proposition,
                    "proposition_type": pv.proposition_kind or "Desconhecido",
                    "mu": pv.mu,
                    "lambda": pv.lam,
                    "certainty": pv.certainty,
                    "contradiction": pv.contradiction,
                    "truth_value": pv.truth_value,
                    "state": pv.state,
                    "confidence": pv.confidence,
                    "heuristic_weight": pv.heuristic_weight,
                    "neural_weight": pv.neural_weight,
                    "ensemble_agreement": pv.ensemble_agreement,
                    "neural_state": pv.neural_state,
                    "neural_truth": pv.neural_truth,
                }
                for pv in pv_list
            ],
            many_valued_routes=self._build_many_valued_routes(pv_list),
            bert_classifications=[
                {
                    "proposition": judgment.proposicao,
                    "priority": judgment.prioridade,
                    "truth": judgment.epistemic_classification.truth,
                    "indeterminacy": judgment.epistemic_classification.indeterminacy,
                    "falsity": judgment.epistemic_classification.falsity,
                    "classification": judgment.epistemic_classification.classification,
                }
                for judgment, _ in filtered[:8]
            ],
            application_context=LogicLMSymbolicSolver.summarize_application_context(concepts),
        )


        self._log("\n[ETAPA 7] L4  Sntese Russelliana")
        layer_start = time.perf_counter()
        l2_priorities = {j.proposicao[:40]: j.prioridade for j, _ in filtered}
        result: SynthesisResult = self.L4.synthesize(pv_list, l2_priorities, prompt)
        l4_result = result
        l5_text = result.response
        cot_trace.add_step(
            "L4",
            LAYER_TITLES["l4"],
            "Sintese por equivalencia russelliana e verificacao Chain-of-Verification (CoVe) sobre a melhor hipotese L3 ponderada por prioridade L2.",
            [
                f"Estado sintetico: {result.state}",
                f"Valor-verdade: {result.truth_value:.3f}",
                f"Verificacoes Chain-of-Verification (CoVe): {len(getattr(result, 'verification_log', []) or [])}",
            ],
            self._clip(result.response, 560),
            self._elapsed_ms(layer_start),
        )

        # Contexto do agente (busca web/local) apenas quando o pipeline está em modo de fallback heurístico.
        agent_context = ""
        heuristic_only_mode = (
            (not self.kb or not concepts or not pv_list)
            or any(
                getattr(pv, "state", "") in {"Indeterminado", "Intermedirio", "Inconsistente_local"}
                or getattr(pv, "truth_value", 0.0) < 0.45
                or getattr(pv, "neural_weight", 0.0) == 0.0
                for pv in pv_list
            )
        )
        if use_agent and run_search_for_context and heuristic_only_mode:
            try:
                agent_context = run_search_for_context(prompt)
                if agent_context and self.verbose:
                    self._log("\n[AGENTE] Contexto de busca obtido para fallback heurístico.")
            except Exception:
                pass

        # L4  nota de fontes e auditoria (agora com contexto RAG disponvel)
        l4_sources_note = self._build_citation_note(concepts, agent_context)
        l4_result.response = self._append_audit_block(
            l4_result.response,
            "L4",
            f"truth={l4_result.truth_value:.4f} certainty={l4_result.certainty:+.4f} contradiction={l4_result.contradiction:+.4f} state={l4_result.state} | {l4_sources_note}",
            pv_list=pv_list,
            hash_source=f"{l4_result.response}|truth={l4_result.truth_value:.4f}|certainty={l4_result.certainty:+.4f}|contradiction={l4_result.contradiction:+.4f}",
            hash_sink=layer_hashes,
        )
        result.response = l4_result.response
        l5_text = result.response

        # L5  Gerao de resposta em texto livre
        gen_cfg = self._config.get("generation", {})
        final_cfg = self._config.get("finalization", {})
        l7_cfg = self._config.get("l7", {})

        gen_resolved = _resolve_provider_settings(self._config, "generation")
        final_resolved = _resolve_provider_settings(self._config, "finalization")
        l7_resolved = _resolve_provider_settings(self._config, "l7")

        base_provider = gen_resolved["provider"]

        if final_cfg.get("provider") and normalize_provider(final_cfg.get("provider")) != base_provider:
            self._log(
                f"[PIPELINE] Ignorando provider de finalizao '{final_cfg.get('provider')}' para usar provider base '{base_provider}'."
            )
        if l7_cfg.get("provider") and normalize_provider(l7_cfg.get("provider")) != base_provider:
            self._log(
                f"[PIPELINE] Ignorando provider L7 '{l7_cfg.get('provider')}' para usar provider base '{base_provider}'."
            )

        provider = base_provider
        if not skip_l5 and l5_generate and provider != "template":
            layer_start = time.perf_counter()
            final_response = l5_generate(
                prompt,
                result,
                provider=provider,
                concepts_summary=concepts_summary,
                top_judgments=top_judgments,
                custom_lm_path=gen_cfg.get("custom_lm_path", ""),
                ollama_model=gen_resolved["resolved_model"],
                ollama_host=gen_resolved["ollama_host"],
                base_url=gen_resolved["base_url"],
                api_key=gen_resolved["api_key"],
            )
            if agent_context and final_response:
                final_response = final_response + "\n\n[Contexto da busca]\n" + agent_context[:800]
            elif agent_context:
                final_response = result.response + "\n\n[Contexto da busca]\n" + agent_context[:800]
            else:
                final_response = final_response or result.response
            result = SynthesisResult(
                response=self._append_audit_block(
                    final_response,
                    "L5",
                    f"provider={provider} model={gen_resolved['resolved_model']} | audit=L1-L5: {self._summarize_judgments(judgments)}",
                    pv_list=pv_list,
                    hash_source=f"{final_response}|provider={provider}|model={gen_resolved['resolved_model']}",
                    hash_sink=layer_hashes,
                ),
                truth_value=result.truth_value,
                certainty=result.certainty,
                contradiction=result.contradiction,
                state=result.state,
                supporting_evidence=result.supporting_evidence,
                falsified_hypotheses=result.falsified_hypotheses,
                confidence_label=result.confidence_label,
            )
            cot_trace.add_step(
                "L5",
                LAYER_TITLES["l5"],
                "Geracao textual intermediaria a partir da sintese L4 e dos resumos L1-L3.",
                [
                    f"Provider: {provider}",
                    f"Modelo: {gen_resolved['resolved_model']}",
                ],
                self._clip(final_response, 560),
                self._elapsed_ms(layer_start),
            )
        elif agent_context and result.response:
            layer_start = time.perf_counter()
            result = SynthesisResult(
                response=self._append_audit_block(
                    result.response + "\n\n[Contexto da busca]\n" + agent_context[:800],
                    "L5",
                    f"provider={provider} model={gen_resolved['resolved_model']} | audit=L1-L5: {self._summarize_judgments(judgments)}",
                    pv_list=pv_list,
                    hash_source=f"{result.response}|agent_context_presente={bool(agent_context)}|provider={provider}",
                    hash_sink=layer_hashes,
                ),
                truth_value=result.truth_value,
                certainty=result.certainty,
                contradiction=result.contradiction,
                state=result.state,
                supporting_evidence=result.supporting_evidence,
                falsified_hypotheses=result.falsified_hypotheses,
                confidence_label=result.confidence_label,
            )
            cot_trace.add_step(
                "L5",
                LAYER_TITLES["l5"],
                "Geracao textual intermediaria em modo fallback, anexando contexto de busca quando disponivel.",
                [f"Provider: {provider}", "Fallback textual usado"],
                self._clip(result.response, 560),
                self._elapsed_ms(layer_start),
            )
        else:
            cot_trace.add_step(
                "L5",
                LAYER_TITLES["l5"],
                "Camada de geracao textual manteve a resposta sintetica de L4 como fallback.",
                [f"Provider: {provider}", "Sem chamada gerativa adicional"],
                self._clip(result.response, 560),
                0.0,
            )
            layer_hashes["L5"] = self._real_hash(f"L5|fallback_l4_sem_geracao|{result.response}")

        l5_text = result.response

        if not skip_l6:
            layer_start = time.perf_counter()
            final_text = self.L6.finalize_response(
                prompt=prompt,
                synthesis_result=result,
                epistemic_context=epistemic_context,
                generated_text=result.response,
                concepts_summary=concepts_summary,
                top_judgments=top_judgments,
                agent_context=agent_context,
            )
            final_text = self.L6.rewrite_response(
                prompt=prompt,
                synthesis_result=result,
                epistemic_context=epistemic_context,
                generated_text=final_text,
                concepts_summary=concepts_summary,
                top_judgments=top_judgments,
                agent_context=agent_context,
                provider=base_provider,
                custom_lm_path=final_cfg.get("custom_lm_path", gen_cfg.get("custom_lm_path", "")),
                ollama_model=final_resolved["resolved_model"],
                ollama_host=final_resolved["ollama_host"],
                base_url=final_resolved["base_url"],
                api_key=final_resolved["api_key"],
            )
            result = SynthesisResult(
                response=self._append_audit_block(
                    final_text,
                    "L6",
                    f"provider={base_provider} model={final_resolved['resolved_model']} | epistemic={self._summarize_paraconsistent(pv_list)}",
                    pv_list=pv_list,
                    hash_source=f"{final_text}|provider={base_provider}|model={final_resolved['resolved_model']}",
                    hash_sink=layer_hashes,
                ),
                truth_value=result.truth_value,
                certainty=result.certainty,
                contradiction=result.contradiction,
                state=result.state,
                supporting_evidence=result.supporting_evidence,
                falsified_hypotheses=result.falsified_hypotheses,
                confidence_label=result.confidence_label,
            )
            cot_trace.add_step(
                "L6",
                LAYER_TITLES["l6"],
                "Refinamento da resposta intermediaria para clareza, coerencia e proporcionalidade epistemica.",
                [
                    f"Provider: {base_provider}",
                    f"Modelo: {final_resolved['resolved_model']}",
                ],
                self._clip(final_text, 560),
                self._elapsed_ms(layer_start),
            )
        else:
            cot_trace.add_step(
                "L6",
                LAYER_TITLES["l6"],
                "Refinamento final foi pulado por configuracao da chamada.",
                ["skip_l6=True"],
                self._clip(result.response, 420),
                0.0,
            )
            layer_hashes["L6"] = self._real_hash(f"L6|skip_l6|{result.response}")

        l3_summary = ""
        if epistemic_context is not None and epistemic_context.proposition_states:
            top_states = epistemic_context.proposition_states[:3]
            l3_summary = "; ".join(
                f"{item.get('proposition', 'desconhecida')}  {item.get('state', 'n/a')} ({item.get('truth_value', 0):.2f}, conf={item.get('confidence', 0) or 0:.2f}, h/n={item.get('heuristic_weight', 0) or 0:.2f}/{item.get('neural_weight', 0) or 0:.2f})"
                for item in top_states
            )
            if epistemic_context.many_valued_routes:
                l3_summary += f"; rotas paraconsistentes: {len(epistemic_context.many_valued_routes)}"

        l7_cfg = self._config.get("l7", {})
        # Coletar alertas de incompatibilidade cannica gerados durante L1
        canonical_alerts = LogicLMSymbolicSolver.get_canonical_alerts() if LogicLMSymbolicSolver else []
        
        # === L7  Texto Final Definitivo (Automtico e Integrado) ===
        # Usa o agente de sntese final quando disponvel, com fallback para a engine L7 original.
        if synthesize_final_agent is not None:
            layer_start = time.perf_counter()
            final_text_l7 = synthesize_final_agent(
                prompt=prompt,
                l1_summary=concepts_summary,
                l2_summary=top_judgments,
                l3_summary=l3_summary,
                l4_response=l4_result.response,
                l5_text=l5_text,
                l6_text=result.response,
                provider=base_provider,
                model=l7_resolved["resolved_model"],
                temperature=l7_cfg.get("temperature", 0.7),
                max_tokens=l7_cfg.get("max_tokens", 4096),
                cot_context=cot_trace.to_markdown(),
            )
        else:
            layer_start = time.perf_counter()
            final_text_l7 = self.L7.finalize_text(
                prompt=prompt,
                l1_summary=concepts_summary,
                l2_summary=top_judgments,
                l3_summary=l3_summary,
                l4_response=l4_result.response,
                l5_text=l5_text,
                l6_text=result.response,
                synthesis_result=l4_result,
                provider=base_provider,
                model=l7_resolved["resolved_model"],
                custom_lm_path=l7_cfg.get("custom_lm_path", gen_cfg.get("custom_lm_path", "")),
                canonical_alerts=canonical_alerts,
                cot_context=cot_trace.to_markdown(),
                temperature=l7_cfg.get("temperature", 0.7),
                max_tokens=l7_cfg.get("max_tokens", 4096),
            )
        cot_trace.add_step(
            "L7",
            LAYER_TITLES["l7"],
            "Sintese final integra os resumos L1-L6 e o rastro auditavel para produzir a resposta definitiva.",
            [
                f"Provider: {base_provider}",
                f"Modelo: {l7_resolved['resolved_model']}",
            ],
            self._clip(final_text_l7, 620),
            self._elapsed_ms(layer_start),
        )
        cot_trace.final_synthesis = final_text_l7
        cot_trace.overall_confidence = float(getattr(result, "truth_value", 0.0) or 0.0)

        # Cadeia de hashes REAL (hash chain) das camadas L1-L6: cada hash em
        # layer_hashes foi calculado por _real_hash() sobre o conteudo efetivo
        # que aquela camada produziu nesta execucao especifica — nunca um valor
        # fixo ou estimado pelo LLM. So entram na cadeia as camadas que de fato
        # executaram e registraram hash nesta chamada.
        hash_chain_source = "|".join(
            f"{layer}={layer_hashes[layer]}" for layer in ("L1", "L2", "L3", "L4", "L5", "L6") if layer in layer_hashes
        )
        hash_chain_l1_l6 = self._real_hash(hash_chain_source, length=32) if hash_chain_source else "indisponivel"

        l7_details = (
            f"provider={base_provider} model={l7_resolved['resolved_model']} | "
            f"sources={'; '.join(self._collect_canonical_sources(concepts)[:6]) or 'nenhuma fonte local'} | "
            f"L2={self._summarize_judgments(judgments)} | "
            f"L3={self._summarize_paraconsistent(pv_list)} | "
            f"hash_chain_L1-L6={hash_chain_l1_l6}"
        )

        result = SynthesisResult(
            response=self._append_audit_block(
                final_text_l7,
                "L7",
                l7_details,
                pv_list=pv_list,
                hash_source=f"{final_text_l7}|{hash_chain_l1_l6}",
                hash_sink=layer_hashes,
            ),
            truth_value=result.truth_value,
            certainty=result.certainty,
            contradiction=result.contradiction,
            state=result.state,
            supporting_evidence=result.supporting_evidence,
            falsified_hypotheses=result.falsified_hypotheses,
            confidence_label=result.confidence_label,
        )
        result.cot_trace = cot_trace  # type: ignore[attr-defined]
        result.cot_markdown = cot_trace.to_markdown()  # type: ignore[attr-defined]
        result.layer_hashes = dict(layer_hashes)  # type: ignore[attr-defined]

        elapsed = (time.perf_counter() - t0) * 1000
        pipeline_finished_at = datetime.datetime.now(datetime.timezone.utc)
        self._log(f"\n[ETAPA 10] L7  Texto Final Definitivo  ({elapsed:.1f} ms)\n")
        self._log(
            f"[PIPELINE] Fim real (UTC): {pipeline_finished_at.isoformat(timespec='seconds')} "
            f"| hash_chain_L1-L6={hash_chain_l1_l6}"
        )
        self._log(str(result))

        # Mostrar os valores auditáveis de todas as camadas
        self._log("\n=== Valores Auditáveis por Camada ===")
        self._log("\n[L1 - Extrao de Conceitos]")
        self._log(f"Conceitos identificados: {concepts_summary or 'Nenhum conceito estruturado encontrado.'}")
        self._log(f"Domínio inferido: {domain}")

        self._log("\n[L2 - Juízos Kantianos]")
        self._log(f"Juízos gerados: {len(judgments)}")
        self._log(f"Resumo dos juízos: {self._summarize_judgments(judgments)}")

        self._log("\n[L3 - Lógica Paraconsistente]")
        self._log(f"Consistência global: {'Sim' if consistent else 'Não'}")
        self._log(f"Resumo paraconsistente: {l3_cot_summary}")

        self._log("\n[L4 - Síntese Russelliana]")
        self._log(f"Estado sintético: {l4_result.state}")
        self._log(f"Valor-verdade: {l4_result.truth_value:.3f}")
        self._log(f"Certeza: {l4_result.certainty:+.3f}")
        self._log(f"Contradição: {l4_result.contradiction:+.3f}")

        self._log("\n[L5 - Geração de Resposta]")
        self._log(f"Provider: {provider}")
        self._log(f"Modelo: {gen_resolved['resolved_model']}")

        self._log("\n[L6 - Resposta Final em Texto Fluido]")
        self._log(f"Provider: {base_provider}")
        self._log(f"Modelo: {final_resolved['resolved_model']}")

        self._log("\n[L7 - Texto Final Definitivo]")
        self._log(f"Provider: {base_provider}")
        self._log(f"Modelo: {l7_resolved['resolved_model']}")
        self._log(f"Fontes: {'; '.join(self._collect_canonical_sources(concepts)[:6]) or 'nenhuma fonte local'}")

        return result

    def repl(self) -> None:
        print("\n" + "=" * 60)
        print("  MODELO HBRIDO DE LLM  Fonseca")
        print("  Digite 'sair' para encerrar")
        print("=" * 60)
        while True:
            try:
                prompt = input("\nPrompt  ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                continue
            if prompt.lower() in {"sair", "exit", "quit"}:
                break
            self.process(prompt)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IA Doninha — Pipeline Neuro-Simbólico L1-L7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python pipeline.py --prompt "O que é verdade?"
  python pipeline.py --prompt "O que é verdade?" --verbose
  python pipeline.py --prompt "O que é verdade?" --verbose --cot
  python pipeline.py --repl
  python pipeline.py --demo
        """,
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Pergunta única — exibe apenas a resposta final por padrão",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Exibe o processamento detalhado de todas as camadas L1–L7 durante a execução",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Exibe o rastro hierárquico de Chain of Thought (resumo estruturado L1–L7) ao final",
    )
    parser.add_argument("--repl",   action="store_true", help="Modo interativo (verbose ativado automaticamente)")
    parser.add_argument("--demo",   action="store_true", help="Rodar demonstração com prompts fixos")
    parser.add_argument("--config", type=str,            help="Caminho para config.yaml alternativo")
    args, _ = parser.parse_known_args()

    # verbose=True quando: flag explícita, modo repl, modo demo, ou nenhum --prompt fornecido.
    # Com --prompt sozinho mantém o comportamento silencioso original.
    # Com --prompt --verbose exibe todas as camadas durante o processamento.
    verbose = args.verbose or args.repl or args.demo or not args.prompt

    config = load_config(Path(args.config)) if load_config and args.config else (load_config() if load_config else {})
    pipeline = HybridLLMPipeline(config=config, verbose=verbose)

    if args.prompt:
        # return_cot=True sempre — o rastro é gerado; só a exibição é condicional
        r = pipeline.process(args.prompt, return_cot=True)

        print(r.response)

        # ── Opção 3: rastro CoT L1–L7 ──────────────────────────────────────
        if args.cot or args.verbose:
            cot_md = getattr(r, "cot_markdown", None)
            if cot_md:
                print()
                print("=" * 62)
                print("  RASTRO L1–L7 — Chain of Thought Hierárquico Auditável")
                print("=" * 62)
                print(cot_md)

            # Hashes reais por camada (gerados pelo pipeline.py corrigido)
            layer_hashes = getattr(r, "layer_hashes", None)
            if layer_hashes:
                print()
                print("=" * 62)
                print("  HASHES REAIS POR CAMADA (SHA-256 — verificável)")
                print("=" * 62)
                for layer, h in layer_hashes.items():
                    print(f"  {layer:4s}: {h}")
                # hash chain L1-L6 já aparece no bloco [AUDIT L7] da resposta,
                # mas exibimos aqui também para facilitar inspeção isolada
                chain_layers = [
                    f"{la}={layer_hashes[la]}"
                    for la in ("L1", "L2", "L3", "L4", "L5", "L6")
                    if la in layer_hashes
                ]
                if chain_layers:
                    import hashlib
                    chain_hash = hashlib.sha256("|".join(chain_layers).encode("utf-8")).hexdigest()[:32]
                    print(f"  {'CHAIN':4s}: {chain_hash}  (L1–L6 encadeados)")
            print()

        return

    if args.repl:
        pipeline.repl()
        return

    if args.demo:
        for p in [
            "A água a 35 graus está quente ou fria?",
            "O que é a verdade?",
        ]:
            r = pipeline.process(p, return_cot=True)
            print(r.response)
            print()
        return

    # Fallback: comportamento legado quando nenhuma flag é passada
    if "--repl" in sys.argv:
        pipeline.repl()
        return
    for p in [
        "A água a 35 graus está quente ou fria?",
        "O que é a verdade?",
    ]:
        pipeline.process(p, return_cot=True)
        print()


if __name__ == "__main__":
    main()