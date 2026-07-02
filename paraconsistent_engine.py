from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from paraconsistent_rules import (
    ParaconsistentRules,
    state_12_to_simple,
    truth_value_from_annotations,
    load_rules_from_fuzzy_file,
)


# ---------------------------------------------------------------------------
# Vocabulário de L2 (artigo, linha 130): domínio da hipótese e prioridade
# ---------------------------------------------------------------------------

class EpistemicDomain(str, Enum):
    """Quatro categorias operacionais de domínio da hipótese, atribuídas por L2."""
    EMPIRICAMENTE_VERIFICADO = "empiricamente_verificado"
    TEORICAMENTE_PLAUSIVEL = "teoricamente_plausivel"
    LOGICAMENTE_CONTRADITORIO = "logicamente_contraditorio"
    EMPIRICAMENTE_INDETERMINADO = "empiricamente_indeterminado"


class EpistemicPriority(str, Enum):
    """Grau de prioridade epistêmica atribuído por L2 a cada proposição."""
    FORTE = "forte"
    INTERMEDIARIA = "intermediaria"
    FRACA = "fraca"


# Deslocamento de prior bayesiano por domínio: (delta_mu, delta_lambda).
# Valores não especificados no artigo — escolha de engenharia documentada.
_DOMAIN_PRIOR_SHIFT: Dict[EpistemicDomain, Tuple[float, float]] = {
    EpistemicDomain.EMPIRICAMENTE_VERIFICADO: (0.20, -0.10),
    EpistemicDomain.TEORICAMENTE_PLAUSIVEL: (0.08, -0.03),
    EpistemicDomain.LOGICAMENTE_CONTRADITORIO: (-0.10, 0.20),
    EpistemicDomain.EMPIRICAMENTE_INDETERMINADO: (-0.05, -0.05),
}

# Quanto o prior de L2 pesa sobre o score heurístico bruto, por prioridade.
_PRIORITY_BLEND: Dict[EpistemicPriority, float] = {
    EpistemicPriority.FORTE: 0.45,
    EpistemicPriority.INTERMEDIARIA: 0.25,
    EpistemicPriority.FRACA: 0.10,
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_enum(enum_cls, raw):
    """Converte raw (str ou já-enum) para o Enum; retorna None se inválido."""
    if raw is None:
        return None
    if isinstance(raw, enum_cls):
        return raw
    try:
        return enum_cls(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class KBDocument:
    """Documento da base de conhecimento usado pela fonte heurística.

    stance indica se o documento apoia, contradiz ou é neutro em relação à
    proposição sendo avaliada — necessário para que a similaridade semântica
    produza μ (favorável) e λ (contrário) separadamente, em vez de um único
    score indiferenciado.
    """
    text: str
    stance: str = "neutral"  # "supports" | "contradicts" | "neutral"
    weight: float = 1.0


def make_kb(entries: List[Dict[str, Any]]) -> List[KBDocument]:
    """Utilitário para construir uma KB a partir de dicts simples."""
    return [
        KBDocument(
            text=e["text"],
            stance=e.get("stance", "neutral"),
            weight=float(e.get("weight", 1.0)),
        )
        for e in entries
    ]


@dataclass
class ParaconsistentValue:
    truth: float = 0.0
    falsity: float = 0.0
    certainty: float = 0.0          # Gc = μ − λ
    contradiction: float = 0.0      # Gct = μ + λ − 1
    state: str = ""                 # um dos 12 estados do reticulado QUPC
    simple_state: str = ""          # Verdadeiro | Falso | Intermediário | Indeterminado
    mu: float = 0.0
    lam: float = 0.0
    truth_value: float = 0.0        # v = (μ − λ + 1) / 2
    proposition: str = ""
    proposition_kind: str = "Desconhecido"
    confidence: float = 0.0
    heuristic_weight: float = 0.0
    neural_weight: float = 0.0
    ensemble_agreement: float = 0.0
    neural_state: str = ""
    neural_truth: float = 0.0
    needs_evidence_search: bool = False
    ready_for_conclusion: bool = False
    sources: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Interfaces plugáveis (artigo: ChromaDB+HuggingFace Embeddings / TruthScoringModel)
# ---------------------------------------------------------------------------

@runtime_checkable
class SemanticRetriever(Protocol):
    """Interface da fonte heurística (artigo, linha 186)."""

    def retrieve(
        self, proposition_text: str, kb: Optional[List[KBDocument]]
    ) -> List[Tuple[KBDocument, float]]:
        """Retorna pares (documento, score_similaridade[0,1]), ordenados por relevância."""
        ...


@runtime_checkable
class NeuralTruthScorer(Protocol):
    """Interface do TruthScoringModel (artigo, linha 188). Implementação real
    (classificador Transformer treinado) fica fora do escopo deste arquivo;
    qualquer objeto que satisfaça esta interface pode ser injetado no engine."""

    def score(self, proposition_text: str) -> Tuple[float, float, str, float]:
        """Retorna (mu_neural, lambda_neural, estado_neural_simples, confidence)."""
        ...


class SimpleLexicalRetriever:
    """Fallback funcional e honesto quando nenhum SemanticRetriever real está
    configurado. Usa sobreposição de tokens (Jaccard) como proxy de
    similaridade semântica. É claramente mais fraco que recuperação vetorial
    real (ChromaDB + embeddings), mas não finge ser outra coisa."""

    _TOKEN_RE = re.compile(r"[a-zà-úA-ZÀ-Ú0-9]+")

    def _tokens(self, text: str) -> set:
        return {t.lower() for t in self._TOKEN_RE.findall(text)}

    def _similarity(self, a: str, b: str) -> float:
        ta, tb = self._tokens(a), self._tokens(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / union if union else 0.0

    def retrieve(self, proposition_text, kb):
        if not kb:
            return []
        scored = [(doc, self._similarity(proposition_text, doc.text if isinstance(doc, KBDocument) else doc)) for doc in kb]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


class HeuristicScorer:
    """Fonte heurística de L3 (artigo, linha 186): combina recuperação
    semântica contra a KB com os priors epistêmicos de L2 (domínio + prioridade,
    artigo linha 130)."""

    def __init__(self, retriever: Optional[SemanticRetriever] = None, top_k: int = 5):
        self.retriever: SemanticRetriever = retriever or SimpleLexicalRetriever()
        self.top_k = top_k

    def hits(self, proposition_text: str, kb: Optional[List[KBDocument]]):
        return self.retriever.retrieve(proposition_text, kb)[: self.top_k]

    def score_from_hits(
        self,
        hits: List[Tuple[KBDocument, float]],
        domain: Optional[EpistemicDomain],
        priority: Optional[EpistemicPriority],
    ) -> Tuple[float, float]:
        if not hits:
            # Sem evidência recuperável -> ponto neutro do QUPC (Indeterminado).
            mu_raw, lam_raw = 0.5, 0.5
        else:
            support = [score * doc.weight for doc, score in hits if isinstance(doc, KBDocument) and doc.stance == "supports"]
            contra = [score * doc.weight for doc, score in hits if isinstance(doc, KBDocument) and doc.stance == "contradicts"]
            neutral = [score * doc.weight for doc, score in hits if isinstance(doc, KBDocument) and doc.stance == "neutral"]

            mu_raw = max(support, default=0.0)
            lam_raw = max(contra, default=0.0)
            if neutral:
                # Evidência neutra confirma cobertura do tópico na KB, mas não
                # indica direção -- conta parcialmente para os dois lados.
                coverage = max(neutral)
                mu_raw = max(mu_raw, coverage * 0.5)
                lam_raw = max(lam_raw, coverage * 0.3)

        if domain is not None:
            d_mu, d_lam = _DOMAIN_PRIOR_SHIFT[domain]
            blend = _PRIORITY_BLEND[priority] if priority is not None else _PRIORITY_BLEND[EpistemicPriority.INTERMEDIARIA]
            mu_raw += blend * d_mu
            lam_raw += blend * d_lam

        return _clamp01(mu_raw), _clamp01(lam_raw)

    def score(
        self,
        proposition_text: str,
        kb: Optional[List[KBDocument]],
        domain: Optional[EpistemicDomain],
        priority: Optional[EpistemicPriority],
    ) -> Tuple[float, float]:
        return self.score_from_hits(self.hits(proposition_text, kb), domain, priority)


# ---------------------------------------------------------------------------
# Motor paraconsistente
# ---------------------------------------------------------------------------

class ParaconsistentEngine:
    def __init__(
        self,
        t_threshold: float = 0.5,
        f_threshold: float = 0.5,
        retriever: Optional[SemanticRetriever] = None,
        neural_scorer: Optional[NeuralTruthScorer] = None,
        rules_path: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        t_threshold / f_threshold: agora realmente parametrizam os limiares de
        certeza (Vscc/Vicc) e contradição (Vscct/Vicct) usados pelo
        para-analisador, em vez de serem ignorados como na versão anterior.

        rules_path: se fornecido, carrega limiares de um arquivo Fuzzy.txt via
        load_rules_from_fuzzy_file(); caso contrário usa t_threshold/f_threshold.

        neural_scorer: implementação opcional do TruthScoringModel. Quando
        None, o engine opera em modo heurístico puro (heuristic_weight = 1.0),
        exatamente como descrito no artigo para o caso de fallback.
        """
        if rules_path is not None:
            self.rules = load_rules_from_fuzzy_file(rules_path)
        else:
            self.rules = ParaconsistentRules(
                vscc=t_threshold, vicc=-t_threshold,
                vscct=f_threshold, vicct=-f_threshold,
            )

        self.heuristic = HeuristicScorer(retriever=retriever)
        self.neural_scorer = neural_scorer
        self.verbose = verbose

        if self.verbose:
            mode = "heurístico + neural (ensemble)" if neural_scorer else "heurístico puro (fallback)"
            print(
                f"✅ ParaconsistentEngine iniciado (modo={mode}, "
                f"Vscc={self.rules.vscc}, Vscct={self.rules.vscct})"
            )

    # ------------------------------------------------------------------
    # Weighted Dynamic Ensemble (artigo, linhas 182-192)
    # ------------------------------------------------------------------
    @staticmethod
    def _agreement(mu_h: float, lam_h: float, mu_n: float, lam_n: float) -> float:
        """Concordância entre as fontes heurística e neural, em [0,1].
        Fórmula não especificada no artigo (ver nota no topo do arquivo);
        definida aqui como 1 menos a distância média absoluta entre os pares
        (μ, λ) das duas fontes."""
        dist = (abs(mu_h - mu_n) + abs(lam_h - lam_n)) / 2.0
        return _clamp01(1.0 - dist)

    def _ensemble(
        self, proposition_text: str, mu_h: float, lam_h: float
    ) -> Tuple[float, float, float, float, float, str, float]:
        """Retorna (mu, lam, heuristic_weight, neural_weight, agreement, neural_state, neural_truth)."""
        if self.neural_scorer is None:
            # Modo heurístico puro (artigo, linha 192).
            return mu_h, lam_h, 1.0, 0.0, 0.0, "", 0.0

        mu_n, lam_n, neural_state, _neural_conf = self.neural_scorer.score(proposition_text)
        mu_n, lam_n = _clamp01(mu_n), _clamp01(lam_n)

        agreement = self._agreement(mu_h, lam_h, mu_n, lam_n)
        heuristic_weight = _clamp01(0.65 + 0.25 * agreement)  # artigo, linha 190
        neural_weight = 1.0 - heuristic_weight

        mu = heuristic_weight * mu_h + neural_weight * mu_n
        lam = heuristic_weight * lam_h + neural_weight * lam_n
        neural_truth = truth_value_from_annotations(mu_n, lam_n)

        return mu, lam, heuristic_weight, neural_weight, agreement, neural_state, neural_truth

    # ------------------------------------------------------------------
    # Avaliação principal
    # ------------------------------------------------------------------
    def evaluate(
        self,
        propositions: List[Any],
        kb: Optional[List[KBDocument]] = None,
    ) -> List[ParaconsistentValue]:
        """
        Avalia uma lista de proposições usando lógica paraconsistente.

        Cada item de `propositions` pode ser:
          - uma string (texto puro da proposição); ou
          - um dict com chaves opcionais:
              'text' | 'proposition'  texto da proposição (obrigatório se
                                       'mu'/'lam' não forem fornecidos)
              'mu', 'lam'             floats já calculados a montante (pulam
                                      o cálculo heurístico, mas ainda passam
                                      pelo ensemble se houver neural_scorer)
              'domain'                EpistemicDomain ou string equivalente
              'priority'              EpistemicPriority ou string equivalente
              'kind'                  rótulo livre vindo de L2.5
        """
        if self.verbose:
            print(f"🔄 L3 avaliando {len(propositions)} proposições...")

        results: List[ParaconsistentValue] = []

        for prop in propositions:
            text, mu_in, lam_in, domain, priority, kind = self._parse_proposition(prop)

            hits = self.heuristic.hits(text, kb)

            if mu_in is not None and lam_in is not None:
                mu_h, lam_h = _clamp01(mu_in), _clamp01(lam_in)
            else:
                mu_h, lam_h = self.heuristic.score_from_hits(hits, domain, priority)

            mu, lam, hw, nw, agreement, neural_state, neural_truth = self._ensemble(text, mu_h, lam_h)

            gc = self.rules.gc(mu, lam)
            gct = self.rules.gct(mu, lam)
            state_12 = self.rules.state_12(mu, lam)
            simple_state = state_12_to_simple(state_12)
            v = truth_value_from_annotations(mu, lam)

            pv = ParaconsistentValue(
                truth=v,
                falsity=1.0 - v,
                certainty=gc,
                contradiction=gct,
                state=state_12,
                simple_state=simple_state,
                mu=mu,
                lam=lam,
                truth_value=v,
                proposition=text,
                proposition_kind=kind,
                confidence=abs(gc),
                heuristic_weight=hw,
                neural_weight=nw,
                ensemble_agreement=agreement,
                neural_state=neural_state,
                neural_truth=neural_truth,
                sources = [doc.text if hasattr(doc, 'text') else (doc if isinstance(doc, str) else str(doc)) for doc, _score in hits[:3]]
            )
            pv.needs_evidence_search = self._needs_evidence_search(pv)
            pv.ready_for_conclusion = self._ready_for_conclusion(pv)

            results.append(pv)

        return results

    @staticmethod
    def _parse_proposition(prop: Any):
        if isinstance(prop, dict):
            text = prop.get("text") or prop.get("proposition") or str(prop)
            mu_in = prop.get("mu", prop.get("truth"))
            lam_in = prop.get("lam", prop.get("falsity"))
            domain = _safe_enum(EpistemicDomain, prop.get("domain"))
            priority = _safe_enum(EpistemicPriority, prop.get("priority"))
            kind = prop.get("kind", "Desconhecido")
            return text, mu_in, lam_in, domain, priority, kind

        return str(prop), None, None, None, None, "Desconhecido"

    # ------------------------------------------------------------------
    # Regras de controle derivadas do estado paraconsistente
    # (artigo, linha 194 e linhas 252-258 — Agente de Busca Web)
    # ------------------------------------------------------------------
    def _needs_evidence_search(self, pv: ParaconsistentValue) -> bool:
        """
        Três condições do artigo para ativar o agente de busca:
          1) Gct >= Vscct  -> estado Inconsistente (alta contradição)
          2) Gct <= Vicct  -> estado Indeterminado (ausência de evidências)
          3) Fallback heurístico ativo (TruthScoringModel indisponível)
        """
        if pv.contradiction >= self.rules.vscct:
            return True
        if pv.contradiction <= self.rules.vicct:
            return True
        if self.neural_scorer is None:
            return True
        return False

    def _ready_for_conclusion(self, pv: ParaconsistentValue) -> bool:
        """Gct < Vscct e |Gc| >= Vscc -> certeza suficiente para emitir a L4 (artigo, l.194)."""
        return pv.contradiction < self.rules.vscct and abs(pv.certainty) >= self.rules.vscc

    # ------------------------------------------------------------------
    # Consistência global e roteamento
    # ------------------------------------------------------------------
    def check_global_consistency(self, pv_list: List[ParaconsistentValue]) -> bool:
        """Falso se qualquer proposição estiver no ou acima do limiar de
        contradição configurado em ParaconsistentRules (antes era um 0.5
        hardcoded, desconectado de self.rules)."""
        return all(pv.contradiction < self.rules.vscct for pv in pv_list)

    def route_contradictions(self, pv_list: List[ParaconsistentValue]) -> List[ParaconsistentValue]:
        """
        Retorna as proposições que precisam de busca de evidência adicional
        (Inconsistente, Indeterminado, ou fallback heurístico ativo).
        Antes era um stub que sempre retornava [] -- o roteamento para o
        agente de busca (artigo, linhas 252-258) nunca acontecia de fato.
        """
        return [pv for pv in pv_list if pv.needs_evidence_search]

    def combine_global(self, pv_list: List[ParaconsistentValue]) -> Tuple[float, float]:
        """
        Agrega μ/λ de múltiplas proposições em um par global via t-norma
        mínima (μ) / t-conorma máxima (λ) -- combinação padrão de lógica
        paraconsistente/fuzzy para conjunção entre proposições.
        Ver nota no topo do arquivo: esta agregação específica não está
        descrita literalmente no artigo; é uma extensão sinalizada como tal.
        """
        if not pv_list:
            return 0.0, 0.0
        mu_global = min(pv.mu for pv in pv_list)
        lam_global = max(pv.lam for pv in pv_list)
        return _clamp01(mu_global), _clamp01(lam_global)


# ---------------------------------------------------------------------------
# Demonstração executável (prova de que o cálculo é real, não placeholder)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    kb = make_kb([
        {"text": "Pacientes com Korsakoff preservam a continuidade psicológica mesmo sem memória episódica nova.",
         "stance": "supports", "weight": 1.0},
        {"text": "A identidade pessoal depende exclusivamente da memória episódica contínua.",
         "stance": "contradicts", "weight": 0.8},
        {"text": "Korsakoff é causado por deficiência de tiamina associada ao alcoolismo crônico.",
         "stance": "neutral", "weight": 0.5},
    ])

    engine = ParaconsistentEngine(verbose=True)  # modo heurístico puro, sem neural_scorer

    propositions = [
        {"text": "Pacientes com síndrome de Korsakoff mantêm identidade coerente mesmo sem consolidar memórias novas.",
         "domain": "teoricamente_plausivel", "priority": "forte"},
        "A Terra é plana.",  # sem hits relevantes -> deve cair perto de Indeterminado
        {"text": "Identidade pessoal depende exclusivamente da memória episódica.",
         "domain": "logicamente_contraditorio", "priority": "intermediaria"},
    ]

    results = engine.evaluate(propositions, kb=kb)

    for pv in results:
        print(f"\nProposição: {pv.proposition}")
        print(f"  μ={pv.mu:.3f}  λ={pv.lam:.3f}  Gc={pv.certainty:.3f}  Gct={pv.contradiction:.3f}")
        print(f"  estado(12)={pv.state}  estado_simples={pv.simple_state}  v={pv.truth_value:.3f}")
        print(f"  needs_evidence_search={pv.needs_evidence_search}  ready_for_conclusion={pv.ready_for_conclusion}")
        print(f"  fontes={pv.sources}")

    print("\nconsistência global:", engine.check_global_consistency(results))
    print("rotas de contradição (precisam de busca):", [pv.proposition for pv in engine.route_contradictions(results)])
    print("combinação global (μ_global, λ_global):", engine.combine_global(results))