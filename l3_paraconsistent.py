"""
CAMADA L3  Avaliao Paraconsistente
======================================
Implementa a Lgica Anotada de Evidncias (LAE / PAL2v) de da Costa & Abe.

Cada proposio recebe um par de anotaes:
       [0,1]   grau de evidncia FAVORVEL
       [0,1]   grau de evidncia CONTRRIA

Estados resultantes:
  
    Verdadeiro     :  alto,  baixo                
    Falso          :  baixo,  alto                
    Inconsistente  :  alto,  alto  (contradio)  
    Indeterminado  :  baixo,  baixo               
    Intermedirio  : valores mdios  (morno, etc.)  
  

Princpio central:
    "Contradio Local + Consistncia Global  Trivializao"

A exploso  GENTIL: uma contradio local (quente e frio) no
trivializa o sistema  produz o estado "Intermedirio" (morno).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import math
import re

import torch

logger = logging.getLogger(__name__)

try:
    from prompt_engineering import get_layer_prompt
except Exception:
    get_layer_prompt = None  # type: ignore

try:
    from paraconsistent_rules import (
        state_from_rules,
        state_12_to_simple,
        load_rules_from_fuzzy_file,
        ParaconsistentRules,
    )
except Exception:
    state_from_rules = None  # type: ignore
    state_12_to_simple = None  # type: ignore
    load_rules_from_fuzzy_file = None  # type: ignore
    ParaconsistentRules = None  # type: ignore

try:
    # Import opcional do modelo neural; o sistema continua funcional sem ele.
    from neural_truth_model import TruthScoringModel, load_tokenizer, neural_annotations
except Exception:  # pragma: no cover - fallback em ambientes sem transformers
    TruthScoringModel = None  # type: ignore
    load_tokenizer = None  # type: ignore
    neural_annotations = None  # type: ignore


# 
# Constantes de limiar
# 
THRESHOLD_TRUE         = 0.7   #   este valor e   (1 - este)  Verdadeiro
THRESHOLD_FALSE        = 0.3   #   este e   (1 - este)  Falso
THRESHOLD_INCONSISTENT = 0.6   # ambos acima  Inconsistente (contradio local)
THRESHOLD_INDETERMINATE= 0.4   # ambos abaixo  Indeterminado

PROPOSITION_TYPE_A = "A"  # Universal Afirmativa
PROPOSITION_TYPE_E = "E"  # Universal Negativa
PROPOSITION_TYPE_I = "I"  # Particular Afirmativa
PROPOSITION_TYPE_O = "O"  # Particular Negativa

PROPOSITION_TYPE_LABELS = {
    PROPOSITION_TYPE_A: "Universal Afirmativa",
    PROPOSITION_TYPE_E: "Universal Negativa",
    PROPOSITION_TYPE_I: "Particular Afirmativa",
    PROPOSITION_TYPE_O: "Particular Negativa",
}


def infer_proposition_type(text: str) -> Optional[str]:
    """Heurstica simples para inferir o tipo A / E / I / O a partir do texto."""
    normalized = text.lower().strip()
    if not normalized:
        return None

    # Detecta padres de proposies particulares negativas antes das afirmativas.
    if re.search(r"\b(algum no|alguns no|alguma no|algumas no|nem todos|pelo menos um no)\b", normalized):
        return PROPOSITION_TYPE_O
    if re.search(r"\b(nenhum|nenhuma|nunca|jamais|sem nenhum|sem nenhuma|no existe|no h)\b", normalized):
        return PROPOSITION_TYPE_E
    if re.search(r"\b(algum|alguma|alguns|algumas|pelo menos um|h um|h algum|existem|existe)\b", normalized):
        return PROPOSITION_TYPE_I
    if re.search(r"\b(todo|todos|toda|todas|cada|sempre|qualquer)\b", normalized):
        return PROPOSITION_TYPE_A

    return None


def type_label(proposition_type: Optional[str]) -> str:
    return PROPOSITION_TYPE_LABELS.get(proposition_type, "Desconhecido")


@dataclass
class ParaconsistentValue:
    """
    Valor-verdade paraconsistente para uma proposio.
    Baseado na Lgica Anotada de Evidncias (LAE).
    """
    proposition: str
    mu: float          # evidncia favorvel   [0,1]
    lam: float         # evidncia contrria   [0,1]
    proposition_type: Optional[str] = None
    confidence: Optional[float] = None
    heuristic_weight: Optional[float] = None
    neural_weight: Optional[float] = None
    ensemble_agreement: Optional[float] = None
    neural_state: Optional[str] = None
    neural_truth: Optional[float] = None

    @property
    def proposition_kind(self) -> Optional[str]:
        """Retorna o tipo de proposio A/E/I/O, inferido do texto se necessrio."""
        return self.proposition_type or infer_proposition_type(self.proposition)

    @property
    def proposition_type_label(self) -> str:
        return type_label(self.proposition_kind)

    #  Graus derivados  #
    @property
    def certainty(self) -> float:
        """Grau de certeza: Gc =       [1, 1]"""
        return self.mu - self.lam

    @property
    def contradiction(self) -> float:
        """Grau de contradio: Gct =  +   1    [1, 1]"""
        return self.mu + self.lam - 1.0

    @property
    def state(self) -> str:
        """Estado lgico qualitativo. Usa regras do Fuzzy.txt se disponveis."""
        if state_from_rules is not None and state_12_to_simple is not None:
            state_12 = state_from_rules(self.mu, self.lam)
            return state_12_to_simple(state_12)
        # Fallback: limiares fixos
        if self.mu >= THRESHOLD_TRUE and self.lam <= (1 - THRESHOLD_TRUE):
            return "Verdadeiro"
        if self.mu <= THRESHOLD_FALSE and self.lam >= (1 - THRESHOLD_FALSE):
            return "Falso"
        if self.mu >= THRESHOLD_INCONSISTENT and self.lam >= THRESHOLD_INCONSISTENT:
            return "Inconsistente_local"   # exploso GENTIL  no trivializa
        if self.mu <= THRESHOLD_INDETERMINATE and self.lam <= THRESHOLD_INDETERMINATE:
            return "Indeterminado"
        return "Intermedirio"             # ex: morno entre quente e frio

    @property
    def state_12(self) -> Optional[str]:
        """Estado lgico de 12 valores (reticulado) conforme Fuzzy.txt, se regras carregadas."""
        if state_from_rules is not None:
            return state_from_rules(self.mu, self.lam)
        return None

    @property
    def truth_value(self) -> float:
        """Valor-verdade escalar normalizado para sada final."""
        return round((self.mu + (1 - self.lam)) / 2.0, 4)

    def __str__(self) -> str:
        type_label_text = self.proposition_type_label
        ensemble_text = ""
        if self.heuristic_weight is not None and self.neural_weight is not None:
            ensemble_text = (
                f"\n  Ensemble: h={self.heuristic_weight:.3f} "
                f"n={self.neural_weight:.3f}"
            )
            if self.confidence is not None:
                ensemble_text += f" conf={self.confidence:.3f}"
            if self.ensemble_agreement is not None:
                ensemble_text += f" acordo={self.ensemble_agreement:.3f}"
        return (
            f"  ={self.mu:.3f}  ={self.lam:.3f}  "
            f"Gc={self.certainty:+.3f}  Gct={self.contradiction:+.3f}  "
            f"v={self.truth_value:.3f}  [{self.state}]\n"
            f"  Tipo={type_label_text}  \"{self.proposition}\""
            f"{ensemble_text}"
        )


class ManyValuedRouter:
    """Roteador fuzzy para distinguir contradio lgica real de incerteza estatstica."""

    REAL_CONTRADICTION = "Contradio_real"
    STATISTICAL_UNCERTAINTY = "Incerteza_estatstica"
    AMBIGUOUS = "Ambguo"
    UNCLASSIFIED = "No_classificado"
    AMBIGUITY_THRESHOLD = 0.1

    @staticmethod
    def _pair_strength(left: ParaconsistentValue, right: ParaconsistentValue) -> float:
        """Grau fuzzy de suporte conjunto entre duas proposies."""
        return round(min(left.mu, right.mu), 4)

    @classmethod
    def _is_ambiguous_pair(
        cls,
        left: ParaconsistentValue,
        right: ParaconsistentValue,
    ) -> bool:
        """Indica se os scores mu/lambda so estatisticamente indistinguveis."""
        return (
            abs(left.mu - right.mu) < cls.AMBIGUITY_THRESHOLD
            and abs(left.lam - right.lam) < cls.AMBIGUITY_THRESHOLD
        )

    @classmethod
    def route_pair(
        cls,
        left: ParaconsistentValue,
        right: ParaconsistentValue,
    ) -> Tuple[str, float, str]:
        """Classifica o par de proposies como contradio real, incerteza, ambguo ou no classificado."""
        left_type = left.proposition_kind
        right_type = right.proposition_kind
        label = f"{left_type or '?'} vs {right_type or '?'}"

        if {left_type, right_type} == {PROPOSITION_TYPE_A, PROPOSITION_TYPE_I}:
            strength = cls._pair_strength(left, right)
            return cls.REAL_CONTRADICTION, strength, f"Contraposio A/I detectada ({label})"

        if {left_type, right_type} == {PROPOSITION_TYPE_E, PROPOSITION_TYPE_I}:
            strength = cls._pair_strength(left, right)
            return cls.STATISTICAL_UNCERTAINTY, strength, f"Contraposio E/I detectada ({label})"

        if cls._is_ambiguous_pair(left, right):
            strength = cls._pair_strength(left, right)
            return cls.AMBIGUOUS, strength, "Scores estatisticamente indistinguveis"

        if left_type is None or right_type is None:
            return cls.UNCLASSIFIED, 0.0, "Tipo de proposio no identificado"

        return cls.UNCLASSIFIED, 0.0, f"Par {label} no corresponde a A/I nem E/I"

    @classmethod
    def route_pairwise(
        cls,
        values: List[ParaconsistentValue],
    ) -> List[Tuple[ParaconsistentValue, ParaconsistentValue, str, float, str]]:
        """Avalia todos os pares de proposies mantendo categorias informativas explcitas."""
        routes: List[Tuple[ParaconsistentValue, ParaconsistentValue, str, float, str]] = []
        n = len(values)
        for i in range(n):
            for j in range(i + 1, n):
                route, confidence, explanation = cls.route_pair(values[i], values[j])
                routes.append((values[i], values[j], route, confidence, explanation))
        return routes


# 
# Motor paraconsistente
# 

class ParaconsistentEngine:
    """
    Avalia as hipteses kantiana (L2) e atribui valores-verdade
    paraconsistentes a cada uma.

    Pode operar em dois modos:
      - Modo heurstico (padro): usa apenas o banco de conhecimento.
      - Modo neural: se um TruthScoringModel for fornecido, usa o modelo
        para calcular (, ) compatveis com a Lgica Anotada.
    """

    def __init__(
        self,
        neural_model: Optional["TruthScoringModel"] = None,
        neural_tokenizer=None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.neural_model = neural_model
        self.neural_tokenizer = neural_tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.calibration_factor = 0.85
        self._warned_neural_fallback = False
        self.last_cot_prompt: str = ""

    def _warn_neural_fallback(self, reason: str) -> None:
        if not self._warned_neural_fallback:
            logger.warning(
                "Modelo neural L3 indisponível ou falhou (%s); usando heurística paraconsistente para avaliação epistemológica.",
                reason,
            )
            self._warned_neural_fallback = True

    def evaluate(
        self,
        propositions: List[Tuple[str, float]],  # (texto, peso_de_prioridade_L2)
        knowledge_base: Dict[str, float],        # termo  grau de evidncia no BD
    ) -> List[ParaconsistentValue]:
        """
        Para cada proposio:
           = evidncia favorvel extrada do banco de dados
           = evidncia contrria = 1  f(compatibilidade)
        """
        if get_layer_prompt is not None:
            self.last_cot_prompt = get_layer_prompt("l3", "", {"propositions": str(propositions[:8])})
        results: List[ParaconsistentValue] = []
        for prop_text, l2_priority in propositions:
            mu, lam, metadata = self._compute_annotations(prop_text, l2_priority, knowledge_base)
            pv = ParaconsistentValue(
                proposition=prop_text,
                mu=mu,
                lam=lam,
                proposition_type=infer_proposition_type(prop_text),
                confidence=metadata.get("confidence"),
                heuristic_weight=metadata.get("heuristic_weight"),
                neural_weight=metadata.get("neural_weight"),
                ensemble_agreement=metadata.get("ensemble_agreement"),
                neural_state=metadata.get("neural_state"),
                neural_truth=metadata.get("neural_truth"),
            )
            results.append(pv)

        # Ordena por valor-verdade descendente
        results.sort(key=lambda pv: pv.truth_value, reverse=True)
        return results

    def route_contradictions(
        self,
        values: List[ParaconsistentValue],
    ) -> List[Tuple[ParaconsistentValue, ParaconsistentValue, str, float, str]]:
        """Retorna rotas de pares de proposies classificadas pelo roteador fuzzy."""
        return ManyValuedRouter.route_pairwise(values)

    # ------------------------------------------------------------------ #
    # Anotao  /                                                        #
    # ------------------------------------------------------------------ #

    def _compute_annotations(
        self,
        text: str,
        l2_priority: float,
        kb: Dict[str, float],
    ) -> Tuple[float, float, Dict[str, float | str]]:
        """
        Calcula (, ) para uma proposio.

        Quando o TruthScoringModel esta disponivel, combina as anotacoes
        heuristicas e neurais por Weighted Dynamic Ensemble. Caso contrario,
        volta para a heuristica original baseada no banco de conhecimento e
        em contradicoes locais.
        """
        h_mu, h_lam = self._heuristic_annotations(text, l2_priority, kb)

        if self.neural_model is not None and self.neural_tokenizer is not None and neural_annotations is not None:
            try:
                n_mu, n_lam, n_state, n_truth = neural_annotations(
                    self.neural_model.to(self.device),
                    self.neural_tokenizer,
                    text,
                )
                # Pequena modulao pela prioridade de L2 para manter a integrao
                n_mu = min(1.0, n_mu * (0.5 + 0.5 * l2_priority))
                n_lam = max(0.0, n_lam * (1.0 - 0.3 * l2_priority))

                agreement = self._compute_agreement(h_mu, h_lam, n_mu, n_lam)
                h_weight = 0.65 + 0.25 * agreement
                n_weight = 1.0 - h_weight

                final_mu = h_weight * h_mu + n_weight * n_mu
                final_lam = h_weight * h_lam + n_weight * n_lam
                final_mu, final_lam = self._paraconsistent_regularization(final_mu, final_lam)
                confidence = (agreement + (1.0 - abs(final_mu - final_lam))) / 2.0

                return round(final_mu, 4), round(final_lam, 4), {
                    "confidence": round(confidence, 4),
                    "heuristic_weight": round(h_weight, 4),
                    "neural_weight": round(n_weight, 4),
                    "ensemble_agreement": round(agreement, 4),
                    "neural_state": n_state,
                    "neural_truth": round(float(n_truth), 4),
                }
            except Exception as exc:
                self._warn_neural_fallback(str(exc))

        self._warn_neural_fallback("modelo neural não carregado ou indisponível")
        confidence = 1.0 - abs(h_mu - h_lam)
        return round(h_mu, 4), round(h_lam, 4), {
            "confidence": round(confidence, 4),
            "heuristic_weight": 1.0,
            "neural_weight": 0.0,
            "ensemble_agreement": 0.0,
        }

    def _heuristic_annotations(
        self,
        text: str,
        l2_priority: float,
        kb: Dict[str, float],
    ) -> Tuple[float, float]:
        """Calcula as anotacoes heuristicas originais da L3."""
        import re
        tokens = set(re.findall(r"[a-z]+", text.lower()))

        kb_scores = [kb.get(t, 0.0) for t in tokens if kb.get(t, 0.0) > 0]
        mu_kb = sum(kb_scores) / len(kb_scores) if kb_scores else 0.3

        contradiction_detected = self._has_antonym_pair(tokens, kb)
        lam_base = 0.8 if contradiction_detected else (1.0 - mu_kb)

        mu = min(1.0, mu_kb * (0.5 + 0.5 * l2_priority))
        lam = max(0.0, lam_base * (1.0 - 0.3 * l2_priority))

        return round(mu, 4), round(lam, 4)

    @staticmethod
    def _compute_agreement(h_mu: float, h_lam: float, n_mu: float, n_lam: float) -> float:
        """Mede a concordancia fuzzy entre a avaliacao heuristica e a neural."""
        distance = (abs(h_mu - n_mu) + abs(h_lam - n_lam)) / 2.0
        return max(0.0, min(1.0, 1.0 - distance))

    def _paraconsistent_regularization(self, mu: float, lam: float) -> Tuple[float, float]:
        """
        Suaviza os graus nos eixos Gc/Gct sem apagar contradicoes locais.

        A regularizacao atua no espaco paraconsistente:
        Gc = mu - lambda e Gct = mu + lambda - 1. Em seguida os valores
        retornam para o plano (mu, lambda), calibrados para evitar extremos
        espurios quando heuristica e neural divergem.
        """
        mu = max(0.0, min(1.0, mu))
        lam = max(0.0, min(1.0, lam))
        gc = (mu - lam) * self.calibration_factor
        gct = (mu + lam - 1.0) * self.calibration_factor
        regularized_mu = (1.0 + gc + gct) / 2.0
        regularized_lam = (1.0 + gct - gc) / 2.0
        return (
            max(0.0, min(1.0, regularized_mu)),
            max(0.0, min(1.0, regularized_lam)),
        )

    ANTONYM_PAIRS = [
        ("quente", "frio"), ("quente", "gelado"),
        ("verdadeiro", "falso"), ("real", "fictcio"),
        ("afirmativo", "negativo"), ("possvel", "impossvel"),
    ]

    def _has_antonym_pair(self, tokens: set, kb: Dict[str, float]) -> bool:
        for a, b in self.ANTONYM_PAIRS:
            if a in tokens and b in tokens:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Consistncia global: verifica se sistema trivializou                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_global_consistency(values: List[ParaconsistentValue]) -> bool:
        """
        Retorna True se o sistema  globalmente consistente
        (nenhuma trivializao  todos os estados vlidos).
        Uma trivializao ocorre se TODAS as proposies so
        'Inconsistente_local' sem nenhum 'Verdadeiro' ou 'Intermedirio'.
        """
        states = {pv.state for pv in values}
        if states == {"Inconsistente_local"}:
            return False   # trivializao global
        return True        # exploso gentil  sistema consistente
