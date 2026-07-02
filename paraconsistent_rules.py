"""
Conjunto de regras para sistema paraconsistente (LPA)
======================================================
Extrado de data/Fuzzy.txt  Lgica Paraconsistente Anotada (da Costa et al.).

Convenes do documento:
  -  (mu) = grau de crena   [0,1], eixo x no QUPC
  -  (lambda) = grau de descrena  [0,1], eixo y no QUPC
  - Gc  = Grau de Certeza     =       [1, 1]
  - Gct = Grau de Contradio =  +   1  [1, 1]

Valores de controle (Figura 3):
  Vscc  = Valor superior de controle de certeza     = 1/2
  Vicc  = Valor inferior de controle de certeza    = -1/2
  Vscct = Valor superior de controle de contradio = 1/2
  Vicct = Valor inferior de controle de contradio = -1/2

Doze estados lgicos (reticulado discretizado):
  T, V, F, , QV, QF e regies de transio (QFV, F, V, QV, TV, QVT, etc.).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Iterator
import re
import os

#  Constantes extradas do Fuzzy.txt 
VSCC = 0.5   # Valor superior de controle de certeza
VICC = -0.5  # Valor inferior de controle de certeza
VSCC_T = 0.5   # Valor superior de controle de contradio
VICC_T = -0.5  # Valor inferior de controle de contradio

# Doze estados lgicos do reticulado (para-analisador)
STATE_T = "Inconsistente"           # T  = (1,1)
STATE_V = "Verdadeiro"              # V  = (1,0)
STATE_F = "Falso"                   # F  = (0,1)
STATE_BOT = "Indeterminado"         #   = (0,0)
STATE_QV = "Quase_Verdadeiro"       # QV
STATE_QF = "Quase_Falso"            # QF
STATE_QF_TO_V = "QF_to_V"
STATE_BOT_TO_F = "Indeterminado_to_F"
STATE_BOT_TO_V = "Indeterminado_to_V"
STATE_QV_TO_BOT = "QV_to_Indeterminado"
STATE_T_TO_V = "Inconsistente_to_V"
STATE_QV_TO_T = "QV_to_Inconsistente"

ALL_STATES_12: List[str] = [
    STATE_T, STATE_V, STATE_F, STATE_BOT,
    STATE_QV, STATE_QF,
    STATE_QF_TO_V, STATE_BOT_TO_F, STATE_BOT_TO_V,
    STATE_QV_TO_BOT, STATE_T_TO_V, STATE_QV_TO_T,
]


@dataclass
class ParaconsistentRules:
    """Parmetros do sistema paraconsistente (ajustveis)."""
    vscc: float = VSCC
    vicc: float = VICC
    vscct: float = VSCC_T
    vicct: float = VICC_T

    @staticmethod
    def gc(mu: float, lam: float) -> float:
        """Grau de Certeza: Gc =     [1, 1]."""
        return mu - lam

    @staticmethod
    def gct(mu: float, lam: float) -> float:
        """Grau de Contradio: Gct =  +   1  [1, 1]."""
        return mu + lam - 1.0

    def state_12(self, mu: float, lam: float) -> str:
        """
        Para-analisador: discretiza (, ) em um dos 12 estados lgicos.
        Regras conforme Fuzzy.txt  regies no QUPC delimitadas por Vscc, Vicc, Vscct, Vicct.
        """
        gc = self.gc(mu, lam)
        gct = self.gct(mu, lam)

        # Alto grau de contradio positiva  Inconsistente (T)
        if gct >= self.vscct:
            if gc >= self.vscc:
                return STATE_T_TO_V      # transio TV
            if gc <= self.vicc:
                return STATE_QV_TO_T     # transio QVT (ou TF)
            return STATE_T
        # Alto grau de contradio negativa  Indeterminado ()
        if gct <= self.vicct:
            if gc >= self.vscc:
                return STATE_BOT_TO_V
            if gc <= self.vicc:
                return STATE_BOT_TO_F
            return STATE_BOT
        # Contradio baixa (zona central em Gct)
        if gc >= self.vscc:
            return STATE_V if gct <= 0 else STATE_QV
        if gc <= self.vicc:
            return STATE_F if gct <= 0 else STATE_QF
        # Certeza em zona intermediria
        if gct > 0:
            return STATE_QV_TO_BOT
        return STATE_QF_TO_V


# Instncia global com valores padro do documento
DEFAULT_RULES = ParaconsistentRules()


def state_from_rules(mu: float, lam: float, rules: ParaconsistentRules | None = None) -> str:
    """Retorna o estado lgico de 12 valores para (, ) segundo as regras do Fuzzy.txt."""
    r = rules or DEFAULT_RULES
    return r.state_12(mu, lam)


def state_12_to_simple(state_12: str) -> str:
    """
    Mapeia os 12 estados do reticulado para os 4 estados usados pelo
    TruthScoringModel / ParaconsistentValue: Verdadeiro | Falso | Intermedirio | Indeterminado.
    """
    if state_12 in (STATE_V, STATE_QV, STATE_T_TO_V, STATE_BOT_TO_V):
        return "Verdadeiro"
    if state_12 in (STATE_F, STATE_QF, STATE_BOT_TO_F, STATE_QF_TO_V):
        return "Falso"
    if state_12 in (STATE_T, STATE_QV_TO_T, STATE_QV_TO_BOT):
        return "Intermedirio"
    return "Indeterminado"


def truth_value_from_annotations(mu: float, lam: float) -> float:
    """Valor-verdade escalar em [0,1] a partir de (, ), compatvel com L3."""
    return round((mu + (1.0 - lam)) / 2.0, 4)


def parse_rules_from_fuzzy_text(content: str) -> ParaconsistentRules:
    """
    Extrai valores de controle do texto do arquivo Fuzzy.txt quando possvel.
    Se no encontrar, retorna DEFAULT_RULES.
    """
    rules = ParaconsistentRules()
    # Procura padres como "Vscc=1/2", "Vscct= 1/2", "1/2" prximo a Vscc, etc.
    vscc_m = re.search(r"Vscc\s*=\s*Vscct\s*=\s*1/2", content, re.I)
    if vscc_m:
        rules.vscc = 0.5
        rules.vscct = 0.5
    vicc_m = re.search(r"Vicc\s*(?:e|e\s*Vicct)?\s*[=:]?\s*-?\s*1/2", content, re.I)
    if vicc_m or "Vicc" in content and "-1/2" in content:
        rules.vicc = -0.5
        rules.vicct = -0.5
    return rules


def load_rules_from_fuzzy_file(path: str | None = None) -> ParaconsistentRules:
    """
    Carrega o contedo de data/Fuzzy.txt e retorna ParaconsistentRules.
    Se o arquivo no existir ou no for legvel, retorna DEFAULT_RULES.
    """
    if path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "data", "Fuzzy.txt")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return parse_rules_from_fuzzy_text(content)
    except Exception:
        return DEFAULT_RULES


def generate_training_pairs(
    rules: ParaconsistentRules | None = None,
    grid_step: float = 0.1,
) -> Iterator[Tuple[float, float, str, float]]:
    """
    Gera pares (, , estado_12, valor_verdade) para treinar a camada L3
    a partir do conjunto de regras (para-analisador).
    til para criar dataset sinttico que segue exatamente o Fuzzy.txt.
    """
    r = rules or DEFAULT_RULES
    mu = 0.0
    while mu <= 1.0:
        lam = 0.0
        while lam <= 1.0:
            state = r.state_12(mu, lam)
            truth = truth_value_from_annotations(mu, lam)
            yield (mu, lam, state, truth)
            lam = round(lam + grid_step, 2)
        mu = round(mu + grid_step, 2)


def get_rules_training_examples(
    rules: ParaconsistentRules | None = None,
    grid_step: float = 0.1,
) -> List[Tuple[float, float, str, float]]:
    """Lista de (, , estado_12, valor_verdade) para uso no treinamento."""
    return list(generate_training_pairs(rules=rules, grid_step=grid_step))
