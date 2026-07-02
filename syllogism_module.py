"""
MDULO  Silogismo Cientfico Aristotlico + Paradoxo de Hempel + Popper
=========================================================================
Integrado entre L2 e L3 (etapa 4 e 5 do fluxo).

Filtra as hipteses kantianas pelas 8 regras do silogismo cientfico e
aplica o princpio da falseabilidade: toda concluso  tratada como
FALSA at que se encontre evidncia verdadeira equivalente.

Paradoxo de Hempel implementado como filtro negativo:
    Nem toda palavra posterior pode ser inferida da anterior.
    Objetos irrelevantes no validam uma teoria.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from l2_kantian_judgments import KantianJudgment


# 
# Estrutura de um silogismo
# 

@dataclass
class Syllogism:
    major:      str   # premissa maior (universal)
    minor:      str   # premissa menor (particular/singular)
    conclusion: str   # concluso derivada
    valid:      bool  = True
    violations: List[str] = None

    def __post_init__(self):
        if self.violations is None:
            self.violations = []

    def __str__(self) -> str:
        status = " VLIDO" if self.valid else f" INVLIDO ({'; '.join(self.violations)})"
        return (
            f"  Maior : {self.major}\n"
            f"  Menor : {self.minor}\n"
            f"  Concl.: {self.conclusion}\n"
            f"  Status: {status}"
        )


# 
# As 8 regras do silogismo cientfico
# 

class AristotelianSyllogismValidator:
    """
    Valida um silogismo segundo as 8 regras aristotlicas e retorna
    a lista de violaes (vazia se vlido).
    """

    def validate(self, major: str, minor: str, conclusion: str) -> List[str]:
        violations: List[str] = []
        m_neg = self._is_negative(major)
        n_neg = self._is_negative(minor)
        c_neg = self._is_negative(conclusion)
        m_part = self._is_particular(major)
        n_part = self._is_particular(minor)
        c_part = self._is_particular(conclusion)

        # R1  Apenas trs termos, cada um no mesmo sentido
        terms_m = self._extract_key_terms(major)
        terms_n = self._extract_key_terms(minor)
        terms_c = self._extract_key_terms(conclusion)
        all_terms = terms_m | terms_n | terms_c
        if len(all_terms) > 6:          # heurstica liberal
            violations.append("R1: mais de trs termos distintos detectados")

        # R2  Termo mdio no aparece na concluso
        middle = terms_m & terms_n - terms_c
        if not middle and terms_m & terms_n:
            violations.append("R2: termo mdio pode estar na concluso")

        # R3  Concluso no excede extenso das premissas
        if not c_part and (m_part or n_part):
            violations.append("R3: concluso mais extensa que as premissas")

        # R4  Termo mdio deve ser universal pelo menos uma vez
        if m_part and n_part:
            violations.append("R4: termo mdio nunca  universal")

        # R5  De duas negativas, nada se conclui
        if m_neg and n_neg:
            violations.append("R5: duas premissas negativas  concluso invlida")

        # R6  Duas afirmativas  concluso afirmativa
        if not m_neg and not n_neg and c_neg:
            violations.append("R6: premissas afirmativas exigem concluso afirmativa")

        # R7  De duas particulares, nada se conclui
        if m_part and n_part:
            violations.append("R7: duas premissas particulares  concluso invlida")

        # R8  "Parte Fraca": concluso segue a premissa mais fraca
        if (m_neg or n_neg) and not c_neg:
            violations.append("R8: premissa negativa exige concluso negativa")
        if (m_part or n_part) and not c_part and not c_neg:
            violations.append("R8: premissa particular exige concluso particular")

        return violations

    #  helpers  #

    @staticmethod
    def _is_negative(text: str) -> bool:
        neg_markers = {"no", "nunca", "nenhum", "jamais", "nem", "negativo"}
        return any(w in text.lower().split() for w in neg_markers)

    @staticmethod
    def _is_particular(text: str) -> bool:
        part_markers = {"algum", "alguma", "alguns", "algumas", "certo",
                        "parte", "pode", "possvel"}
        return any(w in text.lower().split() for w in part_markers)

    @staticmethod
    def _extract_key_terms(text: str) -> set:
        stop = {"", "so", "de", "do", "da", "em", "com", "por",
                "para", "este", "esta", "esse", "toda", "todo", "um", "uma"}
        import re
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        return {t for t in tokens if t not in stop and len(t) > 2}


# 
# Filtro de Hempel (anti-confirmao espria)
# 

class HempelFilter:
    """
    Paradoxo de Hempel: objetos irrelevantes no devem confirmar hipteses.
    Implementado como deteco de correlaes esprias entre termos do
    prompt e termos do banco de dados sem relao semntica real.
    """

    def __init__(self, relevance_threshold: float = 0.25) -> None:
        self.threshold = relevance_threshold

    def is_spurious(self, judgment: KantianJudgment, prompt_terms: set) -> bool:
        """
        Retorna True se a hiptese  provavelmente espria
        (confirmao por objeto irrelevante).
        """
        import re
        hyp_terms = set(re.findall(
            r"[a-zA-Z]+",
            judgment.proposicao.lower()
        ))
        overlap = len(hyp_terms & prompt_terms) / max(len(hyp_terms), 1)
        return overlap < self.threshold   # pouca sobreposio = provvel esprio


# 
# Princpio da Falseabilidade de Popper
# 

class PopperFalsifiability:
    """
    Toda concluso  tratada como FALSA at que se encontre evidncia
    verdadeira equivalente no banco de dados.

    Implementa o princpio do Cisne Negro: a proposio universal
    "todo cisne  branco"  falsa at que seja falsificada por um cisne preto.
    """

    def __init__(self, falsifiability_floor: float = 0.1) -> None:
        """
        falsifiability_floor : score mnimo de evidncia para aceitar
                               a hiptese como no-falsificada.
        """
        self.floor = falsifiability_floor

    def apply(
        self,
        hypotheses: List[Tuple[KantianJudgment, float]],   # (juzo, score_BD)
    ) -> List[Tuple[KantianJudgment, float, bool]]:
        """
        Retorna triplas (juzo, score, falsificada?).
        Hipteses universais afirmativas partem sempre de score 0
        (falsas at prova em contrrio).
        """
        result = []
        for j, score in hypotheses:
            # Proposies universais: presume falso at evidncia forte
            if j.quantidade == "Universal":
                adjusted = score if score >= self.floor else 0.0
                falsified = adjusted < self.floor
            # Singulares: usa o score direto
            else:
                adjusted = score
                falsified = False
            result.append((j, adjusted, falsified))
        return result


# 
# Pipeline integrado
# 

class ScientificSyllogismPipeline:
    """
    Integra: Silogismo Aristotlico + Filtro de Hempel + Falseabilidade.
    Chamado entre L2 e L3.
    """

    def __init__(self) -> None:
        self.validator = AristotelianSyllogismValidator()
        self.hempel    = HempelFilter()
        self.popper    = PopperFalsifiability()

    def run(
        self,
        judgments: List[KantianJudgment],
        prompt_terms: set,
        kb_scores: dict,          # termo_proposicao  score [0,1]
    ) -> List[Tuple[KantianJudgment, float]]:
        """
        Filtra e pontua as hipteses.
        Retorna lista ordenada de (juzo, score_final).
        """
        # 1. Remove hipteses esprias (Hempel)
        non_spurious = [
            j for j in judgments
            if not self.hempel.is_spurious(j, prompt_terms)
        ]

        # 2. Valida via silogismo (usa prioridade L2 como par maior/menor)
        scored: List[Tuple[KantianJudgment, float]] = []
        for j in non_spurious:
            # Constri silogismo sinttico para validao
            major = f"Universal: {j.proposicao}"
            minor = f"Singular: {j.proposicao}"
            conclusion = j.proposicao
            violations = self.validator.validate(major, minor, conclusion)
            penalty = len(violations) * 0.1
            base_score = kb_scores.get(j.proposicao[:30], j.prioridade)
            scored.append((j, max(0.0, base_score - penalty)))

        # 3. Aplica falseabilidade (Popper)
        with_falsifiability = self.popper.apply(scored)

        # 4. Remove falsificadas e reordena
        valid = [
            (j, score)
            for j, score, falsified in with_falsifiability
            if not falsified
        ]
        valid.sort(key=lambda x: x[1], reverse=True)
        return valid
