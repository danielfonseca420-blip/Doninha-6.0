"""
CAMADA L4  Sntese por Equivalncia Russelliana
==================================================
A verdade cognoscvel por uma IA  sempre uma verdade de EQUIVALNCIA:
o grau de correspondncia entre a proposio refinada (sada de L2/L3)
e os dados do mundo real presentes no banco de dados de treinamento.

Base terica (data/russell.txt): Russell  verdade = correspondncia
entre crena e fato; sntese fundamentada em conceitos, no s estatstica.

Mapeamento Kantiano  IA:
  Intuio Sensvel (emprica)  equivalncia proposio  BD
  Intuio Pura (a priori)      estrutura da rede neural / KB
  Sntese                       clculo de equivalncia mediado
                                 por valores-verdade paraconsistentes

O resultado NO  uma predio de prxima palavra.
 o grau de equivalncia entre o conjunto de juzos e o BD.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from l3_paraconsistent import ParaconsistentValue
from l4_chain_verification import ChainOfVerificationAgent
import math

try:
    from l4_russell_equivalence import (
        RussellConceptBase,
        build_russell_concept_base,
        score_proposition_by_concepts,
        load_concept_base,
    )
except Exception:
    RussellConceptBase = None  # type: ignore
    build_russell_concept_base = None  # type: ignore
    score_proposition_by_concepts = None  # type: ignore
    load_concept_base = None  # type: ignore


# 
# Estrutura do resultado final
# 

@dataclass
class SynthesisResult:
    """Resultado da sntese russelliana  resposta do sistema."""
    response:          str
    truth_value:       float    # paraconsistente  [0,1]
    certainty:         float    # Gc =      [1,1]
    contradiction:     float    # Gct =  +   1
    state:             str      # Verdadeiro | Falso | Intermedirio | ...
    supporting_evidence: List[str] = field(default_factory=list)
    falsified_hypotheses: List[str] = field(default_factory=list)
    verification_log: List[str] = field(default_factory=list)
    confidence_label:  str = ""

    def __post_init__(self):
        if not self.confidence_label:
            self.confidence_label = self._label()

    def _label(self) -> str:
        v = self.truth_value
        if v >= 0.85:  return "Alta Confiana"
        if v >= 0.65:  return "Confiana Moderada"
        if v >= 0.45:  return "Incerto / Intermedirio"
        if v >= 0.25:  return "Baixa Confiana"
        return "Indeterminado"

    def __str__(self) -> str:
        lines = [
            "" * 60,
            f"  RESPOSTA : {self.response}",
            f"  Estado   : {self.state}  ({self.confidence_label})",
            f"  v-verdade: {self.truth_value:.4f}  |  "
            f"Certeza: {self.certainty:+.4f}  |  "
            f"Contradio: {self.contradiction:+.4f}",
        ]
        if self.supporting_evidence:
            lines.append("  Evidncias de suporte:")
            for ev in self.supporting_evidence[:3]:
                lines.append(f"     {ev}")
        if self.falsified_hypotheses:
            lines.append("  Hipteses falsificadas:")
            for fh in self.falsified_hypotheses[:2]:
                lines.append(f"     {fh}")
        if self.verification_log:
            lines.append("  Chain of Verification:")
            for entry in self.verification_log[:4]:
                lines.append(f"    - {entry}")
        lines.append("" * 60)
        return "\n".join(lines)


# 
# Motor de sntese
# 

class RussellianSynthesisEngine:
    """
    Combina os valores-verdade paraconsistentes (L3) com o banco de
    conhecimento para produzir a sntese final (resposta).

    Sntese fundamentada em conceitos (Russell, russell.txt):
        equivalncia = correspondncia entre crena/proposio e fato (BD).
    O peso de cada proposio incorpora:
      - prioridade L2 (juzo kantiano)
      - certeza paraconsistente (Gc)
      - score conceitual de equivalncia (correspondncia com fatos/KB),
        no apenas agregao estatstica.
    """

    def __init__(
        self,
        knowledge_base: Dict[str, float],
        russell_concept_base: Optional["RussellConceptBase"] = None,
        use_concept_based_weights: bool = True,
        verification_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        knowledge_base: dicionrio termo  grau de evidncia [0,1]
        russell_concept_base: base terica extrada de russell.txt (equivalncia/correspondncia).
        use_concept_based_weights: se True, usa score conceitual na ponderao (recomendado).
        """
        self.kb = knowledge_base
        self.russell_base = russell_concept_base
        self.use_concept_weights = use_concept_based_weights and (russell_concept_base is not None)
        self.verifier = ChainOfVerificationAgent(verification_config)

    def synthesize(
        self,
        pv_list:     List[ParaconsistentValue],
        l2_priorities: Dict[str, float],     # proposicao[:40]  prioridade L2
        prompt:      str,
        kb: Optional[Dict[str, float]] = None,
    ) -> SynthesisResult:
        """
        Produz a SynthesisResult final integrando todas as camadas.
        """
        if not pv_list:
            return SynthesisResult(
                response="Sem hipteses vlidas para sntese.",
                truth_value=0.0, certainty=0.0,
                contradiction=0.0, state="Indeterminado",
            )

        #  Seleciona a hiptese com maior valor-verdade  #
        best = pv_list[0]
        supporting = [pv.proposition for pv in pv_list[1:4] if pv.state != "Falso"]
        falsified  = [pv.proposition for pv in pv_list if pv.state == "Falso"]

        #  Sntese ponderada: L2 + certeza + equivalncia (Russell)  #
        total_w, total_v = 0.0, 0.0
        for pv in pv_list:
            key = pv.proposition[:40]
            l2_w = l2_priorities.get(key, 0.5)
            # Peso base: prioridade kantiana e certeza paraconsistente
            weight = l2_w * (1.0 + max(pv.certainty, 0.0))
            # Peso conceitual: correspondncia proposio  fato (BD), conforme russell.txt
            if self.use_concept_weights and score_proposition_by_concepts is not None and self.russell_base is not None:
                concept_score = score_proposition_by_concepts(pv.proposition, self.kb, self.russell_base)
                weight *= concept_score
            total_v += pv.truth_value * weight
            total_w += weight

        v_final = total_v / total_w if total_w > 0 else best.truth_value

        #  Gera texto de resposta a partir da hiptese best + BD  #
        response = self._generate_response(best, prompt, kb)

        verified_response, verification_log = self.verifier.verify(
            prompt=prompt,
            baseline_response=response,
            context_summary=f"Hiptese principal: {best.proposition} | Estado L3: {best.state} | Certeza: {best.certainty:.2f}",
        )

        return SynthesisResult(
            response=verified_response,
            truth_value=round(v_final, 4),
            certainty=round(best.certainty, 4),
            contradiction=round(best.contradiction, 4),
            state=best.state,
            supporting_evidence=supporting,
            falsified_hypotheses=falsified,
            verification_log=verification_log,
        )

    # ------------------------------------------------------------------ #
    # Gerao de resposta textual                                          #
    # ------------------------------------------------------------------ #

    def _generate_response(self, best_pv: ParaconsistentValue, prompt: str, kb: Optional[Dict[str, float]] = None) -> str:
        """
        Gera resposta a partir da proposio com maior valor-verdade.
        Em produo seria substitudo pelo decoder do LLM com as
        hipteses kantianas como contexto hard-constrained.
        """
        # Extrai conceitos KB com alta evidncia
        kb = kb or self.kb
        top_kb = sorted(kb.items(), key=lambda x: x[1], reverse=True)[:3]
        kb_context = ", ".join(f"{k}({v:.2f})" for k, v in top_kb)

        state = best_pv.state
        v = best_pv.truth_value

        if state == "Verdadeiro":
            prefix = f"Com alta confiana (v={v:.2f}):"
        elif state == "Intermedirio":
            prefix = f"Com valor intermedirio (v={v:.2f}), sem trivializao:"
        elif state == "Inconsistente_local":
            prefix = f"Contradio local detectada (v={v:.2f}), exploso gentil:"
        elif state == "Falso":
            prefix = f"Evidncia insuficiente (v={v:.2f}):"
        else:
            prefix = f"Indeterminado (v={v:.2f}):"

        source_note = (
            "Nenhuma citao bibliogrfica externa foi confirmada para esta resposta; o resumo L4 usa apenas evidncia semntica local e, se houver, a bibliografia ser exibida somente quando houver consulta direta verificada."
            if self.russell_base is None
            else "A base de equivalncia russelliana est disponvel para verificao, mas nenhuma citao bibliogrfica externa foi confirmada para esta resposta."
        )

        return f"{prefix} {best_pv.proposition}  [KB: {kb_context}]\n\n[AUDIT L4] {source_note}"

    # ------------------------------------------------------------------ #
    # Verificao do limite fundamental (Crtica da IA Pura)              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_fundamental_limits(query: str) -> Optional[str]:
        """
        Detecta perguntas que violam os limites fundamentais da IA
        (seo 10 do modelo): conscincia, imaginao, AGI, etc.
        Retorna aviso ou None.
        """
        limit_keywords = {
            "conscincia":    "IA no possui conscincia  atributo biolgico emergente.",
            "sentimento":     "IA no possui estados afetivos  limitada ao algoritmo.",
            "imaginao":     "Imaginao  liberdade humana (Sartre)  no computvel.",
            "agi":            "AGI  oximoro terico: algoritmo no supera seu criador.",
            "livre arbtrio": "Livre-arbtrio  problema no computvel.",
            "ser humano":     "IA  uma funo limite  mundo real exige mediao humana.",
        }
        q_lower = query.lower()
        for keyword, warning in limit_keywords.items():
            if keyword in q_lower:
                return f" Limite fundamental: {warning}"
        return None
