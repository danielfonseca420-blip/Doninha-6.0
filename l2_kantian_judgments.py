"""
CAMADA L2  Tbua de Juzos Kantianos
======================================
Antes de qualquer clculo estatstico o prompt  destrinchado nas
doze categorias da Tbua dos Juzos (Kritik der reinen Vernunft, 9).

Dimenses:
  Quantidade   Universal | Particular | Singular
  Qualidade    Afirmativo | Negativo | Infinito
  Relao      Categrico | Hipottico | Disjuntivo
  Modalidade   Problemtico | Assertrico | Apodtico

Cada hiptese gerada recebe um peso de prioridade; o Juzo Singular
Afirmativo Assertrico tem prioridade mxima ( a resposta-alvo).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging
import re

from l1_concept_table import ConceptNode, ConceptTable

logger = logging.getLogger(__name__)

try:
    from prompt_engineering import get_layer_prompt
except Exception:
    get_layer_prompt = None  # type: ignore

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


# 
# Estruturas de dados
# 

@dataclass
class EpistemicClassification:
    """Classificao epistemolgica sem restrio T+I+F=1."""
    truth: float = 0.0           # T  [0,1]  grau de verdade
    indeterminacy: float = 0.0   # I  [0,1]  grau de indeterminao
    falsity: float = 0.0         # F  [0,1]  grau de falsidade
    classification: str = "indeterminado"  # paraconsistncia | incompletude | vagueza | assertiva_confiante | indeterminado

    def __post_init__(self):
        self.classification = self._classify()

    def _classify(self) -> str:
        """Aplica regras epistemolgicas para classificar."""
        if self.truth + self.falsity > 1.0:
            return "paraconsistncia"
        if self.truth + self.indeterminacy + self.falsity < 1.0:
            return "incompletude"
        if self.indeterminacy > 0.6:
            return "vagueza"
        if self.truth > 0.7 and self.indeterminacy < 0.2 and self.falsity < 0.2:
            return "assertiva_confiante"
        return "indeterminado"

    def __str__(self) -> str:
        return (
            f"T={self.truth:.2f} I={self.indeterminacy:.2f} F={self.falsity:.2f} "
            f"[{self.classification}]"
        )


@dataclass
class KantianJudgment:
    """Uma proposio refinada segundo a tbua dos juzos."""
    quantidade:  str   # Universal | Particular | Singular
    qualidade:   str   # Afirmativo | Negativo | Infinito
    relacao:     str   # Categrico | Hipottico | Disjuntivo
    modalidade:  str   # Problemtico | Assertrico | Apodtico
    proposicao:  str   # texto da hiptese
    prioridade:  float = 0.0  # 0.0  1.0  (1.0 = resposta-alvo)
    epistemic_classification: EpistemicClassification = field(default_factory=EpistemicClassification)

    def __str__(self) -> str:
        return (
            f"[{self.quantidade}/{self.qualidade}/"
            f"{self.relacao}/{self.modalidade}] "
            f"(pri={self.prioridade:.2f}) {self.epistemic_classification} {self.proposicao}"
        )


@dataclass
class SyntaxProfile:
    """
    Perfil sinttico mnimo extrado do enunciado segundo a gramtica
    (aproximao heurstica baseada em listas inspiradas em grammar.txt).
    """
    quantifier_subject: Optional[str] = None   # "all", "some", "this", etc.
    quantifier_predicate: Optional[str] = None
    has_negation: bool = False
    has_infinite_like: bool = False           # construes do tipo "not-X"
    is_conditional: bool = False              # presena de "if", "then"
    is_disjunctive: bool = False              # presena de "or"
    modality_markers: Tuple[str, ...] = ()    # "can", "must", "might", etc.


# 
# Regras de prioridade entre modalidades (herana da "parte fraca")
# 
MODALIDADE_PESO = {
    "Apodtico":    1.0,
    "Assertrico":  0.7,
    "Problemtico": 0.4,
}
QUANTIDADE_PESO = {
    "Singular":   1.0,
    "Particular": 0.6,
    "Universal":  0.3,
}
QUALIDADE_PESO = {
    "Afirmativo": 1.0,
    "Infinito":   0.6,
    "Negativo":   0.4,
}
RELACAO_PESO = {
    "Categrico":  1.0,
    "Hipottico":  0.7,
    "Disjuntivo":  0.5,
}


def _priority(j: KantianJudgment) -> float:
    return (
        MODALIDADE_PESO[j.modalidade]
        * QUANTIDADE_PESO[j.quantidade]
        * QUALIDADE_PESO[j.qualidade]
        * RELACAO_PESO[j.relacao]
    )


# 
# Motor de gerao de juzos
# 

class BERTAssertionClassifier:
    """Classificador baseado em BERT para juzos assertricos.

    Processa proposies e retorna (T, I, F) sem restrio T+I+F=1,
    capturando paraconsistncia, incompletude e vagueza.
    """

    DOMAIN_CANDIDATES = {
        "fsico": [
            "empiricamente verificado",
            "teoricamente plausvel",
            "logicamente contraditrio",
            "empiricamente indeterminado",
        ],
        "lgica": [
            "logicamente contraditrio",
            "teoricamente plausvel",
            "empiricamente indeterminado",
            "empiricamente verificado",
        ],
        "cognitivo": [
            "empiricamente verificado",
            "teoricamente plausvel",
            "empiricamente indeterminado",
            "logicamente contraditrio",
        ],
        "filosfico": [
            "teoricamente plausvel",
            "empiricamente indeterminado",
            "logicamente contraditrio",
            "empiricamente verificado",
        ],
        "geral": [
            "teoricamente plausvel",
            "empiricamente verificado",
            "logicamente contraditrio",
            "empiricamente indeterminado",
        ],
    }
    GENERIC_CANDIDATES = [
        "empiricamente verificado",
        "teoricamente plausvel",
        "logicamente contraditrio",
        "empiricamente indeterminado",
    ]

    def __init__(self):
        self.classifier = None
        if pipeline is not None:
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="cross-encoder/nli-MiniLM2-L6-H768",
                )
            except Exception as exc:
                logger.warning(
                    "Classificador BERT L2 indisponível (%s); usando heurística lexical para classificação epistemológica.",
                    exc,
                )
        else:
            logger.warning(
                "Pipeline do Transformers indisponível; usando heurística lexical para classificação epistemológica da L2."
            )

    def classify(self, proposition: str, domain: str = "geral") -> EpistemicClassification:
        """Classifica uma proposio em (T, I, F) usando candidatos por domnio."""
        if self.classifier is None:
            logger.warning(
                "Classificador BERT L2 indisponível; usando heurística lexical para a proposição '%s'.",
                proposition[:120],
            )
            return self._heuristic_classify(proposition)

        candidates = self.DOMAIN_CANDIDATES.get(domain, self.GENERIC_CANDIDATES)
        try:
            result = self.classifier(proposition, candidates, multi_class=True)
            scores = {label: score for label, score in zip(result["labels"], result["scores"])}
            return EpistemicClassification(
                truth=scores.get("empiricamente verificado", 0.0)
                + scores.get("teoricamente plausvel", 0.0) * 0.75,
                indeterminacy=scores.get("empiricamente indeterminado", 0.0),
                falsity=scores.get("logicamente contraditrio", 0.0),
            )
        except Exception as exc:
            logger.warning(
                "Falha ao executar classificador BERT L2 para '%s'; usando heurística lexical. Erro: %s",
                proposition[:120],
                exc,
            )
            return self._heuristic_classify(proposition)

    def _heuristic_classify(self, proposition: str) -> EpistemicClassification:
        """Classificao heurstica quando BERT no est disponvel."""
        text = proposition.lower()
        t, i, f = 0.5, 0.3, 0.2

        if "verdadeiro" in text or "" in text or "sempre" in text:
            t = 0.8
            i = 0.1
            f = 0.1
        elif "falso" in text or "nunca" in text or "no " in text:
            t = 0.1
            i = 0.1
            f = 0.8
        elif "pode" in text or "talvez" in text or "possvel" in text:
            t = 0.4
            i = 0.5
            f = 0.3
        elif "contraditrio" in text or "e" in text and "ou" in text:
            t = 0.6
            i = 0.3
            f = 0.7
        elif "indeterminado" in text or "indefinido" in text:
            t = 0.3
            i = 0.7
            f = 0.3
        elif "incompleto" in text or "insuficiente" in text:
            t = 0.2
            i = 0.6
            f = 0.2

        return EpistemicClassification(truth=round(t, 3), indeterminacy=round(i, 3), falsity=round(f, 3))


class KantianJudgmentEngine:
    """
    Recebe um prompt e a lista de ConceptNodes extrados por L1 e devolve
    as 12 hipteses estruturadas segundo a tbua kantiana.
    
    Para juizos assertricos, aplica classificao BERT com (T, I, F).
    """

    def __init__(self, concept_table: ConceptTable) -> None:
        self.ct = concept_table
        self.bert_classifier = BERTAssertionClassifier()
        self.last_cot_prompt: str = ""

    # ------------------------------------------------------------------ #
    # API pblica                                                          #
    # ------------------------------------------------------------------ #

    def refine(self, prompt: str, concepts: List[ConceptNode]) -> List[KantianJudgment]:
        """
        Gera as hipteses kantianas para o prompt e as ordena por
        prioridade descendente.
        """
        if get_layer_prompt is not None:
            concepts_summary = "; ".join(c.term for c in concepts[:8])
            self.last_cot_prompt = get_layer_prompt("l2", prompt, {"concepts_summary": concepts_summary})
        subject, predicates = self._parse_prompt(prompt, concepts)
        syntax = self._analyze_syntax(prompt)
        judgments: List[KantianJudgment] = []

        domain = self._infer_domain(concepts)
        for pred in predicates:
            antonym = self._antonym_of(pred, concepts)
            hypernym = self._hypernym_of(pred, concepts)

            #  Juzo principal guiado pela gramtica 
            qt = self._infer_quantity(syntax)
            ql = self._infer_quality(syntax)
            rel = self._infer_relation(syntax)
            mod = self._infer_modality(syntax)

            base_prop = f"{subject}  {pred}"
            if syntax.has_negation and antonym:
                base_prop = f"{subject} no  {antonym}"

            j = self._make(qt, ql, rel, mod, base_prop)
            if j.modalidade == "Assertrico":
                j.epistemic_classification = self.bert_classifier.classify(base_prop, domain=domain)
            judgments.append(j)

            #  Variaes cannicas (mantidas, mas ancoradas em L1) 
            judgments.append(self._make(
                "Universal", "Afirmativo", "Categrico", "Apodtico",
                f"Todo(a) {subject} com propriedade extrema  {pred}",
            ))
            judgments.append(self._make(
                "Particular", "Afirmativo", "Hipottico", "Problemtico",
                f"Algum(a) {subject} pode ser {pred}",
            ))
            j1 = self._make(
                "Singular", "Afirmativo", "Categrico", "Assertrico",
                f"Este(a) {subject} especfico  {pred}",
            )
            j1.epistemic_classification = self.bert_classifier.classify(j1.proposicao, domain=domain)
            judgments.append(j1)

            prop2 = (f"Este(a) {subject} no  {antonym}" if antonym else
                     f"Este(a) {subject} no possui a propriedade oposta a {pred}")
            j2 = self._make("Singular", "Negativo", "Categrico", "Assertrico", prop2)
            j2.epistemic_classification = self.bert_classifier.classify(j2.proposicao, domain=domain)
            judgments.append(j2)
            j3 = self._make(
                "Singular", "Infinito", "Categrico", "Assertrico",
                f"Este(a) {subject}  no-{antonym}" if antonym else
                f"Este(a) {subject}  indeterminado em relao a {pred}",
            )
            j3.epistemic_classification = self.bert_classifier.classify(j3.proposicao, domain=domain)
            judgments.append(j3)

            judgments.append(self._make(
                "Universal", "Afirmativo", "Hipottico", "Apodtico",
                f"Se {subject} possui condio X, ento  {pred}",
            ))
            j4 = self._make(
                "Universal", "Afirmativo", "Disjuntivo", "Assertrico",
                f"{subject}  {pred} OU {antonym} OU intermedirio"
                if antonym else f"{subject}  {pred} ou outra propriedade",
            )
            j4.epistemic_classification = self.bert_classifier.classify(j4.proposicao, domain=domain)
            judgments.append(j4)

            judgments.append(self._make(
                "Singular", "Afirmativo", "Categrico", "Problemtico",
                f"Este(a) {subject} pode ser {pred}?",
            ))
            j5 = self._make(
                "Singular", "Afirmativo", "Hipottico", "Assertrico",
                f"Este(a) {subject}  {pred} em razo das condies observadas",
            )
            j5.epistemic_classification = self.bert_classifier.classify(j5.proposicao, domain=domain)
            judgments.append(j5)
            judgments.append(self._make(
                "Universal", "Afirmativo", "Categrico", "Apodtico",
                f"{subject} deve ser {pred} quando condies necessrias presentes",
            ))

            #  HIPTESES COM INTERMEDIRIOS (hiperonmia) 
            if hypernym:
                j6 = self._make(
                    "Singular", "Afirmativo", "Categrico", "Assertrico",
                    f"Este(a) {subject} pertence  categoria {hypernym}",
                )
                j6.epistemic_classification = self.bert_classifier.classify(j6.proposicao, domain=domain)
                judgments.append(j6)
            if antonym:
                j7 = self._make(
                    "Singular", "Negativo", "Disjuntivo", "Assertrico",
                    f"Este(a) {subject} no  {pred} nem {antonym}: "
                    f"admite valor intermedirio",
                )
                j7.epistemic_classification = self.bert_classifier.classify(j7.proposicao, domain=domain)
                judgments.append(j7)

        # Calcula prioridades e ordena
        for j in judgments:
            j.prioridade = _priority(j)
        judgments.sort(key=lambda j: j.prioridade, reverse=True)
        return judgments

    def _infer_domain(self, concepts: List[ConceptNode]) -> str:
        """Inferncia simples de domnio majoritrio a partir dos conceitos extrados."""
        if not concepts:
            return "geral"
        domain_counts = {}
        for concept in concepts:
            domain = concept.domain.lower().strip() if concept.domain else "geral"
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return max(domain_counts, key=domain_counts.get)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make(qt, ql, rel, mod, prop) -> KantianJudgment:
        j = KantianJudgment(
            quantidade=qt, qualidade=ql, relacao=rel,
            modalidade=mod, proposicao=prop,
        )
        j.prioridade = _priority(j)
        return j

    @staticmethod
    def _parse_prompt(prompt: str, concepts: List[ConceptNode]) -> Tuple[str, List[str]]:
        """
        Extrai sujeito e predicados candidatos do prompt de forma simples.
        Em produo seria substitudo por um parser sinttico.
        """
        tokens = re.findall(r"[a-zA-Z]+", prompt.lower())
        known = {c.term.lower() for c in concepts}
        subject = tokens[0] if tokens else "entidade"
        predicates = [t for t in tokens[1:] if t in known] or ["indeterminado"]
        return subject, predicates

    # ------------------------------------------------------------------ #
    # Anlise sinttica inspirada em grammar.txt                         #
    # ------------------------------------------------------------------ #

    def _analyze_syntax(self, prompt: str) -> SyntaxProfile:
        """
        Extrai um perfil sinttico mnimo usando listas de palavras
        alinhadas aos captulos de determiners, modals, negatives e
        conjunctions da grammar COBUILD.
        """
        text = prompt.lower()
        tokens = re.findall(r"[a-z]+", text)

        quant_all = {"all", "every", "each"}
        quant_some = {"some", "many", "several", "few", "a few"}
        quant_singular = {"this", "that", "these", "those", "a", "an", "one"}

        neg_markers = {"not", "no", "never", "none", "nothing", "nowhere"}
        infinite_patterns = {"not-", "non-"}

        cond_markers = {"if", "provided", "unless", "whenever", "as long as"}
        disj_markers = {"or", "either"}

        modal_poss = {"can", "could", "may", "might"}
        modal_necess = {"must", "have to", "need to", "should", "ought"}

        has_neg = any(tok in neg_markers for tok in tokens)
        has_inf = any(pat in text for pat in infinite_patterns)
        is_cond = any(tok in cond_markers for tok in tokens)
        is_disj = any(tok in disj_markers for tok in tokens)

        mods: list[str] = []
        for tok in tokens:
            if tok in modal_poss or tok in modal_necess:
                mods.append(tok)

        q_subj: Optional[str] = None
        q_pred: Optional[str] = None

        if tokens:
            first = tokens[0]
            if first in quant_all:
                q_subj = "all"
            elif first in quant_some:
                q_subj = "some"
            elif first in quant_singular:
                q_subj = "this"

        return SyntaxProfile(
            quantifier_subject=q_subj,
            quantifier_predicate=q_pred,
            has_negation=has_neg,
            has_infinite_like=has_inf,
            is_conditional=is_cond,
            is_disjunctive=is_disj,
            modality_markers=tuple(mods),
        )

    def _infer_quantity(self, syntax: SyntaxProfile) -> str:
        if syntax.quantifier_subject == "all":
            return "Universal"
        if syntax.quantifier_subject == "some":
            return "Particular"
        if syntax.quantifier_subject == "this":
            return "Singular"
        return "Singular"

    def _infer_quality(self, syntax: SyntaxProfile) -> str:
        if syntax.has_infinite_like:
            return "Infinito"
        if syntax.has_negation:
            return "Negativo"
        return "Afirmativo"

    def _infer_relation(self, syntax: SyntaxProfile) -> str:
        if syntax.is_conditional:
            return "Hipottico"
        if syntax.is_disjunctive:
            return "Disjuntivo"
        return "Categrico"

    def _infer_modality(self, syntax: SyntaxProfile) -> str:
        markers = {m for m in syntax.modality_markers}
        if any(m in {"must", "have", "need", "should", "ought"} for m in markers):
            return "Apodtico"
        if any(m in {"can", "could", "may", "might"} for m in markers):
            return "Problemtico"
        return "Assertrico"

    def _antonym_of(self, term: str, concepts: List[ConceptNode]) -> str:
        node = self.ct.get(term)
        if node and node.antonyms:
            return node.antonyms[0]
        return ""

    def _hypernym_of(self, term: str, concepts: List[ConceptNode]) -> str:
        node = self.ct.get(term)
        if node and node.hypernyms:
            return node.hypernyms[0]
        return ""
