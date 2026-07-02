"""
Base terica russelliana para a camada L4  Equivalncia e Correspondncia
============================================================================
Utiliza o arquivo data/russell.txt (Bertrand Russell, The Problems of Philosophy)
para fundamentar a sntese L4 no conceito de EQUIVALNCIA como correspondncia
entre crena/proposio e fato, e no apenas em agregao estatstica.

Conceitos extrados do Cap. XII (Truth and Falsehood):
  - Verdade = correspondncia entre crena e fato.
  - Fato = unidade complexa formada pelos objetos da crena na mesma ordem.
  - Crena verdadeira quando existe fato correspondente; falsa quando no existe.
  - Propriedade extrnseca: a verdade depende da relao da crena com algo externo.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


#  Conceitos russellianos (extrados do texto) 
# Termos que indicam alta relevncia para equivalncia/correspondncia
CORRESPONDENCE_TERMS = [
    "correspondence", "correspond", "corresponding", "corresponds",
    "equivalence", "equivalent", "match", "accord", "agree", "fact",
    "belief", "true", "truth", "false", "falsehood", "beliefs", "facts",
    "object-terms", "object-relation", "complex unity", "constituents",
    "judgement", "judging", "sense-data", "physical object",
]
# Normalizados para matching em portugus/ingls
EQUIVALENCE_CONCEPTS_PT = [
    "correspondncia", "equivalncia", "crena", "fato", "verdade", "falsidade",
    "juzo", "objeto", "termos", "relao", "unidade", "complexo",
    "dado sensvel", "proposio", "conhecimento",
]


@dataclass
class RussellConceptBase:
    """
    Base de conceitos extrada de russell.txt para fundamentar a sntese L4.
    Permite ponderar proposies por alinhamento terico (correspondncia com fatos)
    e no apenas por estatstica.
    """
    # Trechos do texto sobre verdade/correspondncia (cap. XII e adjacentes)
    key_passages: List[str] = field(default_factory=list)
    # Termos do texto com peso conceitual (relevncia para equivalncia)
    term_weights: Dict[str, float] = field(default_factory=dict)
    # Princpio em forma de texto (para auditoria/interpretao)
    principle_summary: str = ""

    def concept_weight_for_terms(self, terms: List[str]) -> float:
        """
        Peso conceitual para um conjunto de termos: quanto mais os termos
        aparecem na base russelliana, mais a proposio  tratada como
        alinhada  teoria da equivalncia (correspondncia crenafato).
        """
        if not self.term_weights:
            return 1.0
        total = 0.0
        count = 0
        for t in terms:
            t_lower = t.lower().strip()
            if t_lower in self.term_weights:
                total += self.term_weights[t_lower]
                count += 1
        if count == 0:
            return 1.0
        return 1.0 + (total / count) * 0.5  # modulao suave


def _normalize_word(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def load_russell_text(path: Optional[str] = None) -> str:
    """Carrega o contedo de data/russell.txt."""
    if path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "data", "russell.txt")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_chapter_xii(content: str) -> str:
    """Extrai o captulo XII (Truth and Falsehood) e trechos adjacentes relevantes."""
    start = content.find("CHAPTER XII")
    if start == -1:
        start = content.find("TRUTH AND FALSEHOOD")
    if start == -1:
        return content[:15000]  # fallback: incio do livro
    end = content.find("CHAPTER XIV", start)
    if end == -1:
        end = content.find("CHAPTER XV", start)
    if end == -1:
        end = len(content)
    return content[start:end]


def extract_equivalence_passages(content: str) -> List[str]:
    """Extrai trechos que definem equivalncia/correspondncia."""
    chapter_xii = extract_chapter_xii(content)
    # Frases que contm os conceitos centrais
    sentences = re.split(r"[.!?]\s+", chapter_xii)
    key = []
    for s in sentences:
        s_lower = s.lower()
        if any(
            x in s_lower
            for x in (
                "correspondence",
                "correspond",
                "belief",
                "fact",
                "true",
                "false",
                "complex unity",
                "object-terms",
                "object-relation",
            )
        ):
            key.append(s.strip())
    return key[:50]  # limite razovel


def build_term_weights_from_russell(content: str) -> Dict[str, float]:
    """
    Constri pesos por termo a partir do texto de Russell: termos que aparecem
    em contextos de verdade/correspondncia recebem peso maior.
    """
    chapter = extract_chapter_xii(content)
    words = re.findall(r"[a-z]+", chapter.lower())
    # Frequncia no captulo de verdade
    freq: Dict[str, int] = {}
    for w in words:
        w = _normalize_word(w)
        if len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
    # Normalizar para [0.2, 1.0] por relevncia conceitual
    concept_set = set(
        _normalize_word(t) for t in CORRESPONDENCE_TERMS + EQUIVALENCE_CONCEPTS_PT
    )
    max_f = max(freq.values()) if freq else 1
    term_weights: Dict[str, float] = {}
    for w, c in freq.items():
        if w in concept_set:
            term_weights[w] = 0.5 + 0.5 * (c / max_f)
        else:
            term_weights[w] = 0.2 + 0.3 * (c / max_f)
    return term_weights


def build_russell_concept_base(path: Optional[str] = None) -> RussellConceptBase:
    """
    Treina/constroi a base de conceitos russellianos a partir de russell.txt.
    Usado pela L4 para sntese fundamentada em equivalncia (correspondncia).
    """
    content = load_russell_text(path)
    passages = extract_equivalence_passages(content)
    term_weights = build_term_weights_from_russell(content)
    summary = (
        "Truth consists in correspondence between belief and fact. "
        "A belief is true when there is a corresponding fact (complex unity of the objects of the belief). "
        "Truth and falsehood are extrinsic properties: they depend on the relation of the belief to outside things."
    )
    return RussellConceptBase(
        key_passages=passages,
        term_weights=term_weights,
        principle_summary=summary,
    )


def score_proposition_by_concepts(
    proposition: str,
    knowledge_base: Dict[str, float],
    concept_base: RussellConceptBase,
) -> float:
    """
    Score conceitual da proposio: grau em que ela se alinha  teoria da
    equivalncia (correspondncia com fatos/BD), no apenas estatstica.

    - Termos da proposio que esto no KB com alta evidncia indicam
      melhor "correspondncia" com o mundo (fatos).
    - Termos que aparecem na base russelliana aumentam o peso terico.
    """
    words = re.findall(r"[a-z]+", proposition.lower())
    terms = [_normalize_word(w) for w in words if len(w) > 2]

    # 1) Alinhamento com fatos (KB): termos da proposio presentes no BD
    kb_match = 0.0
    n = 0
    for t in terms:
        for kb_term, ev in knowledge_base.items():
            if _normalize_word(kb_term) == t or t in _normalize_word(kb_term):
                kb_match += ev
                n += 1
                break
    fact_alignment = (kb_match / n) if n > 0 else 0.5  # neutro se nenhum termo no KB

    # 2) Peso conceitual russelliano (termos da teoria)
    concept_weight = concept_base.concept_weight_for_terms(terms)

    # Combinao: correspondncia com fatos (BD) + alinhamento terico
    return (0.7 * fact_alignment + 0.3 * concept_weight)


def save_concept_base(base: RussellConceptBase, path: str) -> None:
    """Salva a base de conceitos para uso posterior da L4."""
    import json
    data = {
        "principle_summary": base.principle_summary,
        "key_passages": base.key_passages[:20],
        "term_weights": base.term_weights,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_concept_base(path: str) -> RussellConceptBase:
    """Carrega base de conceitos previamente construda."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RussellConceptBase(
        principle_summary=data.get("principle_summary", ""),
        key_passages=data.get("key_passages", []),
        term_weights=data.get("term_weights", {}),
    )
