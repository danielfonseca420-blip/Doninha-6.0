"""
CAMADA L1  Tbua de Conceitos (Aristteles: Categorias)
=========================================================
Mapeia cada termo do prompt a relaes semnticas fixas:
  - Sinonmia   : mesma denotao
  - Antonmia   : oposio semntica direta
  - Hiponmia   : relao especfico  geral
  - Homonmia   : mesma forma, sentidos distintos
  - Paronmia   : semelhana formal, sentidos distintos

As relaes so BINRIAS nesta camada  elimina a necessidade de
defuzzificao posterior na camada L3.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re
import json
import os
from knowledge_base import get_domain_knowledge_base

try:
    from prompt_engineering import get_layer_prompt
except Exception:
    get_layer_prompt = None  # type: ignore

try:
    import spacy
except Exception:
    spacy = None  # type: ignore

try:
    from agente_busca_web import run_search_for_context as _run_search_for_context
except Exception:
    _run_search_for_context = None


@dataclass
class ConceptNode:
    """Um conceito na tbua, com todas as suas relaes."""
    term: str
    definition: str = ""
    synonyms:   List[str] = field(default_factory=list)
    antonyms:   List[str] = field(default_factory=list)
    hyponyms:   List[str] = field(default_factory=list)   # mais especficos
    hypernyms:  List[str] = field(default_factory=list)   # mais gerais
    homonyms:   Dict[str, str] = field(default_factory=dict)  # sentido  definio
    paronyms:   List[str] = field(default_factory=list)
    domain:     str = "geral"
    application_context: str = ""
    canonical_source: str = ""
    canonical_context: Dict[str, str] = field(default_factory=dict)  # Verificao de atribuio cannica


class SpacySemanticTermExtractor:
    """Extrator semantico opcional para apoiar a L1 com lemas e sintagmas."""

    MODEL_NAMES = {
        "pt": "pt_core_news_sm",
        "en": "en_core_web_sm",
    }

    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}

    def extract_terms(
        self,
        text: str,
        languages: Optional[List[str]] = None,
        enabled: bool = True,
    ) -> List[str]:
        if not enabled or spacy is None or not text.strip():
            return []

        ordered_languages = self._ordered_languages(text, languages)
        terms: List[str] = []
        seen = set()
        for language in ordered_languages:
            nlp = self._load(language)
            if nlp is None:
                continue
            doc = nlp(text)
            self._add_noun_chunks(doc, terms, seen)
            for token in doc:
                if self._is_semantic_token(token):
                    self._add_term(token.text, terms, seen)
                    lemma = getattr(token, "lemma_", "")
                    if lemma and lemma != token.text:
                        self._add_term(lemma, terms, seen)
        return terms

    def _ordered_languages(self, text: str, languages: Optional[List[str]]) -> List[str]:
        configured = [lang for lang in (languages or ["pt", "en"]) if lang in self.MODEL_NAMES]
        if not configured:
            configured = ["pt", "en"]
        detected = self._detect_language(text)
        ordered = [detected] if detected in configured else []
        ordered.extend(lang for lang in configured if lang not in ordered)
        return ordered

    def _detect_language(self, text: str) -> str:
        lower = text.lower()
        portuguese_markers = [
            " o ", " a ", " os ", " as ", " que ", " de ", " da ", " do ",
            " para ", " com ", " nao ", " no ", " na ", " conhecimento",
            " verdade", " relacao", " semantico",
        ]
        if any(marker in f" {lower} " for marker in portuguese_markers):
            return "pt"
        return "en"

    def _load(self, language: str) -> Optional[Any]:
        if language in self._models:
            return self._models[language]
        model_name = self.MODEL_NAMES.get(language)
        if not model_name or spacy is None:
            return None
        try:
            self._models[language] = spacy.load(model_name)
        except Exception:
            try:
                self._models[language] = spacy.blank(language)
            except Exception:
                self._models[language] = None
        return self._models[language]

    def _add_noun_chunks(self, doc: Any, terms: List[str], seen: set) -> None:
        try:
            chunks = doc.noun_chunks
        except Exception:
            return
        try:
            for chunk in chunks:
                cleaned = self._clean_term(chunk.text)
                if cleaned and len(cleaned.split()) <= 4:
                    self._add_term(cleaned, terms, seen)
        except Exception:
            return

    def _is_semantic_token(self, token: Any) -> bool:
        text = getattr(token, "text", "")
        if not text or not text.isalpha() or len(text) <= 2:
            return False
        if getattr(token, "is_stop", False):
            return False
        pos = getattr(token, "pos_", "")
        return not pos or pos in {"NOUN", "PROPN", "ADJ", "VERB"}

    def _add_term(self, term: str, terms: List[str], seen: set) -> None:
        cleaned = self._clean_term(term)
        if not cleaned:
            return
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            terms.append(cleaned)

    def _clean_term(self, term: str) -> str:
        return re.sub(r"\s+", " ", term.strip().lower())


class ConceptTable:
    """
    Tbua de conceitos fixos.  Em produo seria alimentada por um
    dicionrio / ontologia formal (WordNet-PT, OpenWordNet-PT, etc.).
    Aqui usamos um conjunto seminal suficiente para demonstrar todas
    as camadas do modelo.
    """

    def __init__(self) -> None:
        self._table: Dict[str, ConceptNode] = {}
        self._spacy_extractor = SpacySemanticTermExtractor()
        self.last_cot_prompt: str = ""
        # Tbua seminal em portugus
        self._build_seed_table()
        # Banco de conceitos em ingls aprendido de dicionrio externo (se existir)
        self._load_external_concepts()

    # ------------------------------------------------------------------ #
    # API pblica                                                          #
    # ------------------------------------------------------------------ #

    def get(self, term: str) -> Optional[ConceptNode]:
        return self._table.get(self._normalize(term))

    def _should_use_web_agent_fallback(
        self,
        text: str,
        llm_context: Optional[str],
        concepts: List[ConceptNode],
        config: Optional[Dict] = None,
    ) -> bool:
        if _run_search_for_context is None:
            return False
        if not text.strip():
            return False
        agent_cfg = (config or {}).get("agent", {}) if config else {}
        if agent_cfg.get("use_agent") is False:
            return False
        return not concepts or len(concepts) < 3

    def _fetch_web_context_for_fallback(
        self,
        text: str,
        llm_context: Optional[str],
        config: Optional[Dict] = None,
    ) -> Optional[str]:
        if not self._should_use_web_agent_fallback(text, llm_context, [], config):
            return None
        try:
            web_context = _run_search_for_context(text)
        except Exception:
            return None
        if not web_context:
            return None
        return web_context.strip()

    def extract_concepts(self, text: str, llm_context: Optional[str] = None, domain: str = "geral", config: Optional[Dict] = None) -> List[ConceptNode]:
        """Extrai e retorna os ns de todos os termos encontrados no texto."""
        if get_layer_prompt is not None:
            self.last_cot_prompt = get_layer_prompt("l1", text, {"concepts_summary": ""})
        result = self._extract_known_concepts(text, config)

        web_context = None
        if not result:
            web_context = self._fetch_web_context_for_fallback(text, llm_context, config)
            if web_context:
                search_text = f"{text} {web_context}"
                result = self._extract_known_concepts(search_text, config)
                if llm_context:
                    llm_context = f"{llm_context} {web_context}"
                else:
                    llm_context = web_context

        if result:
            self._enrich_concepts_with_application_context(result, text, llm_context, domain, config)
            combined_text = f"{llm_context.strip()} {text}" if llm_context else text
            result = [
                node for node in result
                if LogicLMSymbolicSolver.is_context_compatible(node, combined_text)
            ]
        return result

    def add(self, node: ConceptNode) -> None:
        self._table[self._normalize(node.term)] = node

    def _extract_known_concepts(self, text: str, config: Optional[Dict] = None) -> List[ConceptNode]:
        seen, result = set(), []
        for term in self._candidate_terms(text, config):
            key = self._normalize(term)
            if key in seen:
                continue
            node = self._table.get(key)
            if node:
                seen.add(key)
                result.append(self._clone_node(node))
        return result

    def _candidate_terms(self, text: str, config: Optional[Dict] = None) -> List[str]:
        l1_cfg = (config or {}).get("l1", {}) if isinstance(config, dict) else {}
        spacy_enabled = l1_cfg.get("spacy_enabled", True)
        languages = l1_cfg.get("spacy_languages", ["pt", "en"])
        if not isinstance(languages, list):
            languages = ["pt", "en"]

        semantic_terms = self._spacy_extractor.extract_terms(
            text,
            languages=languages,
            enabled=bool(spacy_enabled),
        )
        regex_terms = re.findall(r"[a-zA-Z]+", text)
        return semantic_terms + regex_terms

    def relation_type(self, term_a: str, term_b: str) -> str:
        """Retorna o tipo de relao semntica entre dois termos."""
        a = self._normalize(term_a)
        b = self._normalize(term_b)
        node_a = self._table.get(a)
        if not node_a:
            return "desconhecida"
        if b in [self._normalize(s) for s in node_a.synonyms]:
            return "sinonmia"
        if b in [self._normalize(s) for s in node_a.antonyms]:
            return "antonmia"
        if b in [self._normalize(s) for s in node_a.hyponyms]:
            return "hiponmia"
        if b in [self._normalize(s) for s in node_a.hypernyms]:
            return "hiperonmia"
        if b in [self._normalize(s) for s in node_a.paronyms]:
            return "paronmia"
        if b in [self._normalize(k) for k in node_a.homonyms]:
            return "homonmia"
        return "sem_relao_direta"

    # ------------------------------------------------------------------ #
    # Construo da tbua seminal                                          #
    # ------------------------------------------------------------------ #

    def _clone_node(self, node: ConceptNode) -> ConceptNode:
        return ConceptNode(
            term=node.term,
            definition=node.definition,
            synonyms=list(node.synonyms),
            antonyms=list(node.antonyms),
            hyponyms=list(node.hyponyms),
            hypernyms=list(node.hypernyms),
            homonyms=dict(node.homonyms),
            paronyms=list(node.paronyms),
            domain=node.domain,
            application_context="",
            canonical_source=node.canonical_source,
            canonical_context=dict(node.canonical_context),
        )

    def _enrich_concepts_with_application_context(
        self,
        concepts: List[ConceptNode],
        prompt: str,
        llm_context: Optional[str] = None,
        domain: str = "geral",
        config: Optional[Dict] = None,
    ) -> None:
        """Aplica o solver simblico Logic-LM para adicionar contexto de uso aos conceitos."""
        LogicLMSymbolicSolver.enrich(concepts, prompt, llm_context, domain, config)

    def _build_seed_table(self) -> None:
        entries = [
            ConceptNode(
                term="quente",
                definition="Que possui temperatura alta.",
                synonyms=["aquecido", "clido", "morno", "tpido"],
                antonyms=["frio", "gelado", "fresco"],
                hypernyms=["temperatura"],
                hyponyms=["escaldante", "ardente"],
                domain="fsico",
                canonical_source="Newton - Philosophiae Naturalis Principia Mathematica - Livro I",
            ),
            ConceptNode(
                term="frio",
                definition="Que possui temperatura baixa.",
                synonyms=["gelado", "fresco", "frgido"],
                antonyms=["quente", "aquecido", "clido"],
                hypernyms=["temperatura"],
                hyponyms=["congelado", "glacial"],
                domain="fsico",
                canonical_source="Newton - Philosophiae Naturalis Principia Mathematica - Livro I",
            ),
            ConceptNode(
                term="morno",
                definition="Entre quente e frio; tpido.",
                synonyms=["tpido", "ameno"],
                antonyms=["escaldante", "glacial"],
                hypernyms=["temperatura", "quente", "frio"],
                hyponyms=[],
                domain="fsico",
                canonical_source="Galen - De Temperamentis - Seo 3",
            ),
            ConceptNode(
                term="temperatura",
                definition="Grandeza fsica que mede o grau de calor de um corpo.",
                synonyms=["calor", "grau"],
                antonyms=[],
                hypernyms=["grandeza_fsica"],
                hyponyms=["quente", "frio", "morno"],
                domain="fsico",
                canonical_source="Galileu - Discorsi e Dimostrazioni Matematiche - Seo 2",
            ),
            ConceptNode(
                term="gua",
                definition="Substncia H2O, geralmente em estado lquido.",
                synonyms=["H2O", "lquido"],
                antonyms=[],
                hypernyms=["substncia", "fluido"],
                hyponyms=["vapor", "gelo"],
                domain="fsico",
                canonical_source="Newton - Opticks - Definio 19",
            ),
            ConceptNode(
                term="verdadeiro",
                definition="Que est de acordo com os fatos ou a realidade.",
                synonyms=["correto", "real", "factual"],
                antonyms=["falso", "incorreto", "fictcio"],
                hypernyms=["valor_lgico"],
                domain="lgica",
                canonical_source="Aristteles - Metafsica - Livro Gamma",
                canonical_context={
                    "lgica_clssica": "Aristteles - Metafsica: valor de verdade binrio, NO lgica paraconsistente",
                    "epistemologia": "Plato - Teeteto: correspondncia com realidade, NO coerncia pura"
                }
            ),
            ConceptNode(
                term="falso",
                definition="Que no corresponde aos fatos ou  realidade.",
                synonyms=["incorreto", "errado", "fictcio"],
                antonyms=["verdadeiro", "correto", "real"],
                hypernyms=["valor_lgico"],
                domain="lgica",
                canonical_source="Aristteles - Metafsica - Livro Gamma",
                canonical_context={
                    "lgica_clssica": "Aristteles - Metafsica: negao do verdadeiro, NO dialtica hegeliana"
                }
            ),
            ConceptNode(
                term="banco",
                definition="Mvel para sentar; instituio financeira; repositrio de dados.",
                synonyms=[],
                antonyms=[],
                hypernyms=[],
                homonyms={
                    "assento": "mvel para sentar",
                    "financeiro": "instituio financeira",
                    "dados": "repositrio de dados",
                },
                domain="geral",
            ),
            ConceptNode(
                term="eminente",
                definition="Pessoa ilustre ou notvel.",
                synonyms=["ilustre", "notvel"],
                antonyms=[],
                paronyms=["iminente"],
                domain="geral",
            ),
            ConceptNode(
                term="iminente",
                definition="Que est prestes a acontecer.",
                synonyms=["prximo", "imediato"],
                antonyms=[],
                paronyms=["eminente"],
                domain="geral",
            ),
            ConceptNode(
                term="inteligncia",
                definition="Capacidade de compreender, raciocinar e resolver problemas.",
                synonyms=["cognio", "raciocnio", "entendimento"],
                antonyms=["ignorncia", "estupidez"],
                hypernyms=["capacidade_mental"],
                domain="cognitivo",
            ),
            ConceptNode(
                term="conhecimento",
                definition="Ato ou efeito de conhecer; saber, cincia, erudio.",
                synonyms=["saber", "cincia", "erudio"],
                antonyms=["ignorncia", "desconhecimento"],
                hypernyms=["epistemologia"],
                domain="filosfico",
                canonical_context={
                    "epistemologia": "Plato - Teeteto: justificao verdadeira, NO opinio infundada",
                    "kantiano": "Kant - Crtica da Razo Pura: a priori vs a posteriori, NO empirismo puro"
                }
            ),
            ConceptNode(
                term="verdade",
                definition="Conformidade entre o que se diz e o que .",
                synonyms=["veracidade", "factualidade", "realidade"],
                antonyms=["mentira", "falsidade", "iluso"],
                hypernyms=["epistemologia"],
                domain="filosfico",
                canonical_context={
                    "platnico": "Plato - Repblica: ideias eternas, NO relativismo",
                    "aristotlico": "Aristteles - Metafsica: correspondncia, NO coerncia"
                }
            ),
            ConceptNode(
                term="sntese regulativa",
                definition="Princpio que orienta o conhecimento sem constitu-lo.",
                synonyms=["regulativo", "orientador"],
                antonyms=[],
                hypernyms=["epistemologia", "kantismo"],
                domain="filosfico",
                canonical_source="Kant - Crtica da Razo Pura",
                canonical_context={
                    "kantismo": "Kant, CRP: princpio regulativo do conhecimento, NO Russell"
                }
            ),
        ]
        for node in entries:
            self.add(node)

    @staticmethod
    def _normalize(term: str) -> str:
        return term.strip().lower()

    # ------------------------------------------------------------------ #
    # Carregamento de conceitos externos (ex.: dicionrio em ingls)      #
    # ------------------------------------------------------------------ #

    def _load_external_concepts(self) -> None:
        """
        Carrega conceitos adicionais de um banco gerado a partir do
        dicionrio em ingls (arquivo JSON se existir).

        Formato esperado (lista de objetos):
          {
            "term": "abacus",
            "definition": "Frame with beads for calculating...",
            "synonyms": [],
            "antonyms": [],
            "hyponyms": [],
            "hypernyms": [],
            "domain": "geral"
          }
        """
        base_dir = os.path.dirname(__file__) or "."
        json_path = os.path.join(base_dir, "data", "concepts_en.json")
        if not os.path.exists(json_path):
            return
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            return

        for item in items:
            term = item.get("term")
            if not term:
                continue
            node = ConceptNode(
                term=term,
                definition=item.get("definition", ""),
                synonyms=item.get("synonyms", []),
                antonyms=item.get("antonyms", []),
                hyponyms=item.get("hyponyms", []),
                hypernyms=item.get("hypernyms", []),
                homonyms=item.get("homonyms", {}),
                paronyms=item.get("paronyms", []),
                domain=item.get("domain", "geral"),
                application_context="",
                canonical_source=item.get("canonical_source", ""),
            )
            key = self._normalize(term)
            if key not in self._table:
                self._table[key] = node


class LogicLMSymbolicSolver:
    """Processador simblico inspirado em LLM-Symbolic Solver Logic-LM.

    Este mdulo faz uma pesquisa contextual nos parmetros de entrada da
    LLM base da IA Doninha e acrescenta  definio dos conceitos uma nota
    de aplicao prtica para o prompt atual.
    """

    CONTEXTUAL_KEYWORDS = {
        "fsico": ["temperatura", "calor", "energia", "massa", "volume"],
        "lgica": ["verdade", "falso", "proposio", "argumento", "inferencia", "inferncia"],
        "cognitivo": ["raciocnio", "inteligncia", "compreender", "resolver", "pensar"],
        "filosfico": ["verdade", "conhecimento", "epistemologia", "realidade", "tica"],
        "geral": ["aplicao", "uso", "contexto", "pergunta", "problema"],
    }

    @classmethod
    def enrich(
        cls,
        concepts: List[ConceptNode],
        prompt: str,
        llm_context: Optional[str] = None,
        domain: str = "geral",
        config: Optional[Dict] = None,
    ) -> List[ConceptNode]:
        text = prompt.strip()
        if llm_context:
            text = f"{llm_context.strip()} {text}"
        lower_text = text.lower()

        # Carrega KB especfico do domnio
        kb = get_domain_knowledge_base(domain, config, query_for_rag=text)

        for node in concepts:
            node.application_context = cls._infer_application_context(node, lower_text, concepts, kb)
        return concepts

    @classmethod
    def _infer_application_context(
        cls,
        node: ConceptNode,
        text: str,
        concepts: List[ConceptNode],
        kb: Dict[str, float],
    ) -> str:
        base_context = ""
        if node.term.lower() in text:
            if not cls.is_context_compatible(node, text):
                return ""
            relation = cls._infer_relation(node, text, concepts)
            if relation:
                base_context = relation
        else:
            base_context = cls._default_context(node)

        # Enriquece com termos relevantes do KB do domnio
        relevant_terms = [term for term, score in kb.items() if term.lower() in text.lower() and score > 0.5]
        if relevant_terms:
            kb_context = f" Contexto de conhecimento: {', '.join(relevant_terms[:3])}."
            base_context += kb_context

        return base_context

    @classmethod
    def is_context_compatible(
        cls,
        node: ConceptNode,
        text: str,
    ) -> bool:
        if not node.canonical_source:
            return True
        lower_text = text.lower()
        if node.term.lower() in lower_text:
            return True
        if node.domain:
            domain_keywords = cls.CONTEXTUAL_KEYWORDS.get(node.domain, [])
            if any(keyword in lower_text for keyword in domain_keywords):
                return True
        canonical_keywords = cls._extract_source_keywords(node.canonical_source)
        if any(keyword in lower_text for keyword in canonical_keywords):
            return True
        if node.application_context and any(
            part in lower_text for part in cls._tokenize(node.application_context)
        ):
            return True

        # Verificao de atribuio cannica
        if node.canonical_context:
            return cls._check_canonical_context_compatibility(node, text)
        return False

    @classmethod
    def _check_canonical_context_compatibility(
        cls,
        node: ConceptNode,
        text: str,
    ) -> bool:
        """Verifica se o contexto cannico do conceito  compatvel com o texto atual."""
        lower_text = text.lower()
        for context_key, context_value in node.canonical_context.items():
            # Verifica se o contexto cannico contm indicaes de incompatibilidade
            if "NO" in context_value.upper():
                # Extrai termos proibidos (aps "NO")
                not_parts = context_value.upper().split("NO")[1:]
                for not_part in not_parts:
                    prohibited_terms = cls._extract_prohibited_terms(not_part.strip())
                    if any(term in lower_text for term in prohibited_terms):
                        # Incompatvel - gera alerta para L7
                        cls._generate_canonical_alert(node, context_key, context_value, text)
                        return False
            # Verifica se o contexto cannico requer termos especficos
            elif ":" in context_value:
                required_terms = cls._extract_required_terms(context_value)
                if any(term in lower_text for term in required_terms):
                    return True
        return True  # Compatvel por padro se no h restries especficas

    @classmethod
    def _extract_prohibited_terms(cls, not_part: str) -> List[str]:
        """Extrai termos proibidos de uma parte 'NO ...'."""
        # Remove pontuao e divide por vrgulas ou 'ou'
        terms = re.split(r'[,\s]+ou[\s]+|[,;]', not_part)
        return [term.strip().lower() for term in terms if term.strip()]

    @classmethod
    def _extract_required_terms(cls, context_value: str) -> List[str]:
        """Extrai termos requeridos do contexto cannico."""
        # Assume formato "Fonte: descrio, termos requeridos"
        parts = context_value.split(":")
        if len(parts) > 1:
            description = parts[1].strip()
            terms = re.findall(r"[a-zA-Z]+", description)
            return [term.lower() for term in terms if len(term) > 3]
        return []

    @classmethod
    def _generate_canonical_alert(
        cls,
        node: ConceptNode,
        context_key: str,
        context_value: str,
        text: str,
    ) -> None:
        """Gera um alerta de incompatibilidade cannica para ser passado ao L7."""
        # Armazena o alerta em uma varivel global ou estrutura compartilhada
        # Por simplicidade, vamos usar um dicionrio global para alertas
        if not hasattr(cls, '_canonical_alerts'):
            cls._canonical_alerts = []
        alert = {
            'concept': node.term,
            'canonical_context': f"{context_key}: {context_value}",
            'incompatible_usage': text[:100] + "..." if len(text) > 100 else text,
            'alert_type': 'canonical_incompatibility'
        }
        cls._canonical_alerts.append(alert)

    @classmethod
    def get_canonical_alerts(cls) -> List[Dict]:
        """Retorna e limpa os alertas cannicos gerados."""
        if not hasattr(cls, '_canonical_alerts'):
            cls._canonical_alerts = []
        alerts = cls._canonical_alerts[:]
        cls._canonical_alerts.clear()
        return alerts

    @classmethod
    def _extract_source_keywords(cls, source: str) -> List[str]:
        return [
            token for token in re.findall(r"[a-zA-Z]+", source.lower())
            if len(token) > 3
        ]

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        return [token for token in re.findall(r"[a-zA-Z]+", text.lower()) if len(token) > 3]

    @classmethod
    def _infer_relation(
        cls,
        node: ConceptNode,
        text: str,
        concepts: List[ConceptNode],
    ) -> str:
        domain_keywords = cls.CONTEXTUAL_KEYWORDS.get(node.domain, [])
        for keyword in domain_keywords:
            if keyword in text:
                return cls._build_context_sentence(node, keyword)

        related = cls._related_concepts(node, concepts, text)
        if related:
            return cls._build_related_context(node, related)

        return ""

    @classmethod
    def _related_concepts(
        cls,
        node: ConceptNode,
        concepts: List[ConceptNode],
        text: str,
    ) -> List[str]:
        related = []
        for other in concepts:
            if other.term == node.term:
                continue
            if other.term.lower() in text:
                related.append(other.term)
        return related

    @classmethod
    def _build_context_sentence(cls, node: ConceptNode, keyword: str) -> str:
        return (
            f"No contexto da pergunta, '{node.term}'  aplicado como um conceito de {node.domain}"
            f" relacionado a '{keyword}', indicando como o prompt utiliza seu significado prtico."
        )

    @classmethod
    def _build_related_context(cls, node: ConceptNode, related: List[str]) -> str:
        related_terms = ", ".join(related[:3])
        return (
            f"Neste caso, '{node.term}' aparece em conjunto com {related_terms},"
            f" o que sugere seu papel prtico na anlise do prompt."
        )

    @classmethod
    def _default_context(cls, node: ConceptNode) -> str:
        return (
            f"No contexto atual, '{node.term}' representa {node.definition.lower()}"
            f" e serve como um conceito relevante para o problema expresso no prompt."
        )

    @classmethod
    def summarize_application_context(cls, concepts: List[ConceptNode]) -> str:
        parts = [node.application_context for node in concepts if node.application_context]
        return " ".join(parts)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split()).strip()
