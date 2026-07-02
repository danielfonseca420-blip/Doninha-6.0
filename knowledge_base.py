"""
Base de conhecimento escalvel.
===============================
Carrega KB a partir de arquivo (JSON) e opcionalmente enriquece com
retrieval em ChromaDB (RAG). Mantm interface termo -> grau [0,1] para L3/L4.
"""

from __future__ import annotations
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# KB padro (fallback quando no h arquivo)
SEED_KNOWLEDGE_BASE: Dict[str, float] = {
    "quente": 0.85, "frio": 0.85, "morno": 0.70, "aquecido": 0.80, "gelado": 0.80,
    "temperatura": 0.90, "graus": 0.88, "escaldante": 0.75, "tpido": 0.65,
    "verdadeiro": 0.95, "falso": 0.95, "contradio": 0.80, "proposio": 0.85,
    "silogismo": 0.75, "conhecimento": 0.90, "inteligncia": 0.85, "conscincia": 0.70,
    "razo": 0.88, "verdade": 0.92, "gua": 0.95, "lquido": 0.90, "h2o": 0.90,
}


def _extract_texts_from_doc(doc: Any) -> List[str]:
    if isinstance(doc, str):
        return [doc]
    if isinstance(doc, dict):
        for key in ("text", "content", "body", "summary", "description"):
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                return [value]
        text_parts: List[str] = []
        for value in doc.values():
            if isinstance(value, str) and value.strip():
                text_parts.append(value)
        if text_parts:
            return [" ".join(text_parts)]
    if isinstance(doc, list):
        texts = []
        for item in doc:
            texts.extend(_extract_texts_from_doc(item))
        return texts
    return []


def _term_weights_from_texts(texts: List[str], max_terms: int = 2000) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    for text in texts:
        words = re.findall(r"[a-z]+", text.lower())
        for word in words:
            if len(word) > 3:
                counts[word] += 1
    if not counts:
        return {}
    most_common = counts.most_common(max_terms)
    max_value = most_common[0][1]
    return {term: min(1.0, count / max_value) for term, count in most_common}


def _normalize_counter(counts: Counter[str]) -> Dict[str, float]:
    if not counts:
        return {}
    most_common = counts.most_common(2000)
    max_value = most_common[0][1]
    return {term: min(1.0, count / max_value) for term, count in most_common}


def _count_terms_in_text(text: str, counts: Counter[str]) -> None:
    for word in re.findall(r"[a-z]+", text.lower()):
        if len(word) > 3:
            counts[word] += 1


def _load_kb_from_jsonl(path: Path, max_docs: int = 20000) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for text in _extract_texts_from_doc(record):
                _count_terms_in_text(text, counts)
    return _normalize_counter(counts)


def load_kb_from_file(path: str | Path) -> Dict[str, float]:
    """
    Carrega dicionrio termo -> grau [0,1] de um arquivo JSON.
    Suporta JSON de termo->peso, lista de documentos e NDJSON.
    """
    path = Path(path)
    if not path.exists():
        return {}
    try:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            return _load_kb_from_jsonl(path)

        if path.stat().st_size > 1_000_000_000:
            return _load_kb_from_jsonl(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            if all(isinstance(v, (int, float)) for v in data.values()):
                return {k: float(v) for k, v in data.items()}
            return _term_weights_from_texts(_extract_texts_from_doc(data))

        if isinstance(data, list):
            texts: List[str] = []
            for item in data:
                texts.extend(_extract_texts_from_doc(item))
            return _term_weights_from_texts(texts)
    except Exception:
        try:
            return _load_kb_from_jsonl(path)
        except Exception:
            return {}
    return {}


def merge_kb(base: Dict[str, float], extra: Dict[str, float]) -> Dict[str, float]:
    """Mescla extra em base; em conflito, extra prevalece."""
    out = dict(base)
    for k, v in extra.items():
        out[k] = v
    return out


def enrich_kb_from_chroma(
    query: str,
    chroma_path: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    k: int = 5,
    score_weight: float = 0.8,
) -> Dict[str, float]:
    """
    Busca em ChromaDB por query e retorna um dicionrio termo -> peso
    extrado dos trechos recuperados (palavras relevantes com score_weight).
    """
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        return {}

    chroma_path = Path(chroma_path)
    if not chroma_path.exists() or not chroma_path.is_dir():
        return {}

    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vectorstore = Chroma(persist_directory=str(chroma_path), embedding_function=embeddings)
        docs = vectorstore.similarity_search(query, k=k)
    except Exception:
        return {}

    # Extrai termos dos textos e atribui peso
    term_scores: Dict[str, float] = {}
    for d in docs:
        text = d.page_content if hasattr(d, "page_content") else str(d)
        words = re.findall(r"[a-z]+", text.lower())
        for w in words:
            if len(w) > 2:
                term_scores[w] = term_scores.get(w, 0) + score_weight
    # Normaliza para [0, 1]
    if term_scores:
        m = max(term_scores.values())
        term_scores = {t: min(1.0, s / m) for t, s in term_scores.items()}
    return term_scores


def get_domain_knowledge_base(
    domain: str,
    config: Optional[Dict[str, Any]] = None,
    query_for_rag: Optional[str] = None,
) -> Dict[str, float]:
    """
    Retorna KB especializado para um domnio.
    - Usa KB especfico de domnio se configurado em domain_specific_kbs.
    - Se no existirem arquivos de domnio, usa o KB genrico em knowledge_base.path.
    - Enriquece com ChromaDB especfico do domnio (chroma_path/{domain}/).
    - Fallback: SEED_KNOWLEDGE_BASE.
    """
    PROJECT_ROOT = Path(__file__).resolve().parent
    try:
        from config_loader import load_config, PROJECT_ROOT as _root
        PROJECT_ROOT = _root
        if config is None:
            config = load_config()
    except Exception:
        pass
    if config is None:
        config = {}

    base: Dict[str, float] = {}

    domain_specific = config.get("knowledge_base", {}).get("domain_specific_kbs", {})
    if domain and isinstance(domain_specific, dict) and domain_specific.get(domain):
        domain_path = Path(domain_specific[domain])
        if not domain_path.is_absolute():
            domain_path = PROJECT_ROOT / domain_path
        if domain_path.exists():
            base = load_kb_from_file(domain_path)

    if not base:
        kb_path = config.get("knowledge_base", {}).get("path", "")
        if kb_path:
            path_obj = Path(kb_path) if Path(kb_path).is_absolute() else PROJECT_ROOT / kb_path
            if path_obj.exists():
                if path_obj.is_dir() and domain:
                    candidate = path_obj / f"kb_{domain}.json"
                    if candidate.exists():
                        base = load_kb_from_file(candidate)
                elif path_obj.is_file():
                    base = load_kb_from_file(path_obj)

    if not base:
        default_kb = config.get("knowledge_base", {}).get("default_kb", "")
        if default_kb:
            default_path = Path(default_kb) if Path(default_kb).is_absolute() else PROJECT_ROOT / default_kb
            if default_path.exists():
                base = load_kb_from_file(default_path)

    if not base:
        base = dict(SEED_KNOWLEDGE_BASE)

    # Chroma especfico do domnio
    chroma_base = config.get("knowledge_base", {}).get("chroma_path") or config.get("agent", {}).get("vector_db_path", "")
    if chroma_base:
        domain_chroma_path = Path(chroma_base) / domain
        if domain_chroma_path.exists() and domain_chroma_path.is_dir() and query_for_rag:
            extra = enrich_kb_from_chroma(
                query_for_rag,
                str(domain_chroma_path),
                config.get("agent", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                k=5,
            )
            base = merge_kb(base, extra)

    return base


def get_knowledge_base(
    config: Optional[Dict[str, Any]] = None,
    query_for_rag: Optional[str] = None,
    domain: Optional[str] = None,
) -> Dict[str, float]:
    """
    Retorna KB geral ou domnio-especfico a partir da configurao.

    Se knowledge_base.path for um arquivo JSON, usa-o como fonte genrica.
    """
    config = config or {}
    kb_config = config.get("knowledge_base", {})
    kb_path = kb_config.get("path", "")
    default_kb = kb_config.get("default_kb", "")
    domain_specific = kb_config.get("domain_specific_kbs", {})

    project_root = Path(__file__).resolve().parent

    if isinstance(domain_specific, dict) and domain:
        domain_file = domain_specific.get(domain)
        if domain_file:
            domain_path = Path(domain_file) if Path(domain_file).is_absolute() else project_root / domain_file
            if domain_path.exists():
                return load_kb_from_file(domain_path)

    if kb_path:
        base_path = Path(kb_path) if Path(kb_path).is_absolute() else project_root / kb_path
        if base_path.exists():
            if base_path.is_file():
                return load_kb_from_file(base_path)
            if base_path.is_dir() and domain:
                candidate = base_path / f"kb_{domain}.json"
                if candidate.exists():
                    return load_kb_from_file(candidate)

    if default_kb:
        default_path = Path(default_kb) if Path(default_kb).is_absolute() else project_root / default_kb
        if default_path.exists():
            return load_kb_from_file(default_path)

    return dict(SEED_KNOWLEDGE_BASE)
