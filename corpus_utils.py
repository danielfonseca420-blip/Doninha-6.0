from __future__ import annotations

"""
Utilitários de corpus
=====================

Funções para carregar texto de:
  - arquivos Markdown/TXT (ex.: README)
  - artigo completo em DOCX
  - dataset gigante em JSONL (ex.: dump da Wikipedia baixado do HuggingFace),
    lido em STREAMING para não explodir a RAM em arquivos de várias GB.
"""

from typing import List, Iterator
import json
import os

from docx import Document


def read_text_file(path: str, encoding: str = "utf-8") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo de texto não encontrado: {path}")
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def read_docx_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo DOCX não encontrado: {path}")
    doc = Document(path)
    parts: List[str] = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def iter_jsonl_texts(path: str, text_field: str = "text") -> Iterator[str]:
    """
    Itera sobre um arquivo .json/.jsonl gigante, LINHA POR LINHA, sem nunca
    carregar o arquivo inteiro na memória. Cada linha deve ser um objeto
    JSON contendo o campo `text_field`.

    Linhas malformadas ou sem o campo são simplesmente ignoradas (não
    interrompem o processamento de um arquivo de 20GB por uma linha ruim).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo JSONL não encontrado: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(text_field)
            if text and isinstance(text, str):
                yield text


def build_sp_training_corpus(
    jsonl_path: str,
    output_txt_path: str,
    text_field: str = "text",
    max_chars: int = 200_000_000,
) -> str:
    """
    Gera um .txt intermediário para treinar o SentencePiece, lendo o JSONL
    em streaming e escrevendo em streaming (nunca materializa tudo na RAM).

    max_chars limita quanto texto é usado para treinar o TOKENIZER (não o
    modelo). O tokenizer não precisa ver os 20GB inteiros para aprender um
    bom vocabulário; 200M caracteres já é mais que suficiente para um
    vocab_size de poucos milhares de peças.

    Retorna o caminho do arquivo gerado.
    """
    written = 0
    with open(output_txt_path, "w", encoding="utf-8") as out:
        for text in iter_jsonl_texts(jsonl_path, text_field):
            cleaned = text.replace("\r\n", "\n")
            out.write(cleaned + "\n")
            written += len(cleaned)
            if written >= max_chars:
                break
    return output_txt_path


def load_main_corpus() -> List[str]:
    """
    Carrega o corpus "pequeno" (README + artigo DOCX), usado apenas para
    geração de amostras de texto / contexto do projeto.

    NÃO usa mais os arquivos extra antigos (sep_texts_only.txt, etc).
    O dataset grande (train_all.json) é tratado separadamente via
    iter_jsonl_texts / build_sp_training_corpus, em streaming, e NUNCA
    deve ser carregado inteiro com esta função.
    """
    base_dir = os.path.dirname(__file__) or "."
    readme_path = os.path.join(base_dir, "README.md")
    article_path = os.path.join(
        base_dir,
        "Uma verdadeira Epistemologia para a Inteligência Artificial.docx",
    )

    texts: List[str] = []
    if os.path.exists(readme_path):
        texts.append(read_text_file(readme_path))
    if os.path.exists(article_path):
        texts.append(read_docx_file(article_path))

    if not texts:
        raise FileNotFoundError(
            "Nenhum corpus pequeno encontrado (README.md ou artigo DOCX). "
            "Isso é usado só para amostras; o treino principal usa o JSONL grande."
        )
    return texts
