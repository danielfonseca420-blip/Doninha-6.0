from __future__ import annotations

"""
Pré-treinamento de um pequeno modelo de linguagem
=================================================

Fluxo:
  1. Gera um corpus de treino do tokenizer a partir de um trecho do JSONL
     grande (train_all.json), em streaming.
  2. Treina um tokenizador SentencePiece (BPE) se ainda não existir.
  3. Constrói um Dataset de LM em STREAMING (tokeniza sob demanda, nunca
     carrega o JSONL de 20GB inteiro na memória).
  4. Treina `EpistemicLanguageModel` com cross-entropy e AdamW, na CPU.
  5. Salva pesos do modelo e reutiliza o tokenizador treinado.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterator
import os
import random

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from custom_tokenizer import SPConfig, train_sentencepiece, CustomSPTokenizer
from custom_lm_model import LMConfig, EpistemicLanguageModel, save_lm, generate_text
from corpus_utils import load_main_corpus, iter_jsonl_texts, build_sp_training_corpus


@dataclass
class TrainLMConfig:
    sp_config: SPConfig = field(default_factory=SPConfig)
    max_seq_len: int = 128
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    save_dir: str = "checkpoints_lm"

    # --- Configuração do dataset grande (JSONL) ---
    jsonl_path: str = "train_all.json"
    jsonl_text_field: str = "text"
    # Quantos caracteres usar para TREINAR O TOKENIZER (não precisa ser o
    # dataset inteiro). 200M chars já é mais que suficiente.
    tokenizer_train_max_chars: int = 200_000_000
    # Quantos tokens usar para TREINAR O MODELO nesta rodada. Comece
    # pequeno (ex.: 5_000_000) e aumente depois que confirmar que está
    # tudo funcionando sem crash. Use None para não limitar (não
    # recomendado na primeira tentativa com 8-16GB de RAM).
    max_train_tokens: Optional[int] = 5_000_000
    max_val_tokens: Optional[int] = 200_000
    # Quantos documentos do início do arquivo pular antes de começar a
    # coletar dados de validação (para não usar os mesmos textos do treino).
    val_skip_train_docs: bool = True

    # --- Modo dataset completo, em streaming, sem limite de tokens ---
    # Se True, ignora max_train_tokens e treina sobre o JSONL inteiro,
    # lendo o arquivo do disco em streaming a cada epoch (não usa lista
    # em memória). Recomendado: num_epochs=1 nesse modo, já que cada
    # epoch já implica ler o arquivo de 20GB inteiro uma vez.
    use_full_dataset_streaming: bool = False
    # Tamanho do buffer (em blocos) usado para embaralhar parcialmente o
    # stream. Maior = shuffle melhor, mas mais RAM. 2000 blocos de 128
    # tokens = ~256k tokens no buffer, ainda muito leve em RAM.
    shuffle_buffer_blocks: int = 2000
    # Quantos documentos do início do arquivo reservar para validação,
    # antes de começar a ler os dados de treino (assim treino e validação
    # nunca se sobrepõem mesmo no modo streaming).
    full_dataset_val_docs: int = 500


class LMDataset(Dataset):
    """
    Dataset de linguagem causal "em memória", usado quando token_ids já
    foi limitado a um tamanho razoável (ver max_train_tokens/max_val_tokens).
    Divide o fluxo de tokens em blocos de tamanho fixo, com input_ids e
    labels deslocados em 1.
    """

    def __init__(self, token_ids: List[int], block_size: int) -> None:
        self.block_size = block_size
        n = (len(token_ids) // block_size) * block_size
        self.data = token_ids[:n]

    def __len__(self) -> int:
        return max(len(self.data) // self.block_size - 1, 0)

    def __getitem__(self, idx: int):
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.tensor(self.data[start:end], dtype=torch.long)
        y = torch.tensor(self.data[start + 1 : end + 1], dtype=torch.long)
        return x, y


class StreamingLMDataset(IterableDataset):
    """
    Dataset de linguagem causal que NUNCA carrega o JSONL inteiro na
    memória. Lê o arquivo documento por documento, tokeniza, concatena
    em um buffer pequeno de tokens, e a cada vez que o buffer acumula
    tokens suficientes, fatia em blocos de `block_size` e os entrega.

    Os blocos são acumulados em um buffer de embaralhamento de tamanho
    `shuffle_buffer_blocks` antes de serem entregues em ordem aleatória,
    para evitar que o modelo veja sempre a mesma ordem sequencial do
    arquivo (o que prejudicaria o treino).

    Cada época relê o arquivo do disco desde o início (custo de I/O,
    não de RAM).
    """

    def __init__(
        self,
        jsonl_path: str,
        text_field: str,
        tokenizer: "CustomSPTokenizer",
        block_size: int,
        skip_docs: int = 0,
        shuffle_buffer_blocks: int = 2000,
        seed: int = 42,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.skip_docs = skip_docs
        self.shuffle_buffer_blocks = max(1, shuffle_buffer_blocks)
        self.seed = seed

    def _blocks(self) -> Iterator[Tuple[List[int], List[int]]]:
        buffer: List[int] = []
        doc_iter = iter_jsonl_texts(self.jsonl_path, self.text_field)

        for _ in range(self.skip_docs):
            try:
                next(doc_iter)
            except StopIteration:
                break

        for text in doc_iter:
            buffer.extend(self.tokenizer.encode(text, add_bos=True, add_eos=True))
            while len(buffer) >= self.block_size + 1:
                x = buffer[: self.block_size]
                y = buffer[1 : self.block_size + 1]
                buffer = buffer[self.block_size :]
                yield x, y

    def __iter__(self):
        rng = random.Random(self.seed)
        shuffle_buf: List[Tuple[List[int], List[int]]] = []

        for item in self._blocks():
            shuffle_buf.append(item)
            if len(shuffle_buf) >= self.shuffle_buffer_blocks:
                rng.shuffle(shuffle_buf)
                while shuffle_buf:
                    x, y = shuffle_buf.pop()
                    yield (
                        torch.tensor(x, dtype=torch.long),
                        torch.tensor(y, dtype=torch.long),
                    )

        # Esvazia o que restou no buffer ao final do arquivo.
        rng.shuffle(shuffle_buf)
        while shuffle_buf:
            x, y = shuffle_buf.pop()
            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def estimate_streaming_steps(
    jsonl_path: str, batch_size: int, block_size: int, sample_docs: int = 2000
) -> Optional[int]:
    """
    Estima quantos batches uma epoch terá, lendo só uma amostra do início
    do arquivo e extrapolando pelo tamanho total em bytes. É uma estimativa
    aproximada, usada só para dar contexto de tempo na barra de progresso -
    não afeta o treino em si.
    """
    if not os.path.exists(jsonl_path):
        return None
    try:
        total_bytes = os.path.getsize(jsonl_path)
        sample_bytes = 0
        sample_tokens = 0
        count = 0
        with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                sample_bytes += len(line.encode("utf-8", errors="replace"))
                sample_tokens += max(len(line) // 4, 1)  # ~4 chars por token, grosseiro
                count += 1
                if count >= sample_docs:
                    break
        if sample_bytes == 0:
            return None
        est_total_tokens = int(sample_tokens * (total_bytes / sample_bytes))
        est_total_blocks = est_total_tokens // block_size
        est_total_steps = max(est_total_blocks // batch_size, 1)
        return est_total_steps
    except Exception:
        return None


def ensure_tokenizer(config: TrainLMConfig) -> CustomSPTokenizer:
    model_file = f"{config.sp_config.model_prefix}.model"
    if not os.path.exists(model_file):
        tmp_corpus = "sp_corpus_tmp.txt"
        if os.path.exists(config.jsonl_path):
            print(f"Gerando corpus de treino do tokenizer a partir de {config.jsonl_path} "
                  f"(limite: {config.tokenizer_train_max_chars} chars)...")
            build_sp_training_corpus(
                config.jsonl_path,
                tmp_corpus,
                text_field=config.jsonl_text_field,
                max_chars=config.tokenizer_train_max_chars,
            )
        else:
            # Fallback: usa só o corpus pequeno (README/artigo) se o JSONL
            # grande não existir nesta máquina.
            texts = load_main_corpus()
            with open(tmp_corpus, "w", encoding="utf-8") as f:
                for t in texts:
                    f.write(t.replace("\r\n", "\n") + "\n")

        train_sentencepiece([tmp_corpus], config.sp_config)
        os.remove(tmp_corpus)
    return CustomSPTokenizer(model_prefix=config.sp_config.model_prefix)


def build_token_lists(
    tokenizer: CustomSPTokenizer, config: TrainLMConfig
) -> Tuple[List[int], List[int]]:
    """
    Lê o JSONL grande EM STREAMING, tokenizando documento por documento,
    até acumular max_train_tokens (e depois max_val_tokens para validação).
    Nunca guarda o texto bruto inteiro na memória - só os ids já tokenizados,
    e só até o limite configurado.
    """
    if not os.path.exists(config.jsonl_path):
        print(f"Aviso: {config.jsonl_path} não encontrado, usando corpus pequeno (README/artigo).")
        texts = load_main_corpus()
        train_ids: List[int] = []
        for t in texts:
            train_ids.extend(tokenizer.encode(t, add_bos=True, add_eos=True))
        return train_ids, list(train_ids)  # val = mesmo corpus pequeno

    train_ids: List[int] = []
    val_ids: List[int] = []
    max_train = config.max_train_tokens
    max_val = config.max_val_tokens

    doc_iter = iter_jsonl_texts(config.jsonl_path, config.jsonl_text_field)

    print(f"Tokenizando stream de treino (limite: {max_train} tokens)...")
    for text in tqdm(doc_iter, desc="Tokenizando treino"):
        train_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        if max_train is not None and len(train_ids) >= max_train:
            break

    if max_val is not None and max_val > 0:
        print(f"Tokenizando stream de validação (limite: {max_val} tokens)...")
        for text in tqdm(doc_iter, desc="Tokenizando validação"):
            val_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
            if len(val_ids) >= max_val:
                break

    if not val_ids:
        # Garante que sempre haja algo para validar, mesmo em datasets pequenos.
        split = max(1, int(0.9 * len(train_ids)))
        val_ids = train_ids[split:]
        train_ids = train_ids[:split]

    return train_ids, val_ids


def evaluate_lm(
    model: EpistemicLanguageModel,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn,
) -> float:
    model.eval()
    total_loss, steps = 0.0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += float(loss.item())
            steps += 1
    avg_loss = total_loss / max(steps, 1)
    return avg_loss


def train_lm(config: TrainLMConfig) -> EpistemicLanguageModel:
    # GPU AMD no Windows não é suportada via CUDA/ROCm. Forçamos CPU para
    # evitar comportamento inconsistente; torch.cuda.is_available() já
    # seria False nesse caso, mas deixamos explícito.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    tokenizer = ensure_tokenizer(config)
    print(f"Tokenizer carregado. vocab_size={tokenizer.vocab_size}")

    est_steps = None

    if config.use_full_dataset_streaming:
        if not os.path.exists(config.jsonl_path):
            raise FileNotFoundError(
                f"use_full_dataset_streaming=True mas {config.jsonl_path} não existe."
            )
        print(f"Modo streaming completo ativado: lendo {config.jsonl_path} inteiro, "
              f"sem carregar em memória.")

        # Reserva os primeiros N documentos para validação (em memória,
        # pequeno o suficiente), e o resto do arquivo para treino em stream.
        val_ids: List[int] = []
        val_doc_iter = iter_jsonl_texts(config.jsonl_path, config.jsonl_text_field)
        for _ in range(config.full_dataset_val_docs):
            try:
                text = next(val_doc_iter)
            except StopIteration:
                break
            val_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        print(f"Tokens de validação (primeiros {config.full_dataset_val_docs} docs): {len(val_ids)}")

        val_dataset = LMDataset(val_ids, block_size=config.max_seq_len)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        train_dataset = StreamingLMDataset(
            jsonl_path=config.jsonl_path,
            text_field=config.jsonl_text_field,
            tokenizer=tokenizer,
            block_size=config.max_seq_len,
            skip_docs=config.full_dataset_val_docs,
            shuffle_buffer_blocks=config.shuffle_buffer_blocks,
        )
        # IterableDataset: sem shuffle=True no DataLoader (o shuffle já
        # acontece dentro do dataset via buffer).
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

        est_steps = estimate_streaming_steps(
            config.jsonl_path, config.batch_size, config.max_seq_len
        )
        if est_steps:
            print(f"Estimativa aproximada de steps por epoch: ~{est_steps}")

    else:
        train_ids, val_ids = build_token_lists(tokenizer, config)
        print(f"Tokens de treino: {len(train_ids)} | Tokens de validação: {len(val_ids)}")

        train_dataset = LMDataset(train_ids, block_size=config.max_seq_len)
        val_dataset = LMDataset(val_ids, block_size=config.max_seq_len)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    lm_config = LMConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=config.max_seq_len,
    )
    model = EpistemicLanguageModel(lm_config).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()

    os.makedirs(config.save_dir, exist_ok=True)

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        optimizer.zero_grad()

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        ):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

            loss = loss / max(config.grad_accum_steps, 1)
            loss.backward()

            if (step + 1) % config.grad_accum_steps == 0:
                if config.grad_clip is not None and config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += float(loss.item())
            steps += 1

        scheduler.step()
        avg_train_loss = total_loss / max(steps, 1)

        val_loss = evaluate_lm(
            model.module if isinstance(model, torch.nn.DataParallel) else model,
            val_loader,
            device,
            loss_fn,
        )
        ppl = torch.exp(torch.tensor(val_loss)).item()

        print(
            f"Epoch {epoch+1} - train loss: {avg_train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | ppl: {ppl:.2f}"
        )

        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        prompt = "A inteligência artificial"
        try:
            sample = generate_text(base_model, tokenizer, prompt, max_new_tokens=40)
            print(f"Exemplo de geração: {sample}\n")
        except Exception as e:
            # Nunca deixa a geração de amostra derrubar o treino inteiro.
            print(f"[aviso] Falha ao gerar amostra de texto: {e}\n")

        ckpt_path = os.path.join(config.save_dir, f"epistemic_lm_epoch{epoch+1}.pt")
        save_lm(base_model, ckpt_path)

    return model.module if isinstance(model, torch.nn.DataParallel) else model


def main() -> None:
    config = TrainLMConfig()
    model = train_lm(config)
    save_path = "epistemic_lm.pt"
    save_lm(model, save_path)
    print(f"Modelo de linguagem salvo em '{save_path}'")


if __name__ == "__main__":
    main()