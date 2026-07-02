---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- philosophy
- classical-texts
- humanities
- wikitext
- bpe-tokenizer
- gpt-training
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 2153173168
    num_examples: 13683034
  - name: validation
    num_bytes: 238462992
    num_examples: 1520338
  download_size: 1628332782
  dataset_size: 2391636160
---

# Philosophy & Humanities Corpus

Combined humanities and Wikipedia corpus for training small language models.

## Dataset

| Split | Lines | Size | Description |
|-------|-------|------|-------------|
| **train.txt** | 3.0M | 549 MB | Humanities (368K lines) + WikiText-103 (2.6M lines) |
| **val.txt** | 315K | 57 MB | Matching validation split |

## Sources

### Humanities (368K lines, 66 MB)
54 classical philosophy and humanities texts:

| Category | Works |
|---|---|
| **Plato** | Republic, Apology, Symposium, Phaedo, Crito, Meno, Phaedrus, Timaeus, Laws, Gorgias, Protagoras, Theaetetus |
| **Aristotle** | Categories, Ethics, Rhetoric, Physics, Metaphysics, Poetics, Politics, On the Soul, On the Heavens, Prior/Posterior Analytics, Topics, On Generation & Corruption |
| **Stoics** | Marcus Aurelius Meditations, Epictetus Discourses & Enchiridion, Seneca Moral Essays |
| **Roman** | Lucretius, Cicero (On Duties, Nature of Gods, On Friendship) |
| **Early Modern** | Descartes, Kant, Spinoza, Hobbes, Locke, Bacon |
| **Enlightenment/19th c.** | Hume, Rousseau, Nietzsche, Mill, Machiavelli, Emerson, Thoreau, Montaigne, Schopenhauer |
| **Other** | Boethius, Diogenes/Epicurus, Aeschylus, Latin Grammar, Euclid Elements |

Sources: [Project Gutenberg](https://www.gutenberg.org/), [MIT Internet Classics Archive](http://classics.mit.edu/)

### WikiText-103 (2.6M lines, 481 MB)
Wikipedia articles from [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) (wikitext-103-v1). Cleaned, chunked, and deduplicated.

## Tokenizer

**tokenizer.json** — BPE tokenizer (4000 vocab) trained on the combined corpus.
- Format: HuggingFace tokenizers JSON (GPT-2 ByteLevel BPE)
- Special tokens: `<|pad|>` (id=0), `<|eos|>` (id=1)

## Files

| File | Description |
|------|-------------|
| `train.txt` | Combined training data (one chunk per line) |
| `val.txt` | Combined validation data |
| `tokenizer.json` | BPE tokenizer (vocab_size=4000) |
| `data/*.txt` | Individual source text files |
| `train_enriched.jsonl` | Enriched training data with metadata |
| `train_trivium.txt` | Trivium-phase subset |
| `train_quadrivium.txt` | Quadrivium-phase subset |
| `train_philosophy.txt` | Philosophy-phase subset |

## Usage

Training data for [JuliaGPT](https://github.com/DavinciDreams/JuliaGPT) — small transformer language models in Julia (Flux.jl).

```julia
# Auto-downloaded by juliadistill.ipynb
hf_download("LisaMegaWatts/philosophy-corpus", "train.txt"; repo_type="dataset")
hf_download("LisaMegaWatts/philosophy-corpus", "val.txt"; repo_type="dataset")
hf_download("LisaMegaWatts/philosophy-corpus", "tokenizer.json"; repo_type="dataset")
```
