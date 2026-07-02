"""
Pre-encode corpus into token ID binary files for Julia training.

Saves tokens as Int32 arrays (.bin) that Julia can mmap directly,
avoiding the slow pure-Julia BPE encoding at training time.

Usage:
    python encode_corpus.py                          # Encode train.txt + val.txt
    python encode_corpus.py --tokenizer output/tokenizer_4k.json
"""

import argparse
import logging
import struct
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer

logger = logging.getLogger("encode_corpus")
SCRIPT_DIR = Path(__file__).resolve().parent

BATCH_LINES = 100_000  # encode this many lines at once


def encode_file(tokenizer: Tokenizer, input_path: Path, output_path: Path, offset: int = 1):
    """Encode a text file into a binary token ID file, streaming by line batches."""
    logger.info("Encoding %s → %s", input_path, output_path)

    # First pass: count lines for progress
    n_lines = sum(1 for _ in open(input_path, encoding="utf-8"))
    logger.info("  %d lines to encode", n_lines)

    all_ids = []
    total_tokens = 0

    with open(input_path, encoding="utf-8") as f:
        batch = []
        line_count = 0
        for line in f:
            batch.append(line)
            line_count += 1

            if len(batch) >= BATCH_LINES:
                encoded = tokenizer.encode_batch(batch)
                for enc in encoded:
                    all_ids.extend(enc.ids)
                total_tokens += sum(len(enc.ids) for enc in encoded)
                if line_count % (BATCH_LINES * 5) == 0:
                    logger.info("  %d/%d lines (%.1f%%), %dM tokens so far",
                                line_count, n_lines,
                                100 * line_count / max(n_lines, 1),
                                total_tokens / 1e6)
                batch = []

        # Final batch
        if batch:
            encoded = tokenizer.encode_batch(batch)
            for enc in encoded:
                all_ids.extend(enc.ids)
            total_tokens += sum(len(enc.ids) for enc in encoded)

    # Convert to numpy and apply offset
    arr = np.array(all_ids, dtype=np.int32) + offset

    # Write: magic (4B) + n_tokens (8B) + offset (4B) + int32 data
    with open(output_path, "wb") as f:
        f.write(b"JTOK")
        f.write(struct.pack("<Q", len(arr)))
        f.write(struct.pack("<i", offset))
        arr.tofile(f)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info("  Done: %d tokens (%.1fM), %.1f MB, range [%d, %d]",
                len(arr), len(arr) / 1e6, size_mb, arr.min(), arr.max())
    return len(arr)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Pre-encode corpus for Julia training")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "output"
    tok_path = Path(args.tokenizer) if args.tokenizer else output_dir / "tokenizer.json"

    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

    tokenizer = Tokenizer.from_file(str(tok_path))
    logger.info("Loaded tokenizer: vocab_size=%d from %s", tokenizer.get_vocab_size(), tok_path)

    total = 0
    for name in ["train", "val"]:
        txt_path = output_dir / f"{name}.txt"
        bin_path = output_dir / f"{name}.bin"
        if txt_path.exists():
            total += encode_file(tokenizer, txt_path, bin_path)
        else:
            logger.warning("Skipping %s (not found)", txt_path)

    logger.info("All done! Total: %d tokens (%.1fM)", total, total / 1e6)


if __name__ == "__main__":
    main()
