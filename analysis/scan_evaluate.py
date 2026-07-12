"""
scan_evaluate.py — Positional perplexity scan for AbSynth-A.

Slides a mask window of fixed length across every position of each
heavy/light chain sequence and records the cross-entropy loss at each
position. Saves results to an HDF5 file for plotting with plot_scan.py.

Usage:
    python scan_evaluate.py paired_sequences.csv
    python scan_evaluate.py paired_sequences.csv --mask_len 10 --output_dir results/scan
"""

import argparse
import os
from collections import defaultdict
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import GPT2LMHeadModel, RobertaTokenizerFast

from absynth.utils.seed import set_seeds


def build_scan_batch(
    sequence: str,
    tokenizer: RobertaTokenizerFast,
    mask_len: int,
) -> dict:
    """
    Build a batch where each sample masks one window position of the sequence.

    Returns a dict with:
        input_ids: (n_positions, seq_len) — context + [MASK] + infill target
        labels:    (n_positions, seq_len) — -100 everywhere except the target span
    """
    seq = list(sequence)
    seqs = []

    for start in range(0, len(seq) - mask_len):
        end = start + mask_len
        # Format: prefix + [MASK] + suffix + [SEP] + target + [CLS]
        token_ids = tokenizer.convert_tokens_to_ids(
            seq[:start]
            + [tokenizer.mask_token]
            + seq[end:]
            + [tokenizer.sep_token]
            + seq[start:end]
            + [tokenizer.cls_token]
        )
        seqs.append(token_ids)

    input_ids = torch.tensor(seqs, dtype=torch.long)
    labels = input_ids.clone()
    labels[:, : -(mask_len + 1)] = -100  # only compute loss on infilled region

    return {"input_ids": input_ids, "labels": labels}


def compute_loss_per_sample(lm_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    losses = losses.view(shift_labels.shape)
    mask = shift_labels != -100
    losses = losses * mask
    return losses.sum(dim=1) / mask.sum(dim=1)


def main():
    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(
        description="Compute per-position infilling perplexity for AbSynth-A"
    )
    parser.add_argument("paired_csv", type=str, help="CSV with 'hseq' and 'lseq' columns")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "absynth", "trained_models", "absynth-a"),
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "absynth", "tokenizer"),
    )
    parser.add_argument("--output_dir", type=str, default=f"output_dir/scan_evaluate_{now}")
    parser.add_argument("--mask_len", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    model.eval()

    tokenizer = RobertaTokenizerFast.from_pretrained(args.tokenizer_path)

    dataset = pd.read_csv(args.paired_csv)
    print(f"Loaded {len(dataset)} sequences from {args.paired_csv}")

    h5_path = os.path.join(args.output_dir, "losses.h5")

    with h5py.File(h5_path, "w") as h5_out:
        chain_losses = defaultdict(list)
        chain_positions = defaultdict(list)

        for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Scanning"):
            for chain, col in [("Heavy", "hseq"), ("Light", "lseq")]:
                seq = str(row[col])
                batch = build_scan_batch(seq, tokenizer, args.mask_len)

                with torch.no_grad():
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids, labels=labels)
                    loss_per_pos = compute_loss_per_sample(
                        outputs["logits"], labels
                    ).cpu().numpy()

                chain_losses[chain].append(loss_per_pos)
                chain_positions[chain].append(np.arange(len(loss_per_pos)))

        for chain in ["Heavy", "Light"]:
            all_losses = np.concatenate(chain_losses[chain])
            all_pos = np.concatenate(chain_positions[chain])

            max_pos = int(all_pos.max()) + 1
            loss_sum = np.zeros(max_pos)
            count = np.zeros(max_pos)
            np.add.at(loss_sum, all_pos, all_losses)
            np.add.at(count, all_pos, 1)

            avg_loss = loss_sum / count
            h5_out.create_dataset(f"{chain}_losses", data=avg_loss.astype(np.float32))
            h5_out.create_dataset(f"{chain}_num_samples_per_pos", data=count.astype(np.float32))

    print(f"Saved scan results → {h5_path}")


if __name__ == "__main__":
    main()
