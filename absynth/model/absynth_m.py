"""
AbSynth-M — Masked language model module (RoBERTa)

Handles:
    - Sequence recovery: fill missing residues marked as X
    - Region redesign: mask and resample a (start, end) span
    - Humanness optimization: iteratively resample low-confidence positions
    - Embedding extraction: per-sequence hidden-state vectors
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizer
from typing import List, Tuple, Optional

from .utils import resolve_device


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


class AbSynthM:
    """
    AbSynth-M: masked antibody language model.

    Load once, call recover / redesign / humanize / embed as needed.

    Args:
        model_path:     Path to AbSynth-M HuggingFace model folder.
        tokenizer_path: Path to AbSynth tokenizer folder.
        device:         "auto", "cpu", "cuda", or "mps".

    Example:
        model = AbSynthM("absynth/trained_models/absynth-m", "absynth/tokenizer")
        seqs  = model.recover("EVQLQQSGXELAX...VSA", n=5)
        seqs  = model.redesign("EVQLQQSG...VSA", redesign_range=(95, 110), length_range=(5, 15), n=5)
        opt   = model.humanize("EVQLQQSG...VSA", top_n=5, rounds=3)
        vecs  = model.embed(["EVQLQQSG...VSA"])
    """

    def __init__(self, model_path: str, tokenizer_path: str, device: str = "auto"):
        self.device = resolve_device(device)

        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path, max_len=150)
        self.model = RobertaForMaskedLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Cache valid amino acid token IDs once at load time
        self._valid_ids = torch.tensor(
            [self.tokenizer.encode(aa, add_special_tokens=False)[0] for aa in AMINO_ACIDS],
            dtype=torch.long,
        ).to(self.device)

        print(f"AbSynth-M loaded | device={self.device} | vocab={self.tokenizer.vocab_size} | params={sum(p.numel() for p in self.model.parameters()):,}")

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def recover(
        self,
        sequence: str,
        n: int = 5,
        temperature: float = 0.5,
    ) -> List[str]:
        """
        Recover missing residues marked as X in a sequence.

        All X positions are masked simultaneously and sampled in one
        forward pass.

        Args:
            sequence:    Antibody sequence with X at missing positions.
            n:           Number of recovered sequences to return.
            temperature: Sampling temperature. Lower = picks highest-prob AA.

        Returns:
            List of recovered sequences (X positions filled).
        """
        x_positions = [i for i, aa in enumerate(sequence) if aa == "X"]
        if not x_positions:
            raise ValueError("No X positions found in sequence.")

        masked_seq = sequence.replace("X", self.tokenizer.mask_token)
        input_ids = self.tokenizer.encode(
            masked_seq, add_special_tokens=True, return_tensors="pt"
        ).to(self.device)

        # The masked input never changes across draws, so the forward pass
        # only needs to run once — only the sampling below varies per draw.
        with torch.no_grad():
            logits = self.model(input_ids).logits[0]

        seen = set()
        results = []

        for _ in tqdm(range(n), desc="Recovering"):
            recovered_ids = input_ids[0].clone()
            for pos in x_positions:
                recovered_ids[pos + 1] = self._sample_token(logits[pos + 1], temperature)

            seq = self.tokenizer.decode(recovered_ids, skip_special_tokens=True).replace(" ", "")
            if seq not in seen:
                seen.add(seq)
                results.append(seq)

        return results

    # ------------------------------------------------------------------
    # Redesign
    # ------------------------------------------------------------------

    def redesign(
        self,
        sequence: str,
        redesign_range: Tuple[int, int],
        length_range: Optional[Tuple[int, int]] = None,
        length_tolerance: int = 2,
        n: int = 5,
        temperature: float = 0.6,
    ) -> List[str]:
        """
        Redesign a region of a sequence by masking and resampling.

        Args:
            sequence:          Full antibody sequence.
            redesign_range:    (start, end) character positions to redesign.
            length_range:      (min_len, max_len) for the redesigned region.
                               Defaults to (original_len - length_tolerance,
                               original_len + length_tolerance).
            length_tolerance:  Half-width of the auto length window. Default: 2.
            n:                 Number of redesigned sequences to return.
            temperature:       Sampling temperature.

        Returns:
            List of unique redesigned sequences.
        """
        start, end = redesign_range
        if start < 0 or end > len(sequence) or start >= end:
            raise ValueError(
                f"redesign_range ({start}, {end}) is invalid for sequence of length {len(sequence)}."
            )

        original_region_len = end - start
        if length_range is None:
            lo = max(1, original_region_len - length_tolerance)
            hi = original_region_len + length_tolerance
            length_range = (lo, hi)

        prefix = sequence[:start]
        suffix = sequence[end:]

        seen = set()
        results = []

        for _ in tqdm(range(n), desc="Redesigning"):
            region_length = torch.randint(length_range[0], length_range[1] + 1, (1,)).item()

            masked_seq = prefix + self.tokenizer.mask_token * region_length + suffix
            input_ids = self.tokenizer.encode(
                masked_seq, add_special_tokens=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids).logits[0]

            region = ""
            for pos in range(region_length):
                token_pos = start + pos + 1  # +1 for CLS
                region += self.tokenizer.decode(
                    [self._sample_token(logits[token_pos], temperature)]
                ).strip()

            full_seq = prefix + region + suffix
            if full_seq not in seen:
                seen.add(full_seq)
                results.append(full_seq)

        return results

    # ------------------------------------------------------------------
    # Humanness optimization
    # ------------------------------------------------------------------

    def humanize(
        self,
        sequence: str,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        rounds: int = 1,
        temperature: float = 0.3,
    ) -> Tuple[str, list]:
        """
        Optimize a sequence for humanness by iteratively resampling
        low-confidence positions.

        Provide either threshold OR top_n, not both.

        Args:
            sequence:    Antibody sequence to humanize.
            threshold:   Mask positions where MLM probability < threshold.
            top_n:       Mask the N lowest-confidence positions per round.
            rounds:      Number of iterative optimization rounds.
            temperature: Sampling temperature for resampling.

        Returns:
            Tuple of (optimized_sequence, history).
            history: list of (round, sequence, avg_confidence).
        """
        if threshold is None and top_n is None:
            raise ValueError("Provide either threshold or top_n.")
        if threshold is not None and top_n is not None:
            raise ValueError("Provide either threshold or top_n, not both.")

        current_seq = sequence
        history = []

        for r in range(1, rounds + 1):
            scores = self._score_positions(current_seq)
            avg_conf = sum(s[2] for s in scores) / len(scores)
            history.append((r, current_seq, avg_conf))

            print(f"Round {r}/{rounds} | avg confidence: {avg_conf:.4f}")

            to_mask = (
                [s for s in scores if s[2] < threshold]
                if threshold is not None
                else sorted(scores, key=lambda x: x[2])[:top_n]
            )

            if not to_mask:
                print(f"  No positions below threshold — stopping early at round {r}.")
                break

            mask_positions = [s[0] for s in to_mask]
            print(f"  Masking {len(mask_positions)} positions: {mask_positions}")

            seq_list = list(current_seq)
            for pos in mask_positions:
                seq_list[pos] = self.tokenizer.mask_token
            masked_seq = "".join(seq_list)

            input_ids = self.tokenizer.encode(
                masked_seq, add_special_tokens=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids).logits[0]

            for pos in mask_positions:
                seq_list[pos] = self.tokenizer.decode(
                    [self._sample_token(logits[pos + 1], temperature)]
                ).strip()

            current_seq = "".join(seq_list)

        final_scores = self._score_positions(current_seq)
        final_conf = sum(s[2] for s in final_scores) / len(final_scores)
        history.append((rounds + 1, current_seq, final_conf))
        print(f"\nFinal avg confidence: {final_conf:.4f}")

        return current_seq, history

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def embed(
        self,
        sequences: List[str],
        layer: int = -1,
        pooling: str = "mean",
    ) -> np.ndarray:
        """
        Extract per-sequence embeddings from AbSynth-M hidden states.

        Args:
            sequences: List of amino acid sequence strings.
            layer:     Hidden layer index. -1 = last layer.
            pooling:   'mean' (recommended) or 'cls'.

        Returns:
            np.ndarray of shape (len(sequences), hidden_size).
        """
        embeddings = []

        with torch.no_grad():
            for seq in tqdm(sequences, desc="Embedding"):
                enc = self.tokenizer(
                    seq, return_tensors="pt", truncation=True, max_length=150
                ).to(self.device)

                outputs = self.model(**enc, output_hidden_states=True)
                hidden = outputs.hidden_states[layer]

                if pooling == "cls":
                    vec = hidden[:, 0, :]
                elif pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                else:
                    raise ValueError(f"Unknown pooling: {pooling}. Use 'mean' or 'cls'.")

                embeddings.append(vec.squeeze(0).cpu())

        return torch.stack(embeddings).numpy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sample_token(self, logits: torch.Tensor, temperature: float) -> int:
        filtered = torch.full_like(logits, float("-inf"))
        filtered[self._valid_ids] = logits[self._valid_ids]
        probs = F.softmax(filtered / temperature, dim=-1)
        return torch.multinomial(probs, 1).item()

    def _score_positions(self, sequence: str) -> list:
        scores = []
        for i, aa in enumerate(sequence):
            masked = sequence[:i] + self.tokenizer.mask_token + sequence[i + 1:]
            input_ids = self.tokenizer.encode(
                masked, add_special_tokens=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids).logits[0, i + 1]

            filtered = torch.full_like(logits, float("-inf"))
            filtered[self._valid_ids] = logits[self._valid_ids]
            probs = F.softmax(filtered, dim=-1)

            current_id = self.tokenizer.encode(aa, add_special_tokens=False)[0]
            scores.append((i, aa, probs[current_id].item()))

        return scores

    def __repr__(self):
        return f"AbSynthM(device={self.device})"
