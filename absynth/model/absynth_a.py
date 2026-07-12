"""
AbSynth-A — Autoregressive module (GPT2LMHeadModel)

Handles:
    - De novo sequence generation (prompted or unprompted)
    - Region infilling: user specifies (start, end) + length range
    - Perplexity scoring
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel, RobertaTokenizerFast
from typing import List, Tuple, Optional

from .utils import resolve_device


_SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<mask>", "<unk>", "<<", ">>"]


class AbSynthA:
    """
    AbSynth-A: autoregressive antibody language model.

    Load once, call generate / infill / perplexity as needed.

    Args:
        model_path:     Path to AbSynth-A HuggingFace model folder.
        tokenizer_path: Path to AbSynth tokenizer folder.
        device:         "auto", "cpu", "cuda", or "mps".

    Example:
        model = AbSynthA("models/AbSynth-A", "tokenizer/")
        seqs  = model.generate(n=10, prompt="EVQ")
        seqs  = model.infill("EVQLVESGG...VSS", infill_range=(95, 106), length_range=(5, 15), n=5)
        ppl   = model.perplexity(seqs)
    """

    def __init__(self, model_path: str, tokenizer_path: str, device: str = "auto"):
        self.device = resolve_device(device)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Cache blocked token IDs once at load time
        vocab = self.tokenizer.get_vocab()
        self._blocked_ids = torch.tensor(
            [vocab[t] for t in _SPECIAL_TOKENS if t in vocab],
            dtype=torch.long,
        ).to(self.device)

        print(f"AbSynth-A loaded | device={self.device} | vocab={self.tokenizer.vocab_size} | params={sum(p.numel() for p in self.model.parameters()):,}")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        n: int = 10,
        prompt: Optional[str] = None,
        max_length: int = 150,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.90,
        repetition_penalty: float = 1.2,
    ) -> List[str]:
        """
        Generate de novo antibody sequences.

        Args:
            n:                  Number of sequences to generate.
            prompt:             Optional seed string (e.g. "EVQ", "DIV").
                                If None, generation starts from BOS token.
            max_length:         Maximum total sequence length in tokens.
            temperature:        Sampling temperature. Lower = higher quality,
                                higher = more diversity. Range: 0.7–1.2.
            top_k:              Top-k candidates per step.
            top_p:              Nucleus sampling threshold. Keep at 0.9–0.95.
                                Do NOT lower below 0.85 — causes repetition collapse.
            repetition_penalty: Penalises tokens already in the output (>1.0).
                                Keep modest (1.1–1.3) to avoid quality degradation.

        Returns:
            List of unique generated amino acid sequence strings.
        """
        input_text = prompt if prompt else self.tokenizer.bos_token or ""
        encoded = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        seen = set()
        sequences = []

        with torch.no_grad():
            for _ in tqdm(range(n), desc="Generating"):
                output = self.model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                )
                text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
                if text not in seen:
                    seen.add(text)
                    sequences.append(text)

        return sequences

    # ------------------------------------------------------------------
    # Infilling
    # ------------------------------------------------------------------

    def infill(
        self,
        sequence: str,
        infill_range: Tuple[int, int],
        length_range: Optional[Tuple[int, int]] = None,
        length_tolerance: int = 2,
        n: int = 10,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.92,
        repetition_penalty: float = 1.1,
    ) -> List[str]:
        """
        Infill a region of a sequence using autoregressive generation.

        The model sees the prefix up to infill_range[0], generates tokens
        one-by-one for a sampled length within length_range, then appends
        the original suffix from infill_range[1] onward.

        Args:
            sequence:           Full antibody sequence string.
            infill_range:       (start, end) character positions to replace.
                                E.g. (95, 106) replaces characters at positions 95–105.
            length_range:       (min_len, max_len) for the infilled region.
                                Defaults to (original_len - length_tolerance,
                                original_len + length_tolerance) so output length
                                stays close to the parent sequence.
            length_tolerance:   Half-width of the auto length window around the
                                original region length. Only used when length_range
                                is None. E.g. tolerance=2 on a 20-aa region gives
                                length_range=(18, 22). Default: 2.
            n:                  Number of sequences to generate.
            temperature:        Sampling temperature. Lower = more human-like
                                conservative choices. Default 0.7.
            top_k:              Top-k candidates per step.
            top_p:              Nucleus sampling threshold. Keeps the smallest
                                set of tokens whose cumulative probability >= top_p.
            repetition_penalty: Penalise tokens already generated in this region.
                                Keep close to 1.0 (default 1.1) to avoid forcing
                                unusual amino acids.

        Returns:
            List of unique infilled full-length sequence strings.
        """
        start, end = infill_range

        if start < 0 or end > len(sequence) or start >= end:
            raise ValueError(
                f"infill_range ({start}, {end}) is invalid for sequence "
                f"of length {len(sequence)}."
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

        for _ in tqdm(range(n), desc="Infilling"):
            region_length = torch.randint(
                length_range[0], length_range[1] + 1, (1,)
            ).item()

            generated_region = ""
            generated_token_ids = []

            for _ in range(region_length):
                current_input = prefix + generated_region
                # BOS only — do NOT add EOS here; EOS at end of prefix breaks
                # autoregressive generation by making the model generate after </s>
                token_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(
                    current_input, add_special_tokens=False
                )
                input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    # Always last token position — fixes original notebook IndexError
                    logits = outputs.logits[0, -1].clone()

                # Repetition penalty on all tokens generated so far in this region
                for token_id in set(generated_token_ids):
                    logits[token_id] /= repetition_penalty

                # Block special tokens, then apply temperature + top-k + top-p
                logits[self._blocked_ids] = float("-inf")
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)

                # Top-k filter
                safe_top_k = min(top_k, probs.shape[-1])
                top_k_vals, top_k_idx = torch.topk(probs, safe_top_k)
                filtered = torch.zeros_like(probs)
                filtered[top_k_idx] = top_k_vals

                # Nucleus (top-p) filter — keeps highest-prob tokens summing to top_p
                sorted_probs, sorted_idx = torch.sort(filtered, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                cutoff_mask = cumulative - sorted_probs > top_p
                sorted_probs[cutoff_mask] = 0.0
                filtered = torch.zeros_like(probs)
                filtered[sorted_idx] = sorted_probs

                new_token_id = torch.multinomial(filtered, 1).item()
                new_token_str = self.tokenizer.decode([new_token_id]).strip()

                generated_region += new_token_str
                generated_token_ids.append(new_token_id)

            full_seq = prefix + generated_region + suffix
            if full_seq not in seen:
                seen.add(full_seq)
                results.append(full_seq)

        return results

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    def perplexity(self, sequences: List[str]) -> List[float]:
        """
        Compute per-sequence perplexity under the GPT-2 causal LM.

        Lower perplexity = sequence more consistent with the training distribution.
        Use this to rank and filter generated or infilled sequences.

        Args:
            sequences: List of amino acid sequence strings.

        Returns:
            List of float perplexity values (one per sequence).
        """
        scores = []
        with torch.no_grad():
            for seq in sequences:
                enc = self.tokenizer(
                    seq,
                    return_tensors="pt",
                    truncation=True,
                    max_length=150,
                ).to(self.device)
                input_ids = enc["input_ids"]
                outputs = self.model(**enc, labels=input_ids)
                scores.append(math.exp(outputs.loss.item()))
        return scores

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
        Extract per-sequence embeddings from AbSynth-A hidden states.

        Args:
            sequences: List of amino acid sequence strings.
            layer:     Hidden layer index. -1 = last layer.
            pooling:   'mean' (recommended, averages over all token positions)
                       or 'last' (final non-pad token's hidden state — the
                       summary position for a causal LM).

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

                if pooling == "mean":
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
                elif pooling == "last":
                    lengths = enc["attention_mask"].sum(dim=1) - 1
                    vec = hidden[torch.arange(hidden.size(0)), lengths]
                else:
                    raise ValueError(f"Unknown pooling: {pooling}. Use 'mean' or 'last'.")

                embeddings.append(vec.squeeze(0).cpu())

        return torch.stack(embeddings).numpy()

    def __repr__(self):
        return f"AbSynthA(device={self.device})"
