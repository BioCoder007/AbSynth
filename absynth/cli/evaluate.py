import argparse
import os
from datetime import datetime

import absynth
from absynth import AbSynthA
from absynth.utils.seed import set_seeds

package_dir = os.path.dirname(os.path.realpath(absynth.__file__))
DEFAULT_MODEL_PATH = os.path.join(package_dir, "trained_models", "absynth-a")
DEFAULT_TOKENIZER_PATH = os.path.join(package_dir, "tokenizer")


def main():
    """
    Score antibody sequences by AbSynth-A perplexity.
    """
    parser = argparse.ArgumentParser(
        description="Score antibody sequences by AbSynth-A perplexity"
    )
    parser.add_argument("sequences_file", type=str, help="Text file with one sequence per line")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="If set, also saves a ranked CSV here")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seeds(args.seed)

    with open(args.sequences_file) as f:
        sequences = [line.strip() for line in f if line.strip()]

    model = AbSynthA(args.model_path, args.tokenizer_path)

    print(f"Scoring {len(sequences)} sequences...")
    ppls = model.perplexity(sequences)
    ranked = sorted(zip(ppls, sequences), key=lambda x: x[0])

    print(f"\n{'Rank':>4}  {'PPL':>8}  Sequence")
    print("-" * 80)
    for rank, (ppl, seq) in enumerate(ranked, 1):
        print(f"{rank:>4}  {ppl:>8.3f}  {seq}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_csv = os.path.join(args.output_dir, "perplexity.csv")
        with open(out_csv, "w") as f:
            f.write("rank,perplexity,sequence\n")
            for rank, (ppl, seq) in enumerate(ranked, 1):
                f.write(f"{rank},{ppl:.6f},{seq}\n")
        print(f"\nSaved ranked scores -> {out_csv}")


if __name__ == "__main__":
    main()
