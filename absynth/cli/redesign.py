import argparse
import os
from datetime import datetime

import absynth
from absynth import AbSynthM
from absynth.utils.seed import set_seeds

package_dir = os.path.dirname(os.path.realpath(absynth.__file__))
DEFAULT_MODEL_PATH = os.path.join(package_dir, "trained_models", "absynth-m")
DEFAULT_TOKENIZER_PATH = os.path.join(package_dir, "tokenizer")


def main():
    """
    Redesign a region of an antibody sequence with AbSynth-M.
    """
    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(
        description="Redesign a region of an antibody sequence with AbSynth-M"
    )
    parser.add_argument("--sequence", type=str, required=True, help="Full antibody sequence")
    parser.add_argument("--redesign_range", type=int, nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--length_range", type=int, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--output_dir", type=str, default=f"output_dir/redesign_{now}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model = AbSynthM(args.model_path, args.tokenizer_path)

    print(f"Redesigning region {tuple(args.redesign_range)}...")
    sequences = model.redesign(
        args.sequence,
        redesign_range=tuple(args.redesign_range),
        length_range=tuple(args.length_range) if args.length_range else None,
        n=args.n,
        temperature=args.temperature,
    )

    out_fasta = os.path.join(args.output_dir, "redesigned_seqs.fasta")
    with open(out_fasta, "w") as fasta:
        for i, seq in enumerate(sequences):
            print(f">seq_{i}", file=fasta)
            print(seq, file=fasta)

    print(f"Saved {len(sequences)} sequences -> {out_fasta}")


if __name__ == "__main__":
    main()
