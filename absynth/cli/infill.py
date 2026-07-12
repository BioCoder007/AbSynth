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
    Infill a region of an antibody sequence with AbSynth-A.
    """
    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(
        description="Infill a region of an antibody sequence with AbSynth-A"
    )
    parser.add_argument("--sequence", type=str, required=True, help="Full antibody sequence")
    parser.add_argument("--infill_range", type=int, nargs=2, required=True, metavar=("START", "END"))
    parser.add_argument("--length_range", type=int, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--output_dir", type=str, default=f"output_dir/infill_{now}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model = AbSynthA(args.model_path, args.tokenizer_path)

    print(f"Infilling region {tuple(args.infill_range)}...")
    sequences = model.infill(
        args.sequence,
        infill_range=tuple(args.infill_range),
        length_range=tuple(args.length_range) if args.length_range else None,
        n=args.n,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    out_fasta = os.path.join(args.output_dir, "infilled_seqs.fasta")
    with open(out_fasta, "w") as fasta:
        for i, seq in enumerate(sequences):
            print(f">seq_{i}", file=fasta)
            print(seq, file=fasta)

    print(f"Saved {len(sequences)} sequences -> {out_fasta}")


if __name__ == "__main__":
    main()
