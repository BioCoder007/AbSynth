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
    Generate de novo antibody sequences with AbSynth-A.
    """
    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(
        description="Generate de novo antibody sequences with AbSynth-A"
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--prompt", type=str, default=None, help="Optional seed sequence, e.g. 'EVQ'")
    parser.add_argument("--n", type=int, default=10, help="Number of sequences to generate")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--output_dir", type=str, default=f"output_dir/generate_{now}")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model = AbSynthA(args.model_path, args.tokenizer_path)

    print(f"Generating {args.n} sequences...")
    sequences = model.generate(
        n=args.n,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    out_fasta = os.path.join(args.output_dir, "generated_seqs.fasta")
    with open(out_fasta, "w") as fasta:
        for i, seq in enumerate(sequences):
            print(f">seq_{i}", file=fasta)
            print(seq, file=fasta)

    print(f"Saved {len(sequences)} sequences -> {out_fasta}")


if __name__ == "__main__":
    main()
