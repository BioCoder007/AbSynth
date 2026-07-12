# AbSynth

**Generative Framework for Humanised Antibody Sequence Design**

AbSynth provides two complementary antibody language models:

| Model | Architecture | Capabilities |
|---|---|---|
| **AbSynth-A** | GPT-2 (autoregressive) | De novo generation, CDR infilling, perplexity scoring |
| **AbSynth-M** | RoBERTa (masked LM) | Sequence recovery, region redesign, humanness optimization, embeddings |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/BioCoder007/AbSynth.git
cd AbSynth

# Create conda environment
conda env create -f environment.yml
conda activate absynth

# Install as a package
pip install -e .
```

---

## Quick Start

### AbSynth-A — Autoregressive model

```python
from absynth import AbSynthA

model = AbSynthA(
    model_path="absynth/trained_models/absynth-a",
    tokenizer_path="absynth/tokenizer",
)

# De novo generation
seqs = model.generate(n=10, prompt="EVQ")

# CDR infilling
seqs = model.infill(
    sequence="EVQLVQSGGGLVQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDREDIVVVPAPRGYYYYYYMDVWGQGTTVTVSS",
    infill_range=(105, 125),
    length_range=(5, 15),
    n=5,
)

# Perplexity scoring
ppls = model.perplexity(seqs)
```

### AbSynth-M — Masked model

```python
from absynth import AbSynthM

model = AbSynthM(
    model_path="absynth/trained_models/absynth-m",
    tokenizer_path="absynth/tokenizer",
)

# Recover missing residues (X = unknown)
seqs = model.recover("EVQLQQSGXELAXXGASVXMXCK...", n=5)

# Redesign a region
seqs = model.redesign(
    sequence="EVQLQQSGAELARPGASVKMSCKASGYTFTS...",
    redesign_range=(95, 110),
    length_range=(5, 15),
    n=5,
)

# Humanness optimization
optimized, history = model.humanize(sequence, top_n=5, rounds=3)

# Extract embeddings
embeddings = model.embed(seqs)   # shape: (n, 768)
```

---

## Command-Line Tools

```bash
# Generate sequences
absynth-generate --prompt EVQ --n 10

# Infill a region
absynth-infill --sequence "EVQLVQ...VSS" --infill_range 95 106 --length_range 5 15

# Redesign a region
absynth-redesign --sequence "EVQLQQ...VSA" --redesign_range 95 110 --length_range 5 15

# Score perplexity (one sequence per line)
absynth-evaluate sequences.txt

# Positional perplexity scan
absynth-scan paired_sequences.csv --output_dir results/scan
absynth-plot-scan results/scan/losses.h5 --output_dir results/plots
```

All commands accept `--model_path`/`--tokenizer_path` overrides and default to the
weights bundled under `absynth/trained_models/`. Each writes its results
(FASTA / CSV) into a timestamped `--output_dir`.

---

## Repository Structure

```
AbSynth/
├── absynth/
│   ├── __init__.py
│   ├── model/
│   │   ├── absynth_a.py       # AbSynth-A model class (autoregressive)
│   │   ├── absynth_m.py       # AbSynth-M model class (masked LM)
│   │   └── utils.py           # shared device resolution helper
│   ├── cli/
│   │   ├── generate.py        # absynth-generate
│   │   ├── infill.py          # absynth-infill
│   │   ├── redesign.py        # absynth-redesign
│   │   └── evaluate.py        # absynth-evaluate
│   ├── utils/
│   │   └── seed.py            # reproducibility helper
│   ├── tokenizer/              # shared tokenizer
│   └── trained_models/
│       ├── absynth-a/          # AutoregressiveLM
│       └── absynth-m/          # MaskedLM
├── analysis/
│   ├── scan_evaluate.py         # positional perplexity scan (absynth-scan)
│   ├── plot_scan.py             # scan visualization (absynth-plot-scan)
│   └── plot_embeddings.py       # token embedding PCA / t-SNE
│   
├── AbSynth.ipynb                # interactive demo notebook
├── LICENSE
├── MANIFEST.in
├── environment.yml
├── pyproject.toml
├── setup.cfg
├── README.md
├── scan_demo.ipynb          # end-to-end scan notebook
└── plot_embeddings_demo.ipynb
```

---

## Analysis

### Positional Perplexity Scan

Visualize where AbSynth-A finds sequences hardest to predict — peaks correspond to CDR loops.

```bash
# Step 1: run the scan (saves losses.h5)
python analysis/scan_evaluate.py paired_sequences.csv --output_dir results/scan

# Step 2: plot
python analysis/plot_scan.py results/scan/losses.h5 --output_dir results/plots
```

Or run interactively in [scan_demo.ipynb](scan_demo.ipynb).

### Token Embedding Visualization

Project the learned amino acid embeddings to 2D (PCA / t-SNE), color-coded by
hydropathy class, for either model:

```bash
python analysis/plot_embeddings.py --model autoregressive
python analysis/plot_embeddings.py --model masked
```

Or run interactively in [plot_embeddings_demo.ipynb](plot_embeddings_demo.ipynb).

---

## Models

Pre-trained model weights are **not** included in this repository due to file size.  
Download them separately and place under `absynth/trained_models/`.

| Model | Architecture | Parameters |
|---|---|---|
| absynth-a | GPT-2 | 21.9M |
| absynth-m | RoBERTa | 17.7M |

---

## License

AbSynth is released under a custom non-commercial license.
Free for academic research and education. Commercial use requires a paid license.
See [LICENSE](./LICENSE) for full terms. Contact amman.safeer00@gmail.com for commercial inquiries.

---

## Citation

If you use AbSynth in your work, please cite:

```bibtex
@misc{absynth2024,
  title  = {AbSynth: Generative Framework for Humanised Antibody Sequence Design},
  year   = {2024},
}
```
