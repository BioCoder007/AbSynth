"""
plot_embeddings.py — Visualize token embedding space of AbSynth-A or AbSynth-M.

Extracts the input embedding layer weights for the 20 standard amino acids,
projects them to 2D with PCA and t-SNE, and saves color-coded scatter plots
annotated by amino acid hydropathy class.

Usage:
    # AbSynth-A (GPT-2)
    python plot_embeddings.py --model autoregressive \
        --model_path absynth/trained_models/absynth-a \
        --tokenizer_path absynth/tokenizer

    # AbSynth-M (RoBERTa)
    python plot_embeddings.py --model masked \
        --model_path absynth/trained_models/absynth-m \
        --tokenizer_path absynth/tokenizer
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import GPT2LMHeadModel, RobertaForMaskedLM, RobertaTokenizer

from absynth.utils.seed import set_seeds

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

HYDROPATHY = {
    "Hydrophobic (aliphatic)": ["A", "I", "L", "M", "V"],
    "Hydrophobic (aromatic)":  ["F", "W", "Y"],
    "Polar neutral":           ["N", "Q", "S", "T", "C"],
    "Positive":                ["H", "K", "R"],
    "Negative":                ["D", "E"],
    "Special":                 ["G", "P"],
}

# Invert to aa → class
AA_HYDROPATHY = {aa: cls for cls, aas in HYDROPATHY.items() for aa in aas}


def load_aa_embeddings(model_type: str, model_path: str, tokenizer_path: str):
    """
    Extract the input embedding weights for the 20 standard amino acids.

    Returns:
        weights:  np.ndarray (20, hidden_size)
        labels:   list of 20 single-character AA strings
    """
    if model_type == "autoregressive":
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        emb_matrix = model.get_input_embeddings().weight.detach().cpu()
    elif model_type == "masked":
        model = RobertaForMaskedLM.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        emb_matrix = model.get_input_embeddings().weight.detach().cpu()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'autoregressive' or 'masked'.")

    weights, labels = [], []
    for aa in AMINO_ACIDS:
        token_ids = tokenizer.encode(aa, add_special_tokens=False)
        if not token_ids:
            continue
        weights.append(emb_matrix[token_ids[0]].numpy())
        labels.append(aa)

    return np.array(weights), labels


def build_embedding_dataframe(
    weights: np.ndarray,
    labels: list,
    projection: str,
    x_label: str,
    y_label: str,
    tsne_seed: int = 3,
) -> pd.DataFrame:
    if projection == "tsne":
        proj = TSNE(n_components=2, perplexity=5, random_state=tsne_seed)
    elif projection == "pca":
        proj = PCA(n_components=2)
    else:
        raise ValueError(f"Unsupported projection: {projection}. Use 'tsne' or 'pca'.")

    coords = proj.fit_transform(weights)

    df = pd.DataFrame({
        "Residue": labels,
        x_label: coords[:, 0].astype(float),
        y_label: coords[:, 1].astype(float),
    })
    df["Hydropathy"] = df["Residue"].map(AA_HYDROPATHY).fillna("Special")
    return df


def plot_projection(df: pd.DataFrame, labels: list, x_label: str, y_label: str,
                    title: str, out_path: str):
    plt.figure(figsize=(9, 5))
    ax = sns.scatterplot(
        data=df,
        x=x_label,
        y=y_label,
        hue="Hydropathy",
        style="Residue",
        markers={aa: rf"$\bf {aa}$" for aa in labels},
        s=200,
        legend="full",
    )

    handles, lbls = ax.get_legend_handles_labels()
    n_classes = len(df["Hydropathy"].unique())
    ax.legend(
        handles=handles[1 : n_classes + 1],
        labels=lbls[1 : n_classes + 1],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=11,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, transparent=True)
    plt.show()
    plt.close("all")
    print(f"Saved → {out_path}")


def main():
    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(
        description="Plot PCA/t-SNE of AbSynth amino acid token embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["autoregressive", "masked"],
        default="autoregressive",
        help="Which AbSynth model to visualize",
    )
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
    parser.add_argument("--output_dir", type=str, default=f"output_dir/embeddings_{now}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.model} model from {args.model_path} ...")
    weights, labels = load_aa_embeddings(args.model, args.model_path, args.tokenizer_path)
    print(f"Embedding matrix: {weights.shape}  ({len(labels)} amino acids)")

    axis_labels = {"tsne": ("t-SNE X", "t-SNE Y"), "pca": ("PCA X", "PCA Y")}

    for projection in ["pca", "tsne"]:
        x_label, y_label = axis_labels[projection]
        title = f"AbSynth-{'A' if args.model == 'autoregressive' else 'M'} token embeddings — {projection.upper()}"
        df = build_embedding_dataframe(weights, labels, projection, x_label, y_label)
        out_path = os.path.join(args.output_dir, f"embeddings_{projection}.pdf")
        plot_projection(df, labels, x_label, y_label, title, out_path)


if __name__ == "__main__":
    main()
