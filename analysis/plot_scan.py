"""
plot_scan.py — Visualize positional infilling perplexity from scan_evaluate.py.

Usage:
    python plot_scan.py path/to/losses.h5
    python plot_scan.py path/to/losses.h5 --output_dir results/plots
"""

import argparse
import os
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():
    now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    parser = argparse.ArgumentParser(
        description="Plot per-position infilling perplexity from scan_evaluate.py output"
    )
    parser.add_argument("scan_h5", type=str, help="HDF5 output from scan_evaluate.py")
    parser.add_argument("--output_dir", type=str, default=f"output_dir/plot_scan_{now}")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with h5py.File(args.scan_h5, "r") as f:
        heavy_scores = np.array(f["Heavy_losses"])
        light_scores = np.array(f["Light_losses"])
        heavy_n = np.array(f["Heavy_num_samples_per_pos"])
        light_n = np.array(f["Light_num_samples_per_pos"])

    # Convert loss → perplexity
    heavy_scores = np.exp(heavy_scores)
    light_scores = np.exp(light_scores)

    # Mask positions with insufficient samples
    heavy_scores[heavy_n <= 50] = np.nan
    light_scores[light_n <= 50] = np.nan

    plt.figure(figsize=(12, 5))
    plt.plot(heavy_scores, color="blueviolet", label="Heavy chain")
    plt.plot(light_scores, color="royalblue", label="Light chain")
    plt.legend(fontsize=16)

    max_len = max(len(heavy_scores), len(light_scores))
    skip_10 = np.arange(0, max_len, 10)
    plt.xticks(skip_10, skip_10)

    plt.ylabel("Model infilling perplexity", fontsize=14)
    plt.xlabel("Mask position along sequence", fontsize=14)
    plt.title("AbSynth-A Infilling Perplexity", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, "scan.pdf")
    plt.savefig(out_path, dpi=400, transparent=True)
    plt.show()
    plt.close("all")

    print(f"Plot saved → {out_path}")


if __name__ == "__main__":
    main()
