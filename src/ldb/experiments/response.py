"""Corpus-conditioned response comparisons; similarity is not an equivalence proof."""

import argparse
import json
from pathlib import Path

import numpy as np

from .expert_compaction.selection import choose_map, response_costs


def compare(outputs, routes):
    weighted, distance = response_costs(outputs, routes)
    flat = np.asarray(outputs, dtype=np.float64).reshape(len(outputs), -1)
    norms = np.linalg.norm(flat, axis=1)
    denominator = norms[:, None] * norms[None, :]
    cosine = np.divide(
        flat @ flat.T, denominator, out=np.zeros_like(distance), where=denominator != 0
    )
    return {
        "cosine": cosine.tolist(),
        "mean_squared_distance": distance.tolist(),
        "routing_weighted_error": weighted.tolist(),
        "scope": "corpus_conditioned; functional compatibility unverified",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NPZ: outputs [expert, token, feature], routes [token, expert]",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remove", type=int, default=0)
    parser.add_argument(
        "--method",
        choices=["fingerprint", "frequency", "random", "random_donor"],
        default="fingerprint",
    )
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        result = compare(data["outputs"], data["routes"])
        result["mapping"] = choose_map(
            result["routing_weighted_error"],
            np.count_nonzero(data["routes"], axis=0),
            args.remove,
            args.method,
        )
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
