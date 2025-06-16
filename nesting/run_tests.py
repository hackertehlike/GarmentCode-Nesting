import sys, os
# add project root to sys.path for local package imports
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
from pathlib import Path
import copy
import matplotlib.pyplot as plt
import pandas as pd
from shapely.errors import TopologicalError as TopologyException

from nesting.path_extractor import PatternPathExtractor
from nesting.layout import Layout, Container, Piece
from nesting.placement_engine import RandomDecoder, NFPDecoder
import nesting.config as config


def flush_results(rows: list[dict]) -> None:
    """Write metrics CSV and update the utilization plot."""
    df = pd.DataFrame(rows)
    df.to_csv("utilization_metrics.csv", index=False)

    if df.empty:
        return

    # Scatter plot of each run: bounding-box vs concave-hull utilization
    colors = {'BL': 'blue', 'NFP': 'orange'}
    fig, ax = plt.subplots(figsize=(6, 6))
    for alg, color in colors.items():
        for split_flag, marker in [(False, 'o'), (True, 'x')]:
            subset = df[(df['algorithm'] == alg) & (df['split'] == split_flag)]
            label = f"{alg}{' split' if split_flag else ''}"
            ax.scatter(subset['usage_bb'], subset['concave_hull'],
                       color=color, marker=marker, alpha=0.7, label=label)
    ax.set_xlabel('Bounding-box utilization')
    ax.set_ylabel('Concave-hull utilization')
    ax.set_title('Utilization per run')
    ax.legend()
    # save scatter chart
    fig.savefig('utilization_scatter.png', dpi=300)
    plt.close(fig)

    # Now plot average metrics by algorithm and split flag
    summary = df.groupby(['algorithm', 'split']).agg(
        avg_bb=('usage_bb', 'mean'),
        avg_hull=('concave_hull', 'mean')
    ).reset_index()
    labels = [f"{row.algorithm}{' split' if row.split else ''}" for _, row in summary.iterrows()]
    colors = ['blue' if alg == 'BL' else 'orange' for alg in summary['algorithm']]
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
    # Average bounding-box utilization
    axes2[0].bar(labels, summary['avg_bb'], color=colors)
    axes2[0].set_title('Avg Bounding-box Utilization')
    axes2[0].set_ylabel('Utilization')
    axes2[0].tick_params(axis='x', rotation=45)
    # Average concave-hull utilization
    axes2[1].bar(labels, summary['avg_hull'], color=colors)
    axes2[1].set_title('Avg Concave-hull Utilization')
    axes2[1].tick_params(axis='x', rotation=45)
    fig2.tight_layout()
    fig2.savefig('utilization_avg.png', dpi=300)
    plt.close(fig2)


def load_pieces(json_path: Path) -> dict[str, Piece]:
    extractor = PatternPathExtractor(json_path)
    pieces = extractor.get_all_panel_pieces(samples_per_edge=config.SAMPLES_PER_EDGE)
    for p in pieces.values():
        p.add_seam_allowance(config.SEAM_ALLOWANCE_CM)
    return pieces


def split_pieces(pieces: dict[str, Piece]) -> dict[str, Piece]:
    new_pieces: dict[str, Piece] = {}
    for piece in pieces.values():
        left, right = piece.split()
        new_pieces[left.id] = left
        new_pieces[right.id] = right
    return new_pieces


def run_decoder(layout: Layout, container: Container, decoder_cls) -> dict:
    dec = decoder_cls(copy.deepcopy(layout), container, step=config.GRAVITATE_STEP)
    dec.decode()
    return {
        "usage_bb": dec.usage_BB(),
        "concave_hull": dec.concave_hull_utilization(),
        "rest_length": dec.rest_length(),
    }


def run_random_bl(pieces: dict[str, Piece], container: Container) -> dict:
    return run_decoder(Layout(pieces), container, RandomDecoder)


def run_random_nfp(pieces: dict[str, Piece], container: Container) -> dict:
    ids = list(pieces.keys())
    random.shuffle(ids)
    shuffled = {pid: pieces[pid] for pid in ids}
    return run_decoder(Layout(shuffled), container, NFPDecoder)


def main():
    data_dir = Path("./nesting-assets/garmentcodedata_batch0")
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {data_dir}.")
        return

    container = Container(config.CONTAINER_WIDTH_CM, config.CONTAINER_HEIGHT_CM)
    rows: list[dict] = []

    for json_path in json_files:
        print(f"Processing {json_path}")
        pieces = load_pieces(json_path)

        # BL runs before split, skip runs with topology errors
        for i in range(3):
            res = run_random_bl(copy.deepcopy(pieces), container)
            rows.append({"pattern": json_path.stem, "algorithm": "BL", "split": False, "run": i, **res})
            flush_results(rows)
        # NFP runs before split, skip runs with topology errors
        for i in range(3):
            try:
                res = run_random_nfp(copy.deepcopy(pieces), container)
            except TopologyException as e:
                print(f"Skipping NFP run {i} before split due to topology error: {e}")
                continue
            rows.append({"pattern": json_path.stem, "algorithm": "NFP", "split": False, "run": i, **res})
            flush_results(rows)

        split_dict = split_pieces(copy.deepcopy(pieces))
        # BL runs after split, skip runs with topology errors
        for i in range(3):
            res = run_random_bl(copy.deepcopy(split_dict), container)
            rows.append({"pattern": json_path.stem, "algorithm": "BL", "split": True, "run": i, **res})
            flush_results(rows)
        # NFP runs after split, skip runs with topology errors
        for i in range(3):
            try:
                res = run_random_nfp(copy.deepcopy(split_dict), container)
            except TopologyException as e:
                print(f"Skipping NFP run {i} after split due to topology error: {e}")
                continue
            rows.append({"pattern": json_path.stem, "algorithm": "NFP", "split": True, "run": i, **res})
            flush_results(rows)

    # Final flush to ensure results are saved
    flush_results(rows)
    print("Results saved to utilization_metrics.csv and utilization_metrics.png")


if __name__ == "__main__":
    main()
