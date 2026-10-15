"""Compare 3 policies on the stressed phase diagram grid.

Reads nothing; runs everything fresh. Output goes under
``results_phase_diagram/stress_compare_<tag>/`` with one row per
(policy, N, δ, seed) cell plus side-by-side heatmaps.

Policies:
  - score-softmax        : pure score-driven baseline (herding-prone)
  - v4-original          : frozen V4 trained at scale=1.0, δ=0 (baseline_v4)
  - v4-stressed          : frozen V4 trained at scale=0.1, δ=2 (stressed_v4)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_CKPT = ROOT / "results_phase_diagram/baseline_v4/KAIST/20u/uniform/checkpoint_DSR-MTG-Qpen-v4.pt"
DEFAULT_STRESSED_CKPT = ROOT / "results_phase_diagram/stressed_v4/KAIST/20u/uniform/checkpoint_DSR-MTG-Qpen-v4.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--original-ckpt", default=str(DEFAULT_ORIGINAL_CKPT))
    p.add_argument("--stressed-ckpt", default=str(DEFAULT_STRESSED_CKPT))
    p.add_argument("--dataset", default="KAIST")
    p.add_argument("--distribution", default="uniform")
    p.add_argument("--users", default="10,20,50")
    p.add_argument("--delays", default="0,2,8")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--test-steps", type=int, default=1000)
    p.add_argument("--edge-compute-scale", type=float, default=0.1)
    p.add_argument("--output-dir", default="results_phase_diagram/stress_compare")
    p.add_argument("--policies", default="score-softmax,v4-original,v4-stressed")
    return p.parse_args()


def run_phase(policy_tag: str, args: argparse.Namespace, checkpoint: str) -> Path:
    """Invoke run_phase_diagram.py for one policy, return its output dir."""
    out_dir = Path(args.output_dir) / policy_tag
    cmd = [
        sys.executable, "-m", "mappo_gp_sf_dsr.run_phase_diagram",
        "--dataset", args.dataset,
        "--distribution", args.distribution,
        "--users", args.users,
        "--delays", args.delays,
        "--seeds", args.seeds,
        "--test-steps", str(args.test_steps),
        "--edge-compute-scale", str(args.edge_compute_scale),
        "--output-dir", str(out_dir),
    ]
    if policy_tag == "score-softmax":
        cmd += ["--policy", "score-softmax", "--score-policy-temperature", "5.0"]
    elif policy_tag == "v4-original":
        cmd += ["--policy", "checkpoint", "--checkpoint", checkpoint]
    elif policy_tag == "v4-stressed":
        cmd += ["--policy", "checkpoint", "--checkpoint", checkpoint]
    else:
        raise ValueError(policy_tag)
    print(f"\n=== Running {policy_tag} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    return out_dir / args.dataset / args.distribution


def load_summary(path: Path) -> dict[tuple[int, int], dict]:
    """Returns {(users, delay): row_dict}."""
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(int(row["users"]), int(row["feedback_delay"]))] = row
    return out


METRICS = [
    ("cvar99_mean", "CVaR99 (s)"),
    ("queue_oscillation_mean", "Queue oscillation"),
    ("portfolio_migration_rate_mean", "Portfolio migration rate"),
    ("global_demand_concentration_mean", "Global demand concentration"),
    ("server_selection_gini_mean", "Server selection Gini"),
]


def plot_comparison(all_data: dict[str, dict], users: list[int], delays: list[int],
                    out_dir: Path) -> None:
    """For each metric, plot side-by-side heatmaps (one per policy)."""
    policies = list(all_data.keys())
    for key, title in METRICS:
        fig, axes = plt.subplots(1, len(policies), figsize=(4.2 * len(policies), 4.0),
                                 squeeze=False)
        for ax_idx, policy in enumerate(policies):
            ax = axes[0][ax_idx]
            matrix = np.full((len(users), len(delays)), np.nan, dtype=np.float64)
            cell_data = all_data[policy]
            for (u, d), row in cell_data.items():
                if u in users and d in delays:
                    matrix[users.index(u), delays.index(d)] = float(row[key])
            im = ax.imshow(matrix, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(delays)), labels=delays)
            ax.set_yticks(range(len(users)), labels=users)
            ax.set_xlabel("δ (slots)")
            ax.set_ylabel("N users")
            ax.set_title(f"{policy}\n{title}")
            for i in range(len(users)):
                for j in range(len(delays)):
                    v = matrix[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                                color="white", fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fname = key.replace("_mean", "")
        fig.savefig(out_dir / f"compare_{fname}.png", dpi=150)
        plt.close(fig)


def write_combined_table(all_data: dict[str, dict], users: list[int], delays: list[int],
                         out_dir: Path) -> None:
    """Long-form CSV: policy, N, δ, all metrics — for spreadsheets."""
    rows = []
    for policy, cell_data in all_data.items():
        for u in users:
            for d in delays:
                row = cell_data.get((u, d))
                if row is None:
                    continue
                rows.append({
                    "policy": policy,
                    "users": u,
                    "feedback_delay": d,
                    **{k: row.get(k, "") for k, _ in METRICS},
                })
    if not rows:
        return
    path = out_dir / "combined_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    users = [int(x) for x in args.users.split(",")]
    delays = [int(x) for x in args.delays.split(",")]
    requested = [p.strip() for p in args.policies.split(",") if p.strip()]

    ckpt_map = {
        "score-softmax": "",
        "v4-original": args.original_ckpt,
        "v4-stressed": args.stressed_ckpt,
    }
    all_data: dict[str, dict] = {}
    for policy in requested:
        ckpt = ckpt_map[policy]
        if policy in {"v4-original", "v4-stressed"} and not Path(ckpt).exists():
            print(f"[skip] {policy}: checkpoint not found at {ckpt}")
            continue
        leaf = run_phase(policy, args, ckpt)
        all_data[policy] = load_summary(leaf / "phase_summary.csv")

    if not all_data:
        print("No data collected.")
        return

    plot_comparison(all_data, users, delays, out_dir)
    write_combined_table(all_data, users, delays, out_dir)
    with (out_dir / "compare_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
