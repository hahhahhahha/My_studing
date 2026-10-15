from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results_priceedge_v4_missing_evidence"
MAIN_RAW = ROOT / "results_priceedge_qtbias_paper_ready" / "main_baseline_raw_no_facr.csv"
ROBUST_RAW = (
    ROOT
    / "results_priceedge_qtbias_paper_ready"
    / "fixed_checkpoint_robustness_raw_with_qtbias_no_facr.csv"
)

METRICS = [
    "average_latency",
    "cvar99",
    "queue_oscillation",
    "portfolio_migration_rate",
]


def empirical_cvar(values: np.ndarray, alpha: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(math.ceil((1.0 - alpha) * arr.size)))
    return float(np.mean(np.sort(arr)[-k:]))


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def stratified_seed_bootstrap(
    paired: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    """Resample seeds; retain the full delay block for every sampled seed."""

    seed_blocks = [g["difference"].to_numpy(float) for _, g in paired.groupby("seed")]
    if not seed_blocks:
        return float("nan"), float("nan")
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.integers(0, len(seed_blocks), size=len(seed_blocks))
        draws[b] = np.mean(np.concatenate([seed_blocks[i] for i in chosen]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def exact_one_sided_seed_signflip(seed_means: np.ndarray) -> float:
    seed_means = np.asarray(seed_means, dtype=np.float64)
    seed_means = seed_means[np.isfinite(seed_means)]
    if seed_means.size == 0:
        return float("nan")
    observed = float(np.mean(seed_means))
    total = 1 << seed_means.size
    exceed = 0
    for mask in range(total):
        signs = np.asarray([1.0 if mask & (1 << i) else -1.0 for i in range(seed_means.size)])
        if float(np.mean(seed_means * signs)) >= observed - 1e-15:
            exceed += 1
    return float(exceed / total)


def main_baseline_statistics(out: Path, n_boot: int) -> None:
    raw = pd.read_csv(MAIN_RAW)
    raw["seed"] = pd.to_numeric(raw["seed"], errors="raise").astype(int)
    raw["feedback_delay"] = pd.to_numeric(raw["feedback_delay"], errors="raise").astype(int)
    rng = np.random.default_rng(20260725)
    method_main = "PriceEdge-QT-Bias"

    summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    by_delay_rows: list[dict[str, object]] = []
    for method, group in raw.groupby("method"):
        for metric in METRICS:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            lo, hi = bootstrap_mean(vals, rng, n_boot)
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "runs": vals.size,
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)),
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                }
            )

    main = raw[raw["method"].eq(method_main)]
    for baseline in sorted(set(raw["method"]) - {method_main}):
        base = raw[raw["method"].eq(baseline)]
        for metric in METRICS:
            paired = main[["seed", "feedback_delay", metric]].merge(
                base[["seed", "feedback_delay", metric]],
                on=["seed", "feedback_delay"],
                suffixes=("_main", "_baseline"),
                validate="one_to_one",
            )
            paired["difference"] = paired[f"{metric}_baseline"] - paired[f"{metric}_main"]
            lo, hi = stratified_seed_bootstrap(paired, rng, n_boot)
            seed_means = paired.groupby("seed")["difference"].mean().to_numpy(float)
            sd = float(np.std(seed_means, ddof=1)) if seed_means.size > 1 else float("nan")
            paired_rows.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "paired_cells": len(paired),
                    "independent_seeds": len(seed_means),
                    "mean_improvement_baseline_minus_main": float(paired["difference"].mean()),
                    "relative_improvement_vs_baseline": float(
                        paired["difference"].mean() / paired[f"{metric}_baseline"].mean()
                    ),
                    "stratified_seed_bootstrap95_low": lo,
                    "stratified_seed_bootstrap95_high": hi,
                    "paired_effect_size_dz_seed_level": float(np.mean(seed_means) / sd)
                    if np.isfinite(sd) and sd > 0
                    else float("nan"),
                    "one_sided_exact_seed_signflip_p": exact_one_sided_seed_signflip(seed_means),
                }
            )
            for delay, dgroup in paired.groupby("feedback_delay"):
                vals = dgroup["difference"].to_numpy(float)
                dlo, dhi = bootstrap_mean(vals, rng, n_boot)
                by_delay_rows.append(
                    {
                        "baseline": baseline,
                        "metric": metric,
                        "feedback_delay": int(delay),
                        "paired_seeds": len(vals),
                        "mean_improvement_baseline_minus_main": float(np.mean(vals)),
                        "bootstrap95_low": dlo,
                        "bootstrap95_high": dhi,
                    }
                )

    pd.DataFrame(summary_rows).to_csv(out / "main_baseline_bootstrap_ci.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out / "main_baseline_paired_stratified.csv", index=False)
    pd.DataFrame(by_delay_rows).to_csv(out / "main_baseline_paired_by_delay.csv", index=False)


def feedback_age_diagnostics(out: Path, n_boot: int) -> None:
    raw = pd.read_csv(MAIN_RAW)
    rng = np.random.default_rng(20260726)
    rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for (method, metric), group in raw.melt(
        id_vars=["method", "seed", "feedback_delay"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    ).groupby(["method", "metric"]):
        per_seed = []
        for seed, sg in group.groupby("seed"):
            sg = sg.sort_values("feedback_delay")
            x = sg["feedback_delay"].to_numpy(float)
            y = sg["value"].to_numpy(float)
            slope = float(np.polyfit(x, y, 1)[0])
            y0 = float(sg.loc[sg["feedback_delay"].eq(0), "value"].iloc[0])
            y8 = float(sg.loc[sg["feedback_delay"].eq(8), "value"].iloc[0])
            degradation = float((y8 - y0) / y0) if abs(y0) > 1e-12 else float("nan")
            item = {
                "method": method,
                "metric": metric,
                "seed": int(seed),
                "slope_per_delay_slot": slope,
                "relative_stale_degradation_delta0_to8": degradation,
                "value_delta0": y0,
                "value_delta8": y8,
            }
            seed_rows.append(item)
            per_seed.append(item)
        slopes = np.asarray([r["slope_per_delay_slot"] for r in per_seed], dtype=float)
        deg = np.asarray([r["relative_stale_degradation_delta0_to8"] for r in per_seed], dtype=float)
        slo, shi = bootstrap_mean(slopes, rng, n_boot)
        dlo, dhi = bootstrap_mean(deg, rng, n_boot)
        rows.append(
            {
                "method": method,
                "metric": metric,
                "seeds": len(per_seed),
                "slope_mean": float(np.mean(slopes)),
                "slope_std": float(np.std(slopes, ddof=1)),
                "slope_bootstrap95_low": slo,
                "slope_bootstrap95_high": shi,
                "relative_stale_degradation_mean": float(np.mean(deg)),
                "relative_stale_degradation_std": float(np.std(deg, ddof=1)),
                "degradation_bootstrap95_low": dlo,
                "degradation_bootstrap95_high": dhi,
            }
        )
    pd.DataFrame(seed_rows).to_csv(out / "feedback_age_diagnostics_per_seed.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "feedback_age_diagnostics_summary.csv", index=False)


def _latency_values(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "phase" in df.columns:
        df = df[df["phase"].astype(str).eq("test")]
    return pd.to_numeric(df["latency"], errors="coerce").dropna().to_numpy(float)


def tail_distribution(out: Path) -> None:
    specs = {
        "PriceEdge-QT-Bias": (
            ROOT / "results_priceedge_monoqt_formal_5x5" / "formal" / "PriceEdgeBias-QT-P0p75-Final",
            "seed_*/delay_8/KAIST/30u/poisson_bursty/latency_samples_LCC-MAPPO.csv",
        ),
        "PPO-MV-A": (
            ROOT / "results_paper_algorithms_v4scenario_multiseed" / "Towards-PPO-MV",
            "seed_*/delay_8/Towards-PPO-MV_delay_8/KAIST/30u/poisson_bursty/latency_samples_Towards-PPO-MV.csv",
        ),
        "PPO-DSR-A": (
            ROOT / "results_paper_algorithms_v4scenario_multiseed" / "Towards-PPO-DSR",
            "seed_*/delay_8/Towards-PPO-DSR_delay_8/KAIST/30u/poisson_bursty/latency_samples_Towards-PPO-DSR.csv",
        ),
        "QECO-A": (
            ROOT / "results_recent_mec_baselines_strong_qeco_multiseed" / "QECO-load-topk",
            "seed_*/delay_8/KAIST/30u/poisson_bursty/latency_samples_QECO-load-topk.csv",
        ),
        "TODA-MAB-A": (
            ROOT / "results_recent_mec_baselines_strong_multiseed" / "TODAuction-MAB-tail",
            "seed_*/delay_8/KAIST/30u/poisson_bursty/latency_samples_TODAuction-MAB-tail.csv",
        ),
        "LyFOTO-LC": (
            ROOT / "results_recent_mec_local_lyapunov_multiseed" / "Lyapunov-Q-local-stale",
            "seed_*/delay_8/KAIST/30u/poisson_bursty/latency_samples_Lyapunov-Q-local-stale.csv",
        ),
    }
    colors = {
        "PriceEdge-QT-Bias": "#1f77b4",
        "PPO-MV-A": "#2ca02c",
        "PPO-DSR-A": "#9467bd",
        "QECO-A": "#ff7f0e",
        "TODA-MAB-A": "#d62728",
        "LyFOTO-LC": "#8c564b",
    }
    rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for method, (root, pattern) in specs.items():
        files = sorted(root.glob(pattern))
        arrays = [_latency_values(path) for path in files]
        arrays = [arr for arr in arrays if arr.size]
        if not arrays:
            rows.append({"method": method, "files": 0, "samples": 0, "status": "missing"})
            continue
        values = np.concatenate(arrays)
        values = values[np.isfinite(values)]
        xs = np.sort(values)
        ccdf = np.maximum(1.0 - np.arange(1, xs.size + 1) / xs.size, 1.0 / xs.size)
        ax.step(xs, ccdf, where="post", lw=1.7, label=method, color=colors[method])
        capped = np.minimum(values, 10.0)
        rows.append(
            {
                "method": method,
                "files": len(files),
                "samples": int(values.size),
                "tail99_samples": max(1, int(math.ceil(0.01 * values.size))),
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
                "uncapped_cvar95": empirical_cvar(values, 0.95),
                "uncapped_cvar99": empirical_cvar(values, 0.99),
                "capped_10s_cvar99": empirical_cvar(capped, 0.99),
                "count_latency_ge_10s": int(np.sum(values >= 10.0)),
                "prob_latency_ge_10s": float(np.mean(values >= 10.0)),
                "status": "ok",
            }
        )
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    ax.set_xlabel("Task latency (s)")
    ax.set_ylabel(r"CCDF $P(T>x)$")
    ax.set_title(r"Tail latency at shared feedback delay $\delta=8$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fig_multimethod_latency_ccdf_delta8.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "multimethod_latency_tail_audit_delta8.csv", index=False)


def robustness_decomposition(out: Path) -> None:
    raw = pd.read_csv(ROBUST_RAW)
    dims = [
        ["method", "dataset"],
        ["method", "distribution"],
        ["method", "users"],
        ["method", "feedback_delay"],
        ["method", "dataset", "distribution", "users", "feedback_delay"],
    ]
    names = ["by_dataset", "by_distribution", "by_users", "by_delay", "full_grid"]
    for group_cols, name in zip(dims, names):
        rows = []
        for keys, group in raw.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            row["runs"] = len(group)
            for metric in METRICS:
                vals = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"fixed_checkpoint_robustness_{name}.csv", index=False)


def baseline_adaptation_transparency(out: Path) -> None:
    rows = [
        {
            "Baseline": "PPO-DSR-A",
            "retained_core": "PPO and differential-Sharpe-style risk reward",
            "adaptation": "continuous policy mapped to 35-edge portfolio under the common stale-summary observation",
            "not_reproduced_or_limit": "original local/offload system model is replaced by the paper's portfolio environment",
        },
        {
            "Baseline": "PPO-MV-A",
            "retained_core": "PPO with mean-variance latency-risk objective",
            "adaptation": "same portfolio action space, task traces, seed-delay grid and budget",
            "not_reproduced_or_limit": "not a complete reproduction of the original execution model",
        },
        {
            "Baseline": "QECO-A",
            "retained_core": "D3QN/LSTM/dueling, replay, target network and QoE-oriented reward",
            "adaptation": "local-or-edge action adapted to load-aware top-k multi-edge selection",
            "not_reproduced_or_limit": "no local-execution branch; conclusions apply to the adapted interface only",
        },
        {
            "Baseline": "LyFOTO-LC",
            "retained_core": "stale queue-pressure drift-plus-penalty decision principle",
            "adaptation": "local candidate set and shared stale broadcast; no fresh queue oracle",
            "not_reproduced_or_limit": "federated actor-critic, energy queue and full original system are not reproduced",
        },
        {
            "Baseline": "TODA-MAB-A",
            "retained_core": "incomplete-information auction/MAB exploration and delay-based bidding",
            "adaptation": "tail-aware empirical bid mapped into the common MEC server-selection interface",
            "not_reproduced_or_limit": "market protocol details are simplified; it is a mechanism-level adaptation",
        },
    ]
    pd.DataFrame(rows).to_csv(out / "baseline_adaptation_transparency.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill non-training evidence gaps for PriceEdge V4.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    main_baseline_statistics(out, args.bootstrap)
    feedback_age_diagnostics(out, args.bootstrap)
    tail_distribution(out)
    robustness_decomposition(out)
    baseline_adaptation_transparency(out)
    print(f"Wrote PriceEdge V4 evidence outputs to {out}")


if __name__ == "__main__":
    main()
