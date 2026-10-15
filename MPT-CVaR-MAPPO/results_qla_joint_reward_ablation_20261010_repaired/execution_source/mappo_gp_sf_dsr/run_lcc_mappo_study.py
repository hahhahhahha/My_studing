from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results_lcc_mappo_study"


@dataclass(frozen=True)
class Variant:
    name: str
    extra_args: dict[str, object] = field(default_factory=dict)


PRESCREEN_VARIANTS: tuple[Variant, ...] = (
    Variant("LCC-Full"),
    Variant("LCC-Price", {"lcc-price-observation": True}),
    Variant("LCC-Price-Clean", {"lcc-price-clean-observation": True}),
    Variant("LCC-PricePlus-QT", {"lcc-price-observation": True, "lcc-price-scale-observation": True, "lcc-price-components": "QT"}),
    Variant("LCC-PricePlus-QTZ", {"lcc-price-observation": True, "lcc-price-scale-observation": True, "lcc-price-components": "QTZ"}),
    Variant("LCC-PricePlus-QTH", {"lcc-price-observation": True, "lcc-price-scale-observation": True, "lcc-price-components": "QTH"}),
    Variant("LCC-PricePlus-Full", {"lcc-price-observation": True, "lcc-price-scale-observation": True, "lcc-price-components": "Full"}),
    Variant(
        "LCC-PriceEdge-Full",
        {"lcc-price-observation": True, "lcc-price-scale-observation": True, "lcc-edgewise-price-actor": True, "lcc-price-components": "Full"},
    ),
    Variant(
        "LCC-PriceEdge-NoPrice",
        {"lcc-price-observation": True, "lcc-price-scale-observation": True, "lcc-edgewise-price-actor": True, "lcc-price-components": "NoPrice"},
    ),
    Variant(
        "LCC-PriceEdge-Shuffle",
        {"lcc-price-observation": True, "lcc-price-scale-observation": True, "lcc-edgewise-price-actor": True, "lcc-price-components": "Full", "lcc-price-shuffle": True},
    ),
    Variant(
        "LCC-PriceEdge-QT",
        {"lcc-price-observation": True, "lcc-price-scale-observation": True, "lcc-edgewise-price-actor": True, "lcc-price-components": "QT"},
    ),
    Variant(
        "LCC-PriceEdge-QTZ",
        {"lcc-price-observation": True, "lcc-price-scale-observation": True, "lcc-edgewise-price-actor": True, "lcc-price-components": "QTZ"},
    ),
    Variant(
        "LCC-PriceEdge-QTH",
        {"lcc-price-observation": True, "lcc-price-scale-observation": True, "lcc-edgewise-price-actor": True, "lcc-price-components": "QTH"},
    ),
    Variant(
        "LCC-PriceEdgeBias-Full",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-edgewise-price-actor": True,
            "lcc-price-components": "Full",
            "lcc-price-logit-bias": 1.0,
        },
    ),
    Variant(
        "LCC-PriceEdgeBias-FullSafe",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-edgewise-price-actor": True,
            "lcc-price-components": "FullSafe",
            "lcc-price-logit-bias": 1.0,
        },
    ),
    Variant(
        "LCC-PriceEdgeBias-QT",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-edgewise-price-actor": True,
            "lcc-price-components": "QT",
            "lcc-price-logit-bias": 1.0,
        },
    ),
    Variant(
        "LCC-PriceEdgeBias-Shuffle",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-edgewise-price-actor": True,
            "lcc-price-components": "Full",
            "lcc-price-shuffle": True,
            "lcc-price-logit-bias": 1.0,
        },
    ),
    Variant(
        "PriceEdge-QT-NoBias",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-edgewise-price-actor": True,
            "lcc-price-components": "QT",
            "lcc-price-logit-bias": 0.0,
            "lcc-v": 0.5,
            "lcc-migration-weight": 0.05,
        },
    ),
    Variant(
        "PriceEdgeBias-QT-P0p75-Shuffle",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-edgewise-price-actor": True,
            "lcc-price-components": "QT",
            "lcc-price-shuffle": True,
            "lcc-price-logit-bias": 0.75,
            "lcc-v": 0.5,
            "lcc-migration-weight": 0.05,
        },
    ),
    Variant(
        "Flat-Bias-QT-P0p75",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-flat-price-bias-actor": True,
            "lcc-price-components": "QT",
            "lcc-price-logit-bias": 0.75,
            "lcc-v": 0.5,
            "lcc-migration-weight": 0.05,
        },
    ),
    Variant(
        "QT-PriceOnly-P0p75",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-price-only-policy": True,
            "lcc-price-components": "QT",
            "lcc-price-logit-bias": 0.75,
            "lcc-v": 0.5,
            "lcc-migration-weight": 0.05,
        },
    ),
    Variant(
        "PriceEdgeBias-QT-P0p75-NoStaleMargin",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-edgewise-price-actor": True,
            "lcc-price-components": "QT",
            "lcc-price-logit-bias": 0.75,
            "lcc-v": 0.5,
            "lcc-migration-weight": 0.05,
            "lcc-no-stale-margin": True,
        },
    ),
    Variant(
        "PriceEdgeBias-QT-P0p75-NoTailState",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-edgewise-price-actor": True,
            "lcc-price-components": "QT",
            "lcc-price-logit-bias": 0.75,
            "lcc-v": 0.5,
            "lcc-migration-weight": 0.05,
            "lcc-no-z-queue": True,
        },
    ),
    Variant(
        "PriceEdgeBias-QT-P0p75-StaticReward",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-edgewise-price-actor": True,
            "lcc-price-components": "QT",
            "lcc-price-logit-bias": 0.75,
            "lcc-v": 0.5,
            "lcc-migration-weight": 0.05,
            "lcc-static-reward": True,
            "lcc-no-z-queue": True,
            "lcc-no-h-queue": True,
        },
    ),
    Variant(
        "Local-Critic-Bias-QT-P0p75",
        {
            "lcc-price-observation": True,
            "lcc-price-scale-observation": True,
            "lcc-edgewise-price-actor": True,
            "lcc-price-components": "QT",
            "lcc-price-logit-bias": 0.75,
            "lcc-v": 0.5,
            "lcc-migration-weight": 0.05,
            "lcc-local-critic": True,
        },
    ),
    Variant(
        "LCC-PricePlus-Shuffle",
        {"lcc-price-observation": True, "lcc-price-scale-observation": True, "lcc-price-components": "Full", "lcc-price-shuffle": True},
    ),
    Variant("LCC-w-o-Z", {"lcc-no-z-queue": True}),
    Variant("LCC-w-o-H", {"lcc-no-h-queue": True}),
    Variant("LCC-w-o-StaleMargin", {"lcc-no-stale-margin": True}),
    Variant("StaticReward-MAPPO", {"lcc-static-reward": True, "lcc-no-z-queue": True, "lcc-no-h-queue": True}),
)


def common_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset": "KAIST",
        "users": 30,
        "distribution": "poisson_bursty",
        "train-steps": args.train_steps,
        "test-steps": args.test_steps,
        "broadcast-period": 2,
        "slot-s": 1.0,
        "n-edges": 35,
        "edge-compute-mb-s": 63.0,
        "queue-upper-mb": 630.0,
        "task-compute-mb": 15.0,
        "task-package-mb": 270.0 / 1024.0,
        "max-requests-per-user": 7,
        "poisson-lambda": 3.5,
        "burst-prob": 0.1,
        "burst-multiplier": 2.0,
        "background-load": 0.0,
        "capacity-mode": "homogeneous",
        "edge-compute-scale": 1.0,
        "task-compute-scale": 1.0,
        "max-supported-users": 50,
        "score-resource-weight": 1.0,
        "score-capacity-weight": 0.5,
        "score-task-count-weight": 0.5,
        "score-wait-weight": 1.0,
        "learning-rate": args.learning_rate,
        "rollout-size": args.rollout_size,
        "batch-size": args.batch_size,
        "target-portfolio-entropy": args.target_portfolio_entropy,
        "log-interval": args.log_interval,
        "lcc-cvar-alpha": 0.99,
        "lcc-cvar-budget": args.lcc_cvar_budget,
        "lcc-concentration-factor": args.lcc_concentration_factor,
        "lcc-stale-margin": args.lcc_stale_margin,
    }


def selected_prescreen_variants(names: list[str] | None) -> list[Variant]:
    if not names:
        return list(PRESCREEN_VARIANTS)
    wanted = set(names)
    variants = [variant for variant in PRESCREEN_VARIANTS if variant.name in wanted]
    missing = sorted(wanted - {variant.name for variant in variants})
    if missing:
        raise SystemExit(f"Unknown prescreen variants: {', '.join(missing)}")
    return variants


def price_bias_base_args(components: str, lambda_pi: float) -> dict[str, object]:
    return {
        "lcc-price-observation": True,
        "lcc-price-scale-observation": True,
        "lcc-edgewise-price-actor": True,
        "lcc-price-components": components,
        "lcc-price-logit-bias": lambda_pi,
    }


def flat_price_bias_base_args(components: str, lambda_pi: float) -> dict[str, object]:
    return {
        "lcc-price-observation": True,
        "lcc-price-scale-observation": True,
        "lcc-flat-price-bias-actor": True,
        "lcc-price-components": components,
        "lcc-price-logit-bias": lambda_pi,
    }


def price_bias_component_variants(components_values: list[str], price_bias_values: list[float]) -> list[Variant]:
    variants: list[Variant] = []
    for components in components_values:
        for lam_pi in price_bias_values:
            name = f"PriceEdgeBias-{components}-P{lam_pi:g}".replace(".", "p")
            variants.append(Variant(name, price_bias_base_args(components, lam_pi)))
    return variants


def price_bias_migration_variants(components: str, lambda_pi: float, migration_values: list[float]) -> list[Variant]:
    variants: list[Variant] = []
    for lam_m in migration_values:
        name = f"PriceEdgeBias-{components}-P{lambda_pi:g}-M{lam_m:g}".replace(".", "p")
        variants.append(
            Variant(
                name,
                {
                    **price_bias_base_args(components, lambda_pi),
                    "lcc-migration-weight": lam_m,
                },
            )
        )
    return variants


def parameter_variants(v_values: list[float], migration_values: list[float], components: str, lambda_pi: float) -> list[Variant]:
    variants: list[Variant] = []
    for v in v_values:
        for lam in migration_values:
            name = f"PriceEdgeBias-{components}-P{lambda_pi:g}-V{v:g}-M{lam:g}".replace(".", "p")
            variants.append(
                Variant(
                    name,
                    {
                        **price_bias_base_args(components, lambda_pi),
                        "lcc-v": v,
                        "lcc-migration-weight": lam,
                    },
                )
            )
    return variants


def entropy_variants(entropy_values: list[float], components: str, lambda_pi: float) -> list[Variant]:
    return [
        Variant(
            f"PriceEdgeBias-{components}-P{lambda_pi:g}-H{value:g}".replace(".", "p"),
            {
                **price_bias_base_args(components, lambda_pi),
                "target-portfolio-entropy": value,
            },
        )
        for value in entropy_values
    ]


def summary_path(output_root: Path, phase: str, variant: str, seed: int, delay: int) -> Path:
    return (
        output_root
        / phase
        / variant
        / f"seed_{seed}"
        / f"delay_{delay}"
        / "KAIST"
        / "30u"
        / "poisson_bursty"
        / "summary_LCC-MAPPO.csv"
    )


def build_command(
    args: argparse.Namespace,
    output_root: Path,
    phase: str,
    variant: Variant,
    seed: int,
    delay: int,
) -> list[str]:
    values = {
        **common_args(args),
        "lcc-v": args.lcc_v,
        "lcc-migration-weight": args.lcc_migration_weight,
        **variant.extra_args,
        "seed": seed,
        "feedback-delay": delay,
        "output-dir": str(output_root / phase / variant.name / f"seed_{seed}" / f"delay_{delay}"),
    }
    if args.device:
        values["device"] = args.device
    command = [sys.executable, "-B", "-m", "mappo_gp_sf_dsr.train", "--method", "LCC-MAPPO"]
    for key, value in values.items():
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
        elif value is not None:
            command.extend([f"--{key}", str(value)])
    return command


def read_summary(path: Path, phase: str, variant: str, seed: int, delay: int) -> dict[str, object]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty summary: {path}")
    row: dict[str, object] = dict(rows[0])
    row.update({"phase": phase, "variant": variant, "seed": seed, "delay": delay, "summary_path": str(path)})
    return row


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metric_names = (
        "average_latency",
        "cvar99",
        "queue_oscillation",
        "portfolio_migration_rate",
        "global_demand_concentration",
        "mean_portfolio_entropy",
        "lcc_z_queue",
        "lcc_price_action_spearman",
    )
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["phase"]), str(row["variant"])), []).append(row)
    out: list[dict[str, object]] = []
    for (phase, variant), group in sorted(groups.items()):
        item: dict[str, object] = {"phase": phase, "variant": variant, "runs": len(group)}
        for metric in metric_names:
            vals: list[float] = []
            for row in group:
                try:
                    vals.append(float(row[metric]))
                except Exception:
                    pass
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = var ** 0.5
        out.append(item)
    return out


def run_phase(args: argparse.Namespace, output_root: Path, phase: str, variants: list[Variant], seeds: list[int], delays: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = len(variants) * len(seeds) * len(delays)
    done = 0
    for variant in variants:
        for seed in seeds:
            for delay in delays:
                out = summary_path(output_root, phase, variant.name, seed, delay)
                if out.exists() and not args.force:
                    print(f"[skip] {phase} {variant.name} seed={seed} delay={delay}", flush=True)
                else:
                    print(f"[run] {phase} {variant.name} seed={seed} delay={delay}", flush=True)
                    subprocess.run(build_command(args, output_root, phase, variant, seed, delay), cwd=str(ROOT), check=True)
                if not out.exists():
                    raise RuntimeError(f"Expected summary not found: {out}")
                rows.append(read_summary(out, phase, variant.name, seed, delay))
                done += 1
                print(f"[done] {done}/{total}", flush=True)
    write_rows(output_root / f"{phase}_raw.csv", rows)
    write_rows(output_root / f"{phase}_summary.csv", aggregate_rows(rows))
    return rows


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LCC-MAPPO structural, parameter, and formal experiments.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=["prescreen", "pricebias", "migration", "param", "entropy", "formal", "all"], default="prescreen")
    parser.add_argument("--prescreen-variants", nargs="*", default=None)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--delays", default="0,4,8")
    parser.add_argument("--formal-seeds", default="1,2,3,4,5")
    parser.add_argument("--formal-delays", default="0,1,2,4,8")
    parser.add_argument("--candidate-components", default="QT", choices=["QT", "QTZ", "QTH", "Full", "FullSafe"])
    parser.add_argument("--candidate-price-bias", type=float, default=1.0)
    parser.add_argument("--component-grid", default="QT,FullSafe")
    parser.add_argument("--price-bias-grid", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--v-grid", default="0.5,1.0,2.0")
    parser.add_argument("--migration-grid", default="0.0,0.02,0.05")
    parser.add_argument("--entropy-grid", default="0.90,0.95,0.99")
    parser.add_argument("--lcc-v", type=float, default=1.0)
    parser.add_argument("--lcc-migration-weight", type=float, default=0.02)
    parser.add_argument("--lcc-cvar-budget", type=float, default=1.30)
    parser.add_argument("--lcc-concentration-factor", type=float, default=2.0)
    parser.add_argument("--lcc-stale-margin", type=float, default=1.0)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--test-steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-portfolio-entropy", type=float, default=0.99)
    parser.add_argument("--rollout-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir.resolve()
    all_rows: list[dict[str, object]] = []
    if args.phase in {"prescreen", "all"}:
        all_rows.extend(
            run_phase(
                args,
                output_root,
                "prescreen",
                selected_prescreen_variants(args.prescreen_variants),
                parse_int_list(args.seeds),
                parse_int_list(args.delays),
            )
        )
    if args.phase in {"pricebias", "all"}:
        all_rows.extend(
            run_phase(
                args,
                output_root,
                "pricebias",
                price_bias_component_variants(
                    [item.strip() for item in args.component_grid.split(",") if item.strip()],
                    parse_float_list(args.price_bias_grid),
                ),
                parse_int_list(args.seeds),
                parse_int_list(args.delays),
            )
        )
    if args.phase in {"migration", "all"}:
        all_rows.extend(
            run_phase(
                args,
                output_root,
                "migration",
                price_bias_migration_variants(
                    args.candidate_components,
                    args.candidate_price_bias,
                    parse_float_list(args.migration_grid),
                ),
                parse_int_list(args.seeds),
                parse_int_list(args.delays),
            )
        )
    if args.phase in {"param", "all"}:
        all_rows.extend(
            run_phase(
                args,
                output_root,
                "param",
                parameter_variants(
                    parse_float_list(args.v_grid),
                    parse_float_list(args.migration_grid),
                    args.candidate_components,
                    args.candidate_price_bias,
                ),
                parse_int_list(args.seeds),
                parse_int_list(args.delays),
            )
        )
    if args.phase in {"entropy", "all"}:
        all_rows.extend(
            run_phase(
                args,
                output_root,
                "entropy",
                entropy_variants(parse_float_list(args.entropy_grid), args.candidate_components, args.candidate_price_bias),
                parse_int_list(args.seeds),
                parse_int_list(args.delays),
            )
        )
    if args.phase in {"formal", "all"}:
        all_rows.extend(
            run_phase(
                args,
                output_root,
                "formal",
                [
                    Variant(
                        f"PriceEdgeBias-{args.candidate_components}-P{args.candidate_price_bias:g}-Final".replace(".", "p"),
                        {
                            **price_bias_base_args(args.candidate_components, args.candidate_price_bias),
                            "lcc-v": args.lcc_v,
                            "lcc-migration-weight": args.lcc_migration_weight,
                        },
                    )
                ],
                parse_int_list(args.formal_seeds),
                parse_int_list(args.formal_delays),
            )
        )
    if all_rows:
        write_rows(output_root / "all_completed_raw.csv", all_rows)
        write_rows(output_root / "all_completed_summary.csv", aggregate_rows(all_rows))
    print(f"LCC-MAPPO study finished for phase={args.phase}. Results: {output_root}", flush=True)


if __name__ == "__main__":
    main()
