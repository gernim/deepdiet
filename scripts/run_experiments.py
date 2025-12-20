#!/usr/bin/env python3
"""
Experiment runner for DeepDiet training.

Runs multiple experiments sequentially or in parallel, with automatic
logging to W&B and optional Slack/email notifications.

Usage:
    # Run all defined experiments
    python scripts/run_experiments.py

    # Run specific experiments by name
    python scripts/run_experiments.py --experiments attention_baseline lstm_baseline

    # Dry run (print commands without executing)
    python scripts/run_experiments.py --dry-run

    # Run with custom config file
    python scripts/run_experiments.py --config experiments.yaml
"""

import argparse
import subprocess
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

REPO = Path(__file__).resolve().parents[1]

# Default experiments to run
DEFAULT_EXPERIMENTS = {
    # Aggregation method comparison
    "lstm_baseline": {
        "description": "LSTM aggregation baseline",
        "args": {
            "use_side_frames": True,
            "use_overhead": True,
            "use_depth": True,
            "side_aggregation": "lstm",
            "epochs": 20,
            "batch_size": 4,
            "lr": 1e-4,
        },
        "tags": ["baseline", "lstm"],
    },
    "attention_baseline": {
        "description": "Temporal attention aggregation",
        "args": {
            "use_side_frames": True,
            "use_overhead": True,
            "use_depth": True,
            "side_aggregation": "attention",
            "epochs": 20,
            "batch_size": 4,
            "lr": 1e-4,
        },
        "tags": ["baseline", "attention"],
    },
    "mean_baseline": {
        "description": "Mean pooling aggregation",
        "args": {
            "use_side_frames": True,
            "use_overhead": True,
            "use_depth": True,
            "side_aggregation": "mean",
            "epochs": 20,
            "batch_size": 4,
            "lr": 1e-4,
        },
        "tags": ["baseline", "mean"],
    },

    # Learning rate sweep for attention
    "attention_lr_high": {
        "description": "Attention with higher LR",
        "args": {
            "use_side_frames": True,
            "use_overhead": True,
            "use_depth": True,
            "side_aggregation": "attention",
            "epochs": 20,
            "lr": 5e-4,
        },
        "tags": ["lr_sweep", "attention"],
    },
    "attention_lr_low": {
        "description": "Attention with lower LR",
        "args": {
            "use_side_frames": True,
            "use_overhead": True,
            "use_depth": True,
            "side_aggregation": "attention",
            "epochs": 20,
            "lr": 5e-5,
        },
        "tags": ["lr_sweep", "attention"],
    },

    # Frozen encoder experiments
    "attention_frozen": {
        "description": "Attention with frozen encoders",
        "args": {
            "use_side_frames": True,
            "use_overhead": True,
            "use_depth": True,
            "side_aggregation": "attention",
            "epochs": 20,
            "freeze_encoders": True,
            "unfreeze_epoch": 10,
        },
        "tags": ["frozen", "attention"],
    },

    # Single modality ablations
    "side_only_attention": {
        "description": "Side frames only with attention",
        "args": {
            "use_side_frames": True,
            "use_overhead": False,
            "use_depth": False,
            "side_aggregation": "attention",
            "epochs": 20,
        },
        "tags": ["ablation", "side_only"],
    },
    "overhead_only": {
        "description": "Overhead RGB + depth only",
        "args": {
            "use_side_frames": False,
            "use_overhead": True,
            "use_depth": True,
            "epochs": 20,
        },
        "tags": ["ablation", "overhead_only"],
    },
}


def build_command(experiment_name: str, config: Dict, wandb_project: str = "deepdiet") -> List[str]:
    """Build the training command from experiment config."""
    cmd = [sys.executable, "-m", "src.train"]

    args = config.get("args", {})

    # Boolean flags
    if args.get("use_side_frames"):
        cmd.append("--use-side-frames")
    if args.get("use_overhead"):
        cmd.append("--use-overhead")
    if args.get("use_depth"):
        cmd.append("--use-depth")
    if args.get("freeze_encoders"):
        cmd.append("--freeze-encoders")
    if args.get("allow_missing_modalities"):
        cmd.append("--allow-missing-modalities")

    # Value arguments
    if "side_aggregation" in args:
        cmd.extend(["--side-aggregation", args["side_aggregation"]])
    if "epochs" in args:
        cmd.extend(["--epochs", str(args["epochs"])])
    if "batch_size" in args:
        cmd.extend(["--batch-size", str(args["batch_size"])])
    if "lr" in args:
        cmd.extend(["--lr", str(args["lr"])])
    if "max_frames" in args:
        cmd.extend(["--max-frames", str(args["max_frames"])])
    if "image_size" in args:
        cmd.extend(["--image-size", str(args["image_size"])])
    if "unfreeze_epoch" in args:
        cmd.extend(["--unfreeze-epoch", str(args["unfreeze_epoch"])])
    if "encoder_lr_multiplier" in args:
        cmd.extend(["--encoder-lr-multiplier", str(args["encoder_lr_multiplier"])])

    # W&B integration
    cmd.append("--wandb")
    cmd.extend(["--wandb-project", wandb_project])
    cmd.extend(["--wandb-run-name", experiment_name])

    # Tags
    tags = config.get("tags", [])
    if tags:
        cmd.append("--wandb-tags")
        cmd.extend(tags)

    return cmd


def run_experiment(
    name: str,
    config: Dict,
    wandb_project: str,
    dry_run: bool = False,
    log_dir: Optional[Path] = None,
) -> Dict:
    """Run a single experiment and return results."""
    cmd = build_command(name, config, wandb_project)
    cmd_str = " ".join(cmd)

    result = {
        "name": name,
        "description": config.get("description", ""),
        "command": cmd_str,
        "started_at": datetime.now().isoformat(),
        "status": "pending",
    }

    print(f"\n{'='*70}")
    print(f"Experiment: {name}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Command: {cmd_str}")
    print(f"{'='*70}")

    if dry_run:
        result["status"] = "dry_run"
        print("[DRY RUN] Would execute the above command")
        return result

    # Set up log file
    log_file = None
    if log_dir:
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        print(f"Log file: {log_file}")

    try:
        start_time = time.time()

        if log_file:
            with open(log_file, "w") as f:
                process = subprocess.run(
                    cmd,
                    cwd=REPO,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        else:
            process = subprocess.run(
                cmd,
                cwd=REPO,
                text=True,
            )

        elapsed = time.time() - start_time
        result["elapsed_seconds"] = elapsed
        result["elapsed_human"] = f"{elapsed/60:.1f} minutes"
        result["return_code"] = process.returncode
        result["finished_at"] = datetime.now().isoformat()

        if process.returncode == 0:
            result["status"] = "success"
            print(f"\n✓ Experiment '{name}' completed successfully in {elapsed/60:.1f} minutes")
        else:
            result["status"] = "failed"
            print(f"\n✗ Experiment '{name}' failed with return code {process.returncode}")

    except KeyboardInterrupt:
        result["status"] = "interrupted"
        print(f"\n! Experiment '{name}' interrupted by user")
        raise

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"\n✗ Experiment '{name}' error: {e}")

    return result


def load_experiments(config_file: Optional[Path]) -> Dict:
    """Load experiments from config file or use defaults."""
    if config_file and config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    return DEFAULT_EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description="Run DeepDiet experiments")
    parser.add_argument(
        "--experiments",
        nargs="*",
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML config file with experiment definitions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--wandb-project",
        default="deepdiet",
        help="W&B project name (default: deepdiet)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=REPO / "experiment_logs",
        help="Directory for experiment logs",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit",
    )
    args = parser.parse_args()

    # Load experiments
    experiments = load_experiments(args.config)

    # List experiments and exit
    if args.list:
        print("\nAvailable experiments:")
        print("-" * 70)
        for name, config in experiments.items():
            desc = config.get("description", "No description")
            tags = ", ".join(config.get("tags", []))
            print(f"  {name:<25} {desc}")
            if tags:
                print(f"  {'':25} Tags: {tags}")
        return

    # Filter experiments if specific ones requested
    if args.experiments:
        experiments = {k: v for k, v in experiments.items() if k in args.experiments}
        if not experiments:
            print(f"Error: No matching experiments found for: {args.experiments}")
            print("Use --list to see available experiments")
            sys.exit(1)

    # Create log directory
    if not args.dry_run:
        args.log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# DeepDiet Experiment Runner")
    print(f"# Running {len(experiments)} experiment(s)")
    print(f"# W&B Project: {args.wandb_project}")
    print(f"# Log Directory: {args.log_dir}")
    print(f"{'#'*70}")

    results = []
    start_time = time.time()

    try:
        for name, config in experiments.items():
            result = run_experiment(
                name=name,
                config=config,
                wandb_project=args.wandb_project,
                dry_run=args.dry_run,
                log_dir=args.log_dir if not args.dry_run else None,
            )
            results.append(result)

    except KeyboardInterrupt:
        print("\n\nExperiment run interrupted by user")

    finally:
        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"\nResults:")

        for r in results:
            status_icon = {
                "success": "✓",
                "failed": "✗",
                "interrupted": "!",
                "error": "✗",
                "dry_run": "○",
                "pending": "?",
            }.get(r["status"], "?")

            elapsed = r.get("elapsed_human", "N/A")
            print(f"  {status_icon} {r['name']:<25} {r['status']:<12} {elapsed}")

        # Save results
        if not args.dry_run and results:
            results_file = args.log_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()