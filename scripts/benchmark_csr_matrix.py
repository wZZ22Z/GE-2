#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MODE_FLAGS = {
    "none": {"GEGE_CSR_GATHER": "0", "GEGE_CSR_UPDATE": "0"},
    "gather_only": {"GEGE_CSR_GATHER": "1", "GEGE_CSR_UPDATE": "0"},
    "update_only": {"GEGE_CSR_GATHER": "0", "GEGE_CSR_UPDATE": "1"},
    "both": {"GEGE_CSR_GATHER": "1", "GEGE_CSR_UPDATE": "1"},
}


def parse_metrics(log_text: str):
    runtime_ms = [int(value) for value in re.findall(r"Epoch Runtime:\s+(\d+)ms", log_text)]
    edges_per_second = [float(value) for value in re.findall(r"Edges per Second:\s+([0-9.]+)", log_text)]
    mrr = [float(value) for value in re.findall(r"MRR:\s+([0-9.]+)", log_text)]
    return {
        "epochs_seen": len(runtime_ms),
        "epoch_last_runtime_ms": runtime_ms[-1] if runtime_ms else None,
        "epoch_last_eps": edges_per_second[-1] if edges_per_second else None,
        "avg_runtime_ms_excl_epoch1": float(statistics.mean(runtime_ms[1:])) if len(runtime_ms) > 1 else None,
        "avg_eps_excl_epoch1": float(statistics.mean(edges_per_second[1:])) if len(edges_per_second) > 1 else None,
        "valid_mrr_last": mrr[-2] if len(mrr) >= 2 else None,
        "test_mrr_last": mrr[-1] if mrr else None,
        "csr_update_log_seen": "InMemory::indexAdd using CSR reduce update path" in log_text,
    }


def mean_std(values):
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values))}


def pct_change(new_value, old_value):
    if new_value is None or old_value in (None, 0):
        return None
    return 100.0 * (new_value - old_value) / old_value


def run_train(mode, run_index, train_bin, config_path, repo_root, output_dir, cuda_device):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env["GEGE_CSR_DEBUG"] = "0"
    env["GEGE_CSR_DEBUG_MAX_BATCHES"] = "0"
    env["GEGE_CSR_GATHER"] = MODE_FLAGS[mode]["GEGE_CSR_GATHER"]
    env["GEGE_CSR_UPDATE"] = MODE_FLAGS[mode]["GEGE_CSR_UPDATE"]
    log_path = output_dir / f"{mode}_run{run_index}.log"
    cmd = [str(train_bin), str(config_path)]
    with open(log_path, "w", encoding="utf-8") as log_file:
        header = (
            f"[matrix-bench] mode={mode} run={run_index} "
            f"GEGE_CSR_GATHER={env['GEGE_CSR_GATHER']} GEGE_CSR_UPDATE={env['GEGE_CSR_UPDATE']} "
            f"GEGE_CSR_DEBUG=0 CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n"
        )
        log_file.write(header)
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        full_output = [header]
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
            full_output.append(line)
        return_code = process.wait()
    metrics = parse_metrics("".join(full_output))
    metrics["return_code"] = return_code
    metrics["log_path"] = str(log_path)
    return metrics


def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(repo_root / "gege" / "configs" / "fb15k.yaml"))
    parser.add_argument("--train-bin", default=str(repo_root / "build" / "gege" / "gege_train"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) if args.out_dir else (repo_root / "logs" / f"csr_matrix_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = ["none", "gather_only", "update_only", "both"]
    run_results = {mode: [] for mode in modes}

    for mode in modes:
        for run_index in range(1, args.repeats + 1):
            print(f"[matrix-bench] starting mode={mode} run={run_index}")
            metrics = run_train(
                mode=mode,
                run_index=run_index,
                train_bin=Path(args.train_bin),
                config_path=Path(args.config),
                repo_root=repo_root,
                output_dir=output_dir,
                cuda_device=args.cuda_device,
            )
            run_results[mode].append(metrics)
            print(
                "[matrix-bench] done "
                f"mode={mode} run={run_index} rc={metrics['return_code']} "
                f"avg_eps2+={metrics['avg_eps_excl_epoch1']} test_mrr={metrics['test_mrr_last']}"
            )
            if metrics["return_code"] != 0:
                print(f"[matrix-bench] stopping due to non-zero return code mode={mode} run={run_index}")
                break
        if any(item["return_code"] != 0 for item in run_results[mode]):
            break

    metric_keys = [
        "epoch_last_runtime_ms",
        "epoch_last_eps",
        "avg_runtime_ms_excl_epoch1",
        "avg_eps_excl_epoch1",
        "valid_mrr_last",
        "test_mrr_last",
    ]
    aggregate = {}
    for mode in modes:
        aggregate[mode] = {}
        for key in metric_keys:
            values = [item[key] for item in run_results[mode] if item["return_code"] == 0 and item[key] is not None]
            aggregate[mode][key] = mean_std(values)

    baseline = aggregate["none"]
    delta_vs_none = {}
    for mode in modes:
        if mode == "none":
            continue
        delta_vs_none[mode] = {}
        for key in metric_keys:
            delta_vs_none[mode][key] = pct_change(aggregate[mode][key]["mean"], baseline[key]["mean"])

    summary = {
        "timestamp": timestamp,
        "config": args.config,
        "train_bin": args.train_bin,
        "repeats": args.repeats,
        "cuda_device": args.cuda_device,
        "modes": modes,
        "runs": run_results,
        "aggregate": aggregate,
        "delta_pct_vs_none": delta_vs_none,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("[matrix-bench] avg_eps_excl_epoch1 means:")
    for mode in modes:
        value = aggregate[mode]["avg_eps_excl_epoch1"]["mean"]
        print(f"  {mode}: {value}")
    print("[matrix-bench] delta vs none (avg_eps_excl_epoch1):")
    for mode in modes:
        if mode == "none":
            continue
        value = delta_vs_none[mode]["avg_eps_excl_epoch1"]
        value_str = "n/a" if value is None or math.isnan(value) else f"{value:+.2f}%"
        print(f"  {mode}: {value_str}")
    print(f"[matrix-bench] summary: {summary_path}")


if __name__ == "__main__":
    main()
