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


def parse_metrics(log_text: str):
    runtime_ms = [int(value) for value in re.findall(r"Epoch Runtime:\s+(\d+)ms", log_text)]
    edges_per_second = [float(value) for value in re.findall(r"Edges per Second:\s+([0-9.]+)", log_text)]
    mrr = [float(value) for value in re.findall(r"MRR:\s+([0-9.]+)", log_text)]
    metrics = {
        "epochs_seen": len(runtime_ms),
        "epoch_last_runtime_ms": runtime_ms[-1] if runtime_ms else None,
        "epoch_last_eps": edges_per_second[-1] if edges_per_second else None,
        "avg_runtime_ms_all": float(statistics.mean(runtime_ms)) if runtime_ms else None,
        "avg_eps_all": float(statistics.mean(edges_per_second)) if edges_per_second else None,
        "avg_runtime_ms_excl_epoch1": float(statistics.mean(runtime_ms[1:])) if len(runtime_ms) > 1 else (float(runtime_ms[0]) if runtime_ms else None),
        "avg_eps_excl_epoch1": float(statistics.mean(edges_per_second[1:])) if len(edges_per_second) > 1 else (float(edges_per_second[0]) if edges_per_second else None),
        "valid_mrr_last": mrr[-2] if len(mrr) >= 2 else None,
        "test_mrr_last": mrr[-1] if mrr else None,
        "csr_update_log_seen": "InMemory::indexAdd using CSR reduce update path" in log_text,
    }
    return metrics


def mean_std(values):
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values))}


def pct_change(new_value, old_value):
    if new_value is None or old_value is None:
        return None
    if old_value == 0:
        return None
    return 100.0 * (new_value - old_value) / old_value


def run_train(mode, run_index, train_bin, config_path, repo_root, output_dir, cuda_device, extra_env):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env["GEGE_CSR_DEBUG"] = "0"
    env["GEGE_CSR_DEBUG_MAX_BATCHES"] = "0"
    env["GEGE_CSR_GATHER"] = "1" if mode == "csr" else "0"
    env["GEGE_CSR_UPDATE"] = "1" if mode == "csr" else "0"
    for key, value in extra_env.items():
        env[key] = value
    log_path = output_dir / f"{mode}_run{run_index}.log"
    command = [str(train_bin), str(config_path)]
    with open(log_path, "w", encoding="utf-8") as log_file:
        header = (
            f"[ab-bench] mode={mode} run={run_index} "
            f"GEGE_CSR_GATHER={env['GEGE_CSR_GATHER']} GEGE_CSR_UPDATE={env['GEGE_CSR_UPDATE']} "
            f"GEGE_CSR_DEBUG={env['GEGE_CSR_DEBUG']} CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n"
        )
        log_file.write(header)
        log_file.flush()
        process = subprocess.Popen(
            command,
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
    log_text = "".join(full_output)
    metrics = parse_metrics(log_text)
    metrics["return_code"] = return_code
    metrics["log_path"] = str(log_path)
    return metrics


def resolve_train_bin(repo_root: Path, train_bin_arg: str):
    if train_bin_arg:
        return Path(train_bin_arg)
    default_bin = repo_root / "build" / "gege" / "gege_train"
    if default_bin.exists():
        return default_bin
    return Path("gege_train")


def main():
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(repo_root / "gege" / "configs" / "fb15k.yaml"))
    parser.add_argument("--train-bin", default="")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--extra-env", action="append", default=[])
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    train_bin = resolve_train_bin(repo_root, args.train_bin)
    config_path = Path(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.out_dir) if args.out_dir else (repo_root / "logs" / f"csr_ab_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_env = {}
    for item in args.extra_env:
        if "=" not in item:
            raise ValueError(f"invalid --extra-env value: {item}")
        key, value = item.split("=", 1)
        extra_env[key] = value

    run_results = {"baseline": [], "csr": []}
    for mode in ("baseline", "csr"):
        for run_index in range(1, args.repeats + 1):
            print(f"[ab-bench] starting mode={mode} run={run_index}")
            metrics = run_train(
                mode=mode,
                run_index=run_index,
                train_bin=train_bin,
                config_path=config_path,
                repo_root=repo_root,
                output_dir=output_dir,
                cuda_device=args.cuda_device,
                extra_env=extra_env,
            )
            run_results[mode].append(metrics)
            print(
                "[ab-bench] done "
                f"mode={mode} run={run_index} rc={metrics['return_code']} "
                f"last_runtime_ms={metrics['epoch_last_runtime_ms']} "
                f"last_eps={metrics['epoch_last_eps']} "
                f"test_mrr={metrics['test_mrr_last']}"
            )
            if metrics["return_code"] != 0:
                print(f"[ab-bench] stopping due to non-zero return code for {mode} run {run_index}")
                break
        if any(item["return_code"] != 0 for item in run_results[mode]):
            break

    aggregate = {}
    metric_keys = [
        "epoch_last_runtime_ms",
        "epoch_last_eps",
        "avg_runtime_ms_excl_epoch1",
        "avg_eps_excl_epoch1",
        "valid_mrr_last",
        "test_mrr_last",
    ]
    for mode in ("baseline", "csr"):
        aggregate[mode] = {}
        for key in metric_keys:
            values = [item[key] for item in run_results[mode] if item.get("return_code", 1) == 0 and item.get(key) is not None]
            aggregate[mode][key] = mean_std(values)

    delta = {}
    for key in metric_keys:
        base_mean = aggregate["baseline"][key]["mean"]
        csr_mean = aggregate["csr"][key]["mean"]
        delta[key] = pct_change(csr_mean, base_mean)

    summary = {
        "timestamp": timestamp,
        "train_bin": str(train_bin),
        "config": str(config_path),
        "repeats": args.repeats,
        "cuda_device": args.cuda_device,
        "extra_env": extra_env,
        "runs": run_results,
        "aggregate": aggregate,
        "delta_pct_csr_vs_baseline": delta,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print("[ab-bench] aggregate means:")
    for key in metric_keys:
        base_mean = aggregate["baseline"][key]["mean"]
        csr_mean = aggregate["csr"][key]["mean"]
        change = delta[key]
        change_str = "n/a" if change is None or math.isnan(change) else f"{change:+.2f}%"
        print(f"  {key}: baseline={base_mean} csr={csr_mean} delta={change_str}")
    print(f"[ab-bench] summary: {summary_path}")


if __name__ == "__main__":
    main()
