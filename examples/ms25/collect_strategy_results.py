#!/usr/bin/env python3
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "benchmark_results"
SYSTEMS = [
    "MgO-2x2", "MgO-4x4", "H2O-64", "H2O-192",
    "CHA", "HEA", "Reaction", "Zr-O",
]


def load_json(path):
    if not path.is_file():
        return None
    with open(path) as input_file:
        return json.load(input_file)


def collect_hydragnn(strategy):
    suffix = "mlip" if strategy == "full" else strategy
    collected = {}
    for system in SYSTEMS:
        path = (
            HERE / "logs" / f"{system}_{suffix}_seed0" /
            "benchmark_results" / "checkpoint_benchmark_summary.json"
        )
        value = load_json(path)
        if value is not None:
            collected[system] = value
    return collected


def collect_external(model, strategy):
    collected = {}
    directory = RESULTS / "strategy_runs" / model / strategy
    for system in SYSTEMS:
        value = load_json(directory / f"{system}.json")
        if value is not None:
            collected[system] = value
    return collected


def main():
    summary = {
        "hydragnn": {
            strategy: collect_hydragnn(strategy)
            for strategy in ("full", "frozen_message_passing", "frozen_decoder")
        },
        "uma": {
            strategy: collect_external("uma", strategy)
            for strategy in ("full", "frozen_backbone", "lora")
        },
        "mace": {
            strategy: collect_external("mace", strategy)
            for strategy in ("full", "lora")
        },
    }
    output = RESULTS / "finetuning_strategy_summary.json"
    with open(output, "w") as output_file:
        json.dump(summary, output_file, indent=2)
    for model, strategies in summary.items():
        for strategy, values in strategies.items():
            print(f"{model:9s} {strategy:24s}: {len(values)}/8 systems")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
