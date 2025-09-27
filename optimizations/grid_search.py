# /optimizers/grid_search.py
# =========================================
# 🔍 GRID SEARCH OPTIMIZER
# =========================================
import itertools
import os
import subprocess
from datetime import datetime

import yaml


# =========================================
# 📂 CONFIG HELPERS
# =========================================
def load_base_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_temp_config(config, out_path):
    with open(out_path, "w") as f:
        yaml.dump(config, f)


# =========================================
# 🧠 EVALUATION FUNCTION
# =========================================
def evaluate_run(summary_path="results/summary.txt"):
    if not os.path.exists(summary_path):
        return -999.0  # Penalize failed runs
    with open(summary_path, "r") as f:
        for line in f:
            if "ROI" in line:
                try:
                    return float(line.split(":")[-1].strip().replace("%", ""))
                except Exception:
                    return -999.0
    return -999.0


# =========================================
# 🚀 GRID SEARCH FUNCTION
# =========================================
def run_grid_search():
    c1_list = ["c1_twiggs_money_flow", "c1_rsi", "c1_macd"]
    use_volume = [True, False]
    use_baseline = [True, False]
    sl_mult = [1.0, 1.5, 2.0]
    tp_mult = [1.0, 1.5, 2.0]

    all_runs = []

    for combo in itertools.product(c1_list, use_volume, use_baseline, sl_mult, tp_mult):
        c1, vol, base, sl, tp = combo

        config = load_base_config()
        config["indicators"]["c1"] = c1
        config["indicators"]["use_volume"] = vol
        config["indicators"]["use_baseline"] = base
        config["entry"]["sl_multiplier"] = sl
        config["entry"]["tp_multiplier"] = tp

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = f"config_temp_{stamp}.yaml"
        write_temp_config(config, temp_path)

        print(f"\n🚀 Running config: {c1}, vol={vol}, base={base}, SL={sl}, TP={tp}")
        subprocess.run(["python", "core/backtester.py", temp_path])

        roi = evaluate_run()
        all_runs.append((roi, combo))

    all_runs.sort(reverse=True)
    print("\n🏆 Top Configs by ROI:")
    for roi, combo in all_runs[:5]:
        print(f"ROI: {roi:.2f}% | Params: {combo}")
