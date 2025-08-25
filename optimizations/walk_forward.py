# /optimizers/walk_forward.py
# =========================================
# 🚶 WALK-FORWARD OPTIMIZATION
# =========================================
import os
import argparse
import pandas as pd
from datetime import timedelta
import yaml
import subprocess

# =========================================
# 📂 CONFIG HELPERS
# =========================================
def load_base_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def write_temp_config(config, out_path):
    with open(out_path, "w") as f:
        yaml.dump(config, f)

# =========================================
# 🧠 WALK-FORWARD FUNCTION
# =========================================
def run_walk_forward(pair, window=250, step=50):
    df_path = f"data/daily/{pair}.csv"
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Data file not found: {df_path}")

    df = pd.read_csv(df_path)
    df["date"] = pd.to_datetime(df["date"])

    if len(df) < window + step:
        raise ValueError(f"Not enough data for walk-forward with window={window}, step={step}")

    start_idx = 0
    segment = 0

    while start_idx + window + step <= len(df):
        train = df.iloc[start_idx : start_idx + window]
        test = df.iloc[start_idx + window : start_idx + window + step]

        train.to_csv("data/temp_train.csv", index=False)
        test.to_csv("data/temp_test.csv", index=False)

        config = load_base_config()
        config["pairs"] = [pair]
        config["timeframe"] = "D"

        temp_config_path = f"config_temp_{pair}_{segment}.yaml"
        write_temp_config(config, temp_config_path)

        print(f"\n🚀 Running Walk Forward Segment {segment}...")
        result = subprocess.run(["python", "backtester.py", temp_config_path])
        if result.returncode != 0:
            print(f"❌ Segment {segment} failed.")

        segment += 1
        start_idx += step

    print("\n✅ Walk-forward optimization complete.")

# =========================================
# 🚀 ENTRY POINT
# =========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, required=True)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--step", type=int, default=50)
    args = parser.parse_args()

    run_walk_forward(args.pair, args.window, args.step)
