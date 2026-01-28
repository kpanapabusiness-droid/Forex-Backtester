import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True, help="Path to trades.csv")
    args = ap.parse_args()

    p = Path(args.trades)
    if not p.exists():
        raise SystemExit(f"Missing file: {p}")

    df = pd.read_csv(p)
    print(f"{p}: rows={len(df)}")


if __name__ == "__main__":
    main()


