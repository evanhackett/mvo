"""
Fetches Fama-French factor data from Ken French's Data Library and computes
monthly return assumptions for use in mvo.py.

Covers: MKT, SmB, HmL, RmW, CmA, MOM
Note: ITT/LTT bond factors are not available from this source and remain hardcoded.

Usage:
    python fetch_factors.py
    python fetch_factors.py --start 199001
    python fetch_factors.py --start 196307 --end 202012
"""

import argparse
import io
import json
import time
import pathlib
import urllib.request
import zipfile
import numpy as np

CACHE_DIR = pathlib.Path(__file__).parent / ".cache"
CACHE_MAX_AGE_DAYS = 30

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

# Haircuts on the raw historical mean — same philosophy as mvo_original.py
HAIRCUTS = {
    "Mkt-RF": 1.00,  # no haircut for market
    "SMB":    0.50,
    "HML":    0.50,
    "RMW":    0.50,
    "CMA":    0.50,
    "Mom":    0.25,
}

# French column name -> mvo.py factor name
RENAME = {
    "Mkt-RF": "MKT",
    "SMB":    "SmB",
    "HML":    "HmL",
    "RMW":    "RmW",
    "CMA":    "CmA",
    "Mom":    "MOM",
}


def download_french_csv(url):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / (url.split("/")[-1].replace(".zip", ".csv"))

    if cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days < CACHE_MAX_AGE_DAYS:
            print(f"Using cached {cache_file.name} ({age_days:.0f}d old)")
            return cache_file.read_text(encoding="latin-1")

    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        raw = resp.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        text = zf.open(name).read().decode("latin-1")

    cache_file.write_text(text, encoding="latin-1")
    print(f"Cached to {cache_file}")
    return text


def parse_monthly(text):
    """
    Parse a Ken French CSV file, returning only the monthly data section.

    Returns:
        dates: np.array of int YYYYMM
        data:  dict of column_name -> np.array of monthly returns (in percent)
    """
    lines = text.splitlines()

    # Find the header row — the first line whose first comma-delimited token is 'Date'
    header_idx = None
    col_names = None
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if parts[0].lower() in ("date", "") and len(parts) >= 2 and parts[1].strip():
            header_idx = i
            col_names = parts
            break

    if header_idx is None:
        raise ValueError("Could not find header row — French data format may have changed")

    factor_cols = col_names[1:]  # everything after 'Date'
    dates = []
    data = {col: [] for col in factor_cols}

    for line in lines[header_idx + 1:]:
        stripped = line.strip()

        if not stripped:
            if dates:
                break  # first blank line after data = end of monthly section
            continue

        if stripped.startswith("Annual"):
            break

        parts = [p.strip() for p in line.split(",")]
        # Monthly dates are exactly 6 digits (YYYYMM); annual are 4
        if not parts[0].isdigit() or len(parts[0]) != 6:
            continue

        dates.append(int(parts[0]))
        for i, col in enumerate(factor_cols):
            try:
                val = float(parts[i + 1])
            except (ValueError, IndexError):
                val = np.nan
            data[col].append(val)

    return np.array(dates, dtype=int), {k: np.array(v) for k, v in data.items()}


def filter_dates(dates, data, start=None, end=None):
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    return dates[mask], {k: v[mask] for k, v in data.items()}


def compute_stats(returns):
    valid = returns[~np.isnan(returns)]
    return np.mean(valid), np.std(valid, ddof=1)


def main():
    parser = argparse.ArgumentParser(description="Compute FF factor return assumptions for mvo.py")
    parser.add_argument("--start", type=int, default=None,
                        help="Start date YYYYMM (default: all available, ~196307)")
    parser.add_argument("--end",   type=int, default=None,
                        help="End date YYYYMM (default: all available)")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download, ignoring cache")
    parser.add_argument("--output", type=str, default="factors.json",
                        help="Path to write factor assumptions JSON (default: factors.json)")
    args = parser.parse_args()

    if args.refresh:
        for f in CACHE_DIR.glob("*.csv"):
            f.unlink()
        print("Cache cleared.")

    # --- fetch and parse ---
    ff5_text = download_french_csv(FF5_URL)
    ff5_dates, ff5_data = parse_monthly(ff5_text)
    ff5_dates, ff5_data = filter_dates(ff5_dates, ff5_data, args.start, args.end)

    mom_text = download_french_csv(MOM_URL)
    mom_dates, mom_data = parse_monthly(mom_text)
    mom_dates, mom_data = filter_dates(mom_dates, mom_data, args.start, args.end)

    # align to common date range
    common_dates = np.intersect1d(ff5_dates, mom_dates)
    ff5_mask = np.isin(ff5_dates, common_dates)
    mom_mask = np.isin(mom_dates, common_dates)
    n = len(common_dates)

    print(f"\nSample period: {common_dates[0]} – {common_dates[-1]}  ({n} months, {n/12:.1f} years)\n")

    # --- compute stats ---
    sources = {
        "Mkt-RF": ff5_data["Mkt-RF"][ff5_mask],
        "SMB":    ff5_data["SMB"][ff5_mask],
        "HML":    ff5_data["HML"][ff5_mask],
        "RMW":    ff5_data["RMW"][ff5_mask],
        "CMA":    ff5_data["CMA"][ff5_mask],
        "Mom":    mom_data["Mom"][mom_mask],
    }

    results = {}
    for french_col, returns in sources.items():
        raw_mean, std = compute_stats(returns)
        adj_mean = raw_mean * HAIRCUTS[french_col]
        results[french_col] = (raw_mean, adj_mean, std)

    # --- human-readable summary ---
    print(f"{'Factor':<8} {'Raw mean%':>10} {'Haircut':>8} {'Adj mean%':>10} {'Std%':>8}")
    print("-" * 50)
    for french_col, (raw_mean, adj_mean, std) in results.items():
        mvo_name = RENAME[french_col]
        haircut = HAIRCUTS[french_col]
        print(f"{mvo_name:<8} {raw_mean:>10.4f} {haircut:>8.0%} {adj_mean:>10.4f} {std:>8.4f}")

    # --- write JSON output ---
    # ITT/LTT are not available from French's library; include hardcoded values as a baseline.
    output = {
        "sample_period": f"{common_dates[0]}-{common_dates[-1]}",
        "factors": {
            "TER": {"mean": round(-1 / 12.0, 6), "std": 0.0,    "note": "expense ratio drag, set per asset"},
            "MKT": {"mean": round(results["Mkt-RF"][1], 6), "std": round(results["Mkt-RF"][2], 6)},
            "SmB": {"mean": round(results["SMB"][1],    6), "std": round(results["SMB"][2],    6)},
            "HmL": {"mean": round(results["HML"][1],    6), "std": round(results["HML"][2],    6)},
            "RmW": {"mean": round(results["RMW"][1],    6), "std": round(results["RMW"][2],    6)},
            "CmA": {"mean": round(results["CMA"][1],    6), "std": round(results["CMA"][2],    6)},
            "MOM": {"mean": round(results["Mom"][1],    6), "std": round(results["Mom"][2],    6)},
            "ITT": {"mean": round(1.66 / 12, 6), "std": round(5.67 / 12**0.5, 6), "note": "hardcoded — update manually"},
            "LTT": {"mean": round(2.01 / 12, 6), "std": round(9.79 / 12**0.5, 6), "note": "hardcoded — update manually"},
        },
    }

    out_path = pathlib.Path(args.output)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote factor assumptions to {out_path}")
    print(f"Pass to mvo.py with:  python mvo.py --factors {out_path}")


if __name__ == "__main__":
    main()
