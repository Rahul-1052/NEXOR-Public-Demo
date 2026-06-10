from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEMO_DIR = BASE_DIR / "demo_data"

DEMO_DIR.mkdir(exist_ok=True)

TARGET_STATES = [
    "California",
    "Massachusetts",
    "New Jersey",
    "Pennsylvania",
    "Texas",
    "New York",
    "Illinois",
    "North Carolina",
    "Maryland",
    "Indiana"
]


def filter_csv_by_state(input_file, output_file):
    input_path = DATA_DIR / input_file
    output_path = DEMO_DIR / output_file

    if not input_path.exists():
        print(f"Skipped missing file: {input_path}")
        return

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_path, low_memory=False)

    state_col = None
    for col in df.columns:
        if col.lower() == "state":
            state_col = col
            break

    if state_col:
        before = len(df)
        df = df[df[state_col].isin(TARGET_STATES)].copy()
        after = len(df)
        print(f"Filtered {input_file}: {before:,} rows → {after:,} rows")
    else:
        print(f"No STATE column found in {input_file}. Copying as-is.")

    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}\n")


def copy_csv(input_file, output_file):
    input_path = DATA_DIR / input_file
    output_path = DEMO_DIR / output_file

    if not input_path.exists():
        print(f"Skipped missing file: {input_path}")
        return

    print(f"Copying {input_file}...")
    df = pd.read_csv(input_path, low_memory=False)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} with {len(df):,} rows\n")


def main():
    print("Creating NEXOR public demo dataset...\n")

    # Runtime files used by Insurance Dashboard
    filter_csv_by_state("merged_encounters.csv", "merged_encounters.csv")
    copy_csv("claim_cost_forecast.csv", "claim_cost_forecast.csv")
    copy_csv("payers_lookup.csv", "payers_lookup.csv")

    # Runtime files used by Pharma Dashboard
    copy_csv("pharma_actuals.csv", "pharma_actuals.csv")
    copy_csv("pharma_forecast.csv", "pharma_forecast.csv")
    copy_csv("pharma_opportunity.csv", "pharma_opportunity.csv")

    print("Demo data created successfully.")
    print("States included:")
    for state in TARGET_STATES:
        print(f"- {state}")


if __name__ == "__main__":
    main()