import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

encounters_file = DATA_DIR / "encounters.csv"
output_file = DATA_DIR / "payers_lookup.csv"

# Load encounters
encounters = pd.read_csv(encounters_file)

# Make sure PAYER column exists
if "PAYER" not in encounters.columns:
    raise ValueError("PAYER column not found in encounters.csv")

# Get unique payer IDs
payer_ids = (
    encounters["PAYER"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

# Create readable names
payers_lookup = pd.DataFrame({
    "PAYER_ID": payer_ids,
    "PAYER_NAME": [f"Payer {i+1}" for i in range(len(payer_ids))]
})

# Save file
payers_lookup.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
print(payers_lookup.head(10))