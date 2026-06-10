import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

encounter_cols = [
    "Id", "START", "STOP", "PATIENT", "ORGANIZATION", "PROVIDER", "PAYER",
    "ENCOUNTERCLASS", "CODE", "DESCRIPTION", "BASE_ENCOUNTER_COST",
    "TOTAL_CLAIM_COST", "PAYER_COVERAGE", "REASONCODE", "REASONDESCRIPTION"
]

patient_cols = [
    "Id", "BIRTHDATE", "DEATHDATE", "SSN", "DRIVERS", "PASSPORT", "PREFIX",
    "FIRST", "MIDDLE", "LAST", "SUFFIX", "MAIDEN", "MARITAL", "RACE",
    "ETHNICITY", "GENDER", "BIRTHPLACE", "ADDRESS", "CITY", "STATE",
    "COUNTY", "FIPS", "ZIP", "LAT", "LON", "HEALTHCARE_EXPENSES",
    "HEALTHCARE_COVERAGE", "INCOME"
]

encounters_path = DATA_DIR / "encounters.csv"
patients_path = DATA_DIR / "patients.csv"
output_path = DATA_DIR / "merged_encounters.csv"

if not encounters_path.exists():
    raise FileNotFoundError(f"Missing file: {encounters_path}")

if not patients_path.exists():
    raise FileNotFoundError(f"Missing file: {patients_path}")

encounters = pd.read_csv(
    encounters_path,
    names=encounter_cols,
    header=None,
    low_memory=False
)

patients = pd.read_csv(
    patients_path,
    names=patient_cols,
    header=None,
    low_memory=False
)

# Remove accidental header rows if the CSV already had headers.
encounters = encounters[encounters["Id"].astype(str).str.lower() != "id"].copy()
patients = patients[patients["Id"].astype(str).str.lower() != "id"].copy()

merged = encounters.merge(
    patients[["Id", "CITY", "STATE", "COUNTY", "LAT", "LON"]],
    left_on="PATIENT",
    right_on="Id",
    how="left",
    suffixes=("", "_PATIENT")
)

if "Id_PATIENT" in merged.columns:
    merged.drop(columns=["Id_PATIENT"], inplace=True)

merged["START"] = pd.to_datetime(merged["START"], errors="coerce", utc=True)
merged["STOP"] = pd.to_datetime(merged["STOP"], errors="coerce", utc=True)
merged["MonthStart"] = merged["START"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()

for col in ["BASE_ENCOUNTER_COST", "TOTAL_CLAIM_COST", "PAYER_COVERAGE"]:
    merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

for col in ["STATE", "CITY", "COUNTY"]:
    merged[col] = merged[col].fillna("Unknown").astype(str).str.strip()

merged["LAT"] = pd.to_numeric(merged["LAT"], errors="coerce")
merged["LON"] = pd.to_numeric(merged["LON"], errors="coerce")

merged.to_csv(output_path, index=False)

state_count = merged.loc[merged["STATE"] != "Unknown", "STATE"].nunique()
city_count = merged.loc[merged["CITY"] != "Unknown", "CITY"].nunique()

print("Merged file created successfully!")
print(f"Output file: {output_path}")
print(f"Rows: {len(merged):,}")
print(f"Unique states: {state_count}")
print(f"Unique cities: {city_count}")
print(f"Missing/Unknown states: {(merged['STATE'] == 'Unknown').sum():,}")

if state_count <= 2:
    print("\nWARNING: This still looks like the old limited-state dataset.")
    print("Replace data/encounters.csv and data/patients.csv with your friend's new files, then run this again.")
else:
    print("\nGood: multi-state data is ready for the dashboard.")
