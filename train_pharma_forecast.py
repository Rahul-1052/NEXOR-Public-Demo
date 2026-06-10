from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CONDITIONS_FILE = DATA_DIR / "conditions.csv"
ENCOUNTERS_FILE = DATA_DIR / "encounters.csv"

ACTUALS_OUT = DATA_DIR / "pharma_actuals.csv"
FORECAST_OUT = DATA_DIR / "pharma_forecast.csv"
OPPORTUNITY_OUT = DATA_DIR / "pharma_opportunity.csv"

TOP_N_DISEASES = 30
FORECAST_MONTHS = 6


def safe_read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, low_memory=False)


def pick_col(df, options):
    lower_map = {str(c).lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lower_map:
            return lower_map[opt.lower()]
    return None


def main():
    print("Loading conditions.csv...")
    conditions = safe_read_csv(CONDITIONS_FILE)

    print("Loading encounters.csv...")
    encounters = safe_read_csv(ENCOUNTERS_FILE)

    patient_col = pick_col(conditions, ["PATIENT", "Patient", "patient"])
    disease_col = pick_col(conditions, ["DESCRIPTION", "Disease", "DISEASE", "CODE_DESCRIPTION"])
    start_col = pick_col(conditions, ["START", "Start", "DATE", "Date"])

    if patient_col is None:
        raise ValueError("Could not find patient column in conditions.csv")

    if disease_col is None:
        raise ValueError("Could not find disease/description column in conditions.csv")

    if start_col is None:
        raise ValueError("Could not find date/start column in conditions.csv")

    conditions = conditions.copy()
    conditions["PATIENT_ID"] = conditions[patient_col].astype(str)
    conditions["Disease"] = conditions[disease_col].astype(str).str.strip()
    conditions["MonthStart"] = pd.to_datetime(conditions[start_col], errors="coerce")
    conditions = conditions.dropna(subset=["MonthStart"])
    conditions["MonthStart"] = conditions["MonthStart"].dt.to_period("M").dt.to_timestamp()

    disease_counts = (
        conditions.groupby("Disease", as_index=False)["PATIENT_ID"]
        .nunique()
        .rename(columns={"PATIENT_ID": "UniquePatients"})
        .sort_values("UniquePatients", ascending=False)
    )

    top_diseases = disease_counts.head(TOP_N_DISEASES)["Disease"].tolist()

    conditions = conditions[conditions["Disease"].isin(top_diseases)].copy()

    encounter_patient_col = pick_col(encounters, ["PATIENT", "Patient", "patient"])
    cost_col = pick_col(encounters, ["BASE_ENCOUNTER_COST", "CLAIM_COST", "COST", "TOTAL_CLAIM_COST"])

    if encounter_patient_col and cost_col:
        encounter_costs = encounters.copy()
        encounter_costs["PATIENT_ID"] = encounter_costs[encounter_patient_col].astype(str)
        encounter_costs["ClaimAmount"] = pd.to_numeric(
            encounter_costs[cost_col],
            errors="coerce"
        ).fillna(0)

        patient_cost = (
            encounter_costs.groupby("PATIENT_ID", as_index=False)["ClaimAmount"]
            .mean()
            .rename(columns={"ClaimAmount": "AvgPatientClaimAmount"})
        )

        conditions = conditions.merge(patient_cost, on="PATIENT_ID", how="left")
        conditions["AvgPatientClaimAmount"] = conditions["AvgPatientClaimAmount"].fillna(500)
    else:
        conditions["AvgPatientClaimAmount"] = 500

    actuals = (
        conditions.groupby(["MonthStart", "Disease"], as_index=False)
        .agg(
            PatientCount=("PATIENT_ID", "nunique"),
            ClaimAmount=("AvgPatientClaimAmount", "sum")
        )
        .sort_values(["Disease", "MonthStart"])
    )

    actuals["ClaimAmount"] = pd.to_numeric(
        actuals["ClaimAmount"],
        errors="coerce"
    ).fillna(0)

    # IMPORTANT:
    # Remove the last partial month because it can create a fake sudden drop.
    # This makes the forecast chart cleaner and more realistic.
    if not actuals.empty:
        last_partial_month = actuals["MonthStart"].max()
        actuals = actuals[actuals["MonthStart"] < last_partial_month].copy()

    forecast_rows = []

    for disease in top_diseases:
        disease_df = actuals[actuals["Disease"] == disease].copy()
        disease_df = disease_df.sort_values("MonthStart")

        if disease_df.empty:
            continue

        disease_df["TimeIndex"] = np.arange(len(disease_df))

        last_month = disease_df["MonthStart"].max()

        if len(disease_df) >= 2:
            model_patients = LinearRegression()
            model_patients.fit(disease_df[["TimeIndex"]], disease_df["PatientCount"])

            model_claims = LinearRegression()
            model_claims.fit(disease_df[["TimeIndex"]], disease_df["ClaimAmount"])

            future_index = np.arange(len(disease_df), len(disease_df) + FORECAST_MONTHS)

            patient_predictions = model_patients.predict(future_index.reshape(-1, 1))
            claim_predictions = model_claims.predict(future_index.reshape(-1, 1))
        else:
            patient_predictions = np.repeat(
                disease_df["PatientCount"].iloc[-1],
                FORECAST_MONTHS
            )

            claim_predictions = np.repeat(
                disease_df["ClaimAmount"].iloc[-1],
                FORECAST_MONTHS
            )

        patient_predictions = np.maximum(patient_predictions, 0)
        claim_predictions = np.maximum(claim_predictions, 0)

        for i in range(FORECAST_MONTHS):
            forecast_month = last_month + pd.DateOffset(months=i + 1)

            forecast_rows.append({
                "MonthStart": forecast_month,
                "Disease": disease,
                "ForecastPatientCount": round(float(patient_predictions[i]), 2),
                "ForecastClaimAmount": round(float(claim_predictions[i]), 2)
            })

    forecast = pd.DataFrame(forecast_rows)

    latest_actual_month = actuals["MonthStart"].max()

    current = (
        actuals[actuals["MonthStart"] == latest_actual_month]
        .groupby("Disease", as_index=False)
        .agg(
            CurrentPatientVolume=("PatientCount", "sum"),
            CurrentCost=("ClaimAmount", "sum")
        )
    )

    forecast_value = (
        forecast.groupby("Disease", as_index=False)
        .agg(
            ForecastValue=("ForecastClaimAmount", "sum"),
            ForecastPatientVolume=("ForecastPatientCount", "sum")
        )
    )

    opportunity = current.merge(
        forecast_value,
        on="Disease",
        how="outer"
    ).fillna(0)

    opportunity["OpportunityScore"] = (
        opportunity["CurrentPatientVolume"] * 0.45
        + opportunity["ForecastPatientVolume"] * 0.35
        + opportunity["ForecastValue"] * 0.0005
    )

    opportunity = opportunity.sort_values("OpportunityScore", ascending=False)

    actuals.to_csv(ACTUALS_OUT, index=False)
    forecast.to_csv(FORECAST_OUT, index=False)
    opportunity.to_csv(OPPORTUNITY_OUT, index=False)

    print("Saved pharma_actuals.csv")
    print("Saved pharma_forecast.csv")
    print("Saved pharma_opportunity.csv")

    print("\nTop diseases used:")
    for disease in top_diseases:
        print(f"- {disease}")


if __name__ == "__main__":
    main()