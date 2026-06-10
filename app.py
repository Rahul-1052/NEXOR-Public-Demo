from flask import Flask, render_template, request, jsonify
from pathlib import Path
from functools import lru_cache
from chatbot import get_chatbot_response

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly
import json
import re


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "demo_data"


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def pick_col(df, options):
    lower_map = {str(c).lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lower_map:
            return lower_map[opt.lower()]
    return None


def empty_chart_html(title="No data available"):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=380,
        annotations=[
            dict(
                text="No data available",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
        ]
    )
    return fig.to_html(full_html=False)


def empty_chart_json(title="No data available"):
    fig = go.Figure()
    fig.update_layout(
        title=None,
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        annotations=[
            dict(
                text=title,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
        ]
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def parse_filter_date(date_str, end_of_day=False):
    if not date_str:
        return None

    dt = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(dt):
        return None

    if end_of_day:
        dt = dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    return dt


def normalize_text(val):
    if pd.isna(val):
        return ""
    text = str(val).strip().lower()
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_disease(val):
    text = normalize_text(val)
    text = text.replace("(disorder)", "")
    text = text.replace("(finding)", "")
    text = text.replace("(situation)", "")
    text = text.replace("(procedure)", "")
    text = text.replace("(regime/therapy)", "")
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_forecast_view(val):
    text = normalize_text(val)

    if text in ["both", "actual + forecast", "actual+forecast", "all"]:
        return "Both"
    if text in ["actual only", "actual"]:
        return "Actual Only"
    if text in ["forecast only", "forecast"]:
        return "Forecast Only"

    return "Both"


STATE_ABBREV = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC"
}


# ---------------------------------------------------
# LOAD INSURANCE DATA
# ---------------------------------------------------
@lru_cache(maxsize=1)
def load_insurance_data():
    merged_path = DATA_DIR / "merged_encounters.csv"
    forecast_path = DATA_DIR / "claim_cost_forecast.csv"
    payers_lookup_path = DATA_DIR / "payers_lookup.csv"

    if not merged_path.exists():
        raise FileNotFoundError(f"Missing file: {merged_path}. Run prepare_data.py first.")

    encounters = pd.read_csv(merged_path, low_memory=False)

    payer_col = pick_col(encounters, ["PAYER", "PAYER_ID"])
    cost_col = pick_col(encounters, ["BASE_ENCOUNTER_COST", "CLAIM_COST", "COST", "TOTAL_CLAIM_COST"])
    claim_type_col = pick_col(encounters, ["ENCOUNTERCLASS", "TYPE", "CLAIM_TYPE"])
    date_col = pick_col(encounters, ["START", "DATE", "START_DATE"])
    encounter_patient_col = pick_col(encounters, ["PATIENT"])
    disease_col = pick_col(encounters, ["DESCRIPTION", "REASONDESCRIPTION", "DISEASE", "Disease"])

    if payer_col is None:
        encounters["PAYER"] = "Unknown"
        payer_col = "PAYER"

    if cost_col is None:
        encounters["BASE_ENCOUNTER_COST"] = 0
        cost_col = "BASE_ENCOUNTER_COST"

    if claim_type_col is None:
        encounters["ENCOUNTERCLASS"] = "Unknown"
        claim_type_col = "ENCOUNTERCLASS"

    if date_col is None:
        raise ValueError("No usable date column found in merged_encounters.csv")

    if encounter_patient_col is None:
        encounters["PATIENT"] = "Unknown"
        encounter_patient_col = "PATIENT"

    encounters = encounters.copy()

    encounters["_payer"] = encounters[payer_col].astype(str).str.strip()
    encounters["_cost"] = pd.to_numeric(encounters[cost_col], errors="coerce").fillna(0)
    encounters["_claim_type"] = encounters[claim_type_col].astype(str).str.strip()

    # Clean and validate claim dates
    encounters["_date"] = pd.to_datetime(encounters[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    encounters = encounters.dropna(subset=["_date"])

    # Public demo date guardrail:
    # Removes unrealistic dates such as 1920, 1940, etc.
    encounters = encounters[
        (encounters["_date"] >= pd.Timestamp("2015-01-01")) &
        (encounters["_date"] <= pd.Timestamp("2026-12-31"))
    ].copy()

    encounters["_patient"] = encounters[encounter_patient_col].astype(str).str.strip()
    encounters["MonthStart"] = encounters["_date"].dt.to_period("M").dt.to_timestamp()

    if disease_col:
        encounters["_disease"] = encounters[disease_col].astype(str).str.strip()
        encounters["DiseaseKey"] = encounters["_disease"].apply(normalize_disease)
    else:
        encounters["_disease"] = "Unknown"
        encounters["DiseaseKey"] = "unknown"

    if "STATE" not in encounters.columns:
        encounters["STATE"] = "Unknown"
    if "CITY" not in encounters.columns:
        encounters["CITY"] = "Unknown"
    if "LAT" not in encounters.columns:
        encounters["LAT"] = np.nan
    if "LON" not in encounters.columns:
        encounters["LON"] = np.nan

    encounters["STATE"] = encounters["STATE"].fillna("Unknown").astype(str).str.strip()
    encounters["CITY"] = encounters["CITY"].fillna("Unknown").astype(str).str.strip()
    encounters["LAT"] = pd.to_numeric(encounters["LAT"], errors="coerce")
    encounters["LON"] = pd.to_numeric(encounters["LON"], errors="coerce")

    encounters["PAYER_NAME"] = encounters["_payer"]
    encounters["PAYER_GROUP"] = "Other"

    if payers_lookup_path.exists():
        try:
            payers_lookup = pd.read_csv(payers_lookup_path, low_memory=False)

            payer_id_col = pick_col(payers_lookup, ["PAYER_ID", "PAYER"])
            payer_name_col = pick_col(payers_lookup, ["PAYER_NAME", "NAME"])
            payer_group_col = pick_col(payers_lookup, ["PAYER_GROUP", "GROUP"])

            if payer_id_col and payer_name_col:
                lookup_cols = [payer_id_col, payer_name_col]

                if payer_group_col:
                    lookup_cols.append(payer_group_col)

                lookup = payers_lookup[lookup_cols].copy()
                lookup[payer_id_col] = lookup[payer_id_col].astype(str).str.strip()

                rename_map = {
                    payer_id_col: "_payer_lookup_id",
                    payer_name_col: "_payer_lookup_name"
                }

                if payer_group_col:
                    rename_map[payer_group_col] = "_payer_lookup_group"

                lookup = lookup.rename(columns=rename_map)

                encounters = encounters.merge(
                    lookup,
                    left_on="_payer",
                    right_on="_payer_lookup_id",
                    how="left"
                )

                encounters["PAYER_NAME"] = encounters["_payer_lookup_name"].fillna(encounters["_payer"]).astype(str)

                if "_payer_lookup_group" in encounters.columns:
                    encounters["PAYER_GROUP"] = encounters["_payer_lookup_group"].fillna("Other").astype(str)

        except Exception as e:
            print("Payer lookup error:", str(e))
            encounters["PAYER_NAME"] = encounters["_payer"]
            encounters["PAYER_GROUP"] = "Other"

    if forecast_path.exists():
        forecast = pd.read_csv(forecast_path, low_memory=False)
        print("\n========== FORECAST FILE ==========")
        print(forecast_path)
        print(forecast.head())
        print("===================================\n")

        month_col = pick_col(forecast, ["MonthStart", "MONTH", "DATE", "ForecastMonth"])
        forecast_col = pick_col(forecast, ["ForecastClaimCost", "PredictedClaimCost", "Forecast", "Prediction"])

        if month_col and forecast_col:
            forecast = forecast.copy()
            forecast["MonthStart"] = pd.to_datetime(forecast[month_col], errors="coerce", utc=True).dt.tz_localize(None)
            forecast["MonthStart"] = forecast["MonthStart"].dt.to_period("M").dt.to_timestamp()
            forecast["ForecastClaimCost"] = pd.to_numeric(forecast[forecast_col], errors="coerce").fillna(0)
            forecast = forecast.dropna(subset=["MonthStart"])

            # Keep forecast timeline realistic too
            forecast = forecast[
                (forecast["MonthStart"] >= pd.Timestamp("2015-01-01")) &
                (forecast["MonthStart"] <= pd.Timestamp("2027-12-31"))
            ].copy()

            forecast = forecast.sort_values("MonthStart")
            forecast = forecast[["MonthStart", "ForecastClaimCost"]]
        else:
            forecast = pd.DataFrame(columns=["MonthStart", "ForecastClaimCost"])
    else:
        forecast = pd.DataFrame(columns=["MonthStart", "ForecastClaimCost"])

    return encounters, forecast


@lru_cache(maxsize=1)
def get_insurance_filter_options():
    encounters, _ = load_insurance_data()

    payer_options = sorted(encounters["PAYER_NAME"].dropna().astype(str).unique().tolist())
    claim_type_options = sorted(encounters["_claim_type"].dropna().astype(str).unique().tolist())

    state_options = sorted(
        encounters.loc[
            encounters["STATE"].notna() &
            (encounters["STATE"] != "Unknown") &
            (encounters["STATE"] != ""),
            "STATE"
        ].astype(str).unique().tolist()
    )

    city_by_state = {}
    for state, group in encounters.groupby("STATE"):
        if state and state != "Unknown":
            city_by_state[state] = sorted(
                group.loc[
                    group["CITY"].notna() &
                    (group["CITY"] != "Unknown") &
                    (group["CITY"] != ""),
                    "CITY"
                ].astype(str).unique().tolist()
            )

    disease_counts = (
        encounters.groupby(["DiseaseKey", "_disease"], as_index=False)
        .size()
        .rename(columns={"size": "Count"})
        .sort_values("Count", ascending=False)
    )

    if disease_counts.empty:
        disease_options = []
    else:
        disease_options = (
            disease_counts.groupby("DiseaseKey", as_index=False)
            .agg(
                Disease=("_disease", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
                Count=("Count", "sum")
            )
            .sort_values("Count", ascending=False)
            .head(150)["Disease"]
            .astype(str)
            .sort_values()
            .tolist()
        )

    return payer_options, claim_type_options, state_options, city_by_state, disease_options


# ---------------------------------------------------
# LOAD PHARMA FILES
# ---------------------------------------------------
@lru_cache(maxsize=1)
def load_pharma_files():
    actuals_path = DATA_DIR / "pharma_actuals.csv"
    forecast_path = DATA_DIR / "pharma_forecast.csv"
    opportunity_path = DATA_DIR / "pharma_opportunity.csv"

    missing = []
    for path in [actuals_path, forecast_path, opportunity_path]:
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Missing pharma training output files. Run python train_pharma_forecast.py first.\n"
            + "\n".join(missing)
        )

    actuals = pd.read_csv(actuals_path, low_memory=False)
    forecast = pd.read_csv(forecast_path, low_memory=False)
    opportunity = pd.read_csv(opportunity_path, low_memory=False)

    actuals["MonthStart"] = pd.to_datetime(actuals["MonthStart"], errors="coerce")
    forecast["MonthStart"] = pd.to_datetime(forecast["MonthStart"], errors="coerce")

    actuals["Disease"] = actuals["Disease"].astype(str).str.strip()
    forecast["Disease"] = forecast["Disease"].astype(str).str.strip()
    opportunity["Disease"] = opportunity["Disease"].astype(str).str.strip()

    actuals["DiseaseKey"] = actuals["Disease"].apply(normalize_disease)
    forecast["DiseaseKey"] = forecast["Disease"].apply(normalize_disease)
    opportunity["DiseaseKey"] = opportunity["Disease"].apply(normalize_disease)

    actuals["PatientCount"] = pd.to_numeric(actuals.get("PatientCount", 0), errors="coerce").fillna(0)
    actuals["ClaimAmount"] = pd.to_numeric(actuals.get("ClaimAmount", 0), errors="coerce").fillna(0)

    forecast["ForecastPatientCount"] = pd.to_numeric(forecast.get("ForecastPatientCount", 0), errors="coerce").fillna(0)
    forecast["ForecastClaimAmount"] = pd.to_numeric(forecast.get("ForecastClaimAmount", 0), errors="coerce").fillna(0)

    opportunity["CurrentPatientVolume"] = pd.to_numeric(opportunity.get("CurrentPatientVolume", 0), errors="coerce").fillna(0)
    opportunity["CurrentCost"] = pd.to_numeric(opportunity.get("CurrentCost", 0), errors="coerce").fillna(0)
    opportunity["ForecastValue"] = pd.to_numeric(opportunity.get("ForecastValue", 0), errors="coerce").fillna(0)

    if "OpportunityScore" not in opportunity.columns:
        opportunity["OpportunityScore"] = (
            opportunity["CurrentPatientVolume"] * 0.45
            + opportunity["ForecastValue"] * 0.0005
        )
    else:
        opportunity["OpportunityScore"] = pd.to_numeric(opportunity["OpportunityScore"], errors="coerce").fillna(0)

    actuals = actuals.dropna(subset=["MonthStart"])
    forecast = forecast.dropna(subset=["MonthStart"])

    # Public demo date guardrail
    actuals = actuals[
    (actuals["MonthStart"] >= pd.Timestamp("2015-01-01")) &
    (actuals["MonthStart"] <= pd.Timestamp("2026-12-31"))
    ].copy()

    forecast = forecast[
    (forecast["MonthStart"] >= pd.Timestamp("2026-01-01")) &
    (forecast["MonthStart"] <= pd.Timestamp("2027-12-31"))
    ].copy()

    return actuals, forecast, opportunity


# ---------------------------------------------------
# INSURANCE CHARTS
# ---------------------------------------------------
def build_claim_type_chart(df, top_n=5):
    grp = (
        df.groupby("_claim_type", as_index=False)
        .size()
        .rename(columns={"size": "Count"})
        .sort_values("Count", ascending=False)
        .head(top_n)
    )

    if grp.empty:
        return empty_chart_html("Claim Type Distribution")

    fig = px.bar(
        grp,
        x="_claim_type",
        y="Count",
        title="Claim Type Distribution",
        labels={"_claim_type": "Claim Type", "Count": "Claims"}
    )

    fig.update_traces(
        text=grp["Count"],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Claims: %{y:,}<extra></extra>"
    )

    fig.update_layout(
        template="plotly_white",
        height=350,
        xaxis_tickangle=-20,
        margin=dict(l=40, r=20, t=60, b=60)
    )

    return fig.to_html(full_html=False)


def build_payer_leakage_chart(df, top_n=10):
    grp = (
        df.groupby("PAYER_NAME", as_index=False)["_cost"]
        .sum()
        .sort_values("_cost", ascending=False)
        .head(top_n)
    )

    if grp.empty:
        return empty_chart_html("Payer Revenue Leakage")

    fig = px.bar(
        grp,
        x="_cost",
        y="PAYER_NAME",
        orientation="h",
        title="Payer Revenue Leakage",
        labels={"_cost": "Claim Cost", "PAYER_NAME": "Payer"}
    )

    fig.update_traces(
        text=grp["_cost"].apply(lambda x: f"${x:,.0f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Claim Cost: $%{x:,.0f}<extra></extra>"
    )

    fig.update_layout(
        template="plotly_white",
        height=380,
        yaxis=dict(categoryorder="total ascending"),
        margin=dict(l=100, r=40, t=60, b=40)
    )

    return fig.to_html(full_html=False)




def build_claim_cost_trend(df, forecast):
    actual = (
        df.groupby("MonthStart", as_index=False)["_cost"]
        .sum()
        .rename(columns={"_cost": "ActualClaimCost"})
        .sort_values("MonthStart")
    )

    actual["MonthStart"] = pd.to_datetime(actual["MonthStart"], errors="coerce")
    actual = actual.dropna(subset=["MonthStart"]).sort_values("MonthStart")

    forecast = forecast.copy()

    if not forecast.empty:
        forecast["MonthStart"] = pd.to_datetime(forecast["MonthStart"], errors="coerce")
        forecast = forecast.dropna(subset=["MonthStart"]).sort_values("MonthStart")

    if not actual.empty:
        actual = actual.sort_values("MonthStart").copy()

        # Remove incomplete latest month if it is much lower than recent history
        if len(actual) >= 4:
            last_value = actual.iloc[-1]["ActualClaimCost"]
            recent_avg = actual.iloc[-4:-1]["ActualClaimCost"].mean()

            if last_value < recent_avg * 0.85:
                actual = actual.iloc[:-1].copy()

        latest_actual_month = actual["MonthStart"].max()
        actual_cutoff = latest_actual_month - pd.DateOffset(months=36)
        actual = actual[actual["MonthStart"] >= actual_cutoff].copy()
    else:
        latest_actual_month = None

    if latest_actual_month is not None and not forecast.empty:
        forecast = forecast[forecast["MonthStart"] > latest_actual_month].copy()

        if not forecast.empty:
            last_actual_value = actual.iloc[-1]["ActualClaimCost"]
            first_forecast_value = forecast.iloc[0]["ForecastClaimCost"]

            if first_forecast_value != 0:
                scale_factor = last_actual_value / first_forecast_value
                forecast["ForecastClaimCost"] = forecast["ForecastClaimCost"] * scale_factor

            bridge_row = pd.DataFrame({
                "MonthStart": [latest_actual_month],
                "ForecastClaimCost": [last_actual_value]
            })

            forecast = pd.concat([bridge_row, forecast], ignore_index=True)

    fig = go.Figure()

    if not actual.empty:
        fig.add_trace(go.Scatter(
            x=actual["MonthStart"],
            y=actual["ActualClaimCost"],
            mode="lines+markers",
            name="Actual",
            line=dict(width=3),
            marker=dict(size=6),
            hovertemplate="Month: %{x|%b %Y}<br>Actual claim cost: $%{y:,.0f}<extra></extra>"
        ))

    if not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast["MonthStart"],
            y=forecast["ForecastClaimCost"],
            mode="lines+markers",
            name="Forecast",
            line=dict(dash="dash", width=3),
            marker=dict(size=6),
            hovertemplate="Month: %{x|%b %Y}<br>Forecast claim cost: $%{y:,.0f}<extra></extra>"
        ))

    if actual.empty and forecast.empty:
        return empty_chart_html("Actual vs Forecast Claim Cost")

    all_dates = pd.concat([
        actual["MonthStart"] if not actual.empty else pd.Series(dtype="datetime64[ns]"),
        forecast["MonthStart"] if not forecast.empty else pd.Series(dtype="datetime64[ns]")
    ])

    min_date = all_dates.min()
    max_date = all_dates.max()

    fig.update_layout(
        title="Projected Market Cost Trend",
        template="plotly_white",
        height=420,
        xaxis_title="Month",
        yaxis_title="Claim Cost",
        yaxis_tickprefix="$",
        margin=dict(l=45, r=25, t=60, b=80),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
        xaxis=dict(
            type="date",
            tickformat="%b %Y",
            tickangle=-45,
            showgrid=True,
            tickmode="auto",
            nticks=10,
            range=[min_date, max_date]
        ),
        yaxis=dict(
            rangemode="tozero",
            tickformat=",.0f",
            showgrid=True
        )
    )

    return fig.to_html(full_html=False)


def build_payer_performance_rows(df):
    grp = (
        df.groupby(["PAYER_NAME", "PAYER_GROUP"], as_index=False)
        .agg(
            TotalClaims=("_cost", "count"),
            TotalCost=("_cost", "sum"),
            AverageClaimCost=("_cost", "mean")
        )
        .sort_values("TotalCost", ascending=False)
        .head(20)
    )

    if grp.empty:
        return []

    grp["TotalClaims"] = grp["TotalClaims"].apply(lambda x: f"{x:,}")
    grp["TotalCost"] = grp["TotalCost"].apply(lambda x: f"${x:,.0f}")
    grp["AverageClaimCost"] = grp["AverageClaimCost"].apply(lambda x: f"${x:,.0f}")

    return grp.to_dict(orient="records")


def build_state_map(df):
    grp = (
        df[df["STATE"].notna() & (df["STATE"] != "Unknown") & (df["STATE"] != "")]
        .groupby("STATE", as_index=False)["_cost"]
        .sum()
    )

    if grp.empty:
        return empty_chart_html("Geographic Cost Distribution by State")

    grp["STATE_CODE"] = grp["STATE"].map(STATE_ABBREV)
    grp = grp.dropna(subset=["STATE_CODE"])

    if grp.empty:
        return empty_chart_html("Geographic Cost Distribution by State")

    grp["ColorValue"] = np.log1p(grp["_cost"])

    fig = px.choropleth(
        grp,
        locations="STATE_CODE",
        locationmode="USA-states",
        color="ColorValue",
        scope="usa",
        hover_name="STATE",
        custom_data=["_cost"],
        color_continuous_scale="Blues",
        labels={"ColorValue": "Scaled Cost"},
        title="Geographic Cost Distribution by State"
    )

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Total Claim Cost: $%{customdata[0]:,.0f}<extra></extra>"
    )

    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig.to_html(full_html=False)


def build_city_map(df):
    city_df = df.copy()

    city_df = city_df[
        city_df["CITY"].notna() &
        city_df["STATE"].notna() &
        city_df["LAT"].notna() &
        city_df["LON"].notna() &
        (city_df["CITY"] != "Unknown") &
        (city_df["STATE"] != "Unknown") &
        (city_df["CITY"] != "") &
        (city_df["STATE"] != "")
    ]

    if city_df.empty:
        return empty_chart_html("City-Level Claim Distribution")

    grp = (
        city_df.groupby(["CITY", "STATE", "LAT", "LON"], as_index=False)
        .agg(
            TotalClaimCost=("_cost", "sum"),
            TotalClaims=("_cost", "count")
        )
        .sort_values("TotalClaimCost", ascending=False)
        .head(250)
    )

    if grp.empty:
        return empty_chart_html("City-Level Claim Distribution")

    fig = px.scatter_geo(
        grp,
        lat="LAT",
        lon="LON",
        size="TotalClaimCost",
        hover_name="CITY",
        hover_data={
            "STATE": True,
            "TotalClaimCost": ":,.0f",
            "TotalClaims": True,
            "LAT": False,
            "LON": False
        },
        scope="usa",
        title="City-Level Claim Distribution"
    )

    fig.update_traces(
        marker=dict(opacity=0.65, line=dict(width=1, color="white"))
    )

    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig.to_html(full_html=False)


# ---------------------------------------------------
# PHARMA CHARTS
# ---------------------------------------------------
def build_pharma_forecast_chart_json(actuals, forecast, selected_disease="All", forecast_view="Both"):
    actual_df = actuals.copy()
    forecast_df = forecast.copy()

    actual_df["MonthStart"] = pd.to_datetime(actual_df["MonthStart"], errors="coerce")
    forecast_df["MonthStart"] = pd.to_datetime(forecast_df["MonthStart"], errors="coerce")

    actual_df = actual_df.dropna(subset=["MonthStart"])
    forecast_df = forecast_df.dropna(subset=["MonthStart"])

    actual_df = actual_df[
        (actual_df["MonthStart"] >= pd.Timestamp("2024-01-01")) &
        (actual_df["MonthStart"] <= pd.Timestamp("2026-12-31"))
    ].copy()

    forecast_df = forecast_df[
        (forecast_df["MonthStart"] >= pd.Timestamp("2026-01-01")) &
        (forecast_df["MonthStart"] <= pd.Timestamp("2027-12-31"))
    ].copy()

    actual_df["Disease"] = actual_df["Disease"].astype(str).str.strip()
    forecast_df["Disease"] = forecast_df["Disease"].astype(str).str.strip()

    actual_df["DiseaseKey"] = actual_df["Disease"].apply(normalize_disease)
    forecast_df["DiseaseKey"] = forecast_df["Disease"].apply(normalize_disease)

    if selected_disease != "All":
        selected_key = normalize_disease(selected_disease)
        actual_df = actual_df[actual_df["DiseaseKey"] == selected_key]
        forecast_df = forecast_df[forecast_df["DiseaseKey"] == selected_key]

    actual_grp = (
        actual_df.groupby("MonthStart", as_index=False)
        .agg(PatientCount=("PatientCount", "sum"))
        .sort_values("MonthStart")
    )

    forecast_grp = (
        forecast_df.groupby("MonthStart", as_index=False)
        .agg(PatientCount=("ForecastPatientCount", "sum"))
        .sort_values("MonthStart")
    )

    fig = go.Figure()

    if forecast_view in ["Both", "Actual Only"] and not actual_grp.empty:
        fig.add_trace(go.Scatter(
            x=actual_grp["MonthStart"],
            y=actual_grp["PatientCount"],
            mode="lines+markers",
            name="Historical Demand",
            line=dict(width=3),
            marker=dict(size=6),
            hovertemplate="Month: %{x|%b %Y}<br>Actual patients: %{y:,.0f}<extra></extra>"
        ))

    if forecast_view in ["Both", "Forecast Only"] and not forecast_grp.empty:
        if not actual_grp.empty and forecast_view == "Both":
            last_actual_month = actual_grp["MonthStart"].max()
            last_actual_value = actual_grp.iloc[-1]["PatientCount"]

            forecast_grp = forecast_grp[forecast_grp["MonthStart"] > last_actual_month].copy()

            bridge = pd.DataFrame({
                "MonthStart": [last_actual_month],
                "PatientCount": [last_actual_value]
            })

            forecast_grp = pd.concat([bridge, forecast_grp], ignore_index=True)

        fig.add_trace(go.Scatter(
            x=forecast_grp["MonthStart"],
            y=forecast_grp["PatientCount"],
            mode="lines+markers",
            name="Forecast Demand",
            line=dict(width=3, dash="dash"),
            marker=dict(size=6),
            hovertemplate="Month: %{x|%b %Y}<br>Forecast patients: %{y:,.0f}<extra></extra>"
        ))

    if actual_grp.empty and forecast_grp.empty:
        return empty_chart_json("No patient demand data available")

    fig.update_layout(
        title=None,
        template="plotly_white",
        height=430,
        xaxis_title="Month",
        yaxis_title="Patient Demand",
        margin=dict(l=45, r=25, t=25, b=80),
        legend=dict(orientation="h", y=1.05, x=0),
        hovermode="x unified",
        xaxis=dict(
            type="date",
            tickformat="%b %Y",
            tickangle=-45,
            showgrid=True,
            nticks=8
        ),
        yaxis=dict(
            rangemode="tozero",
            tickformat=",.0f",
            showgrid=True
        )
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def build_pharma_opportunity_matrix_json(opportunity, selected_disease="All"):
    matrix = opportunity.copy()

    matrix["Disease"] = matrix["Disease"].astype(str).str.strip()
    matrix["DiseaseKey"] = matrix["Disease"].apply(normalize_disease)

    matrix["CurrentPatientVolume"] = pd.to_numeric(
        matrix.get("CurrentPatientVolume", 0), errors="coerce"
    ).fillna(0)

    matrix["ForecastValue"] = pd.to_numeric(
        matrix.get("ForecastValue", 0), errors="coerce"
    ).fillna(0)

    matrix["CurrentCost"] = pd.to_numeric(
        matrix.get("CurrentCost", 0), errors="coerce"
    ).fillna(0)

    matrix["OpportunityScore"] = pd.to_numeric(
        matrix.get("OpportunityScore", 0), errors="coerce"
    ).fillna(0)

    if selected_disease != "All":
        selected_key = normalize_disease(selected_disease)
        matrix = matrix[matrix["DiseaseKey"] == selected_key]

    matrix = matrix.sort_values("OpportunityScore", ascending=False).head(10).copy()

    if matrix.empty:
        return empty_chart_json("No opportunity data available")

    matrix["DisplayDisease"] = (
        matrix["Disease"]
        .astype(str)
        .str.replace(r"\s*\(.*?\)", "", regex=True)
        .str.strip()
    )

    max_patients = matrix["CurrentPatientVolume"].max()
    max_forecast = matrix["ForecastValue"].max()
    max_score = matrix["OpportunityScore"].max()

    if max_patients == 0:
        max_patients = 1
    if max_forecast == 0:
        max_forecast = 1
    if max_score == 0:
        max_score = 1

    matrix["DemandIndex"] = (matrix["CurrentPatientVolume"] / max_patients) * 100
    matrix["ValueIndex"] = (matrix["ForecastValue"] / max_forecast) * 100
    matrix["BubbleSize"] = 25 + (matrix["OpportunityScore"] / max_score) * 45

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=matrix["DemandIndex"],
        y=matrix["ValueIndex"],
        mode="markers+text",
        text=matrix["DisplayDisease"],
        textposition="top center",
        customdata=np.stack([
            matrix["Disease"],
            matrix["CurrentPatientVolume"],
            matrix["ForecastValue"],
            matrix["CurrentCost"],
            matrix["OpportunityScore"]
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Current patient volume: %{customdata[1]:,.0f}<br>"
            "Forecast claim value: $%{customdata[2]:,.0f}<br>"
            "Current estimated cost: $%{customdata[3]:,.0f}<br>"
            "Opportunity score: %{customdata[4]:,.1f}"
            "<extra></extra>"
        ),
        marker=dict(
            size=matrix["BubbleSize"],
            color=matrix["OpportunityScore"],
            colorscale="Viridis",
            showscale=True,
            opacity=0.85,
            line=dict(width=2, color="white")
        )
    ))

    fig.add_vline(x=50, line_dash="dash", line_color="rgba(75,85,99,0.45)", line_width=1.5)
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(75,85,99,0.45)", line_width=1.5)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=65, r=45, t=35, b=75),
        xaxis=dict(title="Current demand index", range=[0, 110]),
        yaxis=dict(title="Forecast value index", range=[0, 110]),
        showlegend=False
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def build_pharma_forecast_summary_rows(forecast, selected_disease="All"):
    df = forecast.copy()

    if selected_disease != "All":
        selected_key = normalize_disease(selected_disease)
        df = df[df["DiseaseKey"] == selected_key]

    if df.empty:
        return []

    grp = (
        df.groupby(["DiseaseKey", "Disease"], as_index=False)
        .agg(ForecastClaimAmount=("ForecastClaimAmount", "sum"))
        .sort_values("ForecastClaimAmount", ascending=False)
    )

    return grp[["Disease", "ForecastClaimAmount"]].to_dict(orient="records")


# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html", show_filters=False)


@app.route("/insurance-dashboard")
def insurance_dashboard():
    try:
        encounters, forecast = load_insurance_data()
        payer_options, claim_type_options, state_options, city_by_state, disease_options = get_insurance_filter_options()

        start_date = request.args.get("start_date", "")
        end_date = request.args.get("end_date", "")
        selected_state = request.args.get("state", "All")
        selected_city = request.args.get("city", "All")
        selected_disease = request.args.get("disease", "All")
        payer_value = request.args.get("payer", "All")
        claim_type_value = request.args.get("claim_type", "All")

        selected_payers = [] if payer_value in ["", "All", None] else [payer_value]
        selected_claim_types = [] if claim_type_value in ["", "All", None] else [claim_type_value]

        if selected_state not in state_options:
            selected_state = "All"

        city_options = city_by_state.get(selected_state, []) if selected_state != "All" else []

        if selected_city != "All":
            if selected_state == "All" or selected_city not in city_options:
                selected_city = "All"

        filtered_df = encounters.copy()

        parsed_start = parse_filter_date(start_date, end_of_day=False)
        parsed_end = parse_filter_date(end_date, end_of_day=True)

        if parsed_start is not None:
            filtered_df = filtered_df[filtered_df["_date"] >= parsed_start]

        if parsed_end is not None:
            filtered_df = filtered_df[filtered_df["_date"] <= parsed_end]

        if selected_state != "All":
            filtered_df = filtered_df[filtered_df["STATE"].astype(str) == selected_state]

        if selected_city != "All":
            filtered_df = filtered_df[filtered_df["CITY"].astype(str) == selected_city]

        if selected_payers:
            filtered_df = filtered_df[filtered_df["PAYER_NAME"].isin(selected_payers)]

        if selected_claim_types:
            filtered_df = filtered_df[filtered_df["_claim_type"].isin(selected_claim_types)]

        if selected_disease != "All":
            selected_key = normalize_disease(selected_disease)
            filtered_df = filtered_df[filtered_df["DiseaseKey"] == selected_key]

        total_claims = len(filtered_df)
        total_actual_claim_cost = float(filtered_df["_cost"].sum()) if not filtered_df.empty else 0
        total_forecast_claim_cost = float(forecast["ForecastClaimCost"].sum()) if not forecast.empty else 0

        payer_costs = (
            filtered_df.groupby("PAYER_NAME", as_index=False)["_cost"]
            .sum()
            .sort_values("_cost", ascending=False)
        )
        highest_leakage_payer = payer_costs.iloc[0]["PAYER_NAME"] if not payer_costs.empty else "N/A"

        state_costs = (
            filtered_df.loc[
                filtered_df["STATE"].notna() & (filtered_df["STATE"] != "Unknown")
            ]
            .groupby("STATE", as_index=False)["_cost"]
            .sum()
            .sort_values("_cost", ascending=False)
        )
        top_state = state_costs.iloc[0]["STATE"] if not state_costs.empty else "N/A"

        if not filtered_df.empty:
            min_actual = filtered_df["MonthStart"].min()
            max_actual = filtered_df["MonthStart"].max()
            latest_data_note = (
                f"Historical claims are available through {max_actual.strftime('%B %Y')}. "
                f"Current filtered records span from {min_actual.strftime('%B %Y')} "
                f"to {max_actual.strftime('%B %Y')}."
            )
        else:
            latest_data_note = "No records are available for the selected filter criteria."

        return render_template(
            "insurance_dashboard.html",
            show_filters=False,
            page_type="insurance",
            total_claims=total_claims,
            total_actual_claim_cost=total_actual_claim_cost,
            total_forecast_claim_cost=total_forecast_claim_cost,
            highest_leakage_payer=highest_leakage_payer,
            top_state=top_state,
            latest_data_note=latest_data_note,
            claim_type_chart=build_claim_type_chart(filtered_df, top_n=5),
            payer_leakage_chart=build_payer_leakage_chart(filtered_df, top_n=10),
            forecast_chart=build_claim_cost_trend(filtered_df, forecast),
            state_map=build_state_map(filtered_df),
            city_map=build_city_map(filtered_df),
            payer_table_rows=build_payer_performance_rows(filtered_df),
            start_date=start_date,
            end_date=end_date,
            selected_payers=selected_payers,
            selected_claim_types=selected_claim_types,
            selected_state=selected_state,
            selected_city=selected_city,
            selected_disease=selected_disease,
            payer_options=payer_options,
            claim_type_options=claim_type_options,
            state_options=state_options,
            city_options=city_options,
            disease_options=disease_options
        )

    except Exception as e:
        return f"Error loading insurance dashboard:<br>{str(e)}"


@app.route("/pharma-dashboard")
def pharma_dashboard():
    try:
        actuals, forecast, opportunity = load_pharma_files()

        selected_disease = request.args.get("disease", "All")
        forecast_view = normalize_forecast_view(request.args.get("forecast_view", "Both"))
        selected_start_date = request.args.get("start_date", "")
        selected_end_date = request.args.get("end_date", "")

        disease_options = sorted(opportunity["Disease"].dropna().astype(str).unique().tolist())

        if selected_disease != "All" and selected_disease not in disease_options:
            selected_disease = "All"

        actuals_filtered = actuals.copy()
        forecast_filtered = forecast.copy()
        opportunity_filtered = opportunity.copy()

        parsed_start = parse_filter_date(selected_start_date, end_of_day=False)
        parsed_end = parse_filter_date(selected_end_date, end_of_day=True)

        if parsed_start is not None:
            actuals_filtered = actuals_filtered[actuals_filtered["MonthStart"] >= parsed_start]
            forecast_filtered = forecast_filtered[forecast_filtered["MonthStart"] >= parsed_start]

        if parsed_end is not None:
            actuals_filtered = actuals_filtered[actuals_filtered["MonthStart"] <= parsed_end]
            forecast_filtered = forecast_filtered[forecast_filtered["MonthStart"] <= parsed_end]

        if selected_disease != "All":
            selected_key = normalize_disease(selected_disease)
            actuals_filtered = actuals_filtered[actuals_filtered["DiseaseKey"] == selected_key]
            forecast_filtered = forecast_filtered[forecast_filtered["DiseaseKey"] == selected_key]
            opportunity_filtered = opportunity_filtered[opportunity_filtered["DiseaseKey"] == selected_key]

        total_diseases = int(opportunity_filtered["Disease"].nunique()) if not opportunity_filtered.empty else 0
        total_patients = int(opportunity_filtered["CurrentPatientVolume"].sum()) if not opportunity_filtered.empty else 0
        total_forecast_value = float(opportunity_filtered["ForecastValue"].sum()) if not opportunity_filtered.empty else 0

        forecast_chart = build_pharma_forecast_chart_json(
            actuals,
            forecast,
            selected_disease=selected_disease,
            forecast_view=forecast_view
        )

        opportunity_matrix = build_pharma_opportunity_matrix_json(
            opportunity,
            selected_disease=selected_disease
        )

        forecast_summary = build_pharma_forecast_summary_rows(
            forecast_filtered,
            selected_disease="All"
        )

        if not actuals_filtered.empty:
            first_actual = actuals_filtered["MonthStart"].min()
            latest_actual = actuals_filtered["MonthStart"].max()
            forecast_note = (
                f"Historical patient demand is shown from {first_actual.strftime('%B %Y')} "
                f"through {latest_actual.strftime('%B %Y')}."
            )
        else:
            forecast_note = "Historical patient demand data is not available for the selected filters."

        return render_template(
            "pharma_dashboard.html",
            show_filters=False,
            page_type="pharma",
            total_diseases=total_diseases,
            total_patients=total_patients,
            total_forecast_value=total_forecast_value,
            forecast_chart=forecast_chart,
            opportunity_matrix=opportunity_matrix,
            forecast_summary=forecast_summary,
            disease_options=disease_options,
            selected_disease=selected_disease,
            forecast_view=forecast_view,
            selected_start_date=selected_start_date,
            selected_end_date=selected_end_date,
            forecast_note=forecast_note
        )

    except Exception as e:
        return f"Error loading pharma dashboard:<br>{str(e)}"


# ---------------------------------------------------
# CHATBOT ROUTE
# ---------------------------------------------------
@app.route("/ask", methods=["POST"])
@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get("message", "").strip()

        if not user_message:
            answer = "Please enter a question."
            return jsonify({"response": answer, "reply": answer})

        answer = get_chatbot_response(user_message)

        return jsonify({"response": answer, "reply": answer})

    except Exception as e:
        print("CHATBOT ERROR:", str(e))
        answer = "Sorry, I could not generate a response right now."
        return jsonify({"response": answer, "reply": answer})


if __name__ == "__main__":
    app.run(debug=True)