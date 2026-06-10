import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("demo_data/merged_encounters.csv")

df["START"] = pd.to_datetime(df["START"], errors="coerce")
df["MonthStart"] = df["START"].dt.to_period("M").dt.to_timestamp()

monthly = df.groupby("MonthStart").agg(
    TotalEncounters=("Id", "count"),
    TotalClaimCost=("TOTAL_CLAIM_COST", "sum"),
    TotalPatients=("PATIENT", "nunique")
).reset_index()

monthly = monthly.sort_values("MonthStart")
monthly = monthly[monthly["TotalClaimCost"] > 0]

monthly["TimeIndex"] = range(1, len(monthly) + 1)

X = monthly[["TimeIndex", "TotalPatients", "TotalEncounters"]]
y = monthly["TotalClaimCost"]

model = LinearRegression()
model.fit(X, y)

last_time = monthly["TimeIndex"].max()
avg_patients = monthly["TotalPatients"].tail(3).mean()
avg_encounters = monthly["TotalEncounters"].tail(3).mean()

future_X = pd.DataFrame({
    "TimeIndex": np.arange(last_time + 1, last_time + 7),
    "TotalPatients": [avg_patients] * 6,
    "TotalEncounters": [avg_encounters] * 6
})

forecast_values = model.predict(future_X)

future_dates = pd.date_range(
    start=monthly["MonthStart"].max() + pd.DateOffset(months=1),
    periods=6,
    freq="MS"
)

forecast_df = pd.DataFrame({
    "MonthStart": future_dates,
    "ForecastClaimCost": forecast_values
})

forecast_df.to_csv("demo_data/claim_cost_forecast.csv", index=False)

print("Saved demo forecast to data/claim_cost_forecast.csv")
print("Last actual:")
print(monthly.tail(3)[["MonthStart", "TotalClaimCost"]])
print("New forecast:")
print(forecast_df)