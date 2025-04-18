import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

# === Load Excel ===
input_file = "data/1_dataset.xlsx"
df = pd.read_excel(input_file, sheet_name="Данные", skiprows=2)

df = df.rename(columns={
    df.columns[6]: "ds",
    df.columns[9]: "y"
})
df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna(subset=["ds", "y"])

# === Monthly aggregation ===
df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()
df_grouped = df.groupby("ds").sum(numeric_only=True).reset_index()
df_grouped.to_csv("output/cleaned_data.csv", index=False)

# === SARIMA Forecast ===
series = df_grouped.set_index("ds")["y"]
sarima_model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit(disp=False)

sarima_forecast = sarima_result.get_forecast(steps=12)
sarima_index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS")
sarima_pred = pd.Series(sarima_forecast.predicted_mean.values, index=sarima_index)

# === Combined Plot ===
plt.figure(figsize=(12, 6))
plt.plot(series.index, series.values, label="Observed", marker="o")
plt.plot(sarima_pred.index, sarima_pred.values, label="SARIMA Forecast", linestyle=":")
plt.title("Прогноз")
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("output/forecast_comparison.png")
plt.show()
