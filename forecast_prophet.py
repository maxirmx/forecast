import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# === Step 1: Load Excel file ===
input_file = "data/1_dataset.xlsx"
df = pd.read_excel(input_file, sheet_name="Данные", skiprows=2)

# === Step 2: Rename and clean columns ===
df = df.rename(columns={
    df.columns[6]: "ds",   # Год и месяц оформления документа
    df.columns[9]: "y"     # Кол-во единиц оборудования
})
df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna(subset=["ds", "y"])

# === Step 3: Group monthly and sum quantities ===
df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()
df_grouped = df.groupby("ds").sum(numeric_only=True).reset_index()

# Save cleaned data
df_grouped.to_csv("output/cleaned_data.csv", index=False)

# === Step 4: Forecast with Prophet ===
model = Prophet(yearly_seasonality=True)

# If you want to enforce a custom monthly seasonality:
# model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

model.fit(df_grouped)

future = model.make_future_dataframe(periods=12, freq="M")
forecast = model.predict(future)

# Save forecast
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv("output/forecast_output.csv", index=False)

# === Step 5: Plot forecast ===
fig = model.plot(forecast)
plt.title("Прогноз 12 месяцев")
plt.xlabel("Дата")
plt.ylabel("Количество")
plt.grid(True)
plt.tight_layout()
plt.savefig("output/forecast_plot.png")
plt.show()
