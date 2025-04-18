import pandas as pd
import matplotlib.pyplot as plt
import argparse
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX


def load_and_preprocess_data(input_file, sheet_name="Данные", skip_rows=2):
    """Load and preprocess the input Excel file"""
    # Load data
    df = pd.read_excel(input_file, sheet_name=sheet_name, skiprows=skip_rows)

    # Rename columns
    df = df.rename(columns={
        df.columns[6]: "ds",  # Год и месяц оформления документа
        df.columns[9]: "y"    # Кол-во единиц оборудования
    })

    # Clean data
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"])

    # Group monthly and sum quantities
    df["ds"] = df["ds"].dt.to_period("M").dt.to_timestamp()
    df_grouped = df.groupby("ds").sum(numeric_only=True).reset_index()

    # Save cleaned data
    df_grouped.to_excel("output/cleaned_data.xlsx", index=False)

    return df_grouped


def run_prophet_model(df_grouped, periods=12):
    """Run Prophet forecasting model"""
    # Initialize and fit model
    model = Prophet(yearly_seasonality=True)
    model.fit(df_grouped)

    # Create future dataframe and make predictions
    future = model.make_future_dataframe(periods=periods, freq="M")
    forecast = model.predict(future)

    # Save forecast
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_excel("output/forecast_output_prophet.xlsx", index=False)

    return forecast


def run_sarima_model(df_grouped, periods=12):
    """Run SARIMA forecasting model"""
    # Prepare data
    series = df_grouped.set_index("ds")["y"]

    # Initialize and fit model
    sarima_model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarima_result = sarima_model.fit(disp=False)

    # Forecast
    sarima_forecast = sarima_result.get_forecast(steps=periods)
    sarima_index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS")

    # Create a DataFrame with the forecast results
    forecast_df = pd.DataFrame({
        "ds": sarima_index,
        "yhat": sarima_forecast.predicted_mean,
        "yhat_lower": sarima_forecast.conf_int().iloc[:, 0],
        "yhat_upper": sarima_forecast.conf_int().iloc[:, 1]
    })

    # Save forecast
    forecast_df.to_excel("output/forecast_output_sarima.xlsx", index=False)

    return {
        'forecast_df': forecast_df,
        'sarima_result': sarima_result,
        'sarima_forecast': sarima_forecast,
        'sarima_index': sarima_index,
        'series': series
    }


def plot_prophet_forecast(df_grouped, forecast):
    """Plot Prophet forecast with specified styling"""
    plt.figure(figsize=(12, 6))

    # Get observed data
    df_history = df_grouped.copy()

    # Plot observed data with connected line
    plt.plot(df_history["ds"], df_history["y"], 'k-', label="Observed")
    plt.plot(df_history["ds"], df_history["y"], 'k.', alpha=0.8)

    # Connect last observed point to first forecast point
    last_observed_date = df_history["ds"].iloc[-1]
    last_observed_value = df_history["y"].iloc[-1]
    first_forecast_date = forecast["ds"].iloc[len(df_history)]
    first_forecast_value = forecast["yhat"].iloc[len(df_history)]
    plt.plot([last_observed_date, first_forecast_date],
             [last_observed_value, first_forecast_value], '--', color='blue')

    # Plot Prophet forecast with dashed line and points
    forecast_dates = forecast["ds"].iloc[len(df_history):]
    forecast_values = forecast["yhat"].iloc[len(df_history):]
    plt.plot(forecast_dates, forecast_values, '--', color='blue', label='Prophet Forecast')
    plt.plot(forecast_dates, forecast_values, 'o', color='blue', markersize=4)

    # Plot confidence intervals
    plt.fill_between(forecast_dates,
                     forecast["yhat_lower"].iloc[len(df_history):],
                     forecast["yhat_upper"].iloc[len(df_history):],
                     color='blue',
                     alpha=0.2,
                     label='Prophet Confidence Interval')

    # Add formatting
    plt.title("Прогноз 12 месяцев")
    plt.xlabel("Дата")
    plt.ylabel("Количество")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig("output/forecast_plot_prophet.png")
    plt.show()


def plot_sarima_forecast(sarima_results):
    """Plot SARIMA forecast with specified styling"""
    series = sarima_results['series']
    sarima_forecast = sarima_results['sarima_forecast']
    sarima_index = sarima_results['sarima_index']

    plt.figure(figsize=(12, 6))

    # Plot observed data with connected line
    plt.plot(series.index, series.values, 'k-', label="Observed")
    plt.plot(series.index, series.values, 'k.', alpha=0.8)

    # Connect last observed point to first forecast point
    last_observed_date = series.index[-1]
    first_forecast_date = sarima_index[0]
    last_observed_value = series.values[-1]
    first_forecast_value = sarima_forecast.predicted_mean[0]
    plt.plot([last_observed_date, first_forecast_date],
             [last_observed_value, first_forecast_value], '--', color='green')

    # Plot forecast with dashed line and points
    plt.plot(sarima_index, sarima_forecast.predicted_mean, '--', color='green', label='SARIMA Forecast')
    plt.plot(sarima_index, sarima_forecast.predicted_mean, 'o', color='green', markersize=4)

    # Plot confidence intervals
    plt.fill_between(sarima_index,
                     sarima_forecast.conf_int().iloc[:, 0],
                     sarima_forecast.conf_int().iloc[:, 1],
                     color='green',
                     alpha=0.2,
                     label='SARIMA Confidence Interval')

    # Add formatting
    plt.title("Прогноз 12 месяцев")
    plt.xlabel("Дата")
    plt.ylabel("Количество")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig("output/forecast_plot_sarima.png")
    plt.show()


def plot_combined_forecast(df_grouped, prophet_forecast, sarima_results):
    """Plot combined Prophet and SARIMA forecasts"""
    series = sarima_results['series']
    sarima_forecast = sarima_results['sarima_forecast']
    sarima_index = sarima_results['sarima_index']

    plt.figure(figsize=(12, 6))

    # Plot observed data with connected line
    plt.plot(series.index, series.values, 'k-', label="Observed")
    plt.plot(series.index, series.values, 'k.', alpha=0.8)

    # Connect last observed point to first forecast points for both models
    last_observed_date = series.index[-1]
    last_observed_value = series.values[-1]

    # Connect to Prophet forecast
    first_prophet_date = prophet_forecast["ds"].iloc[len(series)]
    first_prophet_value = prophet_forecast["yhat"].iloc[len(series)]
    plt.plot([last_observed_date, first_prophet_date],
             [last_observed_value, first_prophet_value], '--', color='blue')

    # Connect to SARIMA forecast
    first_sarima_date = sarima_index[0]
    first_sarima_value = sarima_forecast.predicted_mean[0]
    plt.plot([last_observed_date, first_sarima_date],
             [last_observed_value, first_sarima_value], '--', color='green')

    # Get forecast-only sections of Prophet
    prophet_forecast_dates = prophet_forecast["ds"].iloc[len(series):]
    prophet_forecast_values = prophet_forecast["yhat"].iloc[len(series):]

    # Plot Prophet forecast with dashed line and points
    plt.plot(prophet_forecast_dates, prophet_forecast_values, '--', color='blue', label='Prophet Forecast')
    plt.plot(prophet_forecast_dates, prophet_forecast_values, 'o', color='blue', markersize=4)

    # Plot Prophet confidence intervals
    plt.fill_between(prophet_forecast_dates,
                    prophet_forecast["yhat_lower"].iloc[len(series):],
                    prophet_forecast["yhat_upper"].iloc[len(series):],
                    color='blue',
                    alpha=0.2,
                    label='Prophet Confidence Interval')

    # Plot SARIMA forecast with dashed line and points
    plt.plot(sarima_index, sarima_forecast.predicted_mean, '--', color='green', label='SARIMA Forecast')
    plt.plot(sarima_index, sarima_forecast.predicted_mean, 'o', color='green', markersize=4)

    # Plot SARIMA confidence intervals
    plt.fill_between(sarima_index,
                    sarima_forecast.conf_int().iloc[:, 0],
                    sarima_forecast.conf_int().iloc[:, 1],
                    color='green',
                    alpha=0.2,
                    label='SARIMA Confidence Interval')

    # Add formatting
    plt.title("Forecast Comparison: Prophet vs SARIMA")
    plt.xlabel("Дата")
    plt.ylabel("Количество")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig("output/forecast_comparison.png")
    plt.show()


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Forecasting with different models")
    parser.add_argument("--model", type=str, choices=["prophet", "sarima", "combined"],
                         default="combined", help="Which forecasting model to use")
    parser.add_argument("--input", type=str, default="data/1_dataset.xlsx",
                         help="Input Excel file path")
    parser.add_argument("--periods", type=int, default=12,
                         help="Number of periods to forecast")

    # Parse arguments
    args = parser.parse_args()

    # Load and preprocess data
    print(f"Loading data from {args.input}...")
    df_grouped = load_and_preprocess_data(args.input)
    print("Data preprocessing complete.")

    # Run selected model
    if args.model == "prophet" or args.model == "combined":
        print("Running Prophet model...")
        prophet_forecast = run_prophet_model(df_grouped, periods=args.periods)
        print("Prophet forecast complete.")

    if args.model == "sarima" or args.model == "combined":
        print("Running SARIMA model...")
        sarima_results = run_sarima_model(df_grouped, periods=args.periods)
        print("SARIMA forecast complete.")

    # Plot results
    if args.model == "prophet":
        print("Plotting Prophet forecast...")
        plot_prophet_forecast(df_grouped, prophet_forecast)

    elif args.model == "sarima":
        print("Plotting SARIMA forecast...")
        plot_sarima_forecast(sarima_results)

    elif args.model == "combined":
        print("Plotting combined forecast comparison...")
        plot_combined_forecast(df_grouped, prophet_forecast, sarima_results)

    print(f"Forecasting complete! Output files saved to the output directory.")


if __name__ == "__main__":
    main()
