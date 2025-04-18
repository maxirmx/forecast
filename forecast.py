import pandas as pd
import matplotlib.pyplot as plt
import argparse
from statsmodels.tsa.statespace.sarimax import SARIMAX
from copy import copy


def format_and_export_excel(df, output_file, sheet_name="Data"):
    """Format and export DataFrame to Excel with proper styling

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame to export
    output_file : str
        Path to save the Excel file
    sheet_name : str
        Name of the worksheet
    """
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Export with proper formatting
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]

        # Auto-adjust column width based on content
        for idx, col in enumerate(df.columns):
            # Find the maximum length of data in each column
            max_len = min(30, max(
                df[col].astype(str).map(len).max(),  # Max data length
                len(str(col)))  # Header length
            ) + 2  # Add a little extra space

            # Set the column width
            column_letter = chr(65 + idx)
            worksheet.column_dimensions[column_letter].width = max_len

            # Format header cell
            header_cell = f"{column_letter}1"
            from copy import copy
            alignment = copy(worksheet[header_cell].alignment)
            alignment.wrapText = True
            alignment.vertical = 'center'
            alignment.horizontal = 'center'
            worksheet[header_cell].alignment = alignment

def load_and_preprocess_data(input_file, sheet_name="Данные", skip_rows=2):
    """Load and preprocess the input Excel file with two time series groupings"""
    # Load data
    df = pd.read_excel(input_file, sheet_name=sheet_name, skiprows=skip_rows)

    # Rename columns
    df = df.rename(columns={
        df.columns[6]: "ds_document",         # Год и месяц оформления документа
        df.columns[7]: "ds_ownership",        # Год и месяц внесения в документ на оборудование записи о владельце
        df.columns[9]: "y"                    # Кол-во единиц оборудования
    })

    # Clean data
    # Convert date columns to datetime with specific format
    df["ds_document"] = pd.to_datetime(df["ds_document"], format="%Y-%m", errors="coerce")
    df["ds_ownership"] = pd.to_datetime(df["ds_ownership"], format="%Y-%m", errors="coerce")

    # Drop rows where dates don't match the YYYY-MM format
    df = df.dropna(subset=["ds_document", "ds_ownership"], how="all")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Create two dataframes - one for each date column
    # First for document date
    df_document = df.dropna(subset=["ds_document", "y"]).copy()
    df_document["ds"] = df_document["ds_document"]
    df_document["ds"] = df_document["ds"].dt.to_period("M").dt.to_timestamp()
    df_grouped_document = df_document.groupby("ds").sum(numeric_only=True).reset_index()
    df_grouped_document = df_grouped_document.rename(columns={"y": "document_count"})

    # Second for ownership date
    df_ownership = df.dropna(subset=["ds_ownership", "y"]).copy()
    df_ownership["ds"] = df_ownership["ds_ownership"]
    df_ownership["ds"] = df_ownership["ds"].dt.to_period("M").dt.to_timestamp()
    df_grouped_ownership = df_ownership.groupby("ds").sum(numeric_only=True).reset_index()
    df_grouped_ownership = df_grouped_ownership.rename(columns={"y": "ownership_count"})

    # Merge both datasets on the date column
    df_combined = pd.merge(df_grouped_document, df_grouped_ownership, on="ds", how="outer")
    df_combined.fillna(0, inplace=True)
    df_combined["document_count"] = df_combined["document_count"].round().astype(int)
    df_combined["ownership_count"] = df_combined["ownership_count"].round().astype(int)
    df_combined = df_combined.sort_values("ds")

    # Create proper monthly date range with MS frequency
    min_date = df_combined["ds"].min().to_period("M").to_timestamp()
    max_date = df_combined["ds"].max().to_period("M").to_timestamp()
    date_range = pd.date_range(start=min_date, end=max_date, freq="MS")

    # Reindex the combined data to ensure complete monthly sequence
    df_combined_reindexed = df_combined.set_index("ds").reindex(date_range, fill_value=0).reset_index()
    df_combined_reindexed = df_combined_reindexed.rename(columns={"index": "ds"})

    # Create compatible dataframes for the forecasting function with proper frequency
    df_document_compat = pd.DataFrame({
        "ds": df_combined_reindexed["ds"],
        "y": df_combined_reindexed["document_count"]
    })

    df_ownership_compat = pd.DataFrame({
        "ds": df_combined_reindexed["ds"],
        "y": df_combined_reindexed["ownership_count"]
    })

    # Format Excel output for display
    df_combined_excel = df_combined_reindexed.copy()
    df_combined_excel["Месяц"] = df_combined_excel["ds"].dt.strftime("%Y-%m")

    # Rename columns for Excel output
    df_combined_excel = df_combined_excel.rename(columns={
        "document_count": "Количество оформлений документа на оборудование",
        "ownership_count": "Количество созданных записей о владельце"
    })

    # Reorder columns to make "Месяц" the first column
    cols = ["Месяц"] + [col for col in df_combined_excel.columns if col != "Месяц" and col != "ds"]
    df_combined_excel = df_combined_excel[cols]

    # Save Excel with proper formatting
    format_and_export_excel(df_combined_excel, "output/cleaned_data_combined.xlsx", "Data")

    return {
        "document": df_document_compat,
        "ownership": df_ownership_compat,
        "combined": df_combined_reindexed
    }

def run_sarima_model(df_grouped, periods=6, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    """Run SARIMA forecasting model with non-negative integer constraints

    Parameters:
    -----------
    df_grouped : pandas DataFrame
        Input data frame with 'ds' and 'y' columns
    periods : int
        Number of periods to forecast
    order : tuple
        SARIMA order parameters (p, d, q)
        - p: autoregressive order
        - d: differencing order
        - q: moving average order
    seasonal_order : tuple
        SARIMA seasonal order parameters (P, D, Q, S)
        - P: seasonal autoregressive order
        - D: seasonal differencing order
        - Q: seasonal moving average order
        - S: seasonal period (e.g., 12 for monthly data)
    """

    # Filter data to only include entries from January 2023 onwards
    cutoff_date = pd.Timestamp('2023-01-01')
    df_grouped = df_grouped[df_grouped["ds"] >= cutoff_date].copy()

    # Prepare data
    series = df_grouped.set_index("ds")["y"]
    series.index.freq = "MS"  # Set frequency to Month Start explicitly

    # Initialize and fit model
    print(f"SARIMA parameters: order={order}, seasonal_order={seasonal_order}")
    sarima_model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    sarima_result = sarima_model.fit(disp=False, maxiter=500)

    # Forecast
    sarima_forecast = sarima_result.get_forecast(steps=periods)
    sarima_index = pd.date_range(start=series.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS")

    # Get confidence intervals
    conf_int = sarima_forecast.conf_int()

    # Create a DataFrame with the forecast results and apply integer, non-negative constraints
    forecast_df = pd.DataFrame({
        "ds": sarima_index,
        "yhat": sarima_forecast.predicted_mean.clip(lower=0).round().astype(int),
        "yhat_lower": conf_int.iloc[:, 0].clip(lower=0).round().astype(int),
        "yhat_upper": conf_int.iloc[:, 1].clip(lower=0).round().astype(int)
    })

    # Save forecast
    forecast_df.to_excel("output/forecast_output.xlsx", index=False)

    # We'll store both original and processed forecasts for plotting
    return {
        'forecast_df': forecast_df,
        'sarima_result': sarima_result,
        'sarima_forecast': sarima_forecast,
        'sarima_forecast_processed': {
            'mean': forecast_df["yhat"],
            'lower': forecast_df["yhat_lower"],
            'upper': forecast_df["yhat_upper"]
        },
        'sarima_index': sarima_index,
        'series': series
    }


def plot_sarima_forecast(sarima_results):
    """Plot SARIMA forecast with specified styling"""
    series = sarima_results['series']
    sarima_index = sarima_results['sarima_index']

    # Use the integer-constrained values for plotting
    forecast_df = sarima_results['forecast_df']
    forecast_values = forecast_df['yhat']
    forecast_lower = forecast_df['yhat_lower']
    forecast_upper = forecast_df['yhat_upper']

    plt.figure(figsize=(12, 6))

    # Plot observed data with connected line
    plt.plot(series.index, series.values, 'k-', label="Observed")
    plt.plot(series.index, series.values, 'k.', alpha=0.8)

    # Connect last observed point to first forecast point
    last_observed_date = series.index[-1]
    first_forecast_date = sarima_index[0]
    last_observed_value = series.values[-1]
    first_forecast_value = forecast_values.iloc[0]
    plt.plot([last_observed_date, first_forecast_date],
             [last_observed_value, first_forecast_value], '--', color='green')

    # Plot forecast with dashed line and points
    plt.plot(sarima_index, forecast_values, '--', color='green', label='SARIMA Forecast')
    plt.plot(sarima_index, forecast_values, 'o', color='green', markersize=4)

    # Plot confidence intervals
    plt.fill_between(sarima_index,
                     forecast_lower,
                     forecast_upper,
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


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Forecasting with different models")
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

    print("Running SARIMA model [document] ...")
    sarima_results_document = run_sarima_model(
        df_grouped["document"],
        periods=args.periods
    )

    print("Running SARIMA model [ownership] ...")
    sarima_results_ownership = run_sarima_model(
        df_grouped["ownership"],
        periods=args.periods

    )

    print("Plotting SARIMA forecast...")
    plot_sarima_forecast(sarima_results_document)
    plot_sarima_forecast(sarima_results_ownership)

    print(f"Forecasting complete! Output files saved to the output directory.")

if __name__ == "__main__":
    main()
