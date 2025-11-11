import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import glob

# Load all forecast files that match *_Forecast.csv
files = glob.glob("*_Forecast.csv")

results = []

for file in files:
    df = pd.read_csv(file)
    source = file.replace("_Forecast.csv", "")  # Extract dataset name (CPI / GOLD / GDP)

    # Compute metrics for that dataset
    mae = mean_absolute_error(df["Actual"], df["Forecast"])
    rmse = np.sqrt(mean_squared_error(df["Actual"], df["Forecast"]))
    mape = np.mean(np.abs((df["Actual"] - df["Forecast"]) / df["Actual"])) * 100

    ci_coverage = np.mean(
        (df["Actual"] >= df["Lower_CI"]) &
        (df["Actual"] <= df["Upper_CI"])
    ) * 100

    # Save results
    results.append({
        "Dataset": source,
        "Observations": len(df),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "CI Coverage (%)": ci_coverage
    })

# Convert to dataframe for pretty printing
results_df = pd.DataFrame(results)

# Calculate the average MAPE across datasets (unweighted)
average_mape = results_df["MAPE (%)"].mean()

print("===== Forecast Accuracy per Dataset =====")
print(results_df.to_string(index=False))
print("\n===== Summary =====")
print(f"Average MAPE across datasets: {average_mape:.2f}%")