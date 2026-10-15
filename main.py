from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def load_fshd_data():
    """Load the FSHD CSV and return it as a pandas DataFrame."""
    default_path = Path(__file__).resolve().parent / "Module2_FSHD_Dataset.csv"

    if default_path.exists():
        csv_path = default_path
    else:
        entered_path = input(
            "Enter the full path to Module2_FSHD_Dataset.csv: "
        ).strip().strip('"')
        csv_path = Path(entered_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find the CSV file at:\n{csv_path}\n"
            "Check the file path and try again."
        )

    # df is created here and returned for use in main().
    df = pd.read_csv(csv_path)
    print(f"\nLoaded dataset: {csv_path}")
    print(f"Rows in dataset: {len(df)}")
    return df


def main():
    # Load the dataset before using df.
    df = load_fshd_data()

    required_columns = [
        "subject_id",
        "stir_quant_baseline_R", "stir_quant_baseline_L",
        "stir_quant_y1_R", "stir_quant_y1_L",
        "dorsiflexion_qmt_R_baseline", "dorsiflexion_qmt_L_baseline",
        "dorsiflexion_qmt_R_y1", "dorsiflexion_qmt_L_y1",
        "clinical_severity_score_baseline", "clinical_severity_score_y1",
    ]

    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "These required columns are missing from the CSV:\n"
            + "\n".join(missing_columns)
        )

numeric_columns = [column for column in required_columns if column != "subject_id"]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # Average right and left sides; if one side is missing, use the available side.
df["stir_baseline"] = df[
        ["stir_quant_baseline_R", "stir_quant_baseline_L"]
    ].mean(axis=1)
df["stir_y1"] = df[
        ["stir_quant_y1_R", "stir_quant_y1_L"]
    ].mean(axis=1)

df["dorsiflexion_baseline"] = df[
        ["dorsiflexion_qmt_R_baseline", "dorsiflexion_qmt_L_baseline"]
    ].mean(axis=1)
df["dorsiflexion_y1"] = df[
        ["dorsiflexion_qmt_R_y1", "dorsiflexion_qmt_L_y1"]
    ].mean(axis=1)

    # Change is year 1 minus baseline.
df["severity_change"] = (
        df["clinical_severity_score_y1"] - df["clinical_severity_score_baseline"]
    )
df["stir_change"] = df["stir_y1"] - df["stir_baseline"]
df["dorsiflexion_change"] = (
        df["dorsiflexion_y1"] - df["dorsiflexion_baseline"]
    )

    # Use the same complete cases for both comparisons.
analysis_df = df[
        ["subject_id", "severity_change", "stir_change", "dorsiflexion_change"]
    ].replace([np.inf, -np.inf], np.nan).dropna().copy()

if len(analysis_df) < 3:
        raise ValueError(
            "Fewer than 3 complete participants remain. Check missing values "
            "and confirm you selected the correct FSHD CSV."
        )

y = analysis_df["severity_change"]
stir_x = analysis_df[["stir_change"]]
dorsiflexion_x = analysis_df[["dorsiflexion_change"]]

stir_model = LinearRegression().fit(stir_x, y)
dorsiflexion_model = LinearRegression().fit(dorsiflexion_x, y)

stir_r2 = stir_model.score(stir_x, y)
dorsiflexion_r2 = dorsiflexion_model.score(dorsiflexion_x, y)

print("Participants included in both comparisons:", len(analysis_df))

print("\nSTIR quantification change:")
print(f"  Slope: {stir_model.coef_[0]:.4f}")
print(f"  R-squared: {stir_r2:.4f}")
if analysis_df["stir_change"].nunique() > 1:
        stir_r, stir_p = stats.pearsonr(analysis_df["stir_change"], y)
        print(f"  Pearson r: {stir_r:.4f}")
        print(f"  p-value: {stir_p:.4g}")
else:
        print("  Pearson r and p-value unavailable: no variation in STIR change.")

print("\nDorsiflexion QMT change:")
print(f"  Slope: {dorsiflexion_model.coef_[0]:.4f}")
print(f"  R-squared: {dorsiflexion_r2:.4f}")
if analysis_df["dorsiflexion_change"].nunique() > 1:
        dorsiflexion_r, dorsiflexion_p = stats.pearsonr(
            analysis_df["dorsiflexion_change"], y
        )
        print(f"  Pearson r: {dorsiflexion_r:.4f}")
        print(f"  p-value: {dorsiflexion_p:.4g}")
else:
        print(
            "  Pearson r and p-value unavailable: "
            "no variation in dorsiflexion change."
        )

if stir_r2 > dorsiflexion_r2:
        print(
            "\nResult: STIR change has the stronger linear association with "
            "clinical severity change in this sample."
        )
elif dorsiflexion_r2 > stir_r2:
        print(
            "\nResult: Dorsiflexion QMT change has the stronger linear "
            "association with clinical severity change in this sample."
        )
else:
        print(
            "\nResult: Both measures have the same R-squared in this sample."
        )

    # Plot each measure against the change in clinical severity.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_specs = [
        (axes[0], "stir_change", stir_model, "STIR quantification change", stir_r2),
        (
            axes[1],
            "dorsiflexion_change",
            dorsiflexion_model,
            "Dorsiflexion QMT change",
            dorsiflexion_r2,
        ),
    ]

for ax, x_column, model, title, r2 in plot_specs:
        x_values = analysis_df[x_column]
        ax.scatter(x_values, y)
        if x_values.nunique() > 1:
            x_line = np.linspace(x_values.min(), x_values.max(), 100)
            line_df = pd.DataFrame({x_column: x_line})
            ax.plot(x_line, model.predict(line_df), linewidth=2)
        ax.set_title(f"{title}\nR² = {r2:.3f}")
        ax.set_xlabel("Change from baseline to year 1")
        ax.set_ylabel("Change in clinical severity score")
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

output_path = Path(__file__).resolve().parent / "clinical_severity_progression_analysis.csv"
analysis_df.to_csv(output_path, index=False)
print(f"\nSaved patient-level results to:\n{output_path}")


if __name__ == "__main__":
    main()
