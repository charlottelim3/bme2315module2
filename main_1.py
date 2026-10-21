"""
FSHD: clinical severity vs. dorsiflexion / STIR quant, using standardized
12-month and 24-month estimates for EACH individual patient.

For every measure and subject:
  - Baseline is time 0 months.
  - Estimate the value at exactly 12 months (T1) and 24 months (T2).
  - If an observed visit is exactly 12 or 24 months, preserve that observed value.
  - Otherwise, use that subject's own observed measurements and actual visit lags
    to linearly interpolate the value at 12/24 months.
  - If a target falls outside the subject's observed time range, this script uses
    the nearest two valid observations for linear extrapolation so that a target
    estimate can still be produced. These estimates are flagged in the output
    because extrapolation is less reliable than interpolation.
  - At least two valid time/value observations are required for an estimate.

Percent changes:
  Baseline-T1 (baseline to 12 months)
  Baseline-T2 (baseline to 24 months)
  T1-T2 (12 to 24 months)

Percent change = (later - earlier) / abs(earlier) * 100.
Right/left limbs are calculated separately and then averaged when available.

Outputs:
  interpolation_check.png
  interpolation_check.csv
  interpolated_common_timepoints.csv
  percent_differences.csv
  summary_stats.csv
  percent_difference_plots.png
  mean_sd_plots.png

Requires: pandas, numpy, matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "Module2_FSHD_Dataset.csv"
PAUSE_AFTER_CHECK = True

# ------------------------------ Load data ------------------------------------
if not os.path.exists(CSV_PATH):
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    alternative = os.path.join(here, os.path.basename(CSV_PATH))
    if os.path.exists(alternative):
        CSV_PATH = alternative

df = pd.read_csv(CSV_PATH)

required_columns = ["subject_id", "lag_months_y1", "lag_months_y2"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print(f"Loaded {len(df)} subjects from {CSV_PATH}")

T1 = 12.0
T2 = 24.0
TARGET_TIMES = {"baseline": 0.0, "y1": T1, "y2": T2}

# (display name, baseline column, observed Y1 column, observed Y2 column)
TARGETS = [
    ("Dorsiflexion R (QMT)", "dorsiflexion_qmt_R_baseline",
     "dorsiflexion_qmt_R_y1", "dorsiflexion_qmt_R_y2"),
    ("Dorsiflexion L (QMT)", "dorsiflexion_qmt_L_baseline",
     "dorsiflexion_qmt_L_y1", "dorsiflexion_qmt_L_y2"),
    ("Clinical severity score", "clinical_severity_score_baseline",
     "clinical_severity_score_y1", "clinical_severity_score_y2"),
    ("STIR quant R", "stir_quant_baseline_R", "stir_quant_y1_R", "stir_quant_y2_R"),
    ("STIR quant L", "stir_quant_baseline_L", "stir_quant_y1_L", "stir_quant_y2_L"),
]

def get_number(row, column):
    """Read a finite numeric value from a row; missing columns/values become NaN."""
    if column not in row.index:
        return np.nan
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    if pd.isna(value) or not np.isfinite(value):
        return np.nan
    return float(value)

def subject_points(row, baseline_col, y1_col, y2_col):
    """Return valid observations (months since baseline, value) for one subject/measure."""
    points = [(0.0, get_number(row, baseline_col))]
    lag_y1 = get_number(row, "lag_months_y1")
    lag_y2 = get_number(row, "lag_months_y2")
    points.append((lag_y1, get_number(row, y1_col)))
    points.append((lag_y2, get_number(row, y2_col)))
    points = [(t, v) for t, v in points
              if pd.notna(t) and pd.notna(v) and np.isfinite(t) and np.isfinite(v) and t >= 0]

    # If multiple observations share a time, average values at that time.
    if not points:
        return []
    points_df = pd.DataFrame(points, columns=["time", "value"])
    points_df = points_df.groupby("time", as_index=False)["value"].mean().sort_values("time")
    return list(zip(points_df["time"].astype(float), points_df["value"].astype(float)))

def estimate_at_month(points, target_month):
    """
    Estimate one subject's measurement at target_month.
    Exact observed target times are returned unchanged. Otherwise use the two
    closest bracketing observations; if target is outside the observed range,
    use the nearest two points for linear extrapolation.
    Returns (estimate, method, source_times).
    """
    if len(points) < 2:
        # An exact target observation can be returned even if only one valid point exists.
        exact = [v for t, v in points if np.isclose(t, target_month, atol=1e-8)]
        if exact:
            return float(exact[0]), "observed_exact", (target_month, target_month)
        return np.nan, "insufficient_data", (np.nan, np.nan)

    for t, v in points:
        if np.isclose(t, target_month, atol=1e-8):
            return float(v), "observed_exact", (t, t)

    times = np.array([p[0] for p in points], dtype=float)
    values = np.array([p[1] for p in points], dtype=float)

    if target_month < times[0]:
        i0, i1 = 0, 1
        method = "linear_extrapolation_before_first_visit"
    elif target_month > times[-1]:
        i0, i1 = len(times) - 2, len(times) - 1
        method = "linear_extrapolation_after_last_visit"
    else:
        i1 = int(np.searchsorted(times, target_month, side="right"))
        i0 = i1 - 1
        method = "linear_interpolation"

    t0, t1 = times[i0], times[i1]
    v0, v1 = values[i0], values[i1]
    if np.isclose(t0, t1):
        return np.nan, "duplicate_time_error", (t0, t1)

    estimate = v0 + (v1 - v0) * ((target_month - t0) / (t1 - t0))
    return float(estimate), method, (float(t0), float(t1))

# ------------------------- Standardize all subjects --------------------------
interpolated = pd.DataFrame({"subject_id": df["subject_id"]})
audit_rows = []

# Stable keys used in output columns.
MEASURE_KEYS = {
    "Dorsiflexion R (QMT)": "dors_r",
    "Dorsiflexion L (QMT)": "dors_l",
    "Clinical severity score": "severity",
    "STIR quant R": "stir_r",
    "STIR quant L": "stir_l",
}

for display_name, baseline_col, y1_col, y2_col in TARGETS:
    key = MEASURE_KEYS[display_name]

    for output_tp, target_month in TARGET_TIMES.items():
        interpolated[f"{key}_{output_tp}"] = np.nan

    for row_index, row in df.iterrows():
        points = subject_points(row, baseline_col, y1_col, y2_col)

        # Baseline is the measured baseline value, not a fitted estimate.
        baseline = get_number(row, baseline_col)
        interpolated.loc[row_index, f"{key}_baseline"] = baseline

        for output_tp, target_month in [("y1", T1), ("y2", T2)]:
            estimate, method, source_times = estimate_at_month(points, target_month)
            interpolated.loc[row_index, f"{key}_{output_tp}"] = estimate
            audit_rows.append({
                "subject_id": row["subject_id"],
                "measure": display_name,
                "target_time_months": target_month,
                "estimate": estimate,
                "method": method,
                "source_time_1_months": source_times[0],
                "source_time_2_months": source_times[1],
                "observed_y1_lag_months": get_number(row, "lag_months_y1"),
                "observed_y2_lag_months": get_number(row, "lag_months_y2"),
                "number_of_valid_observations": len(points),
            })

interpolation_check = pd.DataFrame(audit_rows)
interpolated.to_csv("interpolated_common_timepoints.csv", index=False)
interpolation_check.to_csv("interpolation_check.csv", index=False)

print("\nInterpolation methods used:")
print(interpolation_check.groupby(["measure", "method"]).size().to_string())
print("\nSaved interpolated_common_timepoints.csv and interpolation_check.csv.")

# -------------------------- Interpolation check figure ------------------------
fig0, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Show raw visit timing for each subject's dorsiflexion R and its estimated 12/24 points.
shown = 0
for _, row in df.iterrows():
    points = subject_points(
        row, "dorsiflexion_qmt_R_baseline",
        "dorsiflexion_qmt_R_y1", "dorsiflexion_qmt_R_y2"
    )
    if len(points) < 2:
        continue

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    axes[0].plot(xs, ys, marker="o", alpha=0.45, linewidth=1)

    vals = interpolated.loc[interpolated.index == row.name]
    y12 = vals["dors_r_y1"].iloc[0]
    y24 = vals["dors_r_y2"].iloc[0]
    if pd.notna(y12):
        axes[0].scatter(12, y12, marker="*", s=100, zorder=4)
    if pd.notna(y24):
        axes[0].scatter(24, y24, marker="D", s=45, zorder=4)
    shown += 1
    if shown >= 30:
        break

axes[0].axvline(12, color="gray", linestyle="--", label="T1 = 12 months")
axes[0].axvline(24, color="gray", linestyle=":", label="T2 = 24 months")
axes[0].set_xlabel("Months since baseline")
axes[0].set_ylabel("Right dorsiflexion QMT")
axes[0].set_title("Patient-specific trajectories (up to 30 subjects)")
axes[0].legend(fontsize=8)

method_counts = interpolation_check.groupby(["measure", "method"]).size().unstack(fill_value=0)
method_counts.plot(kind="bar", stacked=True, ax=axes[1])
axes[1].set_ylabel("Number of estimates")
axes[1].set_title("Interpolation/extrapolation method by measure")
axes[1].tick_params(axis="x", labelrotation=45)
axes[1].legend(fontsize=7, loc="best")

fig0.tight_layout()
fig0.savefig("interpolation_check.png", dpi=150)
if PAUSE_AFTER_CHECK:
    plt.show()

# ----------------------------- Percent changes -------------------------------
COMPARISONS = [
    ("Baseline-T1", "baseline", "y1"),
    ("Baseline-T2", "baseline", "y2"),
    ("T1-T2", "y1", "y2"),
]

def percent_change(earlier, later):
    """(later - earlier) / abs(earlier) * 100; undefined if earlier is zero."""
    return (later - earlier) / earlier.abs().replace(0, np.nan) * 100

def averaged_limb_change(data, right_key, left_key, early, late):
    """Calculate change for each limb independently, then average available limbs."""
    right = percent_change(data[f"{right_key}_{early}"], data[f"{right_key}_{late}"])
    left = percent_change(data[f"{left_key}_{early}"], data[f"{left_key}_{late}"])
    return pd.concat([right, left], axis=1).mean(axis=1, skipna=True)

pct = pd.DataFrame({"subject_id": df["subject_id"]})

for label, early, late in COMPARISONS:
    comparison_key = f"{early}_to_{late}"
    pct[f"dors_{comparison_key}"] = averaged_limb_change(
        interpolated, "dors_r", "dors_l", early, late
    )
    pct[f"stir_{comparison_key}"] = averaged_limb_change(
        interpolated, "stir_r", "stir_l", early, late
    )
    pct[f"severity_{comparison_key}"] = percent_change(
        interpolated[f"severity_{early}"], interpolated[f"severity_{late}"]
    )

pct = pct.replace([np.inf, -np.inf], np.nan)
pct.to_csv("percent_differences.csv", index=False)
print("Saved percent_differences.csv")

# ----------------------- Scatter plots and regressions -----------------------
MEASURES = [("dors", "Dorsiflexion"), ("stir", "STIR quant")]
fig1, axes = plt.subplots(2, 3, figsize=(17, 10))

for r, (measure_key, measure_name) in enumerate(MEASURES):
    for c, (label, early, late) in enumerate(COMPARISONS):
        ax = axes[r, c]
        comparison_key = f"{early}_to_{late}"
        pair = pct[[f"{measure_key}_{comparison_key}",
                    f"severity_{comparison_key}"]].dropna()
        pair.columns = ["x", "y"]

        ax.axhline(0, color="gray", linewidth=0.6)
        ax.axvline(0, color="gray", linewidth=0.6)
        ax.set_xlabel(f"{measure_name} percent change ({label})")
        ax.set_ylabel(f"Clinical severity percent change ({label})")

        if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
            ax.set_title(f"{measure_name}: {label} (n={len(pair)})")
            ax.text(0.5, 0.5, "Insufficient paired data for regression",
                    ha="center", va="center", transform=ax.transAxes)
            print(f"{measure_name}: {label}: n={len(pair)}; insufficient for regression.")
            continue

        ax.scatter(pair["x"], pair["y"], edgecolor="black", linewidth=0.4, alpha=0.75)
        slope, intercept = np.polyfit(pair["x"], pair["y"], 1)
        predicted = slope * pair["x"] + intercept
        ss_res = ((pair["y"] - predicted) ** 2).sum()
        ss_tot = ((pair["y"] - pair["y"].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
        x_line = np.linspace(pair["x"].min(), pair["x"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black",
                label=f"y = {slope:.3f}x {intercept:+.2f}\nR² = {r2:.3f}")
        ax.legend(fontsize=8)
        ax.set_title(f"{measure_name}: {label} (n={len(pair)})")
        print(f"{measure_name:13s} {label:14s} n={len(pair):3d} "
              f"slope={slope:.4f} R²={r2:.4f}")

fig1.suptitle("Clinical severity percent change vs. dorsiflexion / STIR quant")
fig1.tight_layout()
fig1.savefig("percent_difference_plots.png", dpi=150)

# -------------------------- Summary statistics -------------------------------
DATASETS = [
    ("dors", "Dorsiflexion percent change"),
    ("stir", "STIR quant percent change"),
    ("severity", "Clinical severity percent change"),
]

summary_rows = []
for prefix, title in DATASETS:
    for label, early, late in COMPARISONS:
        values = pct[f"{prefix}_{early}_to_{late}"].dropna()
        summary_rows.append({
            "dataset": title,
            "comparison": label,
            "n": len(values),
            "mean_percent_change": values.mean() if len(values) else np.nan,
            "sd_percent_change": values.std(ddof=1) if len(values) > 1 else np.nan,
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv("summary_stats.csv", index=False)
print("\nSummary statistics:")
print(summary.round(2).to_string(index=False))

fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5.5))
rng = np.random.default_rng(3)

for ax, (prefix, title) in zip(axes2, DATASETS):
    ax.axhline(0, color="gray", linewidth=0.6)
    for i, (label, early, late) in enumerate(COMPARISONS):
        values = pct[f"{prefix}_{early}_to_{late}"].dropna()
        if values.empty:
            continue

        mean = values.mean()
        sd = values.std(ddof=1) if len(values) > 1 else 0.0
        ax.bar(i, mean, alpha=0.5, edgecolor="black", width=0.6)
        ax.errorbar(i, mean, yerr=sd, fmt="none", ecolor="black", capsize=6)
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(values)), values,
                   s=18, color="black", alpha=0.5, zorder=3)
        ax.text(i, mean + sd, f"n={len(values)}\nmean={mean:.1f}\nSD={sd:.1f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(COMPARISONS)))
    ax.set_xticklabels([c[0] for c in COMPARISONS], rotation=15)
    ax.set_ylabel("Percent change (%)")
    ax.set_title(title)
    ax.margins(y=0.3)

fig2.suptitle("Mean ± SD of patient-level percent changes at 12 and 24 months")
fig2.tight_layout()
fig2.savefig("mean_sd_plots.png", dpi=150)

plt.show()
