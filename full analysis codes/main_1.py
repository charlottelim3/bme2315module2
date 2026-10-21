# fixes:
# keep  outlier filter, but don't remove patients that don't have a y2 data point.
# perform analysis and the graph for only baseline to y1 % change.
# confirming that interpolated data points are used for all of the analysis?
# do not print the interpolation methods. what does insufficient observations mean in the interpolation methods?
# also, for the %change per time point graphs put the n, mean, and sd values at the top of the graph so they are readable.
# zoom in the mean %change graphs so that the outliers are closer to the top and bottom of the plot,
# and add more tick marks on the y-axis.




"""
FSHD patient-level interpolation and percent-change analysis.

Question: which measure is more strongly tied to progression of the clinical
severity score - dorsiflexion strength progression or STIR quant progression?

T1 = minimum positive lag_months_y1 in the dataset.
T2 = minimum positive lag_months_y2 in the dataset.
Each patient's values are estimated at those common times using only that
patient's baseline, Y1, and Y2 observations. Exact target-time observations
are kept. No extrapolation is performed; values outside the observed time
range remain missing.

STIR analyses use only patients with a usable T2 STIR value (patients without
one are removed and counted), and then drop STIR outliers (> 3 SD from mean).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "Module2_FSHD_Dataset.csv"
PAUSE_AFTER_CHECK = True      # show the interpolation check plot before continuing
N_EXAMPLE_PATIENTS = 4        # number of patients drawn in the interpolation check
OUTLIER_SD = 3                # outlier cutoff in standard deviations

# ---------------------------------------------------------------- load data
# If the CSV isn't in the working directory, look next to this script.
if not os.path.exists(CSV_PATH):
    folder = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    candidate = os.path.join(folder, os.path.basename(CSV_PATH))
    if os.path.exists(candidate):
        CSV_PATH = candidate

df = pd.read_csv(CSV_PATH)

# Basic input checks: required columns present and one row per subject.
required = ["subject_id", "lag_months_y1", "lag_months_y2"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
if df["subject_id"].duplicated().any():
    raise ValueError("subject_id must be unique; duplicate subject IDs were found.")

# Make sure the visit lags are numeric (bad entries become NaN).
for c in ["lag_months_y1", "lag_months_y2"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Common target times = the shortest positive lag observed for Y1 and Y2.
y1_lags = df.loc[df["lag_months_y1"] > 0, "lag_months_y1"]
y2_lags = df.loc[df["lag_months_y2"] > 0, "lag_months_y2"]
if y1_lags.empty or y2_lags.empty:
    raise ValueError("Positive Year-1 and Year-2 visit lags are required.")

T1, T2 = float(y1_lags.min()), float(y2_lags.min())
if T1 <= 0 or T2 <= T1:
    raise ValueError(
        f"Invalid common targets: T1={T1:g}, T2={T2:g}. "
        "T1 and T2 must be positive, and T2 must be later than T1."
    )

print(f"Loaded {len(df)} subjects from {CSV_PATH}")
print(f"Common T1={T1:g} months; T2={T2:g} months")
print(f"Elapsed intervals: baseline-T1={T1:g}, baseline-T2={T2:g}, T1-T2={T2-T1:g} months")

# Display name, baseline column, observed Y1 column, observed Y2 column, output key
TARGETS = [
    ("Dorsiflexion R (QMT)", "dorsiflexion_qmt_R_baseline", "dorsiflexion_qmt_R_y1", "dorsiflexion_qmt_R_y2", "dors_r"),
    ("Dorsiflexion L (QMT)", "dorsiflexion_qmt_L_baseline", "dorsiflexion_qmt_L_y1", "dorsiflexion_qmt_L_y2", "dors_l"),
    ("Clinical severity score", "clinical_severity_score_baseline", "clinical_severity_score_y1", "clinical_severity_score_y2", "severity"),
    ("STIR quant R", "stir_quant_baseline_R", "stir_quant_y1_R", "stir_quant_y2_R", "stir_r"),
    ("STIR quant L", "stir_quant_baseline_L", "stir_quant_y1_L", "stir_quant_y2_L", "stir_l"),
]

# helpers ----------------------------------------------------------------
def number(row, col):
    """Return a row's value as a finite float, or NaN if missing/invalid."""
    if col not in row.index:
        return np.nan
    v = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) and np.isfinite(v) else np.nan

def observations(row, base_col, y1_col, y2_col):
    """List a patient's valid (time, value, label) points: baseline, Y1, Y2."""
    raw = [
        (0.0, number(row, base_col), "baseline"),
        (number(row, "lag_months_y1"), number(row, y1_col), "Y1"),
        (number(row, "lag_months_y2"), number(row, y2_col), "Y2"),
    ]
    raw = [(float(t), float(v), label) for t, v, label in raw
           if pd.notna(t) and pd.notna(v) and np.isfinite(t) and np.isfinite(v) and t >= 0]
    # Average any observations that share the same time.
    grouped = {}
    for t, v, label in raw:
        grouped.setdefault(t, []).append((v, label))
    return [(t, float(np.mean([v for v, _ in grouped[t]])),
             "+".join(label for _, label in grouped[t]))
            for t in sorted(grouped)]

def estimate(points, target):
    """Exact observation or linear interpolation between bracketing points; never extrapolate."""
    # Use the observed value if one exists at the target time.
    for t, v, _ in points:
        if np.isclose(t, target, atol=1e-8):
            return v, "observed_exact", t, t
    if len(points) < 2:
        return np.nan, "insufficient_observations", np.nan, np.nan
    times = np.array([p[0] for p in points], dtype=float)
    values = np.array([p[1] for p in points], dtype=float)
    # Target outside the observed time range -> leave missing (no extrapolation).
    if target < times[0] or target > times[-1]:
        return np.nan, "outside_observed_range_no_extrapolation", np.nan, np.nan
    # Find the observations just before (left) and after (right) the target.
    right = int(np.searchsorted(times, target, side="right"))
    left = right - 1
    t0, t1 = times[left], times[right]
    if t1 <= t0:
        return np.nan, "invalid_nonpositive_time_interval", t0, t1
    val = values[left] + (values[right] - values[left]) * ((target - t0) / (t1 - t0))
    return float(val), "linear_interpolation", float(t0), float(t1)

# interpolate ----------------------------------------------------------------
# Build one table holding each patient's baseline, T1 and T2 value per measure.
standard = pd.DataFrame({"subject_id": df["subject_id"]})
audit_rows = []   # log of how every estimate was produced
for display, base_col, y1_col, y2_col, key in TARGETS:
    for tp in ["baseline", "y1", "y2"]:
        standard[f"{key}_{tp}"] = np.nan

    for idx, row in df.iterrows():
        pts = observations(row, base_col, y1_col, y2_col)
        standard.loc[idx, f"{key}_baseline"] = number(row, base_col)
        for tp, target in [("y1", T1), ("y2", T2)]:
            val, method, source_start, source_end = estimate(pts, target)
            standard.loc[idx, f"{key}_{tp}"] = val
            audit_rows.append({
                "subject_id": row["subject_id"], "measure": display,
                "target_timepoint": tp.upper(), "target_months": target,
                "estimated_value": val, "method": method,
                "source_start_months": source_start, "source_end_months": source_end,
                "observed_y1_lag_months": number(row, "lag_months_y1"),
                "observed_y2_lag_months": number(row, "lag_months_y2"),
                "valid_observation_count": len(pts),
            })

audit = pd.DataFrame(audit_rows)
standard.to_csv("interpolated_common_timepoints.csv", index=False)
audit.to_csv("interpolation_check.csv", index=False)
print("\nInterpolation methods:")
print(audit.groupby(["measure", "target_timepoint", "method"]).size().to_string())
print("\nNo extrapolation was performed; unestimable values remain missing.")

# ---------------------------------------------------------------- interpolation check plot
# Plot only a few patients (right dorsiflexion) to visually confirm the
# interpolation: lines = observed data, star = estimate at T1, diamond = estimate at T2.
fig0, ax0 = plt.subplots(figsize=(8, 5.5))
shown = 0
for idx, row in df.iterrows():
    pts = observations(row, "dorsiflexion_qmt_R_baseline",
                       "dorsiflexion_qmt_R_y1", "dorsiflexion_qmt_R_y2")
    if len(pts) < 2:          # need at least 2 points for a visible trajectory
        continue
    line, = ax0.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                     linewidth=1.2, label=f"Subject {row['subject_id']}")
    v1, v2 = standard.loc[idx, "dors_r_y1"], standard.loc[idx, "dors_r_y2"]
    if pd.notna(v1):
        ax0.scatter(T1, v1, marker="*", s=180, color=line.get_color(), edgecolor="black", zorder=4)
    if pd.notna(v2):
        ax0.scatter(T2, v2, marker="D", s=70, color=line.get_color(), edgecolor="black", zorder=4)
    shown += 1
    if shown >= N_EXAMPLE_PATIENTS:
        break
ax0.axvline(T1, color="gray", linestyle="--", label=f"T1={T1:g} months")
ax0.axvline(T2, color="gray", linestyle=":", label=f"T2={T2:g} months")
ax0.set_xlabel("Months since baseline")
ax0.set_ylabel("Right dorsiflexion QMT")
ax0.set_title(f"Interpolation check ({shown} patients; * = T1 estimate, \u25c6 = T2 estimate)")
ax0.legend(fontsize=8)
fig0.tight_layout()
fig0.savefig("interpolation_check.png", dpi=150)
if PAUSE_AFTER_CHECK:
    plt.show()

# ---------------------------------------------------------------- percent changes
# The three chronological comparisons, each with its (positive) elapsed time.
COMPARISONS = [
    ("Baseline-T1", "baseline", "y1", T1),
    ("Baseline-T2", "baseline", "y2", T2),
    ("T1-T2", "y1", "y2", T2 - T1),
]
if any(months <= 0 for _, _, _, months in COMPARISONS):
    raise ValueError("All comparison time differences must be positive.")

def percent_change(earlier, later):
    """Percent change relative to |earlier|; a zero baseline gives NaN (avoids divide-by-zero)."""
    return (later - earlier) / earlier.abs().replace(0, np.nan) * 100

def limb_average(data, right, left, early, late):
    """Average the right and left limb percent changes (uses one limb if the other is missing)."""
    r = percent_change(data[f"{right}_{early}"], data[f"{right}_{late}"])
    l = percent_change(data[f"{left}_{early}"], data[f"{left}_{late}"])
    return pd.concat([r, l], axis=1).mean(axis=1, skipna=True)

# One row per patient: dorsiflexion, STIR and severity percent change for each comparison.
pct = pd.DataFrame({"subject_id": df["subject_id"]})
for _, early, late, _ in COMPARISONS:
    key = f"{early}_to_{late}"
    pct[f"dors_{key}"] = limb_average(standard, "dors_r", "dors_l", early, late)
    pct[f"stir_{key}"] = limb_average(standard, "stir_r", "stir_l", early, late)
    pct[f"severity_{key}"] = percent_change(
        standard[f"severity_{early}"], standard[f"severity_{late}"]
    )
pct = pct.replace([np.inf, -np.inf], np.nan)

STIR_COLS = [f"stir_{e}_to_{l}" for _, e, l, _ in COMPARISONS]

# ---------------------------------------------------------------- STIR filter 1: T2 data point
# Keep only patients with a usable T2 STIR value in at least one limb.
# Everyone else is removed from ALL STIR analyses (dorsiflexion/severity unaffected).
has_stir_t2 = standard[["stir_r_y2", "stir_l_y2"]].notna().any(axis=1)
n_no_t2 = int((~has_stir_t2).sum())
pct.loc[~has_stir_t2, STIR_COLS] = np.nan
print(f"\nSTIR T2 filter: {n_no_t2} of {len(pct)} patients removed (no T2 STIR data point); "
      f"{int(has_stir_t2.sum())} remain.")

# ---------------------------------------------------------------- STIR filter 2: outlier test
# Outlier analysis (3-SD rule):
#  1. For each STIR percent-change column (Baseline-T1, Baseline-T2, T1-T2),
#     compute the mean and sample SD (ddof=1) across the remaining patients.
#  2. Convert each value to a z-score: z = (value - mean) / SD, i.e. how many
#     SDs that patient sits from the group mean.
#  3. Flag a patient as an outlier if |z| > 3 in ANY of the STIR columns.
#  4. Remove flagged patients from all STIR analyses (their STIR values are set
#     to NaN) so the cohort is the same across the three STIR comparisons.
#  The test runs once (not iteratively), so the mean/SD are not recomputed
#  after removal. Dorsiflexion and severity data are not touched.
outlier_mask = pd.Series(False, index=pct.index)
for col in STIR_COLS:
    vals = pct[col]
    sd = vals.std(ddof=1)
    if vals.notna().sum() > 2 and sd > 0:
        z = (vals - vals.mean()) / sd
        outlier_mask |= z.abs() > OUTLIER_SD       # NaN z-scores compare False
n_outliers = int(outlier_mask.sum())
pct.loc[outlier_mask, STIR_COLS] = np.nan
print(f"STIR outlier test (|z| > {OUTLIER_SD}): {n_outliers} patient(s) removed"
      + (f" (subject_id: {', '.join(map(str, pct.loc[outlier_mask, 'subject_id']))})" if n_outliers else "")
      + ".")
print(f"Total STIR patients removed: {n_no_t2 + n_outliers}")

pct.to_csv("percent_differences.csv", index=False)
print("Saved percent_differences.csv")

# ---------------------------------------------------------------- scatter / regression
# Regress severity percent change on dorsiflexion (top row) and STIR (bottom row)
# percent change for each comparison. Higher R² = that measure tracks severity more closely.
MEASURES = [("dors", "Dorsiflexion"), ("stir", "STIR quant")]
reg_rows = []   # store results for the final comparison table
fig1, axes = plt.subplots(2, 3, figsize=(17, 10))
for r, (mkey, mname) in enumerate(MEASURES):
    for c, (label, early, late, _) in enumerate(COMPARISONS):
        ax = axes[r, c]
        key = f"{early}_to_{late}"
        pair = pct[[f"{mkey}_{key}", f"severity_{key}"]].dropna()
        pair.columns = ["x", "y"]
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.axvline(0, color="gray", linewidth=0.6)
        ax.set_xlabel(f"{mname} percent change ({label})")
        ax.set_ylabel(f"Clinical severity percent change ({label})")

        # Need enough varied points to fit a line.
        if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
            ax.set_title(f"{mname}: {label} (n={len(pair)})")
            ax.text(0.5, 0.5, "Insufficient paired data for regression",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        ax.scatter(pair["x"], pair["y"], edgecolor="black", linewidth=0.4, alpha=0.75)
        slope, intercept = np.polyfit(pair["x"], pair["y"], 1)       # least-squares line
        predicted = slope * pair["x"] + intercept
        ss_res = ((pair["y"] - predicted) ** 2).sum()
        ss_tot = ((pair["y"] - pair["y"].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
        corr = np.corrcoef(pair["x"], pair["y"])[0, 1]               # Pearson r
        xs = np.linspace(pair["x"].min(), pair["x"].max(), 100)
        ax.plot(xs, slope * xs + intercept, color="black",
                label=f"y={slope:.3f}x {intercept:+.2f}\nR\u00b2={r2:.3f}")
        ax.legend(fontsize=8)
        ax.set_title(f"{mname}: {label} (n={len(pair)})")
        reg_rows.append({"measure": mname, "comparison": label, "n": len(pair),
                         "slope": slope, "pearson_r": corr, "R2": r2})

fig1.suptitle("Clinical severity percent change vs. dorsiflexion / STIR quant")
fig1.tight_layout()
fig1.savefig("percent_difference_plots.png", dpi=150)

# Side-by-side regression results to answer the main question.
reg = pd.DataFrame(reg_rows)
reg.to_csv("regression_results.csv", index=False)
print("\nRegression of severity % change on each measure's % change:")
print(reg.round(4).to_string(index=False))

# ---------------------------------------------------------------- summary stats and mean +/- SD plots
DATASETS = [
    ("dors", "Dorsiflexion percent change"),
    ("stir", "STIR quant percent change"),
    ("severity", "Clinical severity percent change"),
]
summary_rows = []
for prefix, title in DATASETS:
    for label, early, late, elapsed in COMPARISONS:
        values = pct[f"{prefix}_{early}_to_{late}"].dropna()
        summary_rows.append({
            "dataset": title, "comparison": label, "elapsed_months": elapsed,
            "n": len(values),
            "mean_percent_change": values.mean() if len(values) else np.nan,
            "sd_percent_change": values.std(ddof=1) if len(values) > 1 else np.nan,
        })
summary = pd.DataFrame(summary_rows)
summary.to_csv("summary_stats.csv", index=False)
print("\nSummary statistics:")
print(summary.round(2).to_string(index=False))

# Bar = mean, error bar = SD, dots = individual patients (jittered horizontally).
fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5.5))
rng = np.random.default_rng(3)
for ax, (prefix, title) in zip(axes2, DATASETS):
    ax.axhline(0, color="gray", linewidth=0.6)
    for i, (label, early, late, _) in enumerate(COMPARISONS):
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
fig2.suptitle("Mean \u00b1 SD of patient-level percent changes")
fig2.tight_layout()
fig2.savefig("mean_sd_plots.png", dpi=150)
plt.show()