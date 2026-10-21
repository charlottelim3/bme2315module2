"""
FSHD patient-level interpolation and percent-change analysis (right and left legs averaged).

Question: which measure is more strongly tied to progression of the clinical
severity score - dorsiflexion strength progression or STIR quant progression?

T1 = minimum positive lag_months_y1 in the dataset.
T2 = minimum positive lag_months_y2 in the dataset.
Each patient's value at T1 / T2 is estimated from that patient's own baseline,
Y1 and Y2 observations: an exact observation is kept if one exists at the target
time, otherwise the value is linearly interpolated between the two bracketing
observations. No extrapolation is performed; unestimable values stay missing.

Analyses:
  - Dorsiflexion and clinical severity: Baseline-T1, Baseline-T2, T1-T2
  - STIR quant: Baseline-T1 only (no Y2 STIR data exist)
  - Right and left limb percent changes are averaged into one value per patient.
  - Outliers (> 2 SD from the mean) are removed from the interpolated values (every measure
    and time point) BEFORE percent changes are calculated.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

CSV_PATH = "Module2_FSHD_Dataset.csv"
PAUSE_AFTER_CHECK = True      # show the interpolation check plot before continuing
N_EXAMPLE_PATIENTS = 4        # number of patients drawn in the interpolation check
OUTLIER_SD = 2                # outlier cutoff in standard deviations

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
if T2 <= T1:
    raise ValueError(f"T2 ({T2:g}) must be later than T1 ({T1:g}).")

print(f"Loaded {len(df)} subjects from {CSV_PATH}")
print(f"Common T1={T1:g} months; T2={T2:g} months")
print(f"Elapsed intervals: Baseline-T1={T1:g}, Baseline-T2={T2:g}, T1-T2={T2-T1:g} months")

# Which target times to estimate: dorsiflexion/severity get T1 and T2, STIR only T1.
TP_ALL = [("y1", T1), ("y2", T2)]
TP_STIR = [("y1", T1)]

# Display name, baseline column, observed Y1 column, observed Y2 column, output key, target times
TARGETS = [
    ("Dorsiflexion R (QMT)", "dorsiflexion_qmt_R_baseline", "dorsiflexion_qmt_R_y1", "dorsiflexion_qmt_R_y2", "dors_r", TP_ALL),
    ("Dorsiflexion L (QMT)", "dorsiflexion_qmt_L_baseline", "dorsiflexion_qmt_L_y1", "dorsiflexion_qmt_L_y2", "dors_l", TP_ALL),
    ("Clinical severity score", "clinical_severity_score_baseline", "clinical_severity_score_y1", "clinical_severity_score_y2", "severity", TP_ALL),
    ("STIR quant R", "stir_quant_baseline_R", "stir_quant_y1_R", "stir_quant_y2_R", "stir_r", TP_STIR),
    ("STIR quant L", "stir_quant_baseline_L", "stir_quant_y1_L", "stir_quant_y2_L", "stir_l", TP_STIR),
]

# ---------------------------------------------------------------- helpers
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
    # Fewer than 2 valid points (and none at the target) -> nothing to interpolate between.
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

# ---------------------------------------------------------------- interpolate to T1 / T2
# Table of each patient's baseline value and T1/T2 values (observed or interpolated) per measure.
standard = pd.DataFrame({"subject_id": df["subject_id"]})
audit_rows = []   # log of how every estimate was produced (saved to CSV, not printed)
for display, base_col, y1_col, y2_col, key, tps in TARGETS:
    standard[f"{key}_baseline"] = np.nan
    for tp, _ in tps:
        standard[f"{key}_{tp}"] = np.nan
    for idx, row in df.iterrows():
        pts = observations(row, base_col, y1_col, y2_col)
        standard.loc[idx, f"{key}_baseline"] = number(row, base_col)
        for tp, target in tps:
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

pd.DataFrame(audit_rows).to_csv("interpolation_check.csv", index=False)
standard.to_csv("interpolated_common_timepoints.csv", index=False)

# ---------------------------------------------------------------- interpolation check plot
# A few patients only (right dorsiflexion): lines = observed data,
# star = value used at T1, diamond = value used at T2 (observed or interpolated).
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
ax0.set_title(f"Interpolation check ({shown} patients; star = T1, diamond = T2)")
ax0.legend(fontsize=8)
fig0.tight_layout()
fig0.savefig("interpolation_check.png", dpi=150)
if PAUSE_AFTER_CHECK:
    plt.show()

# ---------------------------------------------------------------- outlier test (interpolated data points)
# Outlier analysis (2-SD rule), run on the interpolated values BEFORE percent changes are calculated.
# It is applied separately to every column of the interpolated table: each measure
# (dorsiflexion R, dorsiflexion L, clinical severity, STIR R, STIR L) at each time point
# (baseline, T1, T2 where available):
#  1. For one column, compute the mean and sample SD (ddof=1) across patients who have a value.
#  2. Convert each patient's value to a z-score: z = (value - mean) / SD,
#     i.e. how many SDs the patient sits from the group mean.
#  3. A patient with |z| > 2 is an outlier for that column.
#  4. Outlier values are set to NaN in a cleaned copy of the table. Any percent change that
#     needs that value becomes missing for that patient (for that limb, when limbs are
#     averaged), but the patient is kept everywhere else.
#  All z-scores are computed on the original data first, then removals are applied together.
#  The test runs once; the mean/SD are not recomputed after removal.
NAMES = {"dors": "Dorsiflexion", "stir": "STIR quant", "severity": "Clinical severity"}
COL_NAMES = {"dors_r": "Dorsiflexion R", "dors_l": "Dorsiflexion L", "severity": "Clinical severity",
             "stir_r": "STIR quant R", "stir_l": "STIR quant L"}
TP_NAMES = {"baseline": "Baseline", "y1": "T1", "y2": "T2"}
test_cols = [c for c in standard.columns if c != "subject_id"]

outlier_masks = {}
for col in test_cols:
    vals = standard[col]
    sd = vals.std(ddof=1)
    if vals.notna().sum() > 2 and sd > 0:
        outlier_masks[col] = ((vals - vals.mean()) / sd).abs() > OUTLIER_SD   # NaN compares False
    else:
        outlier_masks[col] = pd.Series(False, index=standard.index)

print(f"\nOutlier test on interpolated values (|z| > {OUTLIER_SD}), patients removed per measure/time point:")
outlier_rows = []
for col in test_cols:
    mask = outlier_masks[col]
    mkey, tp = col.rsplit("_", 1) if col.rsplit("_", 1)[-1] in ("y1", "y2") else (col.rsplit("_", 1)[0], "baseline")
    n_before = int(standard[col].notna().sum())
    outlier_rows.append({"measure": COL_NAMES[mkey], "time_point": TP_NAMES[tp],
                         "n_before": n_before, "n_removed": int(mask.sum()),
                         "n_remaining": n_before - int(mask.sum()),
                         "removed_subject_ids": ", ".join(map(str, df.loc[mask, "subject_id"]))})
any_outlier = pd.concat(outlier_masks.values(), axis=1).any(axis=1)
clean = standard.copy()                      # cleaned copy used for all percent changes
for col in test_cols:
    clean.loc[outlier_masks[col], col] = np.nan
outlier_table = pd.DataFrame(outlier_rows)
outlier_table.to_csv("outlier_summary.csv", index=False)
clean.to_csv("interpolated_common_timepoints_outliers_removed.csv", index=False)
print(outlier_table.to_string(index=False))
print(f"Total value removals: {int(outlier_table['n_removed'].sum())}; "
      f"unique patients with at least one removal: {int(any_outlier.sum())}")

# ---------------------------------------------------------------- percent changes
# Calculated from the cleaned (outlier-removed) interpolated values.
# (label, earlier timepoint, later timepoint, elapsed months, short key)
COMPARISONS = [
    ("Baseline-T1", "baseline", "y1", T1, "b_t1"),
    ("Baseline-T2", "baseline", "y2", T2, "b_t2"),
    ("T1-T2", "y1", "y2", T2 - T1, "t1_t2"),
]
if any(c[3] <= 0 for c in COMPARISONS):
    raise ValueError("All comparison time differences must be positive.")

def percent_change(earlier, later):
    """Percent change relative to |earlier|; a zero baseline gives NaN (avoids divide-by-zero)."""
    return (later - earlier) / earlier.abs().replace(0, np.nan) * 100

def limb_average(data, right, left, early, late):
    """Average right and left limb percent changes into one value per patient
    (uses the single available limb if the other is missing)."""
    r = percent_change(data[f"{right}_{early}"], data[f"{right}_{late}"])
    l = percent_change(data[f"{left}_{early}"], data[f"{left}_{late}"])
    return pd.concat([r, l], axis=1).mean(axis=1, skipna=True)

# One row per patient. Columns are named <measure>_<comparison>, e.g. dors_b_t1.
pct = pd.DataFrame({"subject_id": df["subject_id"]})
for _, early, late, _, ckey in COMPARISONS:
    pct[f"dors_{ckey}"] = limb_average(clean, "dors_r", "dors_l", early, late)
    pct[f"severity_{ckey}"] = percent_change(clean[f"severity_{early}"], clean[f"severity_{late}"])
pct["stir_b_t1"] = limb_average(clean, "stir_r", "stir_l", "baseline", "y1")   # STIR: Baseline-T1 only
pct = pct.replace([np.inf, -np.inf], np.nan)
pct.to_csv("percent_differences.csv", index=False)

# ---------------------------------------------------------------- scatter / regression plots
# One panel per measure per comparison: severity % change (y) vs. the measure's % change (x).
# Top row: dorsiflexion (3 comparisons). Bottom-left: STIR (Baseline-T1 only).
# Higher R^2 = that measure tracks severity progression more closely.
fig1, axes = plt.subplots(2, 3, figsize=(17, 10))
panels = [(axes[0, i], "dors", "Dorsiflexion", c[0], c[4]) for i, c in enumerate(COMPARISONS)]
panels.append((axes[1, 0], "stir", "STIR quant", "Baseline-T1", "b_t1"))
axes[1, 1].set_axis_off()
axes[1, 2].set_axis_off()

reg_rows = []
for ax, mkey, mname, label, ckey in panels:
    pair = pct[[f"{mkey}_{ckey}", f"severity_{ckey}"]].dropna()
    pair.columns = ["x", "y"]
    row = {"measure": mname, "comparison": label, "n": len(pair),
           "slope": np.nan, "pearson_r": np.nan, "R2": np.nan}
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlabel(f"{mname} percent change ({label})")
    ax.set_ylabel(f"Clinical severity percent change ({label})")

    # Need enough varied points to fit a line.
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        ax.set_title(f"{mname}: {label} (n={len(pair)})")
        ax.text(0.5, 0.5, "Insufficient paired data for regression",
                ha="center", va="center", transform=ax.transAxes)
        reg_rows.append(row)
        continue

    ax.scatter(pair["x"], pair["y"], edgecolor="black", linewidth=0.4, alpha=0.75)
    slope, intercept = np.polyfit(pair["x"], pair["y"], 1)       # least-squares line
    predicted = slope * pair["x"] + intercept
    ss_res = ((pair["y"] - predicted) ** 2).sum()
    ss_tot = ((pair["y"] - pair["y"].mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    xs = np.linspace(pair["x"].min(), pair["x"].max(), 100)
    ax.plot(xs, slope * xs + intercept, color="black",
            label=f"y={slope:.3f}x {intercept:+.2f}\nR\u00b2={r2:.3f}")
    ax.legend(fontsize=8)
    ax.set_title(f"{mname}: {label} (n={len(pair)})")
    row.update(slope=slope, pearson_r=np.corrcoef(pair["x"], pair["y"])[0, 1], R2=r2)
    reg_rows.append(row)

fig1.suptitle("Clinical severity percent change vs. dorsiflexion / STIR quant")
fig1.tight_layout()
fig1.savefig("percent_difference_plots.png", dpi=150)

reg = pd.DataFrame(reg_rows)
reg.to_csv("regression_results.csv", index=False)
print("\nRegression of severity % change on each measure's % change:")
print(reg.round(4).to_string(index=False))

# ---------------------------------------------------------------- mean +/- SD plots
# One separate plot (panel) per measure. Each bar = mean, error bar = SD,
# dots = individual patients. STIR only has Baseline-T1.
fig2, axes2 = plt.subplots(1, 3, figsize=(17, 6.5))
rng = np.random.default_rng(3)
summary_rows = []
for ax, mkey in zip(axes2, ["dors", "stir", "severity"]):
    mname = NAMES[mkey]
    comps = [c for c in COMPARISONS if mkey != "stir" or c[4] == "b_t1"]
    stat_trans = ax.get_xaxis_transform()        # x in data units, y in axes fraction
    ax.axhline(0, color="gray", linewidth=0.6)
    lo_all, hi_all = np.inf, -np.inf
    for i, (label, early, late, elapsed, ckey) in enumerate(comps):
        values = pct[f"{mkey}_{ckey}"].dropna()
        mean = values.mean() if len(values) else np.nan
        sd = values.std(ddof=1) if len(values) > 1 else np.nan
        summary_rows.append({"measure": mname, "comparison": label, "elapsed_months": elapsed,
                             "n": len(values), "mean_percent_change": mean, "sd_percent_change": sd})
        if values.empty:
            continue
        sd0 = 0.0 if np.isnan(sd) else sd
        ax.bar(i, mean, width=0.6, alpha=0.5, edgecolor="black")
        ax.errorbar(i, mean, yerr=sd0, fmt="none", ecolor="black", capsize=6)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values,
                   s=18, color="black", alpha=0.5, zorder=3)
        # n / mean / SD printed above the plot area so they never overlap the data.
        ax.text(i, 1.01, f"n={len(values)}\nmean={mean:.1f}\nSD={sd0:.1f}",
                transform=stat_trans, ha="center", va="bottom", fontsize=9)
        lo_all = min(lo_all, values.min(), mean - sd0)
        hi_all = max(hi_all, values.max(), mean + sd0)

    # Zoom the y-axis to this measure's data (points and SD bars) so extremes sit near top/bottom.
    if np.isfinite(lo_all):
        pad = 0.04 * (hi_all - lo_all) if hi_all > lo_all else 1.0
        ax.set_ylim(lo_all - pad, hi_all + pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=15))   # more y-axis tick marks
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([f"{c[0]}\n({c[3]:g} mo)" for c in comps], fontsize=10)
    ax.set_xlim(-0.6, len(comps) - 0.4)
    ax.set_ylabel("Percent change (%)")
    ax.set_title(f"{mname} percent change", pad=55)
fig2.suptitle("Mean \u00b1 SD of patient-level percent changes (limbs averaged)")
fig2.tight_layout()
fig2.savefig("mean_sd_plots.png", dpi=150)

summary = pd.DataFrame(summary_rows)
summary.to_csv("summary_stats.csv", index=False)
print("\nSummary statistics:")
print(summary.round(2).to_string(index=False))
plt.show()