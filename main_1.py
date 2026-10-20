"""
FSHD: clinical severity vs dorsiflexion / STIR quant, as PERCENT DIFFERENCES.

Order of operations:
  1. Interpolate missing Year-1 values (time-based, using the actual visit lags)
     wherever baseline AND Year-2 exist. Nothing is extrapolated.
     (STIR quant has no Year-2 data, so it cannot be interpolated.)
  2. ONLY THEN compute percent differences for three comparisons:
        Baseline-Y1,  Baseline-Y2,  Y1-Y2
     percent difference = (later - earlier) / |earlier| * 100
     (the earlier timepoint, named first, is the reference.)
     Right/left limbs are computed separately, then averaged when both exist.
  3. Figure 1: scatter plots (x = dorsiflexion or STIR % difference,
     y = clinical severity % difference) with a regression line, its equation
     and R^2.
  4. Figure 2: mean +/- standard deviation (error bars) of every percent-difference
     data set, with individual subjects overlaid.

Outputs: percent_differences.csv, summary_stats.csv,
         percent_difference_plots.png, mean_sd_plots.png

Requires: pandas, numpy, matplotlib
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# DATA PATH -- edit this line to point at your CSV, then run.
#   Windows example: r"C:\Users\you\Downloads\Module2_FSHD_Dataset.csv"
#   Mac/Linux example: "/Users/you/Downloads/Module2_FSHD_Dataset.csv"
#   Or put the CSV in the same folder as this script and leave as is.
# ============================================================================
CSV_PATH = "Module2_FSHD_Dataset.csv"

if not os.path.exists(CSV_PATH):
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    alt = os.path.join(here, os.path.basename(CSV_PATH))
    if os.path.exists(alt):
        CSV_PATH = alt
    else:
        sys.exit(f"Could not find the data file at '{CSV_PATH}'.\n"
                 f"Edit CSV_PATH at the top of the script to the full path of your CSV.")

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} subjects from {CSV_PATH}\n")

# (display label, earlier timepoint, later timepoint)
COMPARISONS = [
    ("Baseline-Y1", "baseline", "y1"),
    ("Baseline-Y2", "baseline", "y2"),
    ("Y1-Y2", "y1", "y2"),
]


# ============================================================================
# STEP 1: INTERPOLATION (before any analysis)
# ============================================================================
def interpolate_y1_values(data):
    """Fill missing Y1 values by linear interpolation between baseline and Y2,
    placed at the actual Y1 visit time. Returns (filled copy, log)."""
    out = data.copy()
    med_t1 = data["lag_months_y1"].median()
    t0 = data["lag_months_baseline"].fillna(0)
    t1 = data["lag_months_y1"].fillna(med_t1)    # cohort median Y1 time if lag missing
    t2 = data["lag_months_y2"]
    frac = (t1 - t0) / (t2 - t0)
    log = []

    targets = [
        ("dorsiflexion_qmt_R_baseline", "dorsiflexion_qmt_R_y1", "dorsiflexion_qmt_R_y2"),
        ("dorsiflexion_qmt_L_baseline", "dorsiflexion_qmt_L_y1", "dorsiflexion_qmt_L_y2"),
        ("clinical_severity_score_baseline", "clinical_severity_score_y1", "clinical_severity_score_y2"),
    ]
    for base, y1, y2 in targets:
        can = (data[y1].isna() & data[base].notna() & data[y2].notna()
               & t2.notna() & (t2 > t0) & (t1 < t2))
        out.loc[can, y1] = data.loc[can, base] + (data.loc[can, y2] - data.loc[can, base]) * frac[can]
        for i in data.index[can]:
            log.append((data.loc[i, "subject_id"], y1, round(out.loc[i, y1], 2)))
    return out, log


df_filled, log = interpolate_y1_values(df)
print("Values filled by interpolation:")
for subj, col, val in log:
    print(f"  {subj}: {col} = {val}")
print(f"Total values filled: {len(log)}\n")
interp_subjects = {s for s, _, _ in log}


# ============================================================================
# STEP 2: PERCENT DIFFERENCES (computed on the interpolated data)
# ============================================================================
def pct_diff(later, earlier):
    """(later - earlier) / |earlier| * 100 ; undefined (NaN) when earlier == 0."""
    earlier = earlier.where(earlier != 0)
    return (later - earlier) / earlier.abs() * 100


def col_name(measure, side, tp):
    if measure == "dors":
        return f"dorsiflexion_qmt_{side}_{tp}"
    if measure == "stir":
        return f"stir_quant_{tp}_{side}"
    raise ValueError(measure)


def limb_averaged_pct(data, measure, early, late):
    """Percent difference per limb, then averaged over whichever limbs are available."""
    per_limb = []
    for side in ["R", "L"]:
        c_e, c_l = col_name(measure, side, early), col_name(measure, side, late)
        if c_e in data.columns and c_l in data.columns:
            per_limb.append(pct_diff(data[c_l], data[c_e]))
    if not per_limb:                                  # e.g. STIR has no Y2 columns
        return pd.Series(np.nan, index=data.index)
    return pd.concat(per_limb, axis=1).mean(axis=1)


def severity_pct(data, early, late):
    return pct_diff(data[f"clinical_severity_score_{late}"],
                    data[f"clinical_severity_score_{early}"])


pct = pd.DataFrame({"subject_id": df_filled["subject_id"]})
for label, early, late in COMPARISONS:
    key = f"{early}_to_{late}"
    pct[f"dors_{key}"] = limb_averaged_pct(df_filled, "dors", early, late)
    pct[f"stir_{key}"] = limb_averaged_pct(df_filled, "stir", early, late)
    pct[f"sev_{key}"] = severity_pct(df_filled, early, late)

pct = pct.replace([np.inf, -np.inf], np.nan)
pct.to_csv("percent_differences.csv", index=False)
print("Saved per-subject percent differences to percent_differences.csv\n")


# ============================================================================
# FIGURE 1: scatter + regression line, equation and R^2
# ============================================================================
MEASURES = [("dors", "Dorsiflexion", "#2a7ab9"), ("stir", "STIR quant", "#d9703a")]


def line_label(slope, icpt, r2):
    sign = "+" if icpt >= 0 else "−"
    return f"y = {slope:.3f}x {sign} {abs(icpt):.2f}   (R² = {r2:.3f})"


fig1, axes = plt.subplots(2, 3, figsize=(17, 10))
for r, (mkey, mname, color) in enumerate(MEASURES):
    for c, (label, early, late) in enumerate(COMPARISONS):
        ax = axes[r, c]
        key = f"{early}_to_{late}"
        s = pct[["subject_id", f"{mkey}_{key}", f"sev_{key}"]].dropna()
        s.columns = ["subject_id", "x", "y"]

        ax.axhline(0, c="gray", lw=.5)
        ax.axvline(0, c="gray", lw=.5)
        ax.set_xlabel(f"{mname} % difference ({label})")
        ax.set_ylabel(f"Clinical severity % difference ({label})")

        if len(s) < 3 or s["x"].nunique() < 2:
            ax.set_title(f"{mname}: {label}  (n={len(s)})")
            msg = ("No Year-2 STIR quant data" if mkey == "stir" and "y2" in key
                   else "Not enough data points")
            ax.text(.5, .5, msg, ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color="gray")
            print(f"{mname:13s} {label:12s} n={len(s):2d}  -> not plotted ({msg})")
            continue

        ax.scatter(s["x"], s["y"], color=color, edgecolor="k", lw=.4, alpha=.8, label="observed")

        # highlight points that involve an interpolated Y1 value
        if "y1" in key:
            fl = s[s["subject_id"].isin(interp_subjects)]
            if len(fl):
                ax.scatter(fl["x"], fl["y"], marker="*", s=260, color="gold",
                           edgecolor="k", label="involves interpolated Y1")

        slope, icpt = np.polyfit(s["x"], s["y"], 1)
        r2 = np.corrcoef(s["x"], s["y"])[0, 1] ** 2 if s["y"].nunique() > 1 else np.nan
        xs = np.linspace(s["x"].min(), s["x"].max(), 50)
        ax.plot(xs, slope * xs + icpt, "k", lw=1.6, label=line_label(slope, icpt, r2))

        ax.set_title(f"{mname}: {label}  (n={len(s)})")
        ax.legend(fontsize=8)
        print(f"{mname:13s} {label:12s} n={len(s):2d}  "
              f"y = {slope:.3f}x {'+' if icpt >= 0 else '-'} {abs(icpt):.2f}   R² = {r2:.3f}")

fig1.suptitle("Clinical severity vs. dorsiflexion / STIR quant — percent differences "
              "(after interpolation)", fontsize=13)
fig1.tight_layout()
fig1.savefig("percent_difference_plots.png", dpi=150)
print("\nSaved figure: percent_difference_plots.png\n")


# ============================================================================
# FIGURE 2: mean +/- SD of each percent-difference data set
# ============================================================================
DATASETS = [("dors", "Dorsiflexion % difference", "#2a7ab9"),
            ("stir", "STIR quant % difference", "#d9703a"),
            ("sev", "Clinical severity % difference", "#4c9a6a")]

rows = []
for pre, title, _ in DATASETS:
    for label, early, late in COMPARISONS:
        v = pct[f"{pre}_{early}_to_{late}"].dropna()
        rows.append(dict(dataset=title, comparison=label, n=len(v),
                         mean=v.mean() if len(v) else np.nan,
                         sd=v.std(ddof=1) if len(v) > 1 else np.nan))
summary = pd.DataFrame(rows)
summary.to_csv("summary_stats.csv", index=False)
print("Mean and standard deviation of each data set:")
print(summary.round(2).to_string(index=False))
print("\nSaved summary to summary_stats.csv")

fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5.5))
jit = np.random.default_rng(3)
for ax, (pre, title, color) in zip(axes2, DATASETS):
    ax.axhline(0, c="gray", lw=.6)
    ymax_text = []
    for i, (label, early, late) in enumerate(COMPARISONS):
        v = pct[f"{pre}_{early}_to_{late}"].dropna()
        if len(v) == 0:
            ax.text(i, 0, "no data", ha="center", va="bottom", color="gray")
            continue
        m = v.mean()
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        ax.bar(i, m, color=color, alpha=.55, edgecolor="k", width=.6)
        ax.errorbar(i, m, yerr=sd, fmt="none", ecolor="k", elinewidth=1.6, capsize=8)
        ax.scatter(i + jit.uniform(-.18, .18, len(v)), v, s=18, color="k", alpha=.5, zorder=3)
        ax.text(i, m + sd, f"n={len(v)}\nmean={m:.1f}\nSD={sd:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(COMPARISONS)))
    ax.set_xticklabels([c[0] for c in COMPARISONS])
    ax.set_ylabel("Percent difference (%)")
    ax.set_title(f"{title}\n(bars = mean, error bars = ±1 SD, dots = subjects)", fontsize=10)
    ax.margins(y=.25)

fig2.suptitle("Mean ± standard deviation of percent differences (after interpolation)", fontsize=13)
fig2.tight_layout()
fig2.savefig("mean_sd_plots.png", dpi=150)
print("Saved figure: mean_sd_plots.png")

plt.show()