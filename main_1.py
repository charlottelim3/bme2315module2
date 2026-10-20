"""
FSHD: clinical severity vs dorsiflexion / STIR quant, as PERCENT DIFFERENCES.

Order of operations:
  1. Interpolate missing Year-1 values (time-based, using the actual visit lags)
     wherever baseline AND Year-2 exist. Nothing is extrapolated.
     (STIR quant has no Year-2 data, so it cannot be interpolated.)
  2. INTERPOLATION CHECK FIGURE (generated before any analysis) -- see below.
  3. ONLY THEN compute percent differences for three comparisons:
        Baseline-Y1,  Baseline-Y2,  Y1-Y2
     percent difference = (later - earlier) / |earlier| * 100
     (the earlier timepoint, named first, is the reference.)
     Right/left limbs are computed separately, then averaged when both exist.
  4. Figure: scatter plots (x = dorsiflexion or STIR % difference,
     y = clinical severity % difference) with a regression line, its equation
     and R^2.
  5. Figure: mean +/- standard deviation (error bars) of every percent-difference
     data set, with individual subjects overlaid.

Interpolation check figure (interpolation_check.png)
  Top row    : every subject's raw trajectory over real time (gray = subjects with
               all 3 visits observed). Subjects with an interpolated Y1 are colored,
               with the interpolated value shown as a star. A sensible interpolation
               sits on the line between that subject's baseline and Y2 and looks
               like the neighbouring gray trajectories.
  Bottom row : how well does a straight line predict Y1 in subjects who DO have all
               three visits? (observed Y1 minus straight-line Y1). If these
               deviations are small relative to the typical change, linear
               interpolation is a reasonable assumption; their spread is the
               realistic uncertainty on each interpolated value.

Outputs: interpolation_check.png, interpolation_check.csv, percent_differences.csv,
         summary_stats.csv, percent_difference_plots.png, mean_sd_plots.png

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

# True  = show the interpolation check figure and PAUSE until you close it,
#         so you can inspect it before any analysis runs.
# False = save it and carry straight on (all figures are shown at the end).
PAUSE_AFTER_CHECK = True

if not os.path.exists(CSV_PATH):
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    alt = os.path.join(here, os.path.basename(CSV_PATH))
    if os.path.exists(alt):
        CSV_PATH = alt

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} subjects from {CSV_PATH}\n")

# (display label, earlier timepoint, later timepoint)
COMPARISONS = [
    ("Baseline-Y1", "baseline", "y1"),
    ("Baseline-Y2", "baseline", "y2"),
    ("Y1-Y2", "y1", "y2"),
]

# Columns that can be interpolated: (name, baseline col, Y1 col, Y2 col)
TARGETS = [
    ("Dorsiflexion R (QMT)", "dorsiflexion_qmt_R_baseline", "dorsiflexion_qmt_R_y1", "dorsiflexion_qmt_R_y2"),
    ("Dorsiflexion L (QMT)", "dorsiflexion_qmt_L_baseline", "dorsiflexion_qmt_L_y1", "dorsiflexion_qmt_L_y2"),
    ("Clinical severity score", "clinical_severity_score_baseline", "clinical_severity_score_y1",
     "clinical_severity_score_y2"),
]


# ============================================================================
# STEP 1: INTERPOLATION (before any analysis)
# ============================================================================
def visit_times(data):
    """Actual visit times in months, and the fraction of the baseline->Y2 interval
    at which the Y1 visit happened."""
    med_t1 = data["lag_months_y1"].median()
    t0 = data["lag_months_baseline"].fillna(0)
    t1 = data["lag_months_y1"].fillna(med_t1)    # cohort median Y1 time if lag missing
    t2 = data["lag_months_y2"]
    frac = (t1 - t0) / (t2 - t0)
    return t0, t1, t2, frac


def interpolate_y1_values(data):
    """Fill missing Y1 values by linear interpolation between baseline and Y2,
    placed at the actual Y1 visit time. Returns (filled copy, log)."""
    out = data.copy()
    t0, t1, t2, frac = visit_times(data)
    log = []
    for _, base, y1, y2 in TARGETS:
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
# STEP 2: INTERPOLATION CHECK FIGURE (generated before any analysis)
# ============================================================================
def interpolation_check(orig, filled):
    t0, t1, t2, frac = visit_times(orig)
    valid_time = t2.notna() & (t2 > t0) & (t1 < t2)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    check_rows = []

    for c, (name, base, y1, y2) in enumerate(TARGETS):
        ax_top, ax_bot = axes[0, c], axes[1, c]

        complete = orig[[base, y1, y2]].notna().all(axis=1) & valid_time
        interp_idx = orig.index[orig[y1].isna() & filled[y1].notna()]

        # ---- top row: trajectories over real time ----
        for i in orig.index[complete]:
            ax_top.plot([t0[i], t1[i], t2[i]], orig.loc[i, [base, y1, y2]].values,
                        color="gray", lw=.8, alpha=.45, zorder=1)
        ax_top.plot([], [], color="gray", lw=1, alpha=.6,
                    label=f"subjects with all 3 visits (n={int(complete.sum())})")

        obs_y1 = orig[y1].dropna()
        lo_y1, hi_y1 = (obs_y1.min(), obs_y1.max()) if len(obs_y1) else (np.nan, np.nan)

        # expected error of a straight-line estimate, from subjects with all 3 visits
        expected = orig[base] + (orig[y2] - orig[base]) * frac
        dev = (orig[y1] - expected)[complete]
        dev_sd = dev.std(ddof=1) if len(dev) > 1 else np.nan

        for k, i in enumerate(interp_idx):
            col = cmap(k % 10)
            xs = [t0[i], t1[i], t2[i]]
            ys = [orig.loc[i, base], filled.loc[i, y1], orig.loc[i, y2]]
            ax_top.plot(xs, ys, color=col, lw=2.2, zorder=3)
            ax_top.scatter([xs[0], xs[2]], [ys[0], ys[2]], color=col, edgecolor="k",
                           s=45, zorder=4)
            inside = lo_y1 <= ys[1] <= hi_y1
            ax_top.scatter(xs[1], ys[1], marker="*", s=330, color=col,
                           edgecolor="k" if inside else "red",
                           linewidth=1.2 if inside else 2.2, zorder=5)
            ax_top.annotate(str(orig.loc[i, "subject_id"]), (xs[1], ys[1]),
                            textcoords="offset points", xytext=(6, 8), fontsize=8)
            check_rows.append(dict(
                measure=name, subject_id=orig.loc[i, "subject_id"],
                baseline=orig.loc[i, base], interpolated_y1=ys[1], y2=orig.loc[i, y2],
                y1_time_months=round(xs[1], 1),
                cohort_observed_y1_min=lo_y1, cohort_observed_y1_max=hi_y1,
                within_cohort_y1_range=bool(inside),
                expected_error_sd=dev_sd))
        if len(interp_idx):
            ax_top.scatter([], [], marker="*", s=200, color="white", edgecolor="k",
                           label="interpolated Y1 (red outline = outside cohort Y1 range)")
        else:
            ax_top.text(.5, .5, "no values interpolated", ha="center", va="center",
                        transform=ax_top.transAxes, fontsize=12, color="gray")
        ax_top.set_xlabel("Months since baseline")
        ax_top.set_ylabel(name)
        ax_top.set_title(f"{name}: trajectories  ({len(interp_idx)} interpolated)")
        ax_top.legend(fontsize=7, loc="best")

        # ---- bottom row: how good is a straight line? ----
        ax_bot.axhline(0, c="k", lw=.8)
        if len(dev) >= 2:
            jit = np.random.default_rng(5 + c).uniform(-.15, .15, len(dev))
            ax_bot.scatter(jit, dev, color="gray", edgecolor="k", lw=.4, alpha=.8,
                           label="observed Y1 − straight-line Y1")
            ax_bot.axhspan(-dev_sd, dev_sd, color="gray", alpha=.15, label=f"±1 SD = {dev_sd:.2f}")
        if len(interp_idx):
            ax_bot.scatter(np.full(len(interp_idx), 1.0), np.zeros(len(interp_idx)),
                           marker="*", s=300, color="gold", edgecolor="k",
                           label="interpolated (0 by construction)", zorder=5)
        ax_bot.set_xticks([0, 1])
        ax_bot.set_xticklabels(["subjects with all\n3 visits (observed)", "interpolated"])
        ax_bot.set_xlim(-.6, 1.6)
        ax_bot.set_ylabel("Deviation from straight line (Y1 units)")
        total_sd = orig[y1].std(ddof=1)
        rel = f"  ({100 * dev_sd / total_sd:.0f}% of cohort Y1 SD)" if total_sd and not np.isnan(dev_sd) else ""
        ax_bot.set_title(f"{name}: is linear reasonable?{rel}", fontsize=10)
        ax_bot.legend(fontsize=7, loc="best")

    fig.suptitle("Interpolation check — do the interpolated Y1 values make sense in context?",
                 fontsize=13)
    fig.tight_layout()
    return fig, pd.DataFrame(check_rows)


fig0, check_df = interpolation_check(df, df_filled)
fig0.savefig("interpolation_check.png", dpi=150)
if len(check_df):
    check_df.to_csv("interpolation_check.csv", index=False)
    print("Interpolation check (each interpolated value vs. the cohort):")
    print(check_df.round(2).to_string(index=False))
    n_out = int((~check_df["within_cohort_y1_range"]).sum())
    print(f"\n{n_out} of {len(check_df)} interpolated values fall outside the cohort's "
          f"observed Y1 range.")
    print("Saved: interpolation_check.png, interpolation_check.csv\n")
else:
    print("No values were interpolated, so the check figure is empty.\n")

if PAUSE_AFTER_CHECK:
    print("Showing interpolation check figure. Close the window to continue with the analysis...\n")
    plt.show()


# ============================================================================
# STEP 3: PERCENT DIFFERENCES (computed on the interpolated data)
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