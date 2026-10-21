"""
FSHD: Which measure is more associated with clinical severity progression --
dorsiflexion (QMT) progression or STIR quant progression?

What this script does (all in one run):
  1. Computes progression = Year 1 value - baseline value (R/L limbs averaged;
     STIR quant has no Year 2 data).
  2. Runs the full analysis on the ORIGINAL data (correlations, matched sample,
     bootstrap comparison, combined standardized regression).
  3. Linearly interpolates missing Year-1 values (time-based, using the actual
     visit lags) wherever baseline AND Year-2 exist. Nothing is extrapolated.
     STIR quant has no Year 2, so it cannot be interpolated.
  4. Re-runs the full analysis on the interpolated data.
  5. Saves two figures and shows them.

"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


CSV_PATH = "Module2_FSHD_Dataset.csv"
here = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(here, os.path.basename(CSV_PATH))

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} subjects from {CSV_PATH}\n")

COLORS = {"dors_prog": "#2a7ab9", "stir_prog": "#d9703a"}
NAMES = {"dors_prog": "Dorsiflexion", "stir_prog": "STIR quant"}


# ============================================================================
# 1. Interpolation of missing Year-1 values
# ============================================================================
def interpolate_y1_values(data):
    """Fill missing Y1 values by linear interpolation between baseline and Y2,
    placed at the actual Y1 visit time. Returns (filled copy, log of filled values)."""
    out = data.copy()
    med_t1 = data["lag_months_y1"].median()
    t0 = data["lag_months_baseline"].fillna(0)
    t1 = data["lag_months_y1"].fillna(med_t1)   # median Y1 time if lag missing
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


# ============================================================================
# 2. Progression (Y1 - baseline), limbs averaged
# ============================================================================
def add_progression(x):
    x = x.copy()
    x["stir_prog"] = pd.concat([x["stir_quant_y1_R"] - x["stir_quant_baseline_R"],
                                x["stir_quant_y1_L"] - x["stir_quant_baseline_L"]], axis=1).mean(axis=1)
    x["dors_prog"] = pd.concat([x["dorsiflexion_qmt_R_y1"] - x["dorsiflexion_qmt_R_baseline"],
                                x["dorsiflexion_qmt_L_y1"] - x["dorsiflexion_qmt_L_baseline"]], axis=1).mean(axis=1)
    x["sev_prog"] = x["clinical_severity_score_y1"] - x["clinical_severity_score_baseline"]
    return x


# ============================================================================
# 3. Full statistical analysis
# ============================================================================
def analyze(x, label, seed=1):
    print("=" * 72)
    print(label)
    print("=" * 72)

    # --- each measure vs severity progression (all available subjects) ---
    results = {}
    for k in ["dors_prog", "stir_prog"]:
        s = x[[k, "sev_prog"]].dropna()
        r, p = stats.pearsonr(s[k], s["sev_prog"])
        rho, p_s = stats.spearmanr(s[k], s["sev_prog"])
        results[k] = dict(n=len(s), r=r, p=p, rho=rho, p_s=p_s)
        print(f"{NAMES[k]:13s} n={len(s):2d}  Pearson r={r:+.3f} (p={p:.3f})  "
              f"Spearman rho={rho:+.3f} (p={p_s:.3f})")

    # --- matched sample: only subjects with all three measures ---
    m = x[["dors_prog", "stir_prog", "sev_prog"]].dropna()
    print(f"\nMatched sample (all three measures) n={len(m)}")
    if len(m) < 8:
        print("  Too few matched subjects for bootstrap/regression.\n")
        return results

    rho_d = stats.spearmanr(m["dors_prog"], m["sev_prog"])[0]
    rho_s = stats.spearmanr(m["stir_prog"], m["sev_prog"])[0]
    print(f"  Spearman rho  dorsiflexion={rho_d:+.3f}   STIR={rho_s:+.3f}")

    # --- bootstrap: |rho_dors| - |rho_stir| ---
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(5000):
        b = m.sample(len(m), replace=True, random_state=int(rng.integers(1e9)))
        if b["sev_prog"].nunique() < 2:
            continue
        d = (abs(stats.spearmanr(b["dors_prog"], b["sev_prog"])[0])
             - abs(stats.spearmanr(b["stir_prog"], b["sev_prog"])[0]))
        if not np.isnan(d):
            diffs.append(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "includes 0 -> no clear winner" if lo <= 0 <= hi else "excludes 0 -> a clear difference"
    print(f"  Bootstrap 95% CI for |rho_dors| - |rho_stir|: [{lo:+.3f}, {hi:+.3f}]  ({verdict})")

    # --- combined standardized regression (numpy only) ---
    z = (m - m.mean()) / m.std()
    X = np.column_stack([np.ones(len(z)), z["dors_prog"], z["stir_prog"]])
    y = z["sev_prog"].values
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    dof = len(y) - 3
    se = np.sqrt(np.diag(resid @ resid / dof * np.linalg.inv(X.T @ X)))
    pvals = 2 * stats.t.sf(np.abs(beta / se), dof)
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    print("\n  Combined model (standardized betas):")
    for name, b_, s_, p_ in zip(["intercept", "dorsiflexion", "STIR quant"], beta, se, pvals):
        print(f"    {name:13s} beta={b_:+.3f}  se={s_:.3f}  p={p_:.3f}")
    print(f"    R^2 = {r2:.3f}")
    print(f"  Correlation between the two predictors: "
          f"{m['dors_prog'].corr(m['stir_prog']):+.3f}\n")
    return results


# ============================================================================
# 4. Plotting
# ============================================================================
def scatter_panels(axes, x, results, title_suffix, newly_filled=None, seed=2):
    jit = np.random.default_rng(seed)
    for i, k in enumerate(["dors_prog", "stir_prog"]):
        s = x[[k, "sev_prog"]].dropna()
        axes[i].scatter(s[k], s["sev_prog"] + jit.uniform(-.08, .08, len(s)),
                        color=COLORS[k], edgecolor="k", lw=.4, alpha=.8, label="observed")
        if newly_filled is not None and k == "dors_prog":
            n_ = x.loc[newly_filled, [k, "sev_prog"]].dropna()
            axes[i].scatter(n_[k], n_["sev_prog"], marker="*", s=260, color="gold",
                            edgecolor="k", label="interpolated")
            axes[i].legend()
        slope, icpt = np.polyfit(s[k], s["sev_prog"], 1)
        xs = np.linspace(s[k].min(), s[k].max(), 50)
        axes[i].plot(xs, slope * xs + icpt, "k", lw=1.5)
        axes[i].axhline(0, c="gray", lw=.5)
        axes[i].axvline(0, c="gray", lw=.5)
        axes[i].set_xlabel(f"{NAMES[k]} change (baseline→Y1)")
        axes[i].set_ylabel("Clinical severity score change (Y1 − baseline)")
        res = results[k]
        axes[i].set_title(f"{NAMES[k]}{title_suffix}: ρ={res['rho']:.2f}, "
                          f"p={res['p_s']:.3f}, n={res['n']}", fontsize=10)


# ============================================================================
# RUN
# ============================================================================
# ---- A. Original data ----
orig = add_progression(df)
res_orig = analyze(orig, "A. ORIGINAL DATA (no interpolation)")

# ---- B. Interpolated data ----
df_filled, log = interpolate_y1_values(df)
print("Values filled by interpolation:")
for subj, col, val in log:
    print(f"  {subj}: {col} = {val}")
print(f"Total values filled: {len(log)}\n")

filled = add_progression(df_filled)
res_fill = analyze(filled, "B. AFTER INTERPOLATION")

# ---- Figure 1: original data ----
fig1, ax1 = plt.subplots(1, 3, figsize=(16, 4.8))
scatter_panels(ax1[:2], orig, res_orig, "")
rhos = [res_orig["dors_prog"]["rho"], res_orig["stir_prog"]["rho"]]
ps = [res_orig["dors_prog"]["p_s"], res_orig["stir_prog"]["p_s"]]
bars = ax1[2].bar(["Dorsiflexion", "STIR quant"], rhos,
                  color=[COLORS["dors_prog"], COLORS["stir_prog"]])
ax1[2].axhline(0, c="k", lw=.8)
ax1[2].set_ylim(-.6, .6)
ax1[2].set_ylabel("Spearman ρ with clinical severity change")
ax1[2].set_title("Strength of association (original data)")
for bar, v, p in zip(bars, rhos, ps):
    ax1[2].text(bar.get_x() + bar.get_width() / 2, v + (0.02 if v >= 0 else -0.08),
                f"ρ={v:.2f}\np={p:.3f}", ha="center")
fig1.tight_layout()
fig1.savefig("progression_vs_severity.png", dpi=150)

# ---- Figure 2: after interpolation + before/after comparison ----
newly = filled["dors_prog"].notna() & orig["dors_prog"].isna()
fig2, ax2 = plt.subplots(1, 3, figsize=(16, 4.8))
scatter_panels(ax2[:2], filled, res_fill, " (after interp.)", newly_filled=newly)
w = 0.35
x_ = np.arange(2)
before = [res_orig["dors_prog"]["rho"], res_orig["stir_prog"]["rho"]]
after = [res_fill["dors_prog"]["rho"], res_fill["stir_prog"]["rho"]]
ax2[2].bar(x_ - w / 2, before, w, label="Original", color="#999999")
ax2[2].bar(x_ + w / 2, after, w, label="After interpolation", color="#4c9a6a")
ax2[2].set_xticks(x_)
ax2[2].set_xticklabels(["Dorsiflexion", "STIR quant"])
ax2[2].axhline(0, c="k", lw=.8)
ax2[2].set_ylim(-.6, .6)
ax2[2].set_ylabel("Spearman ρ with severity change")
ax2[2].set_title("Before vs after interpolation")
ax2[2].legend()
fig2.tight_layout()
fig2.savefig("interpolation_results.png", dpi=150)

print("Saved figures: progression_vs_severity.png, interpolation_results.png")
plt.show()