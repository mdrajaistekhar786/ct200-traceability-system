import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CrossEncoder Score Distribution Analysis
# Dataset: crossencoder_calibration_dataset.csv
# ============================================================

DATASET = "crossencoder_calibration_dataset.csv"

df = pd.read_csv(DATASET)

# Basic validation
required_columns = {
    "case_id",
    "node_id",
    "crossencoder_score",
    "label",
    "label_text",
}

missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["crossencoder_score"] = pd.to_numeric(
    df["crossencoder_score"], errors="coerce"
)

df = df.dropna(subset=["crossencoder_score", "label"])

# ------------------------------------------------------------
# 1. Basic dataset summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. DATASET SUMMARY")
print("=" * 70)

print(f"Total samples       : {len(df)}")
print(f"Unique Gold Cases   : {df['case_id'].nunique()}")
print(f"Relevant samples    : {(df['label'] == 1).sum()}")
print(f"Irrelevant samples  : {(df['label'] == 0).sum()}")

# ------------------------------------------------------------
# 2. Distribution statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. SCORE DISTRIBUTION")
print("=" * 70)

distribution = (
    df.groupby("label")["crossencoder_score"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        minimum="min",
        maximum="max",
    )
    .rename(index={0: "Irrelevant", 1: "Relevant"})
)

print(distribution.round(4))

# ------------------------------------------------------------
# 3. Percentiles
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. SCORE PERCENTILES")
print("=" * 70)

percentiles = [0, 10, 25, 50, 75, 90, 100]

for label, name in [(1, "Relevant"), (0, "Irrelevant")]:
    scores = df.loc[df["label"] == label, "crossencoder_score"]

    print(f"\n{name}:")
    for p in percentiles:
        print(f"  P{p:>3}: {np.percentile(scores, p):8.4f}")

# ------------------------------------------------------------
# 4. Range overlap
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. RANGE OVERLAP")
print("=" * 70)

rel = df.loc[df["label"] == 1, "crossencoder_score"]
irr = df.loc[df["label"] == 0, "crossencoder_score"]

rel_min, rel_max = rel.min(), rel.max()
irr_min, irr_max = irr.min(), irr.max()

overlap_min = max(rel_min, irr_min)
overlap_max = min(rel_max, irr_max)

if overlap_min <= overlap_max:
    print(
        f"Relevant range   : [{rel_min:.4f}, {rel_max:.4f}]"
    )
    print(
        f"Irrelevant range : [{irr_min:.4f}, {irr_max:.4f}]"
    )
    print(
        f"OVERLAP          : [{overlap_min:.4f}, {overlap_max:.4f}]"
    )
    print("Conclusion       : Raw score ranges overlap.")
else:
    print("Conclusion       : No raw-score range overlap.")

# ------------------------------------------------------------
# 5. Pairwise ordering violations
# ------------------------------------------------------------
# A violation occurs when an irrelevant candidate has a higher
# CrossEncoder score than a relevant candidate from the dataset.

print("\n" + "=" * 70)
print("5. RELEVANT vs IRRELEVANT SCORE ORDERING")
print("=" * 70)

violations = 0
total_pairs = 0

for r in rel:
    for i in irr:
        total_pairs += 1
        if i > r:
            violations += 1

violation_rate = violations / total_pairs if total_pairs else 0

print(f"Relevant-Irrelevant pairs : {total_pairs}")
print(f"Ordering violations       : {violations}")
print(f"Violation rate            : {violation_rate:.4f}")
print(f"Violation percentage      : {violation_rate * 100:.2f}%")

# ------------------------------------------------------------
# 6. Examples of difficult scores
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. DIFFICULT / OVERLAPPING EXAMPLES")
print("=" * 70)

# Highest-scoring irrelevant candidates
print("\nHighest-scoring IRRELEVANT candidates:")
print(
    df[df["label"] == 0]
    .sort_values("crossencoder_score", ascending=False)
    [["case_id", "node_id", "crossencoder_score"]]
    .head(10)
    .to_string(index=False)
)

# Lowest-scoring relevant candidates
print("\nLowest-scoring RELEVANT candidates:")
print(
    df[df["label"] == 1]
    .sort_values("crossencoder_score")
    [["case_id", "node_id", "crossencoder_score"]]
    .head(10)
    .to_string(index=False)
)

# ------------------------------------------------------------
# 7. Simple threshold analysis
# ------------------------------------------------------------
# This is NOT the final calibration.
# It only demonstrates why a single raw-score threshold may fail.

print("\n" + "=" * 70)
print("7. RAW SCORE THRESHOLD ANALYSIS")
print("=" * 70)

thresholds = [-5, -3, -2, -1, 0, 1, 2, 3, 4, 5]

print(
    f"{'Threshold':>10} {'Precision':>12} "
    f"{'Recall':>10} {'Predicted Positive':>20}"
)

for threshold in thresholds:
    predicted = df["crossencoder_score"] >= threshold

    tp = ((predicted) & (df["label"] == 1)).sum()
    fp = ((predicted) & (df["label"] == 0)).sum()
    fn = ((~predicted) & (df["label"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    print(
        f"{threshold:10.1f} "
        f"{precision:12.4f} "
        f"{recall:10.4f} "
        f"{predicted.sum():20d}"
    )

# ------------------------------------------------------------
# 8. Visualization: histogram
# ------------------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    rel,
    bins=12,
    alpha=0.65,
    label="Relevant",
)

plt.hist(
    irr,
    bins=12,
    alpha=0.65,
    label="Irrelevant",
)

plt.xlabel("CrossEncoder Raw Score")
plt.ylabel("Number of Candidates")
plt.title("CrossEncoder Score Distribution")
plt.legend()
plt.tight_layout()

plt.savefig(
    "crossencoder_score_distribution.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()

# ------------------------------------------------------------
# 9. Visualization: boxplot
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.boxplot(
    [irr, rel],
    labels=["Irrelevant", "Relevant"],
)

plt.ylabel("CrossEncoder Raw Score")
plt.title("CrossEncoder Score Distribution by Label")
plt.tight_layout()

plt.savefig(
    "crossencoder_score_boxplot.png",
    dpi=150,
    bbox_inches="tight",
)

plt.show()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(
    "Next step: use these labeled scores to fit a calibration model "
    "(Logistic Regression / Platt Scaling)."
)







