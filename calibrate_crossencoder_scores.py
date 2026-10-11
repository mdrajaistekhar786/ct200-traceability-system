import json
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.model_selection import LeaveOneGroupOut


# ============================================================
# CT-200 CrossEncoder Calibration + Cost-Sensitive Filtering
# ============================================================
#
# Pipeline:
#
# CrossEncoder Raw Score
#        ↓
# Logistic Regression / Platt Scaling
#        ↓
# P(Relevant)
#        ↓
# P(Irrelevant) = 1 - P(Relevant)
#        ↓
# Cost-Sensitive Decision
#        ↓
# KEEP / REMOVE
#
# Cost design:
#
# False Negative:
# Removing relevant evidence
# → Cost = 3
#
# False Positive:
# Keeping irrelevant evidence
# → Cost = 1
#
# Therefore we prioritize not losing relevant evidence.
# ============================================================


INPUT_FILE = "crossencoder_calibration_dataset.csv"

OUTPUT_FILE = "crossencoder_cost_sensitive_results.csv"

MODEL_FILE = "crossencoder_cost_sensitive_model.json"


# ------------------------------------------------------------
# COST CONFIGURATION
# ------------------------------------------------------------

FALSE_NEGATIVE_COST = 4.0
FALSE_POSITIVE_COST = 1.0


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "case_id",
    "node_id",
    "crossencoder_score",
    "label",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )


df["crossencoder_score"] = pd.to_numeric(
    df["crossencoder_score"],
    errors="coerce"
)

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "case_id",
        "crossencoder_score",
        "label",
    ]
).copy()

df["label"] = df["label"].astype(int)


X = df[["crossencoder_score"]].values

y = df["label"].values


print("\n" + "=" * 70)
print("CT-200 COST-SENSITIVE CROSSENCODER FILTER")
print("=" * 70)

print(f"Total samples      : {len(df)}")
print(f"Relevant           : {(y == 1).sum()}")
print(f"Irrelevant         : {(y == 0).sum()}")
print(f"Gold Standard cases: {df['case_id'].nunique()}")


# ------------------------------------------------------------
# 2. PLATT SCALING
# ------------------------------------------------------------
#
# Learn:
#
# P(Relevant | CrossEncoder score)
#
# ------------------------------------------------------------

calibrator = LogisticRegression(
    C=1e6,
    solver="lbfgs",
    max_iter=1000,
)

calibrator.fit(X, y)


a = float(calibrator.coef_[0][0])

b = float(calibrator.intercept_[0])


print("\nLearned calibration function:")

print(
    f"P(relevant) = sigmoid("
    f"{a:.6f} * score + {b:.6f})"
)


# ------------------------------------------------------------
# 3. CALIBRATED PROBABILITIES
# ------------------------------------------------------------

df["prob_relevant"] = calibrator.predict_proba(X)[:, 1]

df["prob_irrelevant"] = (
    1.0 - df["prob_relevant"]
)


# ------------------------------------------------------------
# 4. COST-SENSITIVE DECISION
# ------------------------------------------------------------
#
# If we KEEP a context:
#
#     Expected cost =
#     P(irrelevant) × False Positive Cost
#
#
# If we REMOVE a context:
#
#     Expected cost =
#     P(relevant) × False Negative Cost
#
#
# KEEP when:
#
# P(irrelevant) × FP_cost
# <
# P(relevant) × FN_cost
#
# ------------------------------------------------------------

df["cost_keep"] = (
    df["prob_irrelevant"]
    * FALSE_POSITIVE_COST
)

df["cost_remove"] = (
    df["prob_relevant"]
    * FALSE_NEGATIVE_COST
)


df["decision"] = np.where(
    df["cost_keep"] < df["cost_remove"],
    "KEEP",
    "REMOVE",
)


# ------------------------------------------------------------
# 5. COST-BASED PROBABILITY THRESHOLD
# ------------------------------------------------------------
#
# Remove when:
#
# P(relevant) × FN_cost
# <
# P(irrelevant) × FP_cost
#
# Therefore:
#
# P(irrelevant) >
# FN_cost / (FN_cost + FP_cost)
#
# With:
#
# FN = 3
# FP = 1
#
# threshold = 3 / 4 = 0.75
#
# So:
#
# P(irrelevant) >= 0.75 → REMOVE
#
# ------------------------------------------------------------

removal_threshold = (
    FALSE_NEGATIVE_COST
    / (
        FALSE_NEGATIVE_COST
        + FALSE_POSITIVE_COST
    )
)


print("\n" + "=" * 70)
print("COST-SENSITIVE DECISION")
print("=" * 70)

print(
    f"False Negative cost : {FALSE_NEGATIVE_COST}"
)

print(
    f"False Positive cost : {FALSE_POSITIVE_COST}"
)

print(
    f"P(irrelevant) removal threshold : "
    f"{removal_threshold:.2f}"
)


# ------------------------------------------------------------
# 6. FINAL DECISION
# ------------------------------------------------------------

df["final_decision"] = np.where(
    df["prob_irrelevant"] >= removal_threshold,
    "REMOVE",
    "KEEP",
)


# ------------------------------------------------------------
# 7. SHOW RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL CONTEXT FILTER RESULTS")
print("=" * 70)

print(
    df[
        [
            "case_id",
            "node_id",
            "crossencoder_score",
            "label",
            "prob_relevant",
            "prob_irrelevant",
            "final_decision",
        ]
    ]
    .sort_values(
        "prob_irrelevant",
        ascending=False,
    )
    .to_string(index=False)
)


# ------------------------------------------------------------
# 8. SUMMARY
# ------------------------------------------------------------

kept = (
    df["final_decision"] == "KEEP"
).sum()

removed = (
    df["final_decision"] == "REMOVE"
).sum()


print("\n" + "=" * 70)
print("FILTER SUMMARY")
print("=" * 70)

print(f"Total contexts : {len(df)}")
print(f"KEEP           : {kept}")
print(f"REMOVE         : {removed}")

print(
    f"Removal rate   : "
    f"{removed / len(df) * 100:.2f}%"
)


# ------------------------------------------------------------
# 9. ERROR ANALYSIS
# ------------------------------------------------------------

actual_relevant = df["label"] == 1

predicted_remove = (
    df["final_decision"] == "REMOVE"
)

predicted_keep = (
    df["final_decision"] == "KEEP"
)


false_negative = (
    actual_relevant
    & predicted_remove
).sum()

false_positive = (
    (~actual_relevant)
    & predicted_keep
).sum()

true_positive = (
    actual_relevant
    & predicted_keep
).sum()

true_negative = (
    (~actual_relevant)
    & predicted_remove
).sum()


print("\n" + "=" * 70)
print("COST-SENSITIVE ERROR ANALYSIS")
print("=" * 70)

print(
    f"Relevant kept      : {true_positive}"
)

print(
    f"Relevant removed   : {false_negative}"
)

print(
    f"Irrelevant kept    : {false_positive}"
)

print(
    f"Irrelevant removed : {true_negative}"
)


# ------------------------------------------------------------
# 10. CALIBRATION DIAGNOSTICS
# ------------------------------------------------------------

probabilities = df["prob_relevant"].values

print("\n" + "=" * 70)
print("CALIBRATION DIAGNOSTICS")
print("=" * 70)

print(
    f"ROC-AUC  : "
    f"{roc_auc_score(y, probabilities):.4f}"
)

print(
    f"Log Loss : "
    f"{log_loss(y, probabilities):.4f}"
)

print(
    f"Brier    : "
    f"{brier_score_loss(y, probabilities):.4f}"
)


# ------------------------------------------------------------
# 11. SAVE RESULTS
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------
# 12. SAVE MODEL INFORMATION
# ------------------------------------------------------------

model_info = {
    "method": "Logistic Regression / Platt Scaling",

    "input": "CrossEncoder raw score",

    "output": [
        "P(relevant)",
        "P(irrelevant)",
    ],

    "coefficient": a,

    "intercept": b,

    "false_negative_cost": FALSE_NEGATIVE_COST,

    "false_positive_cost": FALSE_POSITIVE_COST,

    "probability_remove_threshold": (
        removal_threshold
    ),

    "decision_rule":
        "REMOVE if P(irrelevant) >= threshold",

    "training_samples": len(df),

    "relevant_samples": int(
        (y == 1).sum()
    ),

    "irrelevant_samples": int(
        (y == 0).sum()
    ),

    "gold_standard_cases": int(
        df["case_id"].nunique()
    ),
}


with open(
    MODEL_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        model_info,
        f,
        indent=4,
    )


print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"\nSaved results:"
    f"\n{OUTPUT_FILE}"
)

print(
    f"\nSaved model:"
    f"\n{MODEL_FILE}"
)

print(
    "\nFinal rule:"
    f"\nP(irrelevant) >= "
    f"{removal_threshold:.2f}"
    "\n→ REMOVE context"
    "\n"
    f"P(irrelevant) < "
    f"{removal_threshold:.2f}"
    "\n→ KEEP context"
)