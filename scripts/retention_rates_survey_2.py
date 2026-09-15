from statsmodels.stats.proportion import proportions_ztest
import pandas as pd

# ── Input data ────────────────────────────────────────────────────
data = pd.DataFrame({
    "Group": ["Control", "Treatment"],
    "Start Count": [383, 390],
    "Passed Knowledge Check": [284, 309],
    "Passed Attention Check": [353, 361],
    "Passed History Check": [299, 313],
    "Final Count": [219, 257]
})

# ── Define tests ──────────────────────────────────────────────────
tests = {
    "Knowledge Check": "Passed Knowledge Check",
    "Attention Check": "Passed Attention Check",
    "History Check": "Passed History Check",
    "Retention": "Final Count"
}

control = data[data["Group"] == "Control"].iloc[0]
treatment = data[data["Group"] == "Treatment"].iloc[0]

results = []

# ── Run two-proportion z-tests ────────────────────────────────────
for name, column in tests.items():

    successes = [
        control[column],
        treatment[column]
    ]

    totals = [
        control["Start Count"],
        treatment["Start Count"]
    ]

    z, p = proportions_ztest(successes, totals)

    control_rate = successes[0] / totals[0]
    treatment_rate = successes[1] / totals[1]

    results.append({
        "Measure": name,
        "Control Rate": control_rate,
        "Treatment Rate": treatment_rate,
        "Difference (pp)": (treatment_rate - control_rate) * 100,
        "z": z,
        "p": p
    })

# ── Display results ───────────────────────────────────────────────
results_df = pd.DataFrame(results)

results_df["Control Rate"] = (
    results_df["Control Rate"] * 100
).round(2)

results_df["Treatment Rate"] = (
    results_df["Treatment Rate"] * 100
).round(2)

results_df["Difference (pp)"] = (
    results_df["Difference (pp)"]
).round(2)

results_df["z"] = results_df["z"].round(3)
results_df["p"] = results_df["p"].round(4)

print(results_df.to_string(index=False))