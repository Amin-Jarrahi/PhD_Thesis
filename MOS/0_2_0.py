import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MOS")
MSI_INPUT_FOLDER = os.path.join(BASE_DIR, "MOS/dummy_data_50/msi/")
RNA_INPUT_FOLDER = os.path.join(BASE_DIR, "MOS/dummy_data_50/rna/")
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'MOS/rotation_analysis')

FILE_PATH = os.path.join(BASE_DIR, "rotation_analysis/rotation_scores_optimized.csv")
ANGLE_MIN, ANGLE_MAX = 340, 20
PLOT_RESULTS = True

# Plot saving
SAVE_PLOT = True
PLOT_SAVE_PATH = os.path.join(BASE_DIR, "rotation_analysis/optimized_accuracy_by_angle.png")
PLOT_DPI = 300

# Weight export, so 1_0_0_fast_new.py can load the optimized weights directly
SAVE_WEIGHTS = True
WEIGHTS_SAVE_PATH = os.path.join(BASE_DIR, "rotation_analysis/optimized_weights.json")


METRIC_ROOTS = [
    "Value_Correlation", "Importance_IoU", "Importance_Correlation",
    "Value_Hist_Corr", "Spatial_Hist_Corr", "Radial_Corr", "Quadrant_Corr",
]


def get_ground_truth(gene_name):
    return f"MZ_{gene_name.replace('Gene_', '')}"


def load_filtered_data(file_path, angle_min, angle_max):
    """Load the CSV and keep only rows within [angle_min, angle_max].
    Handles windows that wrap around 0/360 (e.g. 340 -> 20)."""
    df_full = pd.read_csv(file_path)
    angle = df_full["Rotation_Angle"]

    if angle_min > angle_max:  # wraps around 360/0
        mask = (angle >= angle_min) | (angle <= angle_max)
    else:
        mask = (angle >= angle_min) & (angle <= angle_max)

    df = df_full[mask].copy()
    df["Ground_Truth"] = df["Gene"].apply(get_ground_truth)
    print(f"Filtered data: {len(df)} rows (angles {angle_min}° to {angle_max}°)")
    return df


def get_combined_prediction(row, weights):
    """Weighted vote across all metrics for one row -> predicted m/z."""
    votes = {}
    for metric, weight in zip(METRIC_ROOTS, weights):
        best_mz = row[f"Best_MZ_{metric}"]
        score = row[f"Max_{metric}"]
        votes[best_mz] = votes.get(best_mz, 0) + score * weight
    return max(votes, key=votes.get) if votes else None


def evaluate_accuracy(weights, df):
    if len(df) == 0:
        return 0.0
    preds = df.apply(lambda row: get_combined_prediction(row, weights), axis=1)
    return (preds == df["Ground_Truth"]).mean()


def optimize_weights(df):
    """Find metric weights that maximize accuracy on df."""
    def objective(w):
        w = np.maximum(w, 0)
        if w.sum() > 0:
            w = w / w.sum()
        return -evaluate_accuracy(w, df)

    n = len(METRIC_ROOTS)
    result = minimize(objective, np.ones(n) / n, method="Powell", bounds=[(0, 1)] * n)

    weights = np.maximum(result.x, 0)
    weights /= weights.sum()

    print("\nOptimal weights:")
    print("-" * 40)
    for metric, w in zip(METRIC_ROOTS, weights):
        print(f"{metric:<25}: {w:.4f}")
    print(f"\nAverage accuracy: {evaluate_accuracy(weights, df):.2%}")

    return weights


def save_weights(weights, save_path):
    """Save optimized weights as JSON, keyed the same way 1_0_0_fast_new.py's
    DEFAULT_WEIGHTS expects (lowercase metric names), so that script can load
    them directly instead of using its hardcoded defaults."""
    weights_dict = {metric.lower(): float(w) for metric, w in zip(METRIC_ROOTS, weights)}
    with open(save_path, "w") as f:
        json.dump(weights_dict, f, indent=2)
    print(f"\nSaved weights to: {save_path}")
    return weights_dict


def build_accuracy_table(df, weights):
    """Per-(gene, angle) accuracy for each metric plus the combined vote."""
    df = df.copy()
    df["Combined_Prediction"] = df.apply(lambda row: get_combined_prediction(row, weights), axis=1)

    rows = []
    for (gene, angle), group in df.groupby(["Gene", "Rotation_Angle"]):
        ground_truth = get_ground_truth(gene)
        total = len(group)

        row = {"Gene": gene, "Rotation_Angle": angle}
        for metric in METRIC_ROOTS:
            row[metric] = (group[f"Best_MZ_{metric}"] == ground_truth).sum() / total
        row["Optimized_Combined"] = (group["Combined_Prediction"] == ground_truth).sum() / total
        rows.append(row)

    return pd.DataFrame(rows)


def plot_accuracy_by_angle(df, weights, angle_min, angle_max, save_path=None, dpi=300):
    """Faceted line plot (one panel per gene): accuracy vs. rotation angle."""
    df_acc = build_accuracy_table(df, weights)

    # Shift angles below angle_min by +360 so a wrap-around band (e.g. 340->20)
    # plots as one continuous line instead of jumping.
    df_acc["Plot_Angle"] = df_acc["Rotation_Angle"].apply(lambda x: x if x >= angle_min else x + 360)

    plot_cols = METRIC_ROOTS + ["Optimized_Combined"]
    df_long = df_acc.melt(
        id_vars=["Gene", "Rotation_Angle", "Plot_Angle"],
        value_vars=plot_cols, var_name="Method", value_name="Accuracy",
    ).sort_values("Plot_Angle")

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    palette = dict(zip(METRIC_ROOTS, sns.color_palette("tab10", len(METRIC_ROOTS))))
    palette["Optimized_Combined"] = "black"

    g = sns.FacetGrid(df_long, col="Gene", col_wrap=2, height=4.5, aspect=1.5)
    g.map_dataframe(sns.lineplot, x="Plot_Angle", y="Accuracy", hue="Method",
                     linewidth=2, palette=palette, alpha=0.6)

    def overlay_combined(**kwargs):
        data = kwargs.pop("data")
        sns.lineplot(data=data[data["Method"] == "Optimized_Combined"],
                     x="Plot_Angle", y="Accuracy", color="black",
                     linewidth=4, label="Optimized Combined", zorder=10)

    g.map_dataframe(overlay_combined)
    g.map(plt.axhline, y=1.0, ls=":", c="green", alpha=0.5)

    # X-axis ticks/labels that wrap 360 back to 0
    tick_locs = np.arange(angle_min, angle_min + 61, 10)
    tick_labels = [f"{int(x - 360 if x >= 360 else x)}°" for x in tick_locs]
    span = angle_max - angle_min if angle_max >= angle_min else 360 - abs(angle_max - angle_min)

    for ax in g.axes.flat:
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(angle_min, angle_min + span)

    g.add_legend(title="Method", bbox_to_anchor=(1.02, 0.5), loc="center left")
    g.set_axis_labels("Rotation Angle", "Accuracy")
    g.fig.suptitle(f"Optimized Combined Accuracy ({angle_min}° - {angle_max}°)",
                    y=1.02, fontsize=16, fontweight="bold")

    plt.tight_layout()

    if save_path:
        g.fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    plt.show()


if __name__ == "__main__":

    df = load_filtered_data(FILE_PATH, ANGLE_MIN, ANGLE_MAX)
    weights = optimize_weights(df)

    if SAVE_WEIGHTS:
        save_weights(weights, WEIGHTS_SAVE_PATH)

    if PLOT_RESULTS:
        plot_accuracy_by_angle(
            df, weights, ANGLE_MIN, ANGLE_MAX,
            save_path=PLOT_SAVE_PATH if SAVE_PLOT else None,
            dpi=PLOT_DPI,
        )