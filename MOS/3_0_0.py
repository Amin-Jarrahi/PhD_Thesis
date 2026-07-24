import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MOS")

INPUT_FILE = os.path.join(BASE_DIR,'gene_to_mz_results/gene_to_mz_top_k_matches_all_scores.csv')
VALIDATION_FILE = os.path.join(BASE_DIR,"mz_isotope_results/parent_children_hierarchy.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "gene_to_mz_results")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "gene_consensus_detailed_samples.csv")
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "gene_mz_consensus_heatmap.png")

K_FACTOR = 1
SYNERGY_WEIGHT = 0.2
PRECISION = 4
GROUPS = ['AAD', 'YC', 'AC', 'YAD']
TOP_N_RANKS = 10


# =============================================================================
# DATA LOADING & HIERARCHY MAPPING
# =============================================================================
def load_hierarchy_maps(path):
    try:
        val_df = pd.read_csv(path)
        pairs = set()
        child_to_parent = {}
        child_cols = [c for c in val_df.columns if 'Child' in c]

        for _, row in val_df.iterrows():
            if pd.isna(row['Parent_MZ']):
                continue
            p_mz = round(float(row['Parent_MZ']), PRECISION)
            row_members = [p_mz]

            for col in child_cols:
                if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                    c_mz = round(float(row[col]), PRECISION)
                    row_members.append(c_mz)
                    child_to_parent[c_mz] = p_mz

            if len(row_members) > 1:
                for combo in combinations(row_members, 2):
                    pairs.add(frozenset(combo))
        return pairs, child_to_parent
    except Exception as e:
        print(f"Error loading validation: {e}")
        return set(), {}


VALID_PAIRS, CHILD_TO_PARENT = load_hierarchy_maps(VALIDATION_FILE)


def identify_group(sample_name):
    for g in GROUPS:
        if g in sample_name:
            return g
    return "Other"


# =============================================================================
# CONSENSUS ALGORITHM (single source of truth — computes everything both
# the heatmap and the detailed table need, so scoring logic only lives once)
# =============================================================================
def calculate_consensus_all(df, gene_name):
    subset = df[df['Gene'] == gene_name]
    if subset.empty:
        return []

    all_samples = subset['Sample'].unique()
    total_samples_overall = len(all_samples)

    # ---------------------------------------------------------
    # Build per-sample rank storage
    # ---------------------------------------------------------
    mz_sample_ranks = {}
    for _, row in subset.iterrows():
        mz = round(row['MZ_Feature'], PRECISION)
        sample = row['Sample']
        rank = row['Rank']
        mz_sample_ranks.setdefault(mz, {})[sample] = rank

    candidates = list(mz_sample_ranks.keys())

    # ---------------------------------------------------------
    # Remove children from final candidates (they feed parent)
    # ---------------------------------------------------------
    to_remove = {
        mz for mz in candidates
        if mz in CHILD_TO_PARENT and CHILD_TO_PARENT[mz] in mz_sample_ranks
    }
    remaining_candidates = [c for c in candidates if c not in to_remove]

    final_stats = []

    for parent_mz in remaining_candidates:

        total_score = 0.0
        detected_samples = set()
        surrogate_sample_count = 0          # for heatmap stars
        parent_only_samples = []            # for detailed table
        children_participation = {}         # for detailed table

        children = [c for c, p in CHILD_TO_PARENT.items() if p == parent_mz]

        for sample in all_samples:

            parent_present = (
                parent_mz in mz_sample_ranks and
                sample in mz_sample_ranks[parent_mz]
            )
            child_present = [
                c for c in children
                if c in mz_sample_ranks and sample in mz_sample_ranks[c]
            ]

            if not parent_present and not child_present:
                continue

            detected_samples.add(sample)

            # ------------------------------
            # Case 1: Parent present -> NO STAR
            # ------------------------------
            if parent_present:
                parent_rank = mz_sample_ranks[parent_mz][sample]
                Rp = 1.0 / (parent_rank + K_FACTOR)
                sample_score = Rp
                parent_only_samples.append(sample)

                for c in child_present:
                    child_rank = mz_sample_ranks[c][sample]
                    Rc = 1.0 / (child_rank + K_FACTOR)
                    sample_score += Rc * SYNERGY_WEIGHT
                    children_participation.setdefault(c, []).append(sample)

                total_score += sample_score

            # ------------------------------
            # Case 2 & 3: Parent absent -> ONE STAR
            # ------------------------------
            else:
                surrogate_sample_count += 1

                child_scores_sorted = sorted(
                    (1.0 / (mz_sample_ranks[c][sample] + K_FACTOR) for c in child_present),
                    reverse=True
                )
                sample_score = child_scores_sorted[0] * SYNERGY_WEIGHT
                for extra in child_scores_sorted[1:]:
                    sample_score += extra * SYNERGY_WEIGHT

                total_score += sample_score

                for c in child_present:
                    children_participation.setdefault(c, []).append(sample)

        child_details = [
            f"{c_mz} ({', '.join(samples)})"
            for c_mz, samples in children_participation.items()
        ]

        final_stats.append({
            'Gene': gene_name,
            'mz': parent_mz,
            'final_score': round(total_score, 4),
            'found': len(detected_samples),
            'total': total_samples_overall,
            'confidence_pct': len(detected_samples) / total_samples_overall,
            'stars': surrogate_sample_count,
            'Parent_Samples': ", ".join(parent_only_samples) if parent_only_samples else "None",
            'Participating_Children': " | ".join(child_details) if child_details else "None",
        })

    final_stats.sort(key=lambda x: x['final_score'], reverse=True)
    return final_stats


# =============================================================================
# OUTPUT 1: HEATMAP
# =============================================================================
def build_heatmap(ranked_results):
    if not ranked_results:
        print("No heatmap data to display.")
        return

    h_df = pd.DataFrame(ranked_results)
    conf_matrix = h_df.pivot(index='Rank', columns='Gene', values='confidence_pct')

    def format_label(row):
        total_stars = int(row['stars'])
        if total_stars <= 10:
            star_block = '*' * total_stars
        else:
            star_block = f"{'*' * 10}\n{'*' * (total_stars - 10)}"
        return (
            f"{row['mz']:.4f}\n"
            f"S:{row['final_score']}\n"
            f"({int(row['found'])}/{int(row['total'])})\n"
            f"{star_block}"
        )

    h_df['Label'] = h_df.apply(format_label, axis=1)
    label_matrix = h_df.pivot(index='Rank', columns='Gene', values='Label')

    n_genes = h_df['Gene'].nunique()
    plt.figure(figsize=(max(14, n_genes * 1.5), TOP_N_RANKS * 1.2))
    sns.heatmap(conf_matrix, annot=label_matrix, fmt="", cmap="YlGnBu",
                cbar_kws={'label': 'Confidence (Overlap %)'},
                linewidths=1.0,
                annot_kws={"size": 12, "weight": "bold"})
    plt.title("Gene-to-m/z Detailed Consensus\n"
              "(Stars = Isotopes'/Adducts' Presence without Parent per Sample | "
              "S = Score | (X/Y) = Animals Detected)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=200, bbox_inches='tight')
    plt.show()


# =============================================================================
# OUTPUT 2: DETAILED TABLE / CSV
# =============================================================================
def build_detailed_table(ranked_results):
    if not ranked_results:
        return pd.DataFrame()

    detailed_df = pd.DataFrame(ranked_results).rename(columns={
        'mz': 'Parent_MZ',
        'final_score': 'Final_Score',
        'found': 'Sample_Count',
    })
    cols = ['Gene', 'Rank', 'Parent_MZ', 'Final_Score', 'Sample_Count',
            'Parent_Samples', 'Participating_Children']
    return detailed_df[cols]


# =============================================================================
# EXECUTION — compute once per gene, feed both outputs
# =============================================================================
if __name__ == "__main__":
    try:
        df = pd.read_csv(INPUT_FILE)
        df.columns = [c.title() for c in df.columns]
        if 'Rna_Sample' in df.columns:
            df.rename(columns={'Rna_Sample': 'Sample'}, inplace=True)
        if 'Mz_Feature' in df.columns:
            df.rename(columns={'Mz_Feature': 'MZ_Feature'}, inplace=True)
    except FileNotFoundError:
        print(f"File {INPUT_FILE} not found.")
        raise SystemExit(1)

    all_ranked = []
    for gene in df['Gene'].unique():
        ranked_list = calculate_consensus_all(df, gene)
        for i, cand in enumerate(ranked_list[:TOP_N_RANKS]):
            cand['Rank'] = i + 1
            all_ranked.append(cand)

    # --- Heatmap ---
    build_heatmap(all_ranked)

    # --- Detailed CSV ---
    summary_df = build_detailed_table(all_ranked)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[Success] Detailed list with sample counts saved to '{OUTPUT_CSV}'")