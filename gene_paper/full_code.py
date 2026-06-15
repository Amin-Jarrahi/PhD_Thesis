# =============================================================================
# Unified Spatial Transcriptomics DE Pipeline
# =============================================================================
# Workflow:
#   0. Imports & Setup
#   1. Read Visium data, annotate, save individual h5ad files
#   2. Load h5ad files, concatenate, save aggregated object
#   3. Quality Control
#   4. Normalization
#   5. Feature Selection (HVGs)
#   6. PCA with Elbow-Based PC Selection
#   7. PCA-Loading-Based Gene Filtering
#   8. Clustering
#   9. Pseudobulk Construction
#  10. Differential Expression
#  11. Per-Comparison Figures (dotplot + volcano + stats)
#  12. Combined Overview Figures
#  13. Spatial Variability (Moran's I)
#  14. Summary Report
# =============================================================================

# =============================================================================
# 0. Imports & Setup
# =============================================================================

import scanpy as sc
import squidpy as sq
import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy import sparse, stats
from statsmodels.stats.multitest import multipletests
from kneed import KneeLocator
import os
import warnings

# --- Paths -------------------------------------------------------------------
VISIUM_INPUT_FOLDER = "/opt/Data/RNA Seq 2025"
H5AD_OUTPUT_PATH = "/home/ajarrah/PhD_Thesis/gene_paper/h5ad_data/"
AGGREGATED_H5AD_OUTPUT_PATH = "/home/ajarrah/PhD_Thesis/gene_paper/aggregated_h5ad_data/"
FILE_NAME_LIST_CSV = "/home/ajarrah/PhD_Thesis/gene_paper/csv_data/file_name_list.csv"
FIGURE_DIR = "figures"
RESULTS_DIR = "results"

os.makedirs(H5AD_OUTPUT_PATH, exist_ok=True)
os.makedirs(AGGREGATED_H5AD_OUTPUT_PATH, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Sample definitions (single source of truth) -----------------------------
# Maps original folder name -> (short_key, group, condition, age, sample_code)
SAMPLE_DEFS = {
    "A1_Young_Control_Mouse_Brain_202502": ("YC_1",  "YC",  "Control", "Young", "1-1"),
    "B1_Young_Control_Mouse_Brain_202502": ("YC_2",  "YC",  "Control", "Young", "1-2"),
    "C1_Young_Control_Mouse_Brain_202502": ("YC_3",  "YC",  "Control", "Young", "1-3"),
    "D1_Young_Control_Mouse_Brain_202502": ("YC_4",  "YC",  "Control", "Young", "1-4"),
    "A1_Young_AD_Mouse_Brain_202502":      ("YAD_1", "YAD", "AD",      "Young", "2-1"),
    "B1_Young_AD_Mouse_Brain_202502":      ("YAD_2", "YAD", "AD",      "Young", "2-2"),
    "C1_Young_AD_Mouse_Brain_202502":      ("YAD_3", "YAD", "AD",      "Young", "2-3"),
    "D1_Young_AD_Mouse_Brain_202502":      ("YAD_4", "YAD", "AD",      "Young", "2-4"),
    "A1_Aged_Control_Mouse_Brain_202502":  ("AC_1",  "AC",  "Control", "Aged",  "3-1"),
    "B1_Aged_Control_Mouse_Brain_202502":  ("AC_2",  "AC",  "Control", "Aged",  "3-2"),
    "C1_Aged_Control_Mouse_Brain_202502":  ("AC_3",  "AC",  "Control", "Aged",  "3-3"),
    "D1_Aged_Control_Mouse_Brain_202502":  ("AC_4",  "AC",  "Control", "Aged",  "3-4"),
    "A1_Aged_AD_Mouse_Brain_202502":       ("AAD_1", "AAD", "AD",      "Aged",  "4-1"),
    "B1_Aged_AD_Mouse_Brain_202502":       ("AAD_2", "AAD", "AD",      "Aged",  "4-2"),
    "C1_Aged_AD_Mouse_Brain_202502":       ("AAD_3", "AAD", "AD",      "Aged",  "4-3"),
    "D1_Aged_AD_Mouse_Brain_202502":       ("AAD_4", "AAD", "AD",      "Aged",  "4-4"),
}

# Ordered keys for concatenation
SAMPLE_ORDER = [
    "YC_1", "YC_2", "YC_3", "YC_4",
    "YAD_1", "YAD_2", "YAD_3", "YAD_4",
    "AC_1", "AC_2", "AC_3", "AC_4",
    "AAD_1", "AAD_2", "AAD_3", "AAD_4",
]

# Reverse lookup: short_key -> original folder name
SHORT_TO_ORIG = {v[0]: k for k, v in SAMPLE_DEFS.items()}

# --- DE method toggle --------------------------------------------------------
DE_METHOD = "limma_voom"

_HAS_RPY2 = False
if DE_METHOD == "limma_voom":
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri, Formula
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter

        pandas2ri.activate()
        numpy2ri.activate()

        _limma = importr("limma")
        _edgeR = importr("edgeR")
        _stats_r = importr("stats")
        _base = importr("base")
        _HAS_RPY2 = True
        print("✓ rpy2 + limma/edgeR loaded — using limma-voom for DE")
    except (ImportError, Exception) as e:
        print(f"⚠ limma-voom unavailable ({e}). Falling back to Welch t-test.")
        DE_METHOD = "welch"

warnings.filterwarnings("ignore", category=FutureWarning)

sc.settings.verbosity = 2
sc.settings.set_figure_params(dpi=150, facecolor="white", frameon=False)
sc.settings.figdir = FIGURE_DIR

# =============================================================================
# 1. Read Visium Data, Annotate, Save Individual h5ad Files
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: Reading Visium data and saving individual h5ad files")
print("=" * 70)

file_name_list = sorted(os.listdir(VISIUM_INPUT_FOLDER))
pd.DataFrame(file_name_list, columns=["file_name"]).to_csv(FILE_NAME_LIST_CSV, index=False)
print(f"Found {len(file_name_list)} samples in {VISIUM_INPUT_FOLDER}")

for sample_folder in file_name_list:
    if sample_folder not in SAMPLE_DEFS:
        print(f"  ⚠ Skipping unrecognized folder: {sample_folder}")
        continue

    short_key, group, condition, age, code = SAMPLE_DEFS[sample_folder]
    print(f"  Reading {sample_folder} -> {short_key}")

    a = sc.read_visium(os.path.join(VISIUM_INPUT_FOLDER, sample_folder, "outs"))
    a.obs["Original_Name"] = sample_folder
    a.obs["Sample_Code"] = code
    a.obs["Group"] = group
    a.var_names_make_unique()
    a.write(os.path.join(H5AD_OUTPUT_PATH, f"{sample_folder}.h5ad"))

# =============================================================================
# 2. Load h5ad Files, Concatenate, Save Aggregated Object
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: Loading h5ad files, concatenating, and building unified AnnData")
print("=" * 70)

adatas_for_agg = []  # for aggregated spatial object (with uns["spatial"])
adatas_for_analysis = []  # for main analysis object (with full metadata)

for short_key in SAMPLE_ORDER:
    orig_name = SHORT_TO_ORIG[short_key]
    _, group, condition, age, code = SAMPLE_DEFS[orig_name]
    filepath = os.path.join(H5AD_OUTPUT_PATH, f"{orig_name}.h5ad")

    a = sc.read_h5ad(filepath)

    # --- For aggregated object (preserving spatial data in uns) ---
    adatas_for_agg.append(a.copy())

    # --- For analysis object (with full metadata, unique obs_names) ---
    a.obs["Sample"] = short_key
    a.obs["Group"] = group
    a.obs["Condition"] = condition
    a.obs["Age"] = age
    a.obs_names = [f"{short_key}_{x}" for x in a.obs_names]
    a.obs_names_make_unique()
    adatas_for_analysis.append(a)

# --- Save aggregated object with spatial info ---
adata_agg = sc.concat(
    adatas_for_agg, label="Sample", keys=SAMPLE_ORDER, uns_merge="unique"
)

# Rename spatial keys from original names to short keys
for short_key, a_orig in zip(SAMPLE_ORDER, adatas_for_agg):
    if "spatial" not in a_orig.uns:
        continue
    inner_key = next(iter(a_orig.uns["spatial"].keys()))
    if inner_key in adata_agg.uns.get("spatial", {}):
        spatial_dict = adata_agg.uns["spatial"]
        if short_key != inner_key:
            spatial_dict[short_key] = spatial_dict.pop(inner_key)

adata_agg.write(os.path.join(AGGREGATED_H5AD_OUTPUT_PATH, "aggregated_mouse_brain_202502.h5ad"))
print(f"Aggregated object saved: {adata_agg.shape}")
del adata_agg, adatas_for_agg  # free memory

# --- Build main analysis object ---
adata = ad.concat(adatas_for_analysis, join="inner", merge="first")
adata.obs_names_make_unique()
del adatas_for_analysis
print(f"Analysis object: {adata.shape}")

# =============================================================================
# 3. Quality Control
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: Quality Control")
print("=" * 70)

if sparse.issparse(adata.X):
    mask = np.array(adata.X.sum(axis=0) != 0).ravel()
else:
    mask = adata.X.sum(axis=0) != 0
adata = adata[:, mask].copy()

adata.var["mt"] = adata.var_names.str.startswith("mt-")
adata.var["ribo"] = adata.var_names.str.match("^(Rpl|Rps)")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], inplace=True)

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=10)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

print(f"After QC: {adata.shape}")

# =============================================================================
# 4. Normalization
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: Normalization")
print("=" * 70)

adata.layers["raw_counts"] = adata.X.copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata.layers["log_norm"] = adata.X.copy()
adata.raw = adata.copy()

# =============================================================================
# 5. Feature Selection (HVGs)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: Feature Selection")
print("=" * 70)

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=3000,
    flavor="seurat_v3",
    batch_key="Sample",
    layer="raw_counts",
)

print(f"HVGs selected: {adata.var['highly_variable'].sum()}")

# =============================================================================
# 6. PCA with Elbow-Based PC Selection
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: PCA with Elbow Detection")
print("=" * 70)

adata_hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, n_comps=50)

# Copy PCA results back to main object
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
adata.uns["pca"] = adata_hvg.uns["pca"]
adata.varm["PCs"] = np.zeros((adata.n_vars, 50))
hvg_idx = np.where(adata.var["highly_variable"].values)[0]
adata.varm["PCs"][hvg_idx, :] = adata_hvg.varm["PCs"]

# Elbow detection
variance_ratio = adata.uns["pca"]["variance_ratio"]
cumulative_var = np.cumsum(variance_ratio)

kneedle = KneeLocator(
    range(1, len(variance_ratio) + 1),
    variance_ratio,
    curve="convex",
    direction="decreasing",
    S=1.0,
)
optimal_pcs = kneedle.elbow if kneedle.elbow is not None else 20
print(f"Elbow detected at PC {optimal_pcs}")

# --- Publication figure: PCA variance explained ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(range(1, 51), variance_ratio, "o-", markersize=3, color="#2c3e50")
axes[0].axvline(optimal_pcs, color="#e74c3c", linestyle="--", linewidth=1.5,
                label=f"Elbow = PC {optimal_pcs}")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Ratio")
axes[0].set_title("A. Scree Plot", fontweight="bold", loc="left")
axes[0].legend()

axes[1].plot(range(1, 51), cumulative_var * 100, "o-", markersize=3, color="#2c3e50")
axes[1].axvline(optimal_pcs, color="#e74c3c", linestyle="--", linewidth=1.5)
axes[1].axhline(cumulative_var[optimal_pcs - 1] * 100, color="#95a5a6",
                linestyle=":", linewidth=1)
axes[1].set_xlabel("Principal Component")
axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].set_title("B. Cumulative Variance", fontweight="bold", loc="left")
axes[1].annotate(
    f"{cumulative_var[optimal_pcs - 1]*100:.1f}%",
    xy=(optimal_pcs, cumulative_var[optimal_pcs - 1] * 100),
    xytext=(optimal_pcs + 5, cumulative_var[optimal_pcs - 1] * 100 + 5),
    arrowprops=dict(arrowstyle="->", color="#e74c3c"),
    fontsize=10, color="#e74c3c",
)

plt.tight_layout()
plt.show()

# =============================================================================
# 7. PCA-Loading-Based Gene Filtering
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: PCA-Loading Gene Filtering")
print("=" * 70)

loadings = adata_hvg.varm["PCs"][:, :optimal_pcs]
loading_scores = np.sqrt(np.sum(loadings**2, axis=1))

threshold = np.mean(loading_scores) + np.std(loading_scores)
pca_important_mask = loading_scores > threshold

pca_filtered_genes = adata_hvg.var_names[pca_important_mask].tolist()
print(f"PCA-loading filtered genes: {len(pca_filtered_genes)} "
      f"(from {adata_hvg.n_vars} HVGs)")

adata.var["pca_filtered"] = False
adata.var.loc[adata.var_names.isin(pca_filtered_genes), "pca_filtered"] = True

# --- Publication figure: PCA loading distribution ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(loading_scores, bins=60, color="#3498db", edgecolor="white", alpha=0.8)
ax.axvline(threshold, color="#e74c3c", linestyle="--", linewidth=2,
           label=f"Threshold (mean + 1σ = {threshold:.3f})")
ax.set_xlabel("PCA Loading Score (L2 norm across top PCs)")
ax.set_ylabel("Number of Genes")
ax.set_title("PCA Loading Distribution — HVG Refinement", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.show()

del adata_hvg  # free memory

# =============================================================================
# 8. Clustering
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: Clustering")
print("=" * 70)

sc.pp.neighbors(adata, n_neighbors=15, n_pcs=optimal_pcs)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sc.pl.umap(adata, color="Group", ax=axes[0], show=False,
           palette=["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"],
           title="UMAP — Group")
sc.pl.umap(adata, color="leiden", ax=axes[1], show=False,
           title="UMAP — Leiden Clusters")
plt.tight_layout()
plt.show()

# =============================================================================
# 9. Pseudobulk Construction
# =============================================================================

print("\n" + "=" * 70)
print("STEP 9: Pseudobulk Construction")
print("=" * 70)


def build_pseudobulk(adata):
    """Aggregate raw counts per sample for pseudobulk DE analysis."""
    samples = adata.obs["Sample"].unique()
    pb_data, pb_obs = [], []

    for sample in samples:
        mask = adata.obs["Sample"].values == sample
        if sparse.issparse(adata.layers["raw_counts"]):
            s = np.array(adata.layers["raw_counts"][mask].sum(axis=0)).ravel()
        else:
            s = adata.layers["raw_counts"][mask].sum(axis=0)
        pb_data.append(s)

        row = adata.obs.loc[mask].iloc[0]
        pb_obs.append({
            "Group": row["Group"],
            "Condition": row["Condition"],
            "Age": row["Age"],
            "n_spots": int(mask.sum()),
        })

    pb = ad.AnnData(
        X=np.array(pb_data, dtype=np.float32),
        obs=pd.DataFrame(pb_obs, index=samples),
        var=adata.var[[]].copy(),
    )
    return pb


pb = build_pseudobulk(adata)
pb_raw = pb.copy()  # raw copy for limma-voom

# CPM normalize pseudobulk (for Welch t-test and visualization)
sc.pp.normalize_total(pb, target_sum=1e6)
sc.pp.log1p(pb)

print(f"Pseudobulk: {pb.shape}")
print(pb.obs[["Group", "Condition", "Age", "n_spots"]])

# =============================================================================
# 10. Differential Expression
# =============================================================================

print("\n" + "=" * 70)
print("STEP 10: Differential Expression")
print("=" * 70)


# ---- Welch t-test DE function ------------------------------------------------

def run_de_welch(pb, group1_mask, group2_mask, gene_list=None):
    """Welch t-test between two pseudobulk groups (CPM-log-normalized input)."""
    X1 = pb[group1_mask].X
    X2 = pb[group2_mask].X

    genes = pb.var_names if gene_list is None else gene_list
    gene_idx = [list(pb.var_names).index(g) for g in genes]

    results = []
    for i, gi in enumerate(gene_idx):
        vals1 = X1[:, gi]
        vals2 = X2[:, gi]

        if np.std(vals1) == 0 and np.std(vals2) == 0:
            continue

        tstat, pval = stats.ttest_ind(vals1, vals2, equal_var=False)
        if np.isnan(pval):
            pval = 1.0

        # Convert log(CPM+1) back to CPM for proper fold change
        cpm1 = np.expm1(vals1)
        cpm2 = np.expm1(vals2)

        mean1 = np.mean(cpm1)
        mean2 = np.mean(cpm2)

        log2fc = np.log2((mean1 + 1e-9) / (mean2 + 1e-9))
        mean_expr = np.mean(np.concatenate([cpm1, cpm2]))

        results.append({
            "gene": genes[i],
            "log2FC": log2fc,
            "mean_expr": mean_expr,
            "pval": pval,
        })

    df = pd.DataFrame(results)
    if df.empty:
        return df
    df["padj"] = multipletests(df["pval"], method="fdr_bh")[1]
    df["method"] = "welch"
    return df


# ---- limma-voom DE function --------------------------------------------------

def run_de_limma_voom(pb_raw, group1_mask, group2_mask, gene_list=None):
    """limma-voom on RAW pseudobulk counts (not normalized)."""
    if not _HAS_RPY2:
        raise RuntimeError("rpy2 + limma/edgeR not available")

    labels = pd.Series("group2", index=pb_raw.obs_names)
    labels[group1_mask] = "group1"

    if gene_list is not None:
        gene_idx = [list(pb_raw.var_names).index(g) for g in gene_list
                     if g in pb_raw.var_names]
        genes_use = [pb_raw.var_names[i] for i in gene_idx]
        counts = pb_raw.X[:, gene_idx].T
    else:
        genes_use = list(pb_raw.var_names)
        counts = pb_raw.X.T

    if sparse.issparse(counts):
        counts = np.array(counts.todense())
    counts = np.asarray(counts, dtype=np.float64)

    n_genes, n_samples = counts.shape

    r_counts = ro.r.matrix(
        ro.FloatVector(counts.flatten()),
        nrow=n_genes, ncol=n_samples, byrow=True,
    )
    r_counts.rownames = ro.StrVector(genes_use)
    r_counts.colnames = ro.StrVector(list(pb_raw.obs_names))

    dge = _edgeR.DGEList(counts=r_counts)

    r_group = ro.FactorVector(list(labels))
    ro.globalenv["group"] = r_group
    design = _stats_r.model_matrix(Formula("~ 0 + group"))
    design.colnames = ro.StrVector(["group1", "group2"])

    keep = _edgeR.filterByExpr(dge, design=design)
    keep_bool = np.array(keep).astype(bool)
    dge = dge.rx(ro.BoolVector(keep_bool), True)
    filtered_genes = [genes_use[i] for i in range(n_genes) if keep_bool[i]]

    if len(filtered_genes) == 0:
        print("  ⚠ No genes passed filterByExpr")
        return pd.DataFrame(columns=["gene", "log2FC", "mean_expr", "pval", "padj", "method"])

    dge = _edgeR.calcNormFactors(dge, method="TMM")
    v = _limma.voom(dge, design=design, plot=False)
    fit = _limma.lmFit(v, design)

    contrast_matrix = _limma.makeContrasts(
        contrasts="group1 - group2", levels=design,
    )
    fit2 = _limma.contrasts_fit(fit, contrast_matrix)
    fit2 = _limma.eBayes(fit2)

    with localconverter(ro.default_converter + pandas2ri.converter):
        top = ro.conversion.rpy2py(
            _limma.topTable(fit2, coef=1, number=len(filtered_genes), sort_by="none")
        )

    df = pd.DataFrame({
        "gene": filtered_genes,
        "log2FC": top["logFC"].values,
        "mean_expr": top["AveExpr"].values,
        "pval": top["P.Value"].values,
        "padj": top["adj.P.Val"].values,
        "t_statistic": top["t"].values,
        "B_statistic": top["B"].values,
    })
    df["method"] = "limma_voom"
    return df


# ---- Unified dispatcher ------------------------------------------------------

def run_de(pb, pb_raw, group1_mask, group2_mask, gene_list=None, method=DE_METHOD):
    """Dispatch DE to either limma-voom or Welch t-test."""
    if method == "limma_voom" and _HAS_RPY2:
        return run_de_limma_voom(pb_raw, group1_mask, group2_mask, gene_list)
    else:
        return run_de_welch(pb, group1_mask, group2_mask, gene_list)


# --- Define comparisons -------------------------------------------------------
comparisons = {
    "YAD": {
        "desc": "YAD vs YC (AD effect, young)",
        "group1": pb.obs["Group"] == "YAD",
        "group2": pb.obs["Group"] == "YC",
    },
    "AC": {
        "desc": "AC vs YC (aging effect)",
        "group1": pb.obs["Group"] == "AC",
        "group2": pb.obs["Group"] == "YC",
    },
    "AAD": {
        "desc": "AAD vs AC (AD effect, aged)",
        "group1": pb.obs["Group"] == "AAD",
        "group2": pb.obs["Group"] == "AC",
    },
}

pca_genes = adata.var_names[adata.var["pca_filtered"]].tolist()

top20_all = {}
bot20_all = {}
all_de_results = {}

print(f"\nDE method: {DE_METHOD}")

for group, comp in comparisons.items():
    print(f"\n{'='*60}")
    print(f"DE: {comp['desc']}")
    print(f"{'='*60}")

    df = run_de(pb, pb_raw, comp["group1"], comp["group2"], gene_list=pca_genes)
    all_de_results[group] = df

    # Significant upregulated genes
    sig_up = df[(df["padj"] < 0.05) & (df["log2FC"] > 0.5)].copy()
    sig_up = sig_up.sort_values("log2FC", ascending=False)

    print(f"Significant upregulated (padj<0.05, log2FC>0.5): {sig_up.shape[0]}")

    top20 = sig_up.head(20)
    top20_all[group] = top20

    if top20.shape[0] > 0:
        print(top20[["gene", "log2FC", "padj"]].to_string(index=False))
    else:
        print("  No significant upregulated genes at this threshold.")
        top20_relaxed = df[df["log2FC"] > 0].sort_values("log2FC", ascending=False).head(20)
        top20_all[group] = top20_relaxed
        print("  Top 20 by log2FC (relaxed, no FDR filter):")
        print(top20_relaxed[["gene", "log2FC", "pval", "padj"]].to_string(index=False))

    # Significant downregulated genes
    sig_dn = df[(df["padj"] < 0.05) & (df["log2FC"] < -0.5)].copy()
    sig_dn = sig_dn.sort_values("log2FC", ascending=True)

    print(f"Significant downregulated (padj<0.05, log2FC<-0.5): {sig_dn.shape[0]}")

    bot20 = sig_dn.head(20)
    bot20_all[group] = bot20

    if bot20.shape[0] > 0:
        print(bot20[["gene", "log2FC", "padj"]].to_string(index=False))
    else:
        print("  No significant downregulated genes at this threshold.")
        bot20_relaxed = df[df["log2FC"] < 0].sort_values("log2FC", ascending=True).head(20)
        bot20_all[group] = bot20_relaxed
        print("  Bot 20 by log2FC (relaxed, no FDR filter):")
        print(bot20_relaxed[["gene", "log2FC", "pval", "padj"]].to_string(index=False))

    # Save full DE results
    df.to_csv(os.path.join(RESULTS_DIR, f"DE_{group}_{comp['desc'].replace(' ', '_')}.csv"),
              index=False)

# Save top 20 summaries
summary_frames = []
for group, df in top20_all.items():
    tmp = df.copy()
    tmp["Group"] = group
    tmp["Direction"] = "Up"
    tmp["Comparison"] = comparisons[group]["desc"]
    summary_frames.append(tmp)
for group, df in bot20_all.items():
    tmp = df.copy()
    tmp["Group"] = group
    tmp["Direction"] = "Down"
    tmp["Comparison"] = comparisons[group]["desc"]
    summary_frames.append(tmp)

summary_df = pd.concat(summary_frames, ignore_index=True)
summary_df.to_csv(os.path.join(RESULTS_DIR, "Top20_Per_Group_Summary.csv"), index=False)

# =============================================================================
# 11. Nature-Style Per-Comparison Figures
# =============================================================================

print("\n" + "=" * 70)
print("STEP 11: Per-Comparison Figures")
print("=" * 70)

# --- Nature styling defaults ---
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
})

GROUP_ORDER = ["YC", "YAD", "AC", "AAD"]
adata.obs["Group"] = pd.Categorical(
    adata.obs["Group"], categories=GROUP_ORDER, ordered=True
)

comp_labels = {
    "YAD": "AD effect (Young)",
    "AC":  "Aging effect",
    "AAD": "AD effect (Aged)",
}
comp_colors = {
    "YAD": "#c0392b",
    "AC":  "#2980b9",
    "AAD": "#8e44ad",
}


def format_pval(p):
    """Compact p-value formatting for tables."""
    if pd.isna(p):
        return "—"
    if p < 1e-10:
        return f"{p:.0e}"
    if p < 0.001:
        return f"{p:.1e}"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def sig_stars(padj):
    if pd.isna(padj):
        return ""
    if padj < 0.001:
        return "***"
    if padj < 0.01:
        return "**"
    if padj < 0.05:
        return "*"
    return "ns"


def render_dotplot_and_stats(adata, gene_list, de_df, comp_desc, direction,
                             comp_key, group_order, comp_labels):
    """
    Render:
      1. Standalone scanpy dotplot
      2. Composite figure with volcano + stats heatmap
    """
    n_genes = len(gene_list)
    if n_genes == 0:
        print(f"  Skipping {direction} dotplot for {comp_desc} — no genes.")
        return

    cmap_dot = "Reds" if direction == "Upregulated" else "Blues"

    # ---- Dotplot (standalone) ----
    dp = sc.pl.dotplot(
        adata, var_names=gene_list, groupby="Group",
        standard_scale="var", use_raw=True, dendrogram=False,
        show=False, return_fig=True,
    )
    dp.style(cmap=cmap_dot, dot_edge_color="0.3", dot_edge_lw=0.3, size_exponent=1.5)
    dp.make_figure()

    fig_dp = plt.gcf()
    fig_dp.set_size_inches(max(7, n_genes * 0.4 + 2), 3.5)
    fig_dp.suptitle(
        f"a   Top {n_genes} {direction.lower()} — {comp_labels[comp_key]}",
        fontsize=9, fontweight="bold", x=0.02, ha="left", y=0.99,
    )
    plt.show()

    # ---- Composite: volcano + stats heatmap ----
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 7.0),
        gridspec_kw={
            "height_ratios": [3, 2],
            "width_ratios": [4, 1.2],
            "hspace": 0.55, "wspace": 0.35,
        },
    )
    ax_vol = axes[0, 0]
    ax_n = axes[0, 1]
    ax_heat = axes[1, 0]
    ax_cbar = axes[1, 1]

    # ---- Volcano ----
    de_df = de_df.copy()
    de_df["neg_log10_padj"] = -np.log10(de_df["padj"].clip(lower=1e-300))

    m_up = (de_df["padj"] < 0.05) & (de_df["log2FC"] > 0.5)
    m_dn = (de_df["padj"] < 0.05) & (de_df["log2FC"] < -0.5)
    m_ns = ~(m_up | m_dn)

    ax_vol.scatter(de_df.loc[m_ns, "log2FC"], de_df.loc[m_ns, "neg_log10_padj"],
                   s=4, c="#bdc3c7", alpha=0.5, linewidths=0, rasterized=True)
    ax_vol.scatter(de_df.loc[m_dn, "log2FC"], de_df.loc[m_dn, "neg_log10_padj"],
                   s=6, c="#2980b9", alpha=0.7, linewidths=0, rasterized=True, label="Down")
    ax_vol.scatter(de_df.loc[m_up, "log2FC"], de_df.loc[m_up, "neg_log10_padj"],
                   s=6, c="#e74c3c", alpha=0.7, linewidths=0, rasterized=True, label="Up")

    for gene in gene_list[:10]:
        match = de_df[de_df["gene"] == gene]
        if match.empty:
            continue
        x, y = match["log2FC"].values[0], match["neg_log10_padj"].values[0]
        ax_vol.annotate(gene, (x, y), fontsize=5, fontstyle="italic",
                        xytext=(4, 3), textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", lw=0.3, color="0.4"), color="0.1")

    ax_vol.axhline(-np.log10(0.05), color="0.5", ls=":", lw=0.5)
    ax_vol.axvline(0.5, color="0.5", ls=":", lw=0.5)
    ax_vol.axvline(-0.5, color="0.5", ls=":", lw=0.5)
    ax_vol.set_xlabel("log$_2$ fold change")
    ax_vol.set_ylabel("$-$log$_{10}$(p$_{adj}$)")
    ax_vol.legend(frameon=False, loc="upper left", markerscale=1.5)
    ax_vol.text(-0.12, 1.06, "b", transform=ax_vol.transAxes,
                fontsize=12, fontweight="bold", va="top")
    ax_vol.text(0.98, 0.98, f"Up: {m_up.sum()}\nDown: {m_dn.sum()}",
                transform=ax_vol.transAxes, fontsize=6, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.4))

    # ---- Sample size sidebar ----
    n_per_group = adata.obs["Group"].value_counts().reindex(group_order)
    ax_n.barh(range(len(group_order)), n_per_group.values,
              color=["#95a5a6", "#c0392b", "#2980b9", "#8e44ad"],
              edgecolor="white", linewidth=0.3, height=0.6)
    ax_n.set_yticks(range(len(group_order)))
    ax_n.set_yticklabels(group_order, fontsize=6)
    ax_n.set_xlabel("n spots", fontsize=6)
    ax_n.invert_yaxis()
    ax_n.set_title("Sample size", fontsize=7, fontweight="bold")
    for idx, v in enumerate(n_per_group.values):
        ax_n.text(v + n_per_group.values.max() * 0.02, idx, f"{v:,}",
                  va="center", fontsize=5)

    # ---- Stats heatmap (panel c) ----
    de_indexed = de_df.set_index("gene")

    log2fc_vals = np.array([
        de_indexed.loc[g, "log2FC"] if g in de_indexed.index else 0.0
        for g in gene_list])
    padj_vals = np.array([
        de_indexed.loc[g, "padj"] if g in de_indexed.index else 1.0
        for g in gene_list])
    neg_log10_p = -np.log10(np.clip(padj_vals, 1e-300, None))

    fc_abs_max = max(np.abs(log2fc_vals).max(), 0.1)
    p_max = max(neg_log10_p.max(), 1.0)

    fc_data = log2fc_vals.reshape(1, -1)
    p_data = neg_log10_p.reshape(1, -1)

    im_fc = ax_heat.imshow(fc_data, cmap="RdBu_r", aspect="auto",
                           vmin=-fc_abs_max, vmax=fc_abs_max,
                           extent=[-0.5, n_genes - 0.5, 0.5, 1.5],
                           interpolation="nearest")
    im_p = ax_heat.imshow(p_data, cmap="YlOrRd", aspect="auto",
                          vmin=0, vmax=p_max,
                          extent=[-0.5, n_genes - 0.5, -0.5, 0.5],
                          interpolation="nearest")

    for j in range(n_genes):
        fc_val = log2fc_vals[j]
        fc_c = "white" if abs(fc_val) > fc_abs_max * 0.5 else "0.2"
        ax_heat.text(j, 1.0, f"{fc_val:+.2f}", ha="center", va="center",
                     fontsize=5, fontweight="bold", color=fc_c)
        nlp = neg_log10_p[j]
        p_c = "white" if nlp > p_max * 0.5 else "0.2"
        ax_heat.text(j, 0.0, f"{format_pval(padj_vals[j])}\n{sig_stars(padj_vals[j])}",
                     ha="center", va="center", fontsize=4.5, color=p_c)

    ax_heat.set_xlim(-0.5, n_genes - 0.5)
    ax_heat.set_ylim(-0.5, 1.5)
    ax_heat.set_xticks(range(n_genes))
    ax_heat.set_xticklabels(gene_list, rotation=90, fontsize=5, fontstyle="italic")
    ax_heat.set_yticks([0.0, 1.0])
    ax_heat.set_yticklabels(["$-$log$_{10}$(p$_{adj}$)", "log$_2$FC"], fontsize=6)
    ax_heat.tick_params(length=0)
    ax_heat.grid(False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.text(-0.06, 1.15, "c", transform=ax_heat.transAxes,
                 fontsize=12, fontweight="bold", va="top")
    ax_heat.set_title(f"DE statistics ({direction.lower()})",
                      fontsize=7, fontweight="bold", loc="left", pad=8)

    ax_cbar.axis("off")
    cbar_ax1 = fig.add_axes([0.82, 0.12, 0.03, 0.15])
    cb1 = fig.colorbar(im_fc, cax=cbar_ax1)
    cb1.set_label("log$_2$FC", fontsize=6)
    cb1.ax.tick_params(labelsize=5)

    cbar_ax2 = fig.add_axes([0.90, 0.12, 0.03, 0.15])
    cb2 = fig.colorbar(im_p, cax=cbar_ax2)
    cb2.set_label("$-$log$_{10}$(p$_{adj}$)", fontsize=6)
    cb2.ax.tick_params(labelsize=5)

    fig.suptitle(f"Figure — {comp_desc} ({direction})",
                 fontsize=9, fontweight="bold", y=0.995)
    plt.show()


# --- Render per-comparison figures ---
for comp_key in ["YAD", "AC", "AAD"]:
    comp = comparisons[comp_key]
    de_df = all_de_results[comp_key]

    genes_up = top20_all[comp_key]["gene"].tolist()
    render_dotplot_and_stats(
        adata, genes_up, de_df, comp["desc"], "Upregulated",
        comp_key, GROUP_ORDER, comp_labels,
    )

    genes_dn = bot20_all[comp_key]["gene"].tolist()
    render_dotplot_and_stats(
        adata, genes_dn, de_df, comp["desc"], "Downregulated",
        comp_key, GROUP_ORDER, comp_labels,
    )

# =============================================================================
# 12. Combined Overview Figures
# =============================================================================

print("\n" + "=" * 70)
print("STEP 12: Combined Overview Figures")
print("=" * 70)


def render_combined_overview(adata, gene_sets, all_de_results, title_suffix,
                             direction, cmap_dot, comp_labels, comp_keys, group_order):
    """
    Render:
      1. Grouped scanpy dotplot (standalone)
      2. Composite: cross-comparison log2FC heatmap + stats table
    """
    gene_dict = {}
    all_genes = []
    for ck in comp_keys:
        top = gene_sets[ck]["gene"].tolist()[:10]
        gene_dict[comp_labels[ck]] = top
        for g in top:
            if g not in all_genes:
                all_genes.append(g)

    n_g = len(all_genes)
    if n_g == 0:
        print(f"  Skipping combined {direction} — no genes.")
        return

    # ---- Grouped dotplot (standalone) ----
    dp = sc.pl.dotplot(
        adata, var_names=gene_dict, groupby="Group",
        standard_scale="var", use_raw=True, dendrogram=False,
        show=False, return_fig=True,
    )
    dp.style(cmap=cmap_dot, dot_edge_color="0.3", dot_edge_lw=0.3, size_exponent=1.5)
    dp.make_figure()

    fig_dp = plt.gcf()
    fig_dp.set_size_inches(max(8, n_g * 0.4 + 3), 4.0)
    fig_dp.suptitle(
        f"a   Top {direction.lower()} genes across comparisons (grouped by contrast)",
        fontsize=9, fontweight="bold", x=0.02, ha="left", y=0.98,
    )
    plt.show()

    # ---- Composite: heatmap + table ----
    fig2, axes2 = plt.subplots(
        2, 1,
        figsize=(max(7.2, n_g * 0.38 + 2), 7.5),
        gridspec_kw={"height_ratios": [2.5, 3.5], "hspace": 0.6},
    )
    ax_hm = axes2[0]
    ax_tb = axes2[1]

    # ---- log2FC heatmap ----
    fc_matrix = np.full((len(comp_keys), n_g), np.nan)
    padj_matrix = np.full((len(comp_keys), n_g), np.nan)

    for ci, ck in enumerate(comp_keys):
        de_idx = all_de_results[ck].set_index("gene")
        for gi, gene in enumerate(all_genes):
            if gene in de_idx.index:
                fc_matrix[ci, gi] = de_idx.loc[gene, "log2FC"]
                padj_matrix[ci, gi] = de_idx.loc[gene, "padj"]

    vmax_fc = np.nanmax(np.abs(fc_matrix))
    if np.isnan(vmax_fc) or vmax_fc == 0:
        vmax_fc = 1.0

    im = ax_hm.imshow(fc_matrix, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax_fc, vmax=vmax_fc, interpolation="nearest")

    for ci in range(len(comp_keys)):
        for gi in range(n_g):
            fc = fc_matrix[ci, gi]
            padj = padj_matrix[ci, gi]
            if np.isnan(fc):
                ax_hm.text(gi, ci, "—", ha="center", va="center", fontsize=5, color="0.5")
                continue
            tc = "white" if abs(fc) > vmax_fc * 0.55 else "0.2"
            ax_hm.text(gi, ci, f"{fc:+.2f}\n{sig_stars(padj)}", ha="center",
                       va="center", fontsize=5, fontweight="bold", color=tc)

    ax_hm.set_xticks(range(n_g))
    ax_hm.set_xticklabels(all_genes, rotation=90, fontsize=5, fontstyle="italic")
    ax_hm.set_yticks(range(len(comp_keys)))
    ax_hm.set_yticklabels([comp_labels[k] for k in comp_keys], fontsize=6)
    ax_hm.tick_params(length=0)
    ax_hm.grid(False)
    for spine in ax_hm.spines.values():
        spine.set_visible(False)

    cb = fig2.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.02, shrink=0.8)
    cb.set_label("log$_2$FC", fontsize=6)
    cb.ax.tick_params(labelsize=5)

    ax_hm.text(-0.04, 1.15, "b", transform=ax_hm.transAxes,
               fontsize=12, fontweight="bold", va="top")
    ax_hm.set_title(f"log$_2$ fold change across comparisons ({direction.lower()})",
                    fontsize=8, fontweight="bold", loc="left", pad=8)

    # ---- Stats table ----
    ax_tb.axis("off")

    row_labels = []
    table_data = []

    for ck in comp_keys:
        de_idx = all_de_results[ck].set_index("gene")
        row_pval, row_padj, row_fc = [], [], []
        for gene in all_genes:
            if gene in de_idx.index:
                row_pval.append(format_pval(de_idx.loc[gene, "pval"]))
                p = de_idx.loc[gene, "padj"]
                row_padj.append(f"{format_pval(p)} {sig_stars(p)}")
                row_fc.append(f"{de_idx.loc[gene, 'log2FC']:+.2f}")
            else:
                row_pval.append("—")
                row_padj.append("—")
                row_fc.append("—")

        short = comp_labels[ck]
        row_labels.extend([f"{short}\npval", f"{short}\npadj", f"{short}\nlog₂FC"])
        table_data.extend([row_pval, row_padj, row_fc])

    tbl = ax_tb.table(
        cellText=table_data, rowLabels=row_labels, colLabels=all_genes,
        loc="upper center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(4.5)
    tbl.scale(1.0, 1.15)

    for ri, rlabel in enumerate(row_labels):
        if "padj" in rlabel:
            ck = comp_keys[ri // 3]
            de_idx = all_de_results[ck].set_index("gene")
            for gi, gene in enumerate(all_genes):
                cell = tbl[ri + 1, gi]
                if gene in de_idx.index:
                    padj = de_idx.loc[gene, "padj"]
                    if padj < 0.001:
                        cell.set_facecolor("#fadbd8")
                    elif padj < 0.05:
                        cell.set_facecolor("#fef9e7")
                    else:
                        cell.set_facecolor("#f5f5f5")

    block_colors = ["#ffffff", "#f8f9fa", "#f0f1f2"]
    for ri, rlabel in enumerate(row_labels):
        block = ri // 3
        if "padj" not in rlabel:
            for gi in range(-1, n_g):
                try:
                    tbl[ri + 1, gi].set_facecolor(block_colors[block])
                except KeyError:
                    pass

    ax_tb.text(-0.02, 1.02, "c", transform=ax_tb.transAxes,
               fontsize=12, fontweight="bold", va="top")
    ax_tb.set_title(f"DE statistics — {direction.lower()} (all comparisons × top genes)",
                    fontsize=7, fontweight="bold", loc="left", pad=2)

    fig2.suptitle(f"Figure — Combined {direction} DE across contrasts",
                  fontsize=9, fontweight="bold", y=0.995)
    plt.show()


comp_keys_fig2 = ["YAD", "AC", "AAD"]

print("\nCombined overview — Upregulated")
render_combined_overview(
    adata, top20_all, all_de_results, "Upregulated",
    "Upregulated", "Reds", comp_labels, comp_keys_fig2, GROUP_ORDER,
)

print("\nCombined overview — Downregulated")
render_combined_overview(
    adata, bot20_all, all_de_results, "Downregulated",
    "Downregulated", "Blues", comp_labels, comp_keys_fig2, GROUP_ORDER,
)

# =============================================================================
# 13. Spatial Variability (Moran's I) on AD Tissue
# =============================================================================

print("\n" + "=" * 70)
print("STEP 13: Spatial Variability (Moran's I)")
print("=" * 70)

ad_genes = set()
for group in ["YAD", "AAD"]:
    if group in top20_all:
        ad_genes.update(top20_all[group]["gene"].tolist())
    if group in bot20_all:
        ad_genes.update(bot20_all[group]["gene"].tolist())

ad_genes = [g for g in ad_genes if g in adata.var_names]

if ad_genes:
    print(f"Computing Moran's I for {len(ad_genes)} AD candidate genes...")

    adata_ad = adata[adata.obs["Condition"] == "AD"].copy()
    adata_ad.X = adata_ad.layers["log_norm"].copy()

    sq.gr.spatial_neighbors(adata_ad, coord_type="generic", n_neighs=6)
    sq.gr.spatial_autocorr(
        adata_ad, mode="moran", genes=ad_genes, n_perms=999, n_jobs=4,
    )

    morans = adata_ad.uns["moranI"].copy()
    morans = morans.sort_values("I", ascending=False)
    morans.to_csv(os.path.join(RESULTS_DIR, "Spatial_MoranI_AD_genes.csv"))

    sv_genes = morans[(morans["pval_norm"] < 0.05) & (morans["I"] > 0.1)].index.tolist()
    print(f"Spatially variable AD genes (Moran's I > 0.1, p < 0.05): {len(sv_genes)}")

    plot_df = morans.head(30).copy()
    plot_df["significant"] = (plot_df["pval_norm"] < 0.05) & (plot_df["I"] > 0.1)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if s else "#bdc3c7" for s in plot_df["significant"]]
    ax.barh(range(len(plot_df)), plot_df["I"].values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.index, fontsize=8)
    ax.set_xlabel("Moran's I")
    ax.set_title("Spatial Autocorrelation of AD Candidate Genes", fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(0.1, color="#2c3e50", linestyle="--", alpha=0.5, label="I = 0.1")
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    sv_genes = []
    print("No AD candidate genes found for spatial analysis.")

# =============================================================================
# 14. Summary Report
# =============================================================================

print("\n" + "=" * 70)
print("PIPELINE SUMMARY")
print("=" * 70)
print(f"DE method used:             {DE_METHOD}")
print(f"Total spots after QC:       {adata.n_obs:,}")
print(f"Total genes after QC:       {adata.n_vars:,}")
print(f"HVGs (Seurat v3):           {adata.var['highly_variable'].sum():,}")
print(f"PCA-filtered genes:         {adata.var['pca_filtered'].sum():,}")
print(f"Optimal PCs (elbow):        {optimal_pcs}")
print(f"Leiden clusters:            {adata.obs['leiden'].nunique()}")
print()
for group in ["YAD", "AC", "AAD"]:
    de = all_de_results[group]
    n_up = de[(de["padj"] < 0.05) & (de["log2FC"] > 0.5)].shape[0]
    n_dn = de[(de["padj"] < 0.05) & (de["log2FC"] < -0.5)].shape[0]
    n_top = top20_all[group].shape[0] if group in top20_all else 0
    n_bot = bot20_all[group].shape[0] if group in bot20_all else 0
    print(f"  {comparisons[group]['desc']}:")
    print(f"    Upregulated:   {n_up} sig. | Top 20 reported: {n_top}")
    print(f"    Downregulated: {n_dn} sig. | Top 20 reported: {n_bot}")
print()
if ad_genes:
    print(f"Spatially variable AD genes: {len(sv_genes)}")
print(f"\nResults saved to: {RESULTS_DIR}/")
print("Pipeline Complete.")
