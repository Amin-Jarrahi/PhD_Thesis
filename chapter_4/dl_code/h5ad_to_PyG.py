#!/usr/bin/env python3
"""
Preprocessing pipeline to convert MSI and Visium h5ad files into PyG Data objects
with neighbor-count diagnostics and automatic radius sweep plots for tuning RADIUS_UM.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------
# USER-CONFIG / DEFAULT PATHS
# ----------------------------
MSI_INPUT_FOLDER = "/home/ajarrah/PhD_Thesis/chapter_2/h5ad_data_processed_4lockmasses_filtered_halfbrain/"
COMMON_MZS_CSV = "/home/ajarrah/PhD_Thesis/chapter_2/csv_data/common_mz_clusters_improved.csv"
MSI_SAMPLE_IDS = [
    "yc_1", "yc_2", "yc_3", "yc_4",
    "yad_1", "yad_2", "yad_3", "yad_4",
    "ac_1", "ac_2", "ac_3", "ac_4",
    "aad_1", "aad_2", "aad_3", "aad_4"
]

RNA_INPUT_FOLDER = "/home/ajarrah/PhD_Thesis/chapter_3/h5ad_data/"
RNA_TISSUE_POSITIONS_CSV = "/home/ajarrah/PhD_Thesis/chapter_4/tissue_positions/tissue_positions.csv"
RNA_SAMPLE_FILES = [
    "A1_Young_Control_Mouse_Brain_202502.h5ad",
    "B1_Young_Control_Mouse_Brain_202502.h5ad",
    "C1_Young_Control_Mouse_Brain_202502.h5ad",
    "D1_Young_Control_Mouse_Brain_202502.h5ad",
    "A1_Young_AD_Mouse_Brain_202502.h5ad",
    "B1_Young_AD_Mouse_Brain_202502.h5ad",
    "C1_Young_AD_Mouse_Brain_202502.h5ad",
    "D1_Young_AD_Mouse_Brain_202502.h5ad",
    "A1_Aged_Control_Mouse_Brain_202502.h5ad",
    "B1_Aged_Control_Mouse_Brain_202502.h5ad",
    "C1_Aged_Control_Mouse_Brain_202502.h5ad",
    "D1_Aged_Control_Mouse_Brain_202502.h5ad",
    "A1_Aged_AD_Mouse_Brain_202502.h5ad",
    "B1_Aged_AD_Mouse_Brain_202502.h5ad",
    "C1_Aged_AD_Mouse_Brain_202502.h5ad",
    "D1_Aged_AD_Mouse_Brain_202502.h5ad"
]
RNA_SAMPLE_IDS = [
    "YC_1","YC_2","YC_3","YC_4",
    "YAD_1","YAD_2","YAD_3","YAD_4",
    "AC_1","AC_2","AC_3","AC_4",
    "AAD_1","AAD_2","AAD_3","AAD_4"
]

OUTPUT_DIR = "processed_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# PARAMETERS
# ----------------------------
RADIUS_UM = 110.0
REDUCE_DIM = True
N_COMPONENTS = 128
LOG_TRANSFORM = True
VISIUM_HVG_N = 2000
MSI_TOP_K_MZ = None
USE_COMMON_MZS = True

# ----------------------------
# Helper functions
# ----------------------------
def ensure_numpy_matrix(X):
    if sparse.issparse(X):
        return X.toarray()
    if isinstance(X, np.ndarray):
        return X
    if hasattr(X, "values"):
        return X.values
    return np.asarray(X)

def tic_normalize(intensity_matrix, tic_values):
    tic = np.array(tic_values, dtype=float)
    tic[tic == 0] = 1.0
    return intensity_matrix / tic[:, None]

def build_radius_edges(coords, radius):
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=radius)
    row_idx, col_idx = [], []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if i == j: continue
            row_idx.append(i)
            col_idx.append(j)
    if len(row_idx) == 0:
        N = coords.shape[0]
        for i in range(N):
            for j in range(N):
                if i != j:
                    row_idx.append(i)
                    col_idx.append(j)
    edge_index = torch.tensor([row_idx, col_idx], dtype=torch.long)
    return edge_index

def reduce_features(X, n_components=N_COMPONENTS):
    scaler = StandardScaler(with_mean=False)
    Xs = scaler.fit_transform(X)
    svd = TruncatedSVD(n_components=min(n_components, Xs.shape[1]-1 or 1), random_state=0)
    return svd.fit_transform(Xs).astype(np.float32)

# ----------------------------
# --- Neighbor count diagnostic ---
# ----------------------------
def diagnostic_neighbor_count(coords_um, radius_um, sample_id):
    tree = cKDTree(coords_um)
    neighbors = tree.query_ball_point(coords_um, r=radius_um)
    counts = np.array([len(n)-1 for n in neighbors])
    print(f"\n=== Neighbor Diagnostic for {sample_id} ===")
    print(f"Radius = {radius_um} µm")
    print(f"N nodes = {len(coords_um)}")
    print(f"Mean neighbors:   {counts.mean():.2f}")
    print(f"Median neighbors: {np.median(counts):.2f}")
    print(f"Min neighbors:    {counts.min()}")
    print(f"Max neighbors:    {counts.max()}")
    print(f"Isolated nodes:   {(counts==0).sum()} ({(counts==0).mean()*100:.2f}%)")
    bins = [0,3,6,10,20,40,80,200,500,1000]
    hist, edges = np.histogram(counts, bins=bins)
    print("\nHistogram (neighbor count bins):")
    for h, e1, e2 in zip(hist, edges[:-1], edges[1:]):
        print(f"{int(e1):4d}-{int(e2):4d}: {h}")
    return counts

# ----------------------------
# --- Radius sweep diagnostic ---
# ----------------------------
def radius_sweep_diagnostic(coords_um, sample_id, radius_list=None):
    if radius_list is None:
        radius_list = np.linspace(20, 300, 15)
    
    mean_neighbors = []
    median_neighbors = []
    isolated_fraction = []

    for r in radius_list:
        tree = cKDTree(coords_um)
        neighbors = tree.query_ball_point(coords_um, r=r)
        counts = np.array([len(n)-1 for n in neighbors])
        mean_neighbors.append(counts.mean())
        median_neighbors.append(np.median(counts))
        isolated_fraction.append((counts==0).mean())

    plt.figure(figsize=(6,4))
    plt.plot(radius_list, mean_neighbors, marker='o', label='Mean neighbors')
    plt.plot(radius_list, median_neighbors, marker='s', label='Median neighbors')
    plt.plot(radius_list, np.array(isolated_fraction)*100, marker='^', label='Isolated nodes (%)')
    plt.xlabel('Radius (µm)')
    plt.ylabel('Neighbors / Isolated %')
    plt.title(f'Neighbor counts vs radius ({sample_id})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return radius_list, mean_neighbors, median_neighbors, isolated_fraction

# ----------------------------
# MSI processing
# ----------------------------
def process_msi_h5ad(path_h5ad, sample_id, common_mzs_csv=None,
                     use_common_mzs=True, top_k_mz=None,
                     radius_um=RADIUS_UM,
                     log_transform=LOG_TRANSFORM,
                     reduce_dim=REDUCE_DIM):
    adata = sc.read_h5ad(path_h5ad)
    X_raw = ensure_numpy_matrix(adata.X)
    coords = np.vstack([adata.obs["x_um"].values, adata.obs["y_um"].values]).T.astype(float)
    tic = adata.obs.get("Processed_TIC", adata.obs.get("TIC", X_raw.sum(axis=1)))
    X_norm = tic_normalize(X_raw.astype(float), tic)
    # Feature selection
    if use_common_mzs and common_mzs_csv and os.path.exists(common_mzs_csv):
        df_mz = pd.read_csv(common_mzs_csv)
        mz_col = "mz" if "mz" in df_mz.columns else df_mz.columns[0]
        common_mz_list = df_mz[mz_col].values.astype(float)
        var_mzs = adata.var.get("mzs", np.arange(adata.shape[1])).astype(float)
        idxs = np.unique([np.argmin(np.abs(var_mzs - mz)) for mz in common_mz_list])
        X_sel = X_norm[:, idxs]
    elif top_k_mz:
        mean_by_mz = X_norm.mean(axis=0)
        idxs = np.argsort(mean_by_mz)[::-1][:top_k_mz]
        X_sel = X_norm[:, idxs]
    else:
        X_sel = X_norm
    if log_transform: X_sel = np.log1p(X_sel)
    X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if reduce_dim and X_sel.shape[1] > N_COMPONENTS:
        X_proc = reduce_features(X_sel)
    else:
        X_proc = X_sel.astype(np.float32)
    edge_index = build_radius_edges(coords, radius_um)
    # --- Neighbor diagnostic ---
    diagnostic_neighbor_count(coords, radius_um, sample_id)
    # --- Radius sweep ---
    radius_sweep_diagnostic(coords, sample_id)
    data = Data(x=torch.tensor(X_proc, dtype=torch.float),
                pos=torch.tensor(coords, dtype=torch.float),
                edge_index=edge_index)
    data.sample_id = sample_id
    data.modality = "MSI"
    data.n_nodes = X_proc.shape[0]
    data.n_features = X_proc.shape[1]
    return data

# ----------------------------
# Visium processing
# ----------------------------
def reconstruct_visium_coords_from_tissue_positions(tissue_positions_csv, spot_spacing_um=100.0):
    df = pd.read_csv(tissue_positions_csv, usecols=["barcode","array_row","array_col"])
    y = df["array_row"].values * (np.sqrt(3)/2.0 * spot_spacing_um)
    x = df["array_col"].values * (spot_spacing_um / 2.0)
    df["x_um"] = x
    df["y_um"] = y
    return df.set_index("barcode")[["x_um","y_um"]]

def process_visium_h5ad(path_h5ad, sample_id, tissue_positions_df,
                        radius_um=RADIUS_UM,
                        log_transform=LOG_TRANSFORM,
                        reduce_dim=REDUCE_DIM,
                        hvg_n=VISIUM_HVG_N):
    adata = sc.read_h5ad(path_h5ad)
    sc.pp.normalize_total(adata, target_sum=1e4)
    if log_transform: sc.pp.log1p(adata)
    if "x_um" in adata.obs.columns and "y_um" in adata.obs.columns:
        coords = np.vstack([adata.obs["x_um"].values, adata.obs["y_um"].values]).T.astype(float)
    else:
        coords = np.array([tissue_positions_df.loc[bc, ["x_um","y_um"]].values
                           if bc in tissue_positions_df.index else (np.nan,np.nan)
                           for bc in adata.obs_names])
    X_raw = ensure_numpy_matrix(adata.X)
    if hvg_n and adata.shape[1] > hvg_n:
        try:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg_n, subset=True)
            X_sel = ensure_numpy_matrix(adata.X)
        except Exception:
            var = X_raw.var(axis=0)
            idx = np.argsort(var)[::-1][:hvg_n]
            X_sel = X_raw[:, idx]
    else:
        X_sel = X_raw
    X_sel = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if reduce_dim and X_sel.shape[1] > N_COMPONENTS:
        X_proc = reduce_features(X_sel)
    else:
        X_proc = X_sel.astype(np.float32)
    valid_mask = ~np.isnan(coords).any(axis=1)
    coords_valid = coords[valid_mask]
    X_valid = X_proc[valid_mask]
    edge_index = build_radius_edges(coords_valid, radius_um)
    # --- Neighbor diagnostic ---
    diagnostic_neighbor_count(coords_valid, radius_um, sample_id)
    # --- Radius sweep ---
    radius_sweep_diagnostic(coords_valid, sample_id)
    data = Data(x=torch.tensor(X_valid, dtype=torch.float),
                pos=torch.tensor(coords_valid, dtype=torch.float),
                edge_index=edge_index)
    data.sample_id = sample_id
    data.modality = "Visium"
    data.n_nodes = X_valid.shape[0]
    data.n_features = X_valid.shape[1]
    return data

# ----------------------------
# Main orchestration
# ----------------------------
def main():
    common_mzs_csv = COMMON_MZS_CSV if (USE_COMMON_MZS and os.path.exists(COMMON_MZS_CSV)) else None
    # MSI
    for sid in tqdm(MSI_SAMPLE_IDS, desc="MSI samples"):
        h5ad_path = os.path.join(MSI_INPUT_FOLDER, f"{sid}.h5ad")
        if not os.path.exists(h5ad_path): continue
        data = process_msi_h5ad(h5ad_path, sample_id=sid, common_mzs_csv=common_mzs_csv)
        torch.save(data, os.path.join(OUTPUT_DIR, f"msi_{sid}.pt"))
    # Visium
    tissue_pos_df = reconstruct_visium_coords_from_tissue_positions(RNA_TISSUE_POSITIONS_CSV)
    for idx, fname in enumerate(tqdm(RNA_SAMPLE_FILES, desc="Visium samples")):
        sample_id = RNA_SAMPLE_IDS[idx]
        h5ad_path = os.path.join(RNA_INPUT_FOLDER, fname)
        if not os.path.exists(h5ad_path): continue
        data = process_visium_h5ad(h5ad_path, sample_id=sample_id, tissue_positions_df=tissue_pos_df)
        torch.save(data, os.path.join(OUTPUT_DIR, f"visium_{sample_id}.pt"))

if __name__ == "__main__":
    main()
