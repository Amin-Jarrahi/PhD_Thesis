"""
Gene-to-m/z Spatial Pattern Matching Pipeline
"""

import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from typing import Dict, Optional
import pandas as pd
import os
import warnings
from dataclasses import dataclass
import pickle
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

RNA_PIXEL_SIZE = 55  # μm (Visium)
MSI_PIXEL_SIZE = 60  # μm
TOP_K_MATCHES = 26

MSI_INPUT_FOLDER = "/home/ajarrah/PhD_Thesis/chapter_2/h5ad_data_processed_4lockmasses_filtered_halfbrain_common_synced/"
MSI_SAMPLE_FILES = [
    "halfbrain_yc_1_filtered_common_synced.h5ad", "halfbrain_yc_2_filtered_common_synced.h5ad",
    "halfbrain_yc_3_filtered_common_synced.h5ad", "halfbrain_yc_4_filtered_common_synced.h5ad",
    "halfbrain_yad_1_filtered_common_synced.h5ad", "halfbrain_yad_2_filtered_common_synced.h5ad",
    "halfbrain_yad_3_filtered_common_synced.h5ad", "halfbrain_yad_4_filtered_common_synced.h5ad",
    "halfbrain_ac_1_filtered_common_synced.h5ad", "halfbrain_ac_2_filtered_common_synced.h5ad",
    "halfbrain_ac_3_filtered_common_synced.h5ad", "halfbrain_ac_4_filtered_common_synced.h5ad",
    "halfbrain_aad_1_filtered_common_synced.h5ad", "halfbrain_aad_2_filtered_common_synced.h5ad",
    "halfbrain_aad_3_filtered_common_synced.h5ad", "halfbrain_aad_4_filtered_common_synced.h5ad"
]
MSI_SAMPLE_IDS = [
    "YC_1", "YC_2", "YC_3", "YC_4",
    "YAD_1", "YAD_2", "YAD_3", "YAD_4",
    "AC_1", "AC_2", "AC_3", "AC_4",
    "AAD_1", "AAD_2", "AAD_3", "AAD_4"
]

RNA_INPUT_FOLDER = "/home/ajarrah/PhD_Thesis/chapter_4/h5ad_data/genes_top_800/"
RNA_SAMPLE_FILES = [
    "YC_1.h5ad", "YC_2.h5ad", "YC_3.h5ad", "YC_4.h5ad",
    "YAD_1.h5ad", "YAD_2.h5ad", "YAD_3.h5ad", "YAD_4.h5ad",
    "AC_1.h5ad", "AC_2.h5ad", "AC_3.h5ad", "AC_4.h5ad",
    "AAD_1.h5ad", "AAD_2.h5ad", "AAD_3.h5ad", "AAD_4.h5ad"
]
RNA_SAMPLE_IDS = [
    "YC_1", "YC_2", "YC_3", "YC_4",
    "YAD_1", "YAD_2", "YAD_3", "YAD_4",
    "AC_1", "AC_2", "AC_3", "AC_4",
    "AAD_1", "AAD_2", "AAD_3", "AAD_4"
]

AAD_TARGET_GENES = ['Mapt']

#AAD_TARGET_GENES = ['Eno1', 'Mapt', 'Thy1', 'Pmch', 'Atp1a3', 'Rac1', 'Rsrp1', 'Snhg11', 'Tubb4a',
#   'Rasgrf1', 'Hsp90ab1', 'Elavl3', 'App', 'Syp', 'AC149090.1',
#   'Aplp1', 'Apoe', 'Meg3', 'Gnas', 'Pkm']
DEFAULT_WEIGHTS = {
    'value_correlation': 0.3743,
    'importance_iou': 0.0412,
    'importance_correlation': 0.0962,
    'spatial_hist_corr': 0.1098,
    'radial_corr': 0.1167,
    'quadrant_corr': 0.1048,
    'morans_similarity': 0.0504,
    'value_hist_corr': 0.1066,
}

def rotate_180(coords: np.ndarray) -> np.ndarray:
    center = coords.mean(axis=0)
    return 2 * center - coords


"""def align_rna_to_msi(rna_coords: np.ndarray, msi_coords: np.ndarray) -> np.ndarray:
    rotated = rotate_180(rna_coords)
    rna_min, rna_max = rotated.min(axis=0), rotated.max(axis=0)
    msi_min, msi_max = msi_coords.min(axis=0), msi_coords.max(axis=0)
    rna_range = rna_max - rna_min
    msi_range = msi_max - msi_min
    scale = msi_range / (rna_range + 1e-8)
    return (rotated - rna_min) * scale + msi_min
"""

def align_rna_to_msi(rna_coords: np.ndarray, msi_coords: np.ndarray) -> np.ndarray:
    rotated = rotate_180(rna_coords)
    rna_min, rna_max = rotated.min(axis=0), rotated.max(axis=0)
    msi_min, msi_max = msi_coords.min(axis=0), msi_coords.max(axis=0)
    rna_range = rna_max - rna_min
    msi_range = msi_max - msi_min
    # ◆ FIX 2: Was: scale = msi_range / (rna_range + 1e-8)  — independent x/y scaling
    #   distorts spatial relationships (circles → ellipses), corrupting all
    #   distance-based metrics downstream.
    #   Now: single isotropic scale factor preserves aspect ratio.
    scale = min(msi_range[0] / (rna_range[0] + 1e-8),  # ◆ FIX 2
                msi_range[1] / (rna_range[1] + 1e-8))   # ◆ FIX 2
    rna_center = (rna_min + rna_max) / 2                 # ◆ FIX 2
    msi_center = (msi_min + msi_max) / 2                 # ◆ FIX 2
    return (rotated - rna_center) * scale + msi_center   # ◆ FIX 2


def compute_value_histogram(values: np.ndarray, n_bins: int = 50) -> np.ndarray:
    hist, _ = np.histogram(values, bins=n_bins, density=False)
    return hist / (hist.sum() + 1e-8)


def compute_spatial_histogram(coords: np.ndarray, values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    coord_min, coord_max = coords.min(axis=0), coords.max(axis=0)
    norm = (coords - coord_min) / (coord_max - coord_min + 1e-8)
    x_bins = np.clip((norm[:, 0] * n_bins).astype(int), 0, n_bins - 1)
    y_bins = np.clip((norm[:, 1] * n_bins).astype(int), 0, n_bins - 1)
    flat_idx = y_bins * n_bins + x_bins
    hist = np.bincount(flat_idx, weights=values, minlength=n_bins * n_bins).reshape(n_bins, n_bins)
    counts = np.bincount(flat_idx, minlength=n_bins * n_bins).reshape(n_bins, n_bins)
    hist = np.divide(hist, counts, where=counts > 0, out=np.zeros_like(hist))
    hist_min, hist_max = hist.min(), hist.max()
    if hist_max > hist_min:
        hist = (hist - hist_min) / (hist_max - hist_min)
    return hist


def compute_radial_profile(coords: np.ndarray, values: np.ndarray, n_rings: int = 10) -> np.ndarray:
    coord_min, coord_max = coords.min(axis=0), coords.max(axis=0)
    norm = (coords - coord_min) / (coord_max - coord_min + 1e-8)
    centroid = norm.mean(axis=0)
    distances = np.linalg.norm(norm - centroid, axis=1)
    max_dist = distances.max() + 1e-8
    ring_idx = np.clip((distances / max_dist * n_rings).astype(int), 0, n_rings - 1)
    profile = np.bincount(ring_idx, weights=values, minlength=n_rings)
    counts = np.bincount(ring_idx, minlength=n_rings)
    profile = np.divide(profile, counts, where=counts > 0, out=np.zeros_like(profile, dtype=float))
    prof_min, prof_max = profile.min(), profile.max()
    if prof_max > prof_min:
        profile = (profile - prof_min) / (prof_max - prof_min)
    return profile


def compute_quadrant_stats(coords: np.ndarray, values: np.ndarray, n_div: int = 3) -> np.ndarray:
    coord_min, coord_max = coords.min(axis=0), coords.max(axis=0)
    norm = (coords - coord_min) / (coord_max - coord_min + 1e-8)
    x_bins = np.clip((norm[:, 0] * n_div).astype(int), 0, n_div - 1)
    y_bins = np.clip((norm[:, 1] * n_div).astype(int), 0, n_div - 1)
    flat_idx = y_bins * n_div + x_bins
    stats = np.zeros(n_div * n_div * 2)
    for q in range(n_div * n_div):
        mask = flat_idx == q
        if mask.sum() > 0:
            q_vals = values[mask]
            stats[q * 2] = np.mean(q_vals)
            stats[q * 2 + 1] = np.std(q_vals)
    stats_min, stats_max = stats.min(), stats.max()
    if stats_max > stats_min:
        stats = (stats - stats_min) / (stats_max - stats_min)
    return stats


def compute_morans_i_vectorized(coords: np.ndarray, values: np.ndarray, indices: np.ndarray) -> float:
    n = len(values)
    mean_val = values.mean()
    deviations = values - mean_val
    denom = np.sum(deviations ** 2)
    if denom == 0:
        return 0.0
    neighbor_deviations = deviations[indices[:, 1:]]
    numer = np.sum(deviations[:, np.newaxis] * neighbor_deviations)
    w_sum = indices.shape[0] * (indices.shape[1] - 1)
    return (n / w_sum) * (numer / denom) if w_sum > 0 else 0.0


@dataclass
class SpatialSignature:
    sample_id: str
    feature_name: str
    feature_type: str
    node_importance: np.ndarray
    value_histogram: np.ndarray = None
    spatial_histogram: np.ndarray = None
    radial_profile: np.ndarray = None
    quadrant_stats: np.ndarray = None
    morans_i: float = 0.0
    coordinates: np.ndarray = None
    raw_values: np.ndarray = None
    aligned_coordinates: Optional[np.ndarray] = None


def compute_coordinate_based_similarity(gene_sig: SpatialSignature, mz_sig: SpatialSignature, grid_res: int = 50) -> dict:
    gene_coords = gene_sig.aligned_coordinates if gene_sig.aligned_coordinates is not None else gene_sig.coordinates
    mz_coords = mz_sig.coordinates
    x_min = min(gene_coords[:, 0].min(), mz_coords[:, 0].min())
    x_max = max(gene_coords[:, 0].max(), mz_coords[:, 0].max())
    y_min = min(gene_coords[:, 1].min(), mz_coords[:, 1].min())
    y_max = max(gene_coords[:, 1].max(), mz_coords[:, 1].max())
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, grid_res), np.linspace(y_min, y_max, grid_res))
    gene_grid = griddata(gene_coords, gene_sig.raw_values, (grid_x, grid_y), method='linear')
    mz_grid = griddata(mz_coords, mz_sig.raw_values, (grid_x, grid_y), method='linear')
    gene_imp_grid = griddata(gene_coords, gene_sig.node_importance, (grid_x, grid_y), method='linear')
    mz_imp_grid = griddata(mz_coords, mz_sig.node_importance, (grid_x, grid_y), method='linear')
    mask = ~(np.isnan(gene_grid) | np.isnan(mz_grid))
    value_corr = 0
    if mask.sum() > 10:
        r, _ = pearsonr(gene_grid[mask], mz_grid[mask])
        value_corr = r if not np.isnan(r) else 0
    mask_imp = ~(np.isnan(gene_imp_grid) | np.isnan(mz_imp_grid))
    importance_iou, imp_corr = 0, 0
    if mask_imp.sum() > 0:
        g_imp = gene_imp_grid.copy()
        m_imp = mz_imp_grid.copy()
        g_imp[np.isnan(g_imp)] = 0
        m_imp[np.isnan(m_imp)] = 0
        g_imp = g_imp / (g_imp.max() + 1e-8)
        m_imp = m_imp / (m_imp.max() + 1e-8)
        importance_iou = (np.minimum(g_imp[mask_imp], m_imp[mask_imp]).sum() / 
                          (np.maximum(g_imp[mask_imp], m_imp[mask_imp]).sum() + 1e-8))
        r, _ = pearsonr(g_imp[mask_imp], m_imp[mask_imp])
        imp_corr = r if not np.isnan(r) else 0
    return {'value_correlation': value_corr, 'importance_iou': importance_iou, 'importance_correlation': imp_corr}


def compute_descriptor_similarity(gene_sig: SpatialSignature, mz_sig: SpatialSignature) -> dict:
    def safe_pearsonr(a, b):
        if a.std() > 0 and b.std() > 0:
            r, _ = pearsonr(a, b)
            return r if not np.isnan(r) else 0
        return 0
    val_hist_corr = safe_pearsonr(gene_sig.value_histogram, mz_sig.value_histogram)
    spatial_hist_corr = safe_pearsonr(gene_sig.spatial_histogram.flatten(), mz_sig.spatial_histogram.flatten())
    radial_corr = safe_pearsonr(gene_sig.radial_profile, mz_sig.radial_profile)
    quad_corr = safe_pearsonr(gene_sig.quadrant_stats, mz_sig.quadrant_stats)
    morans_sim = 1.0 - abs(gene_sig.morans_i - mz_sig.morans_i)
    return {'value_hist_corr': val_hist_corr, 'spatial_hist_corr': spatial_hist_corr, 
            'radial_corr': radial_corr, 'quadrant_corr': quad_corr, 'morans_similarity': morans_sim}


def compute_combined_score(coord_sim: dict, desc_sim: dict, weights: dict = None) -> float:  
    if weights is None: 
        weights = DEFAULT_WEIGHTS  
    coord_score = (weights['value_correlation'] * max(coord_sim['value_correlation'], 0) +
                   weights['importance_iou'] * max(coord_sim['importance_iou'], 0) +
                   weights['importance_correlation'] * max(coord_sim['importance_correlation'], 0))
    desc_score = (weights['spatial_hist_corr'] * max(desc_sim['spatial_hist_corr'], 0) +
                  weights['radial_corr'] * max(desc_sim['radial_corr'], 0) +
                  weights['quadrant_corr'] * max(desc_sim['quadrant_corr'], 0) +
                  weights['morans_similarity'] * max(desc_sim['morans_similarity'], 0) +
                  weights['value_hist_corr'] * max(desc_sim['value_hist_corr'], 0))
    return coord_score + desc_score


class AnalyticPatternMatcher:
    def __init__(self, output_dir: str = './gene_to_mz_results_v1_analytic', n_jobs: int = -1):
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        for subdir in ['descriptors']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        self.rna_data = {}
        self.msi_data = {}
        self._nn_cache = {}

    def load_all_data(self):
        print("Loading RNA-seq data...")
        print(f"  Pixel size: {RNA_PIXEL_SIZE} μm (hexagonal)")
        for file, sample_id in zip(RNA_SAMPLE_FILES, RNA_SAMPLE_IDS):
            path = os.path.join(RNA_INPUT_FOLDER, file)
            if os.path.exists(path):
                self.rna_data[sample_id] = sc.read_h5ad(path)
                print(f"  {sample_id}: {self.rna_data[sample_id].shape}")
        print(f"\nLoading MSI data...")
        print(f"  Pixel size: {MSI_PIXEL_SIZE} μm (Cartesian)")
        for file, sample_id in zip(MSI_SAMPLE_FILES, MSI_SAMPLE_IDS):
            path = os.path.join(MSI_INPUT_FOLDER, file)
            if os.path.exists(path):
                self.msi_data[sample_id] = sc.read_h5ad(path)
                print(f"  {sample_id}: {self.msi_data[sample_id].shape}")

    def _get_nn_indices(self, coords: np.ndarray, k: int, cache_key: str) -> np.ndarray:
        full_key = f"{cache_key}_{k}"
        if full_key not in self._nn_cache:
            coord_min, coord_max = coords.min(axis=0), coords.max(axis=0)
            norm = (coords - coord_min) / (coord_max - coord_min + 1e-8)
            k_actual = min(k + 1, len(coords))
            nn = NearestNeighbors(n_neighbors=k_actual)
            nn.fit(norm)
            _, indices = nn.kneighbors(norm)
            self._nn_cache[full_key] = indices
        return self._nn_cache[full_key]

    def compute_bio_importance(self, coords: np.ndarray, values: np.ndarray, nn_indices: np.ndarray) -> np.ndarray:
        neighbor_vals = values[nn_indices[:, 1:]]
        local_var = np.var(neighbor_vals, axis=1)
        lv_min, lv_max = local_var.min(), local_var.max()
        if lv_max > lv_min:
            local_var = (local_var - lv_min) / (lv_max - lv_min)
        else:
            local_var = np.zeros_like(local_var)
        v_min, v_max = values.min(), values.max()
        if v_max > v_min:
            val_norm = (values - v_min) / (v_max - v_min)
        else:
            val_norm = np.zeros_like(values)
        return 0.5 * local_var + 0.5 * val_norm

    def extract_signature(self, coords: np.ndarray, values: np.ndarray, sample_id: str,
                          feature_name: str, feature_type: str, n_neighbors: int,
                          nn_indices: np.ndarray, aligned_coords: Optional[np.ndarray] = None) -> SpatialSignature:
        bio_imp = self.compute_bio_importance(coords, values, nn_indices)
        return SpatialSignature(
            sample_id=sample_id, feature_name=feature_name, feature_type=feature_type,
            node_importance=bio_imp, value_histogram=compute_value_histogram(values),
            spatial_histogram=compute_spatial_histogram(coords, values),
            radial_profile=compute_radial_profile(coords, values),
            quadrant_stats=compute_quadrant_stats(coords, values),
            morans_i=compute_morans_i_vectorized(coords, values, nn_indices),
            coordinates=coords, raw_values=values, aligned_coordinates=aligned_coords)

    
    def find_matches(self, gene_sig, mz_sigs, top_k=20):
        def compute_match(mz_name, mz_sig):
            coord_sim = compute_coordinate_based_similarity(gene_sig, mz_sig)
            desc_sim = compute_descriptor_similarity(gene_sig, mz_sig)
            combined = compute_combined_score(coord_sim, desc_sim)
            return {'gene': gene_sig.feature_name, 'rna_sample': gene_sig.sample_id,
                    'mz_feature': mz_name, 'msi_sample': mz_sig.sample_id,
                    **coord_sim, **desc_sim, 'combined_score': combined}
        matches = Parallel(n_jobs=self.n_jobs, prefer='threads')(
            delayed(compute_match)(mz_name, mz_sig) for mz_name, mz_sig in mz_sigs.items())
        return pd.DataFrame(matches).sort_values('combined_score', ascending=False).head(top_k)

    def run_analysis(self, top_k=20):
        print("\n" + "="*70)
        print("GENE-TO-M/Z MATCHING V1 (Analytic - Optimized)")
        print(f"RNA: {RNA_PIXEL_SIZE}μm (hexagonal) | MSI: {MSI_PIXEL_SIZE}μm (Cartesian)")
        print("Strategy: Analytic Spatial Descriptors + Heuristic Importance")
        print("="*70)
        gene_avail = {gene: {s: gene in self.rna_data[s].var_names
                             for s in RNA_SAMPLE_IDS if s in self.rna_data}
                      for gene in AAD_TARGET_GENES}
        print("\nGene availability:")
        for gene in AAD_TARGET_GENES:
            n = sum(gene_avail[gene].values())
            print(f"  {gene}: {n}/{len(RNA_SAMPLE_IDS)}")
        all_results = []
        all_topk_results = []
        for gene in AAD_TARGET_GENES:
            print(f"\n{'='*50}")
            print(f"Gene: {gene}")
            print(f"{'='*50}")
            available = [s for s, a in gene_avail[gene].items() if a]
            if not available:
                continue
            for rna_sample in available:
                print(f"\n  {rna_sample}")
                adata = self.rna_data[rna_sample]
                rna_coords = np.column_stack([adata.obs['x_um'].values, adata.obs['y_um'].values])
                gene_idx = list(adata.var_names).index(gene)
                gene_expr = adata.X[:, gene_idx].toarray().flatten() if hasattr(adata.X, 'toarray') else adata.X[:, gene_idx].flatten()
                msi_sample = rna_sample
                if msi_sample not in self.msi_data:
                    print(f"    MSI {msi_sample} not found")
                    continue
                msi_adata = self.msi_data[msi_sample]
                msi_coords = np.column_stack([msi_adata.obs['x_um'].values, msi_adata.obs['y_um'].values])
                aligned_coords = align_rna_to_msi(rna_coords, msi_coords)
                rna_nn_indices = self._get_nn_indices(rna_coords, 6, f"rna_{rna_sample}")
                msi_nn_indices = self._get_nn_indices(msi_coords, 8, f"msi_{msi_sample}")
                gene_sig = self.extract_signature(rna_coords, gene_expr, rna_sample, gene, 'gene', 6, rna_nn_indices, aligned_coords)
                with open(os.path.join(self.output_dir, 'descriptors', f'{gene}_{rna_sample}_descriptors.pkl'), 'wb') as f:
                    pickle.dump({
                        'feature_name': gene_sig.feature_name, 'sample_id': gene_sig.sample_id,
                        'feature_type': gene_sig.feature_type, 'value_histogram': gene_sig.value_histogram,
                        'spatial_histogram': gene_sig.spatial_histogram, 'radial_profile': gene_sig.radial_profile,
                        'quadrant_stats': gene_sig.quadrant_stats, 'morans_i': gene_sig.morans_i,
                        'coordinates': gene_sig.coordinates, 'aligned_coordinates': gene_sig.aligned_coordinates,
                        'expression_stats': {
                            'mean': float(gene_sig.raw_values.mean()), 'std': float(gene_sig.raw_values.std()),
                            'min': float(gene_sig.raw_values.min()), 'max': float(gene_sig.raw_values.max()),
                            'median': float(np.median(gene_sig.raw_values)), 'n_spots': len(gene_sig.raw_values),
                            'n_nonzero': int((gene_sig.raw_values > 0).sum())},
                        'importance_stats': {
                            'mean': float(gene_sig.node_importance.mean()), 'std': float(gene_sig.node_importance.std()),
                            'top_10pct_threshold': float(np.percentile(gene_sig.node_importance, 90))}}, f)
                print(f"    Extracting MSI signatures...")
                msi_data = msi_adata.X.toarray() if hasattr(msi_adata.X, 'toarray') else msi_adata.X
                mz_names = list(msi_adata.var_names)
                def extract_single_mz(i):
                    return mz_names[i], self.extract_signature(msi_coords, msi_data[:, i], msi_sample, mz_names[i], 'mz', 8, msi_nn_indices)
                mz_results = Parallel(n_jobs=self.n_jobs, prefer='threads')(delayed(extract_single_mz)(i) for i in range(len(mz_names)))
                mz_sigs = dict(mz_results)
                print(f"      {len(mz_names)} m/z features processed")
                print(f"    Matching...")
                matches = self.find_matches(gene_sig, mz_sigs, top_k)
                all_results.append(matches)
                topk_matches = matches.head(top_k).copy() # Store top k matches for detailed analysis
                topk_matches['gene'] = gene
                topk_matches['rna_sample'] = rna_sample
                topk_matches['rank'] = range(1, len(topk_matches) + 1)
                all_topk_results.append(topk_matches)
                if len(matches) > 0:
                    print(f"\n    Top {top_k}:")
                    for idx in range(min(top_k, len(matches))):
                        m = matches.iloc[idx]
                        print(f"      {idx+1}. m/z {m['mz_feature']}: {m['combined_score']:.3f}")

                    for idx in range(min(top_k, len(matches))): # Save top k matches' descriptors
                        m = matches.iloc[idx]
                        mz_name = m['mz_feature']
                        mz_sig = mz_sigs[mz_name]
                        mz_filename = mz_name.replace('/', '_').replace('\\', '_')
                        with open(os.path.join(self.output_dir, 'descriptors',
                                               f'{gene}_{rna_sample}_match{idx+1}_{mz_filename}_descriptors.pkl'), 'wb') as f:
                            pickle.dump({
                                'gene': gene, 'gene_sample': rna_sample, 'mz_feature': mz_name,
                                'mz_sample': mz_sig.sample_id, 'match_rank': idx + 1,
                                'combined_score': float(m['combined_score']),
                                'mz_value_histogram': mz_sig.value_histogram, 'mz_spatial_histogram': mz_sig.spatial_histogram,
                                'mz_radial_profile': mz_sig.radial_profile, 'mz_quadrant_stats': mz_sig.quadrant_stats,
                                'mz_morans_i': mz_sig.morans_i, 'gene_value_histogram': gene_sig.value_histogram,
                                'gene_spatial_histogram': gene_sig.spatial_histogram, 'gene_radial_profile': gene_sig.radial_profile,
                                'gene_quadrant_stats': gene_sig.quadrant_stats, 'gene_morans_i': gene_sig.morans_i,
                                'similarity_scores': {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                                                      for k, v in m.items() if k not in ['gene', 'rna_sample', 'mz_feature', 'msi_sample']}}, f)
        if all_results:
            results = pd.concat(all_results, ignore_index=True)
            results = results.sort_values('combined_score', ascending=False)
            results.to_csv(os.path.join(self.output_dir, 'gene_to_mz_matches_v1_analytic.csv'), index=False)
            if all_topk_results:
                topk_df = pd.concat(all_topk_results, ignore_index=True)
                priority_cols = ['gene', 'rna_sample', 'rank', 'mz_feature', 'msi_sample', 'combined_score']
                other_cols = [c for c in topk_df.columns if c not in priority_cols]
                topk_df = topk_df[priority_cols + other_cols]
                topk_df = topk_df.sort_values(['gene', 'rna_sample', 'rank'])
                topk_df.to_csv(os.path.join(self.output_dir, f'gene_to_mz_top{TOP_K_MATCHES}_matches_all_scores.csv'), index=False)
            print(f"\nSaved results to: {self.output_dir}")
            print("\n" + "="*70)
            print("TOP MATCHES")
            print("="*70)
            for gene in AAD_TARGET_GENES:
                gr = results[results['gene'] == gene]
                if len(gr) > 0:
                    t = gr.iloc[0]
                    print(f"\n{gene}: m/z {t['mz_feature']} ({t['rna_sample']}) = {t['combined_score']:.3f}")
            return results
        return None


def main():
    print("="*70)
    print("V1: Analytic Spatial Matching (No GNN/Transformer) - Optimized")
    print(f"RNA: {RNA_PIXEL_SIZE}μm | MSI: {MSI_PIXEL_SIZE}μm")
    print("="*70)
    matcher = AnalyticPatternMatcher(output_dir=f'./{TOP_K_MATCHES}_gene_to_mz_synced_results_v1_analytic_fast', n_jobs=-1)
    matcher.load_all_data()
    results = matcher.run_analysis(top_k=TOP_K_MATCHES)
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    return matcher, results


if __name__ == "__main__":
    matcher, results = main()