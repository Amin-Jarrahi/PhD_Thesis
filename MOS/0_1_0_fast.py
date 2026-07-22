"""
Gene-to-m/z Rotation Analysis - SPEED-OPTIMIZED VERSION
========================================================
Changes marked with # ◆ SPEED
"""
import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from scipy.interpolate import griddata, LinearNDInterpolator  # ◆ SPEED: added LinearNDInterpolator
from scipy.spatial import Delaunay  # ◆ SPEED: for reusable interpolator
import pandas as pd
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import functools
import time  # ◆ SPEED: for timing

warnings.filterwarnings('ignore')

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Note: numba not available. Install with 'pip install numba' for additional speedup.")

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

RNA_PIXEL_SIZE = 55
MSI_PIXEL_SIZE = 60

ANGLE_START = 340    # Start angle in degrees
ANGLE_END = 20      # End angle in degrees
ANGLE_STEP = 2     # Step size in degrees

MSI_INPUT_FOLDER = "/home/ajarrah/PhD_Thesis/chapter_4/dummy_data_50/msi/"
RNA_INPUT_FOLDER = "/home/ajarrah/PhD_Thesis/chapter_4/dummy_data_50/rna/"

MSI_SAMPLE_FILES = [
    "halfbrain_yc_1_filtered_common.h5ad", "halfbrain_yc_2_filtered_common.h5ad",
    "halfbrain_yc_3_filtered_common.h5ad", "halfbrain_yc_4_filtered_common.h5ad",
    "halfbrain_yad_1_filtered_common.h5ad", "halfbrain_yad_2_filtered_common.h5ad",
    "halfbrain_yad_3_filtered_common.h5ad", "halfbrain_yad_4_filtered_common.h5ad",
    "halfbrain_ac_1_filtered_common.h5ad", "halfbrain_ac_2_filtered_common.h5ad",
    "halfbrain_ac_3_filtered_common.h5ad", "halfbrain_ac_4_filtered_common.h5ad",
    "halfbrain_aad_1_filtered_common.h5ad", "halfbrain_aad_2_filtered_common.h5ad",
    "halfbrain_aad_3_filtered_common.h5ad", "halfbrain_aad_4_filtered_common.h5ad"
]
MSI_SAMPLE_IDS = [
    "YC_1", "YC_2", "YC_3", "YC_4",
    "YAD_1", "YAD_2", "YAD_3", "YAD_4",
    "AC_1", "AC_2", "AC_3", "AC_4",
    "AAD_1", "AAD_2", "AAD_3", "AAD_4"
]

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

AAD_TARGET_GENES = [
    # === GRADIENTS (6 patterns) ===
    "Gene_Gradient_X",              # Left to right
    "Gene_Gradient_Y",              # Bottom to top
    "Gene_Gradient_Diagonal_NE",    # Southwest to Northeast
    "Gene_Gradient_Diagonal_NW",    # Southeast to Northwest
    "Gene_Gradient_Radial_In",      # Center bright, edges dark
    "Gene_Gradient_Radial_Out",     # Edges bright, center dark
    
    # === WAVES & STRIPES (8 patterns) ===
    "Gene_Stripes_Vertical",        # Vertical sinusoidal
    "Gene_Stripes_Horizontal",      # Horizontal sinusoidal
    "Gene_Stripes_Diagonal_45",     # 45-degree stripes
    "Gene_Stripes_Diagonal_135",    # 135-degree stripes
    "Gene_Waves_Concentric",        # Concentric circles (bullseye)
    "Gene_Waves_Spiral",            # Spiral pattern
    "Gene_Waves_Interference",      # Two-source interference
    "Gene_Waves_Ripple",            # Decaying ripples from center
    
    # === BLOBS & SPOTS (10 patterns) ===
    "Gene_Blob_Center",             # Gaussian blob center
    "Gene_Blob_TopRight",           # Gaussian blob top-right
    "Gene_Blob_TopLeft",            # Gaussian blob top-left
    "Gene_Blob_BottomRight",        # Gaussian blob bottom-right
    "Gene_Blob_BottomLeft",         # Gaussian blob bottom-left
    "Gene_Spots_Grid_Dense",        # Dense polka dots
    "Gene_Spots_Grid_Sparse",       # Sparse polka dots
    "Gene_Spots_Random_Large",      # Large random spots
    "Gene_Spots_Triangular",        # Triangular arrangement
    "Gene_Spots_Hexagonal",         # Hexagonal arrangement
    
    # === RINGS & DONUTS (6 patterns) ===
    "Gene_Ring_Inner",              # Inner ring
    "Gene_Ring_Outer",              # Outer ring
    "Gene_Ring_Double",             # Double concentric rings
    "Gene_Ring_Eccentric",          # Off-center ring
    "Gene_Ring_Elliptical",         # Elliptical ring
    "Gene_Ring_Partial",            # Partial arc/ring

    # === GEOMETRIC PATTERNS (8 patterns) ===
    "Gene_Checkerboard_Fine",       # Fine checkerboard
    "Gene_Checkerboard_Coarse",     # Coarse checkerboard
    "Gene_Quadrant_Alternating",    # Alternating quadrants
    "Gene_Sectors_4",               # 4 pie sectors
    "Gene_Sectors_8",               # 8 pie sectors
    "Gene_Triangle_Pattern",        # Triangular tessellation
    "Gene_Diamond_Pattern",         # Diamond/rhombus pattern
    "Gene_Honeycomb",               # Honeycomb hexagonal

    # === COMPLEX BIOLOGICAL-LIKE (12 patterns) ===
    "Gene_Cortical_Layers",         # Layered like cortex
    "Gene_Hotspot_Cluster",         # Multiple clustered hotspots
    "Gene_Edge_Enhancement",        # Enhanced at tissue edges
    "Gene_Core_Shell",              # Core-shell structure
    "Gene_Branching",               # Branching/dendritic pattern
    "Gene_Laminar_Curved",          # Curved laminar structure
    "Gene_Mosaic_Irregular",        # Irregular mosaic
    "Gene_Gradient_Sigmoid",        # Sigmoid transition
    "Gene_Bimodal_Distribution",    # Two distinct regions
    "Gene_Punctate_Dense",          # Dense punctate pattern
    "Gene_Periventricular",         # Around a central void
    "Gene_Asymmetric_Lobe",         # Asymmetric lobed structure
]

OUTPUT_FOLDER = f'./rotation_analysis_optimized_{len(AAD_TARGET_GENES)}_{ANGLE_START}_{ANGLE_END}_{ANGLE_STEP}'


# =============================================================================
# OPTIMIZED UTILITY FUNCTIONS
# =============================================================================

def rotate_coords_batch(coords: np.ndarray, angles_degrees: np.ndarray) -> Dict[int, np.ndarray]:
    """Pre-compute rotations for all angles at once."""
    center = coords.mean(axis=0)
    centered = coords - center
    rotated = {}
    
    for angle in angles_degrees:
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array([[c, -s], [s, c]])
        rotated[angle] = (centered @ rotation_matrix.T) + center
    
    return rotated


def align_bounds(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    """Vectorized bounds alignment."""
    s_min, s_max = source_coords.min(axis=0), source_coords.max(axis=0)
    t_min, t_max = target_coords.min(axis=0), target_coords.max(axis=0)
    s_range = s_max - s_min + 1e-8
    t_range = t_max - t_min
    # ◆ FIX 2: Was: scale = t_range / s_range (independent x/y scaling, distorts geometry)
    #   Now: uniform isotropic scale preserves aspect ratio and all distance-based metrics
    scale = min(t_range[0] / s_range[0], t_range[1] / s_range[1])  # ◆ FIX 2
    t_center = (t_min + t_max) / 2  # ◆ FIX 2: center the result in the target bbox
    s_centered = source_coords - (s_min + s_max) / 2  # ◆ FIX 2
    return s_centered * scale + t_center  # ◆ FIX 2


def compute_value_histogram_fast(values: np.ndarray, n_bins: int = 50) -> np.ndarray:
    """Optimized histogram computation."""
    hist, _ = np.histogram(values, bins=n_bins, density=False)
    return hist / (hist.sum() + 1e-8) if hist.sum() > 0 else hist


def compute_spatial_histogram_fast(coords: np.ndarray, values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Vectorized spatial histogram using numpy binning."""
    # Normalize coordinates
    coord_min = coords.min(axis=0)
    coord_range = coords.max(axis=0) - coord_min + 1e-8
    norm = (coords - coord_min) / coord_range
    
    # Compute bin indices
    x_bins = np.clip((norm[:, 0] * n_bins).astype(int), 0, n_bins - 1)
    y_bins = np.clip((norm[:, 1] * n_bins).astype(int), 0, n_bins - 1)
    
    # Use numpy's bincount for fast accumulation
    flat_idx = y_bins * n_bins + x_bins
    hist_sum = np.bincount(flat_idx, weights=values, minlength=n_bins * n_bins)
    hist_count = np.bincount(flat_idx, minlength=n_bins * n_bins)
    
    hist = np.divide(hist_sum, hist_count, where=hist_count > 0, out=np.zeros_like(hist_sum, dtype=float))
    
    # Normalize
    h_min, h_max = hist.min(), hist.max()
    if h_max > h_min:
        hist = (hist - h_min) / (h_max - h_min)
    
    return hist


def compute_radial_profile_fast(coords: np.ndarray, values: np.ndarray, n_rings: int = 10) -> np.ndarray:
    """Vectorized radial profile computation."""
    coord_min = coords.min(axis=0)
    norm = (coords - coord_min) / (coords.max(axis=0) - coord_min + 1e-8)
    centroid = norm.mean(axis=0)
    
    dists = np.linalg.norm(norm - centroid, axis=1)
    max_d = dists.max() + 1e-8
    
    ring_idx = np.clip((dists / max_d * n_rings).astype(int), 0, n_rings - 1)
    
    prof_sum = np.bincount(ring_idx, weights=values, minlength=n_rings)
    prof_count = np.bincount(ring_idx, minlength=n_rings)
    
    prof = np.divide(prof_sum, prof_count, where=prof_count > 0, out=np.zeros_like(prof_sum, dtype=float))
    p_min, p_max = prof.min(), prof.max()
    if p_max > p_min:
        prof = (prof - p_min) / (p_max - p_min)
    
    return prof


def compute_quadrant_stats_fast(coords: np.ndarray, values: np.ndarray, n_div: int = 3) -> np.ndarray:
    """Vectorized quadrant statistics."""
    coord_min = coords.min(axis=0)
    coord_range = coords.max(axis=0) - coord_min + 1e-8
    norm = (coords - coord_min) / coord_range
    
    x_bins = np.clip((norm[:, 0] * n_div).astype(int), 0, n_div - 1)
    y_bins = np.clip((norm[:, 1] * n_div).astype(int), 0, n_div - 1)
    
    stats = []
    for qx in range(n_div):
        for qy in range(n_div):
            mask = (x_bins == qx) & (y_bins == qy)
            if mask.sum() > 0:
                stats.extend([np.mean(values[mask]), np.std(values[mask])])
            else:
                stats.extend([0, 0])
    
    stats = np.array(stats)
    s_min, s_max = stats.min(), stats.max()
    if s_max > s_min:
        stats = (stats - s_min) / (s_max - s_min)
    
    return stats


def compute_morans_i_fast(values: np.ndarray, nn_indices: np.ndarray) -> float:
    """
    Fast Moran's I using precomputed kNN indices.
    Assumes nn_indices includes self as first column.
    """
    n = len(values)
    mean_v = values.mean()
    dev = values - mean_v

    denom = np.dot(dev, dev)
    if denom == 0:
        return 0.0

    # Neighbor deviations (exclude self)
    neighbor_devs = dev[nn_indices[:, 1:]]

    # Vectorized numerator
    numer = np.sum(dev[:, None] * neighbor_devs)

    # Total weight (exact count of edges)
    w_sum = neighbor_devs.size

    if w_sum == 0:
        return 0.0

    return (n / w_sum) * (numer / denom)


def compute_node_importance_fast(values: np.ndarray, nn_indices: np.ndarray) -> np.ndarray:
    neighbor_vals = values[nn_indices[:, 1:]]
    loc_var = np.var(neighbor_vals, axis=1)

    lv_min, lv_max = loc_var.min(), loc_var.max()
    if lv_max > lv_min:
        loc_var = (loc_var - lv_min) / (lv_max - lv_min)
    else:
        loc_var = np.zeros_like(loc_var)

    v_min, v_max = values.min(), values.max()
    if v_max > v_min:
        val_norm = (values - v_min) / (v_max - v_min)
    else:
        val_norm = np.zeros_like(values)

    return 0.5 * loc_var + 0.5 * val_norm


# =============================================================================
# CACHED SIGNATURE CLASS
# =============================================================================

@dataclass
class SpatialSignature:
    sample_id: str
    feature_name: str
    feature_type: str
    node_importance: np.ndarray
    value_histogram: np.ndarray
    spatial_histogram: np.ndarray
    radial_profile: np.ndarray
    quadrant_stats: np.ndarray
    morans_i: float
    coordinates: np.ndarray
    raw_values: np.ndarray
    aligned_coordinates: Optional[np.ndarray] = None
    
    # Pre-computed grids for fast correlation
    grid_values: Optional[np.ndarray] = None
    grid_importance: Optional[np.ndarray] = None


def fast_pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    """Faster Pearson correlation without p-value computation."""
    a = a.flatten()
    b = b.flatten()
    
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    
    numer = np.dot(a_centered, b_centered)
    denom = np.sqrt(np.dot(a_centered, a_centered) * np.dot(b_centered, b_centered))
    
    if denom == 0:
        return 0.0
    
    return numer / denom


# ◆ SPEED: Vectorized Pearson correlation — one gene descriptor vs ALL m/z descriptors at once
def fast_pearsonr_batch(gene_vec: np.ndarray, mz_matrix: np.ndarray) -> np.ndarray:
    """
    ◆ SPEED: Compute Pearson correlation between one gene vector and each row
    of mz_matrix (n_mz × len) in a single vectorized operation.
    Returns array of shape (n_mz,).
    """
    g = gene_vec.flatten().astype(np.float64)
    M = mz_matrix.astype(np.float64)  # (n_mz, D)

    g_c = g - g.mean()
    M_c = M - M.mean(axis=1, keepdims=True)

    g_norm = np.sqrt(np.dot(g_c, g_c))
    M_norms = np.sqrt(np.sum(M_c ** 2, axis=1))

    denom = g_norm * M_norms
    # Avoid division by zero
    safe = denom > 0
    result = np.zeros(M.shape[0])
    if g_norm > 0:
        dots = M_c @ g_c  # (n_mz,)
        result[safe] = dots[safe] / denom[safe]

    return result


# ◆ SPEED: Replaced per-m/z Python loop with fully vectorized scoring
def compute_all_scores_vectorized(
    gene_coords: np.ndarray,
    gene_values: np.ndarray,
    gene_importance: np.ndarray,
    gene_descriptors: Dict[str, np.ndarray],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mz_names: List[str],
    mz_grid_values_stack: np.ndarray,     # (n_mz, grid_res, grid_res)
    mz_grid_importance_stack: np.ndarray,  # (n_mz, grid_res, grid_res)
    mz_value_hist_stack: np.ndarray,       # (n_mz, n_bins)
    mz_spatial_hist_stack: np.ndarray,     # (n_mz, n_bins*n_bins)
    mz_radial_stack: np.ndarray,           # (n_mz, n_rings)
    mz_quadrant_stack: np.ndarray,         # (n_mz, n_div*n_div*2)
    mz_morans_array: np.ndarray,           # (n_mz,)
    gene_interp_func                        # ◆ SPEED: reusable interpolator
) -> Dict[str, Dict[str, object]]:
    """
    ◆ SPEED: Compute scores for ONE gene against ALL m/z features at once
    using matrix operations instead of a Python loop over m/z.
    """
    n_mz = len(mz_names)
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # ◆ SPEED: Use prebuilt interpolator instead of calling griddata each time
    g_grid_flat = gene_interp_func(grid_pts)
    g_grid = g_grid_flat.reshape(grid_x.shape)

    g_imp_interp = LinearNDInterpolator(
        Delaunay(gene_coords), gene_importance, fill_value=np.nan
    )
    g_imp_grid = g_imp_interp(grid_pts).reshape(grid_x.shape)

    # --- Value Correlation: gene grid vs each m/z grid ---
    g_flat = g_grid.ravel()
    g_imp_flat = g_imp_grid.ravel()

    val_corrs = np.zeros(n_mz)
    iou_scores = np.zeros(n_mz)
    imp_corrs = np.zeros(n_mz)

    g_nan_mask = np.isnan(g_flat)
    g_imp_nan_mask = np.isnan(g_imp_flat)

    for i in range(n_mz):
        m_flat = mz_grid_values_stack[i].ravel()
        mask = ~(g_nan_mask | np.isnan(m_flat))
        if mask.sum() > 10:
            val_corrs[i] = fast_pearsonr(g_flat[mask], m_flat[mask])

        m_imp_flat = mz_grid_importance_stack[i].ravel()
        mask_imp = ~(g_imp_nan_mask | np.isnan(m_imp_flat))
        if mask_imp.sum() > 0:
            gi = np.nan_to_num(g_imp_flat[mask_imp], 0.0)
            mi = np.nan_to_num(m_imp_flat[mask_imp], 0.0)
            gi_max = gi.max()
            mi_max = mi.max()
            if gi_max > 0:
                gi = gi / gi_max
            if mi_max > 0:
                mi = mi / mi_max
            iou_scores[i] = np.minimum(gi, mi).sum() / (np.maximum(gi, mi).sum() + 1e-8)
            imp_corrs[i] = fast_pearsonr(gi, mi)

    # ◆ SPEED: Descriptor correlations — batch across all m/z at once
    vhist_corrs = fast_pearsonr_batch(gene_descriptors['value_hist'], mz_value_hist_stack)
    shist_corrs = fast_pearsonr_batch(gene_descriptors['spatial_hist'], mz_spatial_hist_stack)
    radial_corrs = fast_pearsonr_batch(gene_descriptors['radial'], mz_radial_stack)
    quad_corrs = fast_pearsonr_batch(gene_descriptors['quadrant'], mz_quadrant_stack)
    morans_sims = np.maximum(0.0, 1.0 - np.abs(gene_descriptors['morans_i'] - mz_morans_array))

    # --- Find best m/z per metric ---
    metric_arrays = {
        'Value_Correlation': val_corrs,
        'Importance_IoU': iou_scores,
        'Importance_Correlation': imp_corrs,
        'Value_Hist_Corr': vhist_corrs,
        'Spatial_Hist_Corr': shist_corrs,
        'Radial_Corr': radial_corrs,
        'Quadrant_Corr': quad_corrs,
        'Morans_Sim': morans_sims,
    }

    best_per_metric = {}
    for k, arr in metric_arrays.items():
        idx = np.argmax(arr)
        best_per_metric[k] = {'mz': mz_names[idx], 'score': arr[idx]}

    return best_per_metric


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class OptimizedRotationAnalyzer:
    def __init__(self, output_dir: str = OUTPUT_FOLDER, n_workers: int = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.rna = {}
        self.msi = {}
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        
        # Caches
        self.msi_signatures: Dict[str, Dict[str, SpatialSignature]] = {}  # sample -> {mz_name -> sig}
        self.msi_grids: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}  # sample -> {mz_name -> grids}
        self.msi_nn_indices: Dict[str, np.ndarray] = {}  # sample -> nn_indices
        self.rna_nn_indices: Dict[str, np.ndarray] = {}
        self.common_grids: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # sample -> (grid_x, grid_y)
        
        # ◆ SPEED: Stacked m/z descriptor arrays for vectorized correlation
        self.msi_grid_values_stack: Dict[str, np.ndarray] = {}
        self.msi_grid_importance_stack: Dict[str, np.ndarray] = {}
        self.msi_value_hist_stack: Dict[str, np.ndarray] = {}
        self.msi_spatial_hist_stack: Dict[str, np.ndarray] = {}
        self.msi_radial_stack: Dict[str, np.ndarray] = {}
        self.msi_quadrant_stack: Dict[str, np.ndarray] = {}
        self.msi_morans_array: Dict[str, np.ndarray] = {}
        self.msi_mz_names: Dict[str, List[str]] = {}

        # ◆ SPEED: Cached dense RNA matrices and gene index maps
        self.rna_dense: Dict[str, np.ndarray] = {}
        self.rna_gene_idx: Dict[str, Dict[str, int]] = {}
        self.rna_coords: Dict[str, np.ndarray] = {}

        self.grid_res = 50

    def load_data(self):
        print("\n=== LOADING DATA ===")
        print(f"Checking RNA Folder: {RNA_INPUT_FOLDER}")
        if not os.path.exists(RNA_INPUT_FOLDER):
            print("!!! ERROR: RNA folder path does not exist!")
            
        loaded_rna = 0
        for f, s in zip(RNA_SAMPLE_FILES, RNA_SAMPLE_IDS):
            p = os.path.join(RNA_INPUT_FOLDER, f)
            if os.path.exists(p): 
                self.rna[s] = sc.read_h5ad(p)
                loaded_rna += 1
            else:
                print(f"  Missing file: {f}")
        print(f"Successfully loaded {loaded_rna}/{len(RNA_SAMPLE_IDS)} RNA files.")

        print(f"\nChecking MSI Folder: {MSI_INPUT_FOLDER}")
        if not os.path.exists(MSI_INPUT_FOLDER):
            print("!!! ERROR: MSI folder path does not exist!")
            
        loaded_msi = 0
        for f, s in zip(MSI_SAMPLE_FILES, MSI_SAMPLE_IDS):
            p = os.path.join(MSI_INPUT_FOLDER, f)
            if os.path.exists(p): 
                self.msi[s] = sc.read_h5ad(p)
                loaded_msi += 1
            else:
                print(f"  Missing file: {f}")
        print(f"Successfully loaded {loaded_msi}/{len(MSI_SAMPLE_IDS)} MSI files.")

        # ◆ SPEED: Pre-densify RNA matrices and build gene index maps once
        print("\n  Pre-densifying RNA matrices...")
        for s, adata in self.rna.items():
            if hasattr(adata.X, 'toarray'):
                self.rna_dense[s] = adata.X.toarray()
            else:
                self.rna_dense[s] = np.array(adata.X)
            self.rna_gene_idx[s] = {g: i for i, g in enumerate(adata.var_names)}
            self.rna_coords[s] = np.column_stack([
                adata.obs['x_um'].values, adata.obs['y_um'].values
            ])
        print("  Done.")

    def precompute_msi_signatures(self):
        """
        MAJOR OPTIMIZATION: Pre-compute all MSI signatures once.
        MSI data doesn't change with rotation, so we cache everything.
        """
        print("\n=== PRE-COMPUTING MSI SIGNATURES (one-time cost) ===")
        
        for sample_id, m_adata in self.msi.items():
            print(f"  Processing MSI sample: {sample_id}")
            
            m_coords = np.column_stack([m_adata.obs['x_um'].values, m_adata.obs['y_um'].values])
            m_vals_all = m_adata.X.toarray() if hasattr(m_adata.X, 'toarray') else m_adata.X
            mz_names = list(m_adata.var_names)
            
            # ---- NORMALIZE MSI COORDINATES ----
            coord_min = m_coords.min(axis=0)
            coord_range = m_coords.max(axis=0) - coord_min + 1e-8
            norm_coords = (m_coords - coord_min) / coord_range

            # Pre-compute nearest neighbors for this sample
            # ---- BUILD kNN ON NORMALIZED SPACE ----
            k = 8
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(norm_coords)
            _, nn_indices = nn.kneighbors(norm_coords)

            self.msi_nn_indices[sample_id] = nn_indices
            
            # Pre-compute common grid for this sample
            x_min, x_max = m_coords[:, 0].min(), m_coords[:, 0].max()
            y_min, y_max = m_coords[:, 1].min(), m_coords[:, 1].max()
            gx, gy = np.meshgrid(
                np.linspace(x_min, x_max, self.grid_res),
                np.linspace(y_min, y_max, self.grid_res)
            )
            self.common_grids[sample_id] = (gx, gy)
            
            self.msi_signatures[sample_id] = {}
            self.msi_grids[sample_id] = {}

            # ◆ SPEED: Collect per-mz arrays into lists, then stack
            grid_val_list = []
            grid_imp_list = []
            vhist_list = []
            shist_list = []
            radial_list = []
            quad_list = []
            morans_list = []
            
            for i, mz_name in enumerate(mz_names):
                vals = m_vals_all[:, i]
                
                # Compute signature
                importance = compute_node_importance_fast(vals, nn_indices)
                
                vhist = compute_value_histogram_fast(vals)
                shist = compute_spatial_histogram_fast(m_coords, vals)
                radial = compute_radial_profile_fast(m_coords, vals)
                quadrant = compute_quadrant_stats_fast(m_coords, vals)
                morans = compute_morans_i_fast(vals, nn_indices)
                
                sig = SpatialSignature(
                    sample_id=sample_id,
                    feature_name=mz_name,
                    feature_type='mz',
                    node_importance=importance,
                    value_histogram=vhist,
                    spatial_histogram=shist,
                    radial_profile=radial,
                    quadrant_stats=quadrant,
                    morans_i=morans,
                    coordinates=m_coords,
                    raw_values=vals
                )
                
                self.msi_signatures[sample_id][mz_name] = sig
                
                # Pre-compute grids for fast interpolation later
                gv = griddata(m_coords, vals, (gx, gy), method='linear')
                gi = griddata(m_coords, importance, (gx, gy), method='linear')
                self.msi_grids[sample_id][mz_name] = {
                    'values': gv,
                    'importance': gi
                }

                # ◆ SPEED: Accumulate for stacking
                grid_val_list.append(gv)
                grid_imp_list.append(gi)
                vhist_list.append(vhist)
                shist_list.append(shist)
                radial_list.append(radial)
                quad_list.append(quadrant)
                morans_list.append(morans)

            # ◆ SPEED: Stack all m/z descriptors into arrays for vectorized ops
            self.msi_mz_names[sample_id] = mz_names
            self.msi_grid_values_stack[sample_id] = np.array(grid_val_list)      # (n_mz, grid_res, grid_res)
            self.msi_grid_importance_stack[sample_id] = np.array(grid_imp_list)
            self.msi_value_hist_stack[sample_id] = np.array(vhist_list)
            self.msi_spatial_hist_stack[sample_id] = np.array(shist_list)
            self.msi_radial_stack[sample_id] = np.array(radial_list)
            self.msi_quadrant_stack[sample_id] = np.array(quad_list)
            self.msi_morans_array[sample_id] = np.array(morans_list)
            
            print(f"    Cached {len(mz_names)} m/z signatures (stacked for vectorized scoring)")
        
        print("MSI pre-computation complete!")

    def precompute_rna_nn(self):
        """Pre-compute RNA nearest neighbors on normalized coordinates."""
        
        print("\n=== PRE-COMPUTING RNA NEAREST NEIGHBORS (NORMALIZED) ===")
        
        for sample_id, r_adata in self.rna.items():
            print(f"  Processing RNA sample: {sample_id}")
            
            r_coords = self.rna_coords[sample_id]  # ◆ SPEED: use cached coords
            
            # ---- NORMALIZE COORDINATES HERE ----
            coord_min = r_coords.min(axis=0)
            coord_range = r_coords.max(axis=0) - coord_min + 1e-8
            norm_coords = (r_coords - coord_min) / coord_range
            
            # ---- BUILD kNN ON NORMALIZED SPACE ----
            k = 6  # or whatever you use
            nn = NearestNeighbors(n_neighbors=k + 1)
            nn.fit(norm_coords)
            _, nn_indices = nn.kneighbors(norm_coords)
            
            # Store indices
            self.rna_nn_indices[sample_id] = nn_indices
            
        print("RNA normalized kNN pre-computation complete!")

    def process_angle_gene_sample(
        self,
        angle: int,
        gene: str,
        sample_id: str,
    ) -> Optional[Dict]:
        """Process a single (angle, gene, sample) combination."""
        
        if sample_id not in self.msi:
            return None
        
        # ◆ SPEED: Use cached dense matrix and index map instead of repeated .toarray() + list().index()
        r_coords = self.rna_coords[sample_id]
        g_idx = self.rna_gene_idx[sample_id][gene]
        g_vals = self.rna_dense[sample_id][:, g_idx]
        
        # Rotate coordinates
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        center = r_coords.mean(axis=0)
        centered = r_coords - center
        rotation_matrix = np.array([[c, -s], [s, c]])
        rot_coords = (centered @ rotation_matrix.T) + center
        
        # Align to MSI bounds
        m_coords = self.msi_signatures[sample_id][self.msi_mz_names[sample_id][0]].coordinates  # ◆ SPEED: direct list access
        aligned = align_bounds(rot_coords, m_coords)
        
        # Compute gene importance (using original coords for NN structure)
        nn_indices = self.rna_nn_indices[sample_id]
        importance = compute_node_importance_fast(g_vals, nn_indices)
        
        # Compute gene descriptors (these depend on aligned coordinates for spatial patterns)
        gene_descriptors = {
            'value_hist': compute_value_histogram_fast(g_vals),
            'spatial_hist': compute_spatial_histogram_fast(aligned, g_vals),
            'radial': compute_radial_profile_fast(aligned, g_vals),
            'quadrant': compute_quadrant_stats_fast(aligned, g_vals),
            'morans_i': compute_morans_i_fast(g_vals, nn_indices)
        }
        
        # Get common grid
        grid_x, grid_y = self.common_grids[sample_id]

        # ◆ SPEED: Build a reusable LinearNDInterpolator for gene values
        #   (replaces two separate griddata calls inside the scoring function)
        try:
            tri = Delaunay(aligned)
            gene_interp = LinearNDInterpolator(tri, g_vals, fill_value=np.nan)
        except Exception:
            # Fallback if Delaunay fails (degenerate point set)
            gene_interp = lambda pts: griddata(aligned, g_vals, pts, method='linear')

        # ◆ SPEED: Call vectorized scoring — processes ALL m/z at once
        best_per_metric = compute_all_scores_vectorized(
            gene_coords=aligned,
            gene_values=g_vals,
            gene_importance=importance,
            gene_descriptors=gene_descriptors,
            grid_x=grid_x,
            grid_y=grid_y,
            mz_names=self.msi_mz_names[sample_id],
            mz_grid_values_stack=self.msi_grid_values_stack[sample_id],
            mz_grid_importance_stack=self.msi_grid_importance_stack[sample_id],
            mz_value_hist_stack=self.msi_value_hist_stack[sample_id],
            mz_spatial_hist_stack=self.msi_spatial_hist_stack[sample_id],
            mz_radial_stack=self.msi_radial_stack[sample_id],
            mz_quadrant_stack=self.msi_quadrant_stack[sample_id],
            mz_morans_array=self.msi_morans_array[sample_id],
            gene_interp_func=gene_interp
        )
        
        # Build result row
        metric_keys = list(best_per_metric.keys())
        row = {
            'Rotation_Angle': angle,
            'Gene': gene,
            'Sample': sample_id
        }
        for k in metric_keys:
            row[f'Best_MZ_{k}'] = best_per_metric[k]['mz']
            row[f'Max_{k}'] = best_per_metric[k]['score']
        
        return row

    def run(self):
        print("\n=== STARTING OPTIMIZED ROTATION ANALYSIS ===")
        
        if not self.rna:
            print("!!! STOPPING: No RNA data loaded.")
            return
        
        # Pre-compute everything
        self.precompute_rna_nn()
        self.precompute_msi_signatures()
        
        # Check gene availability
        gene_avail = {}
        for g in AAD_TARGET_GENES:
            found_in_samples = {s: g in self.rna[s].var_names for s in RNA_SAMPLE_IDS if s in self.rna}
            gene_avail[g] = found_in_samples
            count = sum(found_in_samples.values())
            print(f"Gene '{g}' found in {count} samples.")
        
        total_genes_found = sum(sum(v.values()) for v in gene_avail.values())
        if total_genes_found == 0:
            print("!!! STOPPING: None of the target genes were found in the loaded data.")
            return
        
        # Build angles list
        if ANGLE_START <= ANGLE_END:
            angles = list(range(ANGLE_START, ANGLE_END + ANGLE_STEP, ANGLE_STEP))
        else:
            angles = list(range(ANGLE_START, 360, ANGLE_STEP)) + \
                     list(range(0, ANGLE_END + ANGLE_STEP, ANGLE_STEP))

        # Build work items
        work_items = []
        for angle in angles:
            for gene in AAD_TARGET_GENES:
                samples = [s for s, a in gene_avail[gene].items() if a]
                for sample_id in samples:
                    if sample_id in self.msi:
                        work_items.append((angle, gene, sample_id))
        
        print(f"\nTotal work items: {len(work_items)}")

        # ◆ SPEED: Process sequentially — no ThreadPoolExecutor overhead
        #   (NumPy releases the GIL for some ops, but the Python-level overhead
        #    of pickling/scheduling dominates for these fine-grained tasks.
        #    If you want true parallelism, use ProcessPoolExecutor with chunked angles.)
        results = []
        
        for angle in angles:
            t0 = time.time()  # ◆ SPEED: timing per angle
            angle_items = [(a, g, s) for a, g, s in work_items if a == angle]
            
            for a, g, s in angle_items:
                r = self.process_angle_gene_sample(a, g, s)
                if r is not None:
                    results.append(r)

            dt = time.time() - t0
            print(f"  Angle {angle:3d}°  —  {len(angle_items)} items in {dt:.1f}s")
        
        if results:
            df = pd.DataFrame(results)
            out_path = os.path.join(self.output_dir, 'rotation_scores_optimized.csv')
            df.to_csv(out_path, index=False)
            print(f"\nSUCCESS: Saved optimized scores to: {out_path}")
            print(f"Rows generated: {len(df)}")
        else:
            print("\n!!! WARNING: Analysis finished but NO matches were found.")


if __name__ == "__main__":
    analyzer = OptimizedRotationAnalyzer()
    analyzer.load_data()
    analyzer.run()
