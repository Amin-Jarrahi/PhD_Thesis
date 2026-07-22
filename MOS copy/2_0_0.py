"""
Combined Pipeline: ML Calibration → m/z Isotope Detection & Hierarchy
======================================================================
Phase 1: Cross-validated calibration of spatial similarity weights using known positive and negative m/z pairs.
Phase 2: m/z-to-m/z matching for isotope/adduct detection using calibrated weights.
Phase 3: Hierarchical organization of identified isotope/adduct candidates.
"""

import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from typing import Dict, Optional
import pandas as pd
import os
import warnings
from dataclasses import dataclass
from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# SHARED CONFIGURATION
# =============================================================================

MSI_PIXEL_SIZE = 60  # μm

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

# Mass differences for isotope and adduct detection
MASS_DIFFS = {
    'Isotope (M+1)': 1.0033,
    'Isotope (M+2)': 2.0067,
    'Adduct (NH4)': 17.0265,
    'Adduct (Na)': 21.982,
    'Adduct (K)': 37.9555
}

MASS_DIFF_TOLERANCE = 0.01

# Isotope identification thresholds
MIN_ANIMALS = 12
MIN_SCORE = 60

# Output directory
OUTPUT_DIR = './mz_isotope_results'

# Stage 3 configuration
HIERARCHY_PRECISION = 4
HIERARCHY_TOLERANCE = 0.01

# =============================================================================
# CALIBRATION CONFIGURATION
# =============================================================================

POSITIVE_CONTROLS = [
    ("760.5856", "761.5864"), ("788.6156", "789.6178"),
    ("810.5997", "811.6054"), ("792.632", "793.5559"),
    ("813.6858", "814.6872"), ("524.3706", "525.3731"),
    ("522.3557", "523.3569"), ("734.5686", "735.572"),
    ("947.6513", "948.6564"), ("560.3341", "561.3381"),
    ("971.6514", "972.6575"), ("504.3442", "506.3606"),
    ("484.2952", "485.3011"), ("627.534", "628.5358"),
    ("455.2904", "456.2954"), ("455.2904", "457.3071"),
    ("666.4319", "667.4356"), ("678.4684", "679.4734"),
    ("703.574", "704.5774"), ("706.5388", "707.5393")
]

NEGATIVE_CONTROLS = [
    ("760.5856", "810.5997"), ("788.6156", "798.5951"),
    ("810.5997", "792.632"), ("792.632", "524.3706"),
    ("813.6858", "522.3557"), ("524.3706", "947.6513"),
    ("740.4725", "739.5955"), ("739.4661", "740.6021"),
    ("629.5422", "630.6165"), ("579.5304", "592.3947"),
    ("759.6355", "760.5856"), ("772.6211", "773.5294"),
    ("788.6156", "797.5895"), ("840.641", "842.6652"),
    ("848.5591", "848.6185"), ("848.5591", "848.6907"),
    ("868.6629", "870.6936"), ("971.6514", "971.6854")
]

N_FOLDS = 10

ORDERED_FEATURES = [
    'intensity_ratio_consistency', 'value_correlation', 'peak_colocalization',
    'importance_iou', 'importance_correlation', 'spatial_hist_corr',
    'value_iou', 'morans_similarity', 'radial_profile_corr'
]


# =============================================================================
# CALIBRATION: CROSS-VALIDATED ML CALIBRATOR
# =============================================================================

@dataclass
class CalibrationSignature:
    """Lightweight signature used only during weight calibration."""
    mz: str
    values: np.ndarray
    histogram: np.ndarray
    importance: np.ndarray
    morans: float
    radial_profile: np.ndarray  


def get_9_metrics(s1: CalibrationSignature, s2: CalibrationSignature) -> dict:
    """Compute all 9 spatial similarity metrics between two calibration signatures."""
    # 1. Intensity Ratio Consistency
    mask = (s1.values > np.percentile(s1.values, 10)) & (s2.values > np.percentile(s2.values, 10))
    r_con = 0
    if mask.sum() > 10:
        rats = np.minimum(s1.values[mask], s2.values[mask]) / (np.maximum(s1.values[mask], s2.values[mask]) + 1e-8)
        r_con = 1 / (1 + (np.std(rats) / (np.mean(rats) + 1e-8)))

    mask = (s1.values > 0) | (s2.values > 0)
    v_corr = 0
    if mask.sum() > 10:
        v_corr = max(0, pearsonr(s1.values[mask], s2.values[mask])[0])

    p1, p2 = s1.values >= np.percentile(s1.values, 80), s2.values >= np.percentile(s2.values, 80)
    peak = (p1 & p2).sum() / ((p1 | p2).sum() + 1e-8)

    imp1 = s1.importance / (s1.importance.max() + 1e-8)
    imp2 = s2.importance / (s2.importance.max() + 1e-8)
    imp_iou = np.minimum(imp1, imp2).sum() / (np.maximum(imp1, imp2).sum() + 1e-8)
    i_corr = max(0, pearsonr(s1.importance, s2.importance)[0])

    # Descriptor Metrics
    h_corr = max(0, pearsonr(s1.histogram.flatten(), s2.histogram.flatten())[0])
    val_iou = np.minimum(s1.values, s2.values).sum() / (np.maximum(s1.values, s2.values).sum() + 1e-8)
    m_sim = max(0, 1 - abs(s1.morans - s2.morans))

    def safe_pearsonr(a, b):
        if a.std() > 0 and b.std() > 0:
            r, _ = pearsonr(a, b)
            return r if not np.isnan(r) else 0
        return 0

    rad_corr = max(0, safe_pearsonr(s1.radial_profile, s2.radial_profile))

    return {
        'intensity_ratio_consistency': r_con,
        'value_correlation': v_corr,
        'peak_colocalization': peak,
        'importance_iou': imp_iou,
        'importance_correlation': i_corr,
        'spatial_hist_corr': h_corr,
        'value_iou': val_iou,
        'morans_similarity': m_sim,
        'radial_profile_corr': rad_corr
    }


class CrossValidatedCalibrator:
    def _compute_bio_importance(self, values: np.ndarray, nn_idx: np.ndarray) -> np.ndarray:
        """
        Shared importance computation matching the pipeline's logic.
        Blends normalized local variance (50%) with normalized value magnitude (50%).
        """
        neighbor_vals = values[nn_idx[:, 1:]]
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

    def get_sig(self, adata, mz, nn_idx, coords_normalized):
        """
        Build a CalibrationSignature using normalized coordinates.
        """
        v = adata[:, mz].X.toarray().flatten() if hasattr(adata.X, 'toarray') else adata[:, mz].X.flatten()

        hist = compute_spatial_histogram(coords_normalized, v)
        imp = self._compute_bio_importance(v, nn_idx)

        morans = compute_morans_i_vectorized(v, nn_idx)

        rad_profile = compute_radial_profile(coords_normalized, v)

        return CalibrationSignature(mz, v, hist, imp, morans, rad_profile)

    def run(self):
        """
       Run calibration and return the learned weights as a dict.

        Approach: Logistic Regression with ReLU rectification
        -----------------------------------------------------
        1. Fit L2-regularized logistic regression (no intercept).
        2. Rectify coefficients: w_i = max(0, coef_i).
           - Negative coefficients arise from collinearity, not true inverse
             relationships, because all 9 metrics are designed so higher = more
             co-localized. Zeroing them is the correct response.
        3. Normalize to sum to 1 → valid convex-combination weights.
        4. Evaluate AUC using the SAME rectified weights (no disconnect).

        Returns:
            dict mapping metric name → calibrated weight (sums to 1.0)
        """
        print("--- PHASE 1: COLLECTING SPATIAL DATA ACROSS ALL COHORTS ---")
        data_rows = []
        for file_name in tqdm(MSI_SAMPLE_FILES):
            path = os.path.join(MSI_INPUT_FOLDER, file_name)
            adata = sc.read_h5ad(path)
            coords = np.column_stack([adata.obs['x_um'], adata.obs['y_um']])

            # Normalize coordinates before k-NN (matching pipeline's _get_nn_indices)
            coord_min, coord_max = coords.min(axis=0), coords.max(axis=0)
            coords_normalized = (coords - coord_min) / (coord_max - coord_min + 1e-8)
            nn_idx = NearestNeighbors(n_neighbors=9).fit(coords_normalized).kneighbors(coords_normalized)[1]

            for m1, m2 in POSITIVE_CONTROLS + NEGATIVE_CONTROLS:
                if m1 in adata.var_names and m2 in adata.var_names:
                    s1 = self.get_sig(adata, m1, nn_idx, coords_normalized)
                    s2 = self.get_sig(adata, m2, nn_idx, coords_normalized)
                    metrics = get_9_metrics(s1, s2)
                    metrics['label'] = 1 if (m1, m2) in POSITIVE_CONTROLS else 0
                    metrics['sample_id'] = file_name
                    data_rows.append(metrics)

        df = pd.DataFrame(data_rows)
        samples = df['sample_id'].unique()

        print(f"\n--- PHASE 2: {N_FOLDS}-FOLD CROSS-VALIDATION (BY ANIMAL) ---")
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        fold_aucs = []
        fold_weights = []
        fold_n_active = []
        fold_n_rectified = []

        n_features = len(ORDERED_FEATURES)


        for fold_i, (train_idx, test_idx) in enumerate(kf.split(samples)):
            train_samples = samples[train_idx]
            test_samples  = samples[test_idx]

            X_train = df[df['sample_id'].isin(train_samples)][ORDERED_FEATURES].values
            y_train = df[df['sample_id'].isin(train_samples)]['label'].values.astype(float)
            X_test  = df[df['sample_id'].isin(test_samples)][ORDERED_FEATURES].values
            y_test  = df[df['sample_id'].isin(test_samples)]['label'].values.astype(float)

            # Step 1: Fit logistic regression (no intercept, L2 penalty)
            model = LogisticRegression(fit_intercept=False, penalty='l2', max_iter=1000)
            model.fit(X_train, y_train)
            raw_coeffs = model.coef_[0].copy()

            # Step 2: Rectify — zero out negative coefficients
            rectified = np.maximum(0.0, raw_coeffs)
            n_neg = int((raw_coeffs < 0).sum())
            fold_n_rectified.append(n_neg)

            # Step 3: Normalize to sum to 1
            total = rectified.sum()
            if total > 0:
                normalized_w = rectified / total
            else:
                # Fallback: all coefficients were negative (extremely unlikely)
                print(f"  ⚠ Fold {fold_i+1}: all coefficients negative, using uniform weights")
                normalized_w = np.ones(n_features) / n_features

            fold_weights.append(normalized_w)
            fold_n_active.append(int((normalized_w > 0).sum()))

            # Step 4: Evaluate AUC using the SAME rectified+normalized weights
            #   Score = X @ w  (weighted sum of metrics, all in [0,1])
            #   Convert to probability via sigmoid for AUC computation
            logits = X_test @ normalized_w
            y_probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            fold_aucs.append(roc_auc_score(y_test, y_probs))

        avg_weights = np.mean(fold_weights, axis=0)
        std_weights = np.std(fold_weights, axis=0)
        avg_auc     = np.mean(fold_aucs)

        self.print_report(
            ORDERED_FEATURES, avg_weights, std_weights, avg_auc,
            fold_n_active, fold_n_rectified
        )

        calibrated_weights = {
            name: float(w) for name, w in zip(ORDERED_FEATURES, avg_weights)
        }
        return calibrated_weights

    def print_report(self, names, avg_w, std_w, auc,
                            fold_n_active=None, fold_n_rectified=None):
        se_w = std_w / np.sqrt(N_FOLDS)

        print("\n" + "=" * 95)
        print("CALIBRATION REPORT: ROBUST SPATIAL WEIGHTS")
        print("=" * 95)
        print(f"{N_FOLDS}-Fold Cross-Validated AUC: {auc:.4f}")

        if fold_n_rectified is not None:
            mean_rect = np.mean(fold_n_rectified)
            print(f"Negative coefficients rectified per fold: "
                  f"{fold_n_rectified}  (mean: {mean_rect:.1f}/{len(names)})")
        if fold_n_active is not None:
            mean_active = np.mean(fold_n_active)
            print(f"Active metrics (weight > 0) per fold:     "
                  f"{fold_n_active}  (mean: {mean_active:.1f}/{len(names)})")

        print(f"\nValidated Weights (Fractional Mean [0-1] ± SD and SE):")
        print("-" * 95)
        print(f"{'Metric Name':<35} | {'Weight (0-1)':<14} | {'SD':<10} | {'Std. Error (SE)':<15}")
        print("-" * 95)
        for i in range(len(names)):
            print(f"{names[i]:<35} | {avg_w[i]:>12.4f}   | {std_w[i]:>8.4f}   | {se_w[i]:>12.4f}")
        print("-" * 95)
        print(f"{'TOTAL (Sum)':<35} | {np.sum(avg_w):>12.4f}   |")
        print("=" * 95)
        print(f"Justification: Logistic regression coefficients rectified (ReLU) to enforce")
        print(f"non-negativity, then normalized to sum to 1. Negative coefficients arise from")
        print(f"collinearity, not true inverse relationships, as all metrics are designed so")
        print(f"higher = more co-localized. AUC evaluated with the same rectified weights.")
        print("=" * 95)


# =============================================================================
# STAGE 1: SPATIAL PATTERN MATCHING FUNCTIONS
# =============================================================================

def compute_spatial_histogram(coords: np.ndarray, values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Compute a 2D spatial histogram of values. Normalizes coordinates internally,
    so works correctly with both raw and pre-normalized coordinates.
    """
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
    """
    Compute a radial intensity profile from centroid outward. Normalizes
    coordinates internally, so works with both raw and pre-normalized coordinates.
    """
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


def compute_morans_i_vectorized(values: np.ndarray, indices: np.ndarray) -> float:
    """
    Compute Moran's I spatial autocorrelation.
    """
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
    spatial_histogram: np.ndarray = None
    radial_profile: np.ndarray = None
    morans_i: float = 0.0
    coordinates: np.ndarray = None
    raw_values: np.ndarray = None


def compute_coordinate_based_similarity(sig1: SpatialSignature, sig2: SpatialSignature) -> dict:
    """
    Compute coordinate-based similarity between two signatures in the same
    coordinate space.
    """
    mask = (sig1.raw_values > 0) | (sig2.raw_values > 0)
    value_corr = 0
    if mask.sum() > 10:
        r, _ = pearsonr(sig1.raw_values[mask], sig2.raw_values[mask])
        value_corr = r if not np.isnan(r) else 0

    imp_corr = 0
    if len(sig1.node_importance) > 10:
        r, _ = pearsonr(sig1.node_importance, sig2.node_importance)
        imp_corr = r if not np.isnan(r) else 0

    imp1 = sig1.node_importance / (sig1.node_importance.max() + 1e-8)
    imp2 = sig2.node_importance / (sig2.node_importance.max() + 1e-8)
    importance_iou = np.minimum(imp1, imp2).sum() / (np.maximum(imp1, imp2).sum() + 1e-8)

    intensity_ratio_consistency = compute_intensity_ratio_consistency(sig1.raw_values, sig2.raw_values)
    peak_colocalization = compute_peak_colocalization(sig1.raw_values, sig2.raw_values)

    value_iou = np.minimum(sig1.raw_values, sig2.raw_values).sum() / (
        np.maximum(sig1.raw_values, sig2.raw_values).sum() + 1e-8)

    return {
        'value_correlation': value_corr,
        'importance_iou': importance_iou,
        'importance_correlation': imp_corr,
        'intensity_ratio_consistency': intensity_ratio_consistency,
        'peak_colocalization': peak_colocalization,
        'value_iou': value_iou,
    }


def compute_intensity_ratio_consistency(values1: np.ndarray, values2: np.ndarray, min_intensity_pct: float = 10) -> float:
    thresh1 = np.percentile(values1, min_intensity_pct)
    thresh2 = np.percentile(values2, min_intensity_pct)
    mask = (values1 > thresh1) & (values2 > thresh2)
    if mask.sum() < 10:
        return 0.0
    v1 = values1[mask]
    v2 = values2[mask]
    ratios = np.minimum(v1, v2) / (np.maximum(v1, v2) + 1e-8)
    ratio_mean = ratios.mean()
    ratio_std = ratios.std()
    if ratio_mean > 0:
        cv = ratio_std / ratio_mean
        consistency = 1 / (1 + cv)
    else:
        consistency = 0.0
    return consistency


def compute_peak_colocalization(values1: np.ndarray, values2: np.ndarray, top_pct: float = 20) -> float:
    thresh1 = np.percentile(values1, 100 - top_pct)
    thresh2 = np.percentile(values2, 100 - top_pct)
    peaks1 = values1 >= thresh1
    peaks2 = values2 >= thresh2
    intersection = (peaks1 & peaks2).sum()
    union = (peaks1 | peaks2).sum()
    if union > 0:
        return intersection / union
    return 0.0


def compute_descriptor_similarity(sig1: SpatialSignature, sig2: SpatialSignature) -> dict:
    """
    Compute descriptor-based similarity metrics.
    """
    def safe_pearsonr(a, b):
        if a.std() > 0 and b.std() > 0:
            r, _ = pearsonr(a, b)
            return r if not np.isnan(r) else 0
        return 0

    spatial_hist_corr = safe_pearsonr(sig1.spatial_histogram.flatten(), sig2.spatial_histogram.flatten())
    morans_sim = 1 - abs(sig1.morans_i - sig2.morans_i)
    radial_corr = max(0, safe_pearsonr(sig1.radial_profile, sig2.radial_profile))

    return {
        'spatial_hist_corr': spatial_hist_corr,
        'morans_similarity': morans_sim,
        'radial_profile_corr': radial_corr
    }


def compute_combined_score(coord_sim: dict, desc_sim: dict, weights: dict) -> float:
    """
    Compute combined score using dynamically calibrated weights.

    Parameters:
    -----------
    coord_sim : dict
        Coordinate-based similarity metrics.
    desc_sim : dict
        Descriptor-based similarity metrics (now includes radial_profile_corr).
    weights : dict
        Calibrated weights from CrossValidatedCalibrator (metric_name → weight).
    """
    score = (
        weights['intensity_ratio_consistency'] * max(coord_sim['intensity_ratio_consistency'], 0) +
        weights['value_correlation']           * max(coord_sim['value_correlation'], 0) +
        weights['peak_colocalization']         * max(coord_sim['peak_colocalization'], 0) +
        weights['importance_iou']              * coord_sim['importance_iou'] +
        weights['importance_correlation']      * max(coord_sim['importance_correlation'], 0) +
        weights['spatial_hist_corr']           * max(desc_sim['spatial_hist_corr'], 0) +
        weights['value_iou']                   * max(coord_sim['value_iou'], 0) +
        weights['morans_similarity']           * max(desc_sim['morans_similarity'], 0) +
        weights['radial_profile_corr']         * max(desc_sim['radial_profile_corr'], 0)
    )
    return score


def classify_mass_difference(mz_diff: float, tolerance: float = MASS_DIFF_TOLERANCE) -> str:
    for label, expected_diff in MASS_DIFFS.items():
        if abs(mz_diff - expected_diff) <= tolerance:
            return label
    return 'None'


class MzIsotopeMatcher:
    def __init__(self, calibrated_weights: dict, output_dir: str = './mz_isotope_matching_results', n_jobs: int = -1):
        self.calibrated_weights = calibrated_weights
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        os.makedirs(output_dir, exist_ok=True)
        self.msi_data = {}
        self._nn_cache = {}

    def load_all_data(self):
        print(f"Loading MSI data...")
        print(f"  Pixel size: {MSI_PIXEL_SIZE} μm (Cartesian)")
        for file, sample_id in zip(MSI_SAMPLE_FILES, MSI_SAMPLE_IDS):
            path = os.path.join(MSI_INPUT_FOLDER, file)
            if os.path.exists(path):
                self.msi_data[sample_id] = sc.read_h5ad(path)
                print(f"  {sample_id}: {self.msi_data[sample_id].shape}")
            else:
                print(f"  {sample_id}: NOT FOUND at {path}")

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

    def compute_bio_importance(self, values: np.ndarray, nn_indices: np.ndarray) -> np.ndarray:
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
                          feature_name: str, n_neighbors: int, nn_indices: np.ndarray) -> SpatialSignature:
        bio_imp = self.compute_bio_importance(values, nn_indices)
        return SpatialSignature(
            sample_id=sample_id, feature_name=feature_name, feature_type='mz',
            node_importance=bio_imp,
            spatial_histogram=compute_spatial_histogram(coords, values),
            radial_profile=compute_radial_profile(coords, values),
            morans_i=compute_morans_i_vectorized(values, nn_indices), 
            coordinates=coords, raw_values=values)

    def compute_pair_similarity(self, sig1: SpatialSignature, sig2: SpatialSignature) -> dict:
        """
        Compute all similarity metrics between two m/z signatures.
        """
        coord_sim = compute_coordinate_based_similarity(sig1, sig2)
        desc_sim = compute_descriptor_similarity(sig1, sig2)
        combined = compute_combined_score(coord_sim, desc_sim, self.calibrated_weights)

        try:
            mz1 = float(sig1.feature_name)
            mz2 = float(sig2.feature_name)
            mz_diff = abs(mz2 - mz1)
        except ValueError:
            mz_diff = np.nan

        return {
            'mz_1': sig1.feature_name,
            'mz_2': sig2.feature_name,
            'mz_difference': mz_diff,
            'sample_id': sig1.sample_id,
            **coord_sim,
            **desc_sim,
            'combined_score': combined,
            'mz_1_morans_i': sig1.morans_i,
            'mz_2_morans_i': sig2.morans_i
        }

    def run_analysis(self):
        print("\n" + "=" * 70)
        print("m/z-to-m/z MATCHING FOR ISOTOPE DETECTION")
        print(f"MSI: {MSI_PIXEL_SIZE}μm (Cartesian)")
        print(f"Mass differences: {list(MASS_DIFFS.keys())}")
        print(f"Tolerance: ±{MASS_DIFF_TOLERANCE} Da")
        print(f"Weights: dynamically calibrated from cross-validation")
        print("=" * 70)

        all_results = []

        for sample_id in tqdm(MSI_SAMPLE_IDS, desc="Samples", unit="sample"):
            if sample_id not in self.msi_data:
                print(f"\n{sample_id}: NOT LOADED, skipping")
                continue

            print(f"\n{'=' * 50}")
            print(f"Sample: {sample_id}")
            print(f"{'=' * 50}")

            msi_adata = self.msi_data[sample_id]
            msi_coords_raw = np.column_stack([msi_adata.obs['x_um'].values, msi_adata.obs['y_um'].values])
            coord_min, coord_max = msi_coords_raw.min(axis=0), msi_coords_raw.max(axis=0)
            coords_normalized = (msi_coords_raw - coord_min) / (coord_max - coord_min + 1e-8)
            msi_data = msi_adata.X.toarray() if hasattr(msi_adata.X, 'toarray') else msi_adata.X
            mz_names = list(msi_adata.var_names)
            n_mz = len(mz_names)

            print(f"  {n_mz} m/z features, {len(coords_normalized)} pixels")

            nn_indices = self._get_nn_indices(coords_normalized, 8, f"msi_{sample_id}")

            print(f"  Finding candidate isotope/adduct pairs...")

            mz_values = []
            for name in mz_names:
                try:
                    mz_values.append(float(name))
                except ValueError:
                    mz_values.append(np.nan)
            mz_values = np.array(mz_values)

            candidate_pairs = []
            for i in range(n_mz):
                if np.isnan(mz_values[i]):
                    continue
                for j in range(i + 1, n_mz):
                    if np.isnan(mz_values[j]):
                        continue
                    mz_diff = abs(mz_values[j] - mz_values[i])
                    for diff_type, expected_diff in MASS_DIFFS.items():
                        if abs(mz_diff - expected_diff) <= MASS_DIFF_TOLERANCE:
                            candidate_pairs.append((i, j, diff_type))
                            break

            n_candidates = len(candidate_pairs)
            print(f"  {n_candidates} candidate pairs (from {n_mz * (n_mz - 1) // 2} total possible)")

            if n_candidates == 0:
                print(f"  No candidate pairs found, skipping sample")
                continue

            involved_indices = set()
            for i, j, _ in candidate_pairs:
                involved_indices.add(i)
                involved_indices.add(j)
            involved_indices = sorted(involved_indices)

            print(f"  Extracting signatures for {len(involved_indices)} involved m/z features...")

            def extract_single_mz(i):
                return i, self.extract_signature(coords_normalized, msi_data[:, i], sample_id, mz_names[i], 8, nn_indices)

            sig_results = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                delayed(extract_single_mz)(i) for i in tqdm(involved_indices, desc="  Signatures", unit="mz"))

            signatures = {i: sig for i, sig in sig_results}
            print(f"  {len(signatures)} signatures extracted")

            print(f"  Computing self-scores...")
            self_scores = {}
            for i in involved_indices:
                self_score_result = self.compute_pair_similarity(signatures[i], signatures[i])
                self_scores[i] = self_score_result['combined_score']

            print(f"  Computing pairwise similarities...")

            def compute_pair(i, j, diff_type):
                result = self.compute_pair_similarity(signatures[i], signatures[j])
                result['mass_diff_type'] = diff_type
                result['mz_1_self_score'] = self_scores[i]
                result['mz_2_self_score'] = self_scores[j]
                max_self_score = max(self_scores[i], self_scores[j])
                result['score_percentage'] = (result['combined_score'] / max_self_score * 100) if max_self_score > 0 else 0
                return result

            pair_results = Parallel(n_jobs=self.n_jobs, prefer='threads')(
                delayed(compute_pair)(i, j, diff_type) for i, j, diff_type in tqdm(candidate_pairs, desc="  Pairs", unit="pair"))

            sample_results = pd.DataFrame(pair_results)
            all_results.append(sample_results)

            print(f"  Sample complete: {len(sample_results)} pairs scored")

        if all_results:
            print("\n" + "=" * 70)
            print("SAVING RESULTS")
            print("=" * 70)

            full_results = pd.concat(all_results, ignore_index=True)
            full_results = full_results.sort_values(['sample_id', 'mass_diff_type', 'combined_score'], ascending=[True, True, False])
            full_path = os.path.join(self.output_dir, 'mz_to_mz_isotope_candidates.csv')
            full_results.to_csv(full_path, index=False)
            print(f"  All candidate pairs: {full_path} ({len(full_results)} rows)")

            summary = []
            for sample_id in MSI_SAMPLE_IDS:
                if sample_id in self.msi_data:
                    sample_data = full_results[full_results['sample_id'] == sample_id]
                    summary.append({
                        'sample_id': sample_id,
                        'n_mz_features': len(self.msi_data[sample_id].var_names),
                        'n_pairs': len(sample_data),
                        'mean_combined_score': sample_data['combined_score'].mean(),
                        'max_combined_score': sample_data['combined_score'].max(),
                        'min_combined_score': sample_data['combined_score'].min(),
                        'std_combined_score': sample_data['combined_score'].std()
                    })

            summary_df = pd.DataFrame(summary)
            summary_path = os.path.join(self.output_dir, 'mz_matching_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"  Summary: {summary_path}")

            return full_results

        return None


# =============================================================================
# STAGE 2: ISOTOPE IDENTIFICATION FUNCTIONS
# =============================================================================

def identify_isotopes(df, min_animals=12, min_score=60):
    print(f"Total rows in dataset: {len(df)}")
    print(f"Unique samples: {df['sample_id'].nunique()}")
    print(f"Mass difference types: {df['mass_diff_type'].unique()}")

    df['mz_1'] = pd.to_numeric(df['mz_1'], errors='coerce')
    df['mz_2'] = pd.to_numeric(df['mz_2'], errors='coerce')
    df['score_percentage'] = pd.to_numeric(df['score_percentage'], errors='coerce')

    df['mz_1_rounded'] = df['mz_1'].round(4)
    df['mz_2_rounded'] = df['mz_2'].round(4)

    print(f"\nOriginal unique mz_1 values: {df['mz_1'].nunique()}")
    print(f"Rounded unique mz_1 values: {df['mz_1_rounded'].nunique()}")
    print(f"Original unique mz_2 values: {df['mz_2'].nunique()}")
    print(f"Rounded unique mz_2 values: {df['mz_2_rounded'].nunique()}")

    df_filtered = df[df['score_percentage'] > min_score].copy()
    print(f"\nRows with score_percentage > {min_score}: {len(df_filtered)}")

    df_filtered['pair_id'] = df_filtered.apply(
        lambda row: f"{row['mz_1_rounded']:.4f}_{row['mz_2_rounded']:.4f}_{row['mass_diff_type']}",
        axis=1
    )

    isotope_stats = []

    for pair_id, group in df_filtered.groupby('pair_id'):
        n_animals = group['sample_id'].nunique()

        if n_animals >= min_animals:
            stats = {
                'mz_1': group['mz_1_rounded'].iloc[0],
                'mz_2': group['mz_2_rounded'].iloc[0],
                'mz_difference': group['mz_difference'].mean(),
                'mass_diff_type': group['mass_diff_type'].iloc[0],
                'n_animals': n_animals,
                'mean_score_percentage': group['score_percentage'].mean(),
                'median_score_percentage': group['score_percentage'].median(),
                'min_score_percentage': group['score_percentage'].min(),
                'max_score_percentage': group['score_percentage'].max(),
                'std_score_percentage': group['score_percentage'].std(),
                'animals': ','.join(sorted(group['sample_id'].unique()))
            }
            isotope_stats.append(stats)

    results_df = pd.DataFrame(isotope_stats)

    if len(results_df) > 0:
        results_df = results_df.sort_values(
            ['n_animals', 'mean_score_percentage'],
            ascending=[False, False]
        ).reset_index(drop=True)

    return results_df, df_filtered


def save_results(results_df, output_file='identified_isotopes.csv'):
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


def print_summary(results_df, min_animals=12, min_score=60):
    print("\n" + "=" * 80)
    print(f"ISOTOPE IDENTIFICATION SUMMARY")
    print(f"Criteria: ≥{min_animals} animals AND score_percentage > {min_score}")
    print("=" * 80)

    if len(results_df) == 0:
        print("\nNo isotopes found meeting the criteria.")
        return

    print(f"\nTotal isotope pairs identified: {len(results_df)}")
    print(f"\nBreakdown by mass difference type:")
    for mass_type, count in results_df['mass_diff_type'].value_counts().items():
        print(f"  {mass_type}: {count}")

    print(f"\nBreakdown by number of animals:")
    for n_animals in sorted(results_df['n_animals'].unique(), reverse=True):
        count = (results_df['n_animals'] == n_animals).sum()
        print(f"  {n_animals} animals: {count} pairs")

    print(f"\nScore statistics for identified isotopes:")
    print(f"  Mean score_percentage: {results_df['mean_score_percentage'].mean():.2f}")
    print(f"  Median score_percentage: {results_df['median_score_percentage'].median():.2f}")
    print(f"  Range: {results_df['min_score_percentage'].min():.2f} - {results_df['max_score_percentage'].max():.2f}")

    print(f"\nTop 10 isotope pairs by mean score_percentage:")
    print("-" * 80)
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        mz1 = float(row['mz_1'])
        mz2 = float(row['mz_2'])
        print(f"{idx + 1}. m/z {mz1:.3f} → {mz2:.3f} ({row['mass_diff_type']})")
        print(f"   Animals: {row['n_animals']}/16 | Mean score: {row['mean_score_percentage']:.2f}%")
        print()


# =============================================================================
# STAGE 3: PARENT-CHILDREN HIERARCHY
# =============================================================================

def build_strict_hierarchy(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Could not find file at {path}")
        return None

    all_mzs = sorted(pd.concat([df['mz_1'], df['mz_2']]).unique())
    all_mzs = np.array([round(x, HIERARCHY_PRECISION) for x in all_mzs])

    rows = []
    assigned_as_child = set()

    for parent in all_mzs:
        if parent in assigned_as_child:
            continue

        row_dict = {'Parent_MZ': parent}
        found_any_child = False

        for i, (label, diff) in enumerate(MASS_DIFFS.items()):
            target_mz = parent + diff
            distances = np.abs(all_mzs - target_mz)
            candidates_mask = distances <= HIERARCHY_TOLERANCE

            if np.any(candidates_mask):
                indices = np.where(candidates_mask)[0]
                best_match_idx = indices[np.argmin(distances[indices])]
                best_match = all_mzs[best_match_idx]
                row_dict[f'Child_{i + 1}'] = best_match
                assigned_as_child.add(best_match)
                found_any_child = True
            else:
                row_dict[f'Child_{i + 1}'] = np.nan

        if found_any_child:
            rows.append(row_dict)

    if not rows:
        print("No matches found based on the provided MASS_DIFFS.")
        return None

    final_df = pd.DataFrame(rows)
    cols = ['Parent_MZ'] + [f'Child_{i + 1}' for i in range(len(MASS_DIFFS))]
    return final_df[cols]


# =============================================================================
# MAIN: CALIBRATION → STAGE 1 → STAGE 2 → STAGE 3
# =============================================================================

def main():
    print("=" * 70)
    print("COMBINED PIPELINE: ML CALIBRATION → ISOTOPE DETECTION & HIERARCHY")
    print(f"MSI: {MSI_PIXEL_SIZE}μm")
    print("=" * 70)

    # --- CALIBRATION PHASE: Learn optimal weights from controls ---
    print("\n" + "=" * 70)
    print("CALIBRATION PHASE: LEARNING OPTIMAL METRIC WEIGHTS")
    print("=" * 70)

    calibrator = CrossValidatedCalibrator()
    calibrated_weights = calibrator.run()

    print("\nCalibrated weights to be used in scoring:")
    for name, w in calibrated_weights.items():
        print(f"  {name}: {w:.4f}")

    # --- STAGE 1: Spatial pattern matching (using calibrated weights) ---
    matcher = MzIsotopeMatcher(
        calibrated_weights=calibrated_weights,
        output_dir=OUTPUT_DIR,
        n_jobs=-1
    )
    matcher.load_all_data()
    matching_results = matcher.run_analysis()

    if matching_results is None:
        print("\nNo matching results produced. Exiting.")
        return calibrated_weights, matcher, None, None, None

    # --- STAGE 2: Isotope identification ---
    print("\n" + "=" * 70)
    print("STAGE 2: ISOTOPE IDENTIFICATION")
    print("=" * 70)
    print(f"\nStarting isotope identification analysis...")

    results_df, filtered_df = identify_isotopes(
        matching_results,
        min_animals=MIN_ANIMALS,
        min_score=MIN_SCORE
    )

    print_summary(results_df, MIN_ANIMALS, MIN_SCORE)

    isotope_csv_path = None
    if len(results_df) > 0:
        isotope_csv_path = os.path.join(OUTPUT_DIR, 'identified_isotopes.csv')
        save_results(results_df, isotope_csv_path)

        detailed_output = isotope_csv_path.replace('.csv', '_detailed.csv')
        filtered_df.to_csv(detailed_output, index=False)
        print(f"Detailed data saved to {detailed_output}")

    # --- STAGE 3: Parent-children hierarchy ---
    print("\n" + "=" * 70)
    print("STAGE 3: PARENT-CHILDREN HIERARCHY")
    print("=" * 70)

    hierarchy_df = None
    if isotope_csv_path is not None and os.path.exists(isotope_csv_path):
        print(f"\nBuilding hierarchy from: {isotope_csv_path}")
        hierarchy_df = build_strict_hierarchy(isotope_csv_path)

        if hierarchy_df is not None:
            print(f"\nSuccessfully grouped {len(hierarchy_df)} parent-centered families.")
            print(f"Logic: Closest match within {HIERARCHY_TOLERANCE} Da selected for each pattern.")
            print(hierarchy_df.head().to_string(float_format=f"%.{HIERARCHY_PRECISION}f"))

            hierarchy_output_path = os.path.join(OUTPUT_DIR, 'parent_children_hierarchy.csv')
            hierarchy_df.to_csv(hierarchy_output_path, index=False, float_format=f"%.{HIERARCHY_PRECISION}f")
            print(f"\nFinal table saved to: {hierarchy_output_path}")
        else:
            print("\nNo hierarchy could be built from the identified isotopes.")
    else:
        print("\nNo identified isotopes CSV available. Skipping hierarchy construction.")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)

    return calibrated_weights, matcher, matching_results, results_df, hierarchy_df


if __name__ == "__main__":
    calibrated_weights, matcher, matching_results, isotope_results, hierarchy_results = main()
