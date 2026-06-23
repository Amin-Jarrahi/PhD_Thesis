import os
import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial.distance import cdist

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_DIR = "dummy_data_50"
RNA_DIR = os.path.join(OUTPUT_DIR, "rna")
MSI_DIR = os.path.join(OUTPUT_DIR, "msi")

GROUPS = ["YC", "YAD", "AC", "AAD"]
SAMPLES_PER_GROUP = 4

RNA_SPOT_DIAMETER = 55
RNA_SPOT_SPACING = 100
MSI_PIXEL_SIZE = 60

PATTERN_SIZE_SCALE_RANGE = (0.9, 1.1)

# =============================================================================
# NOISE DISTORTION CONTROLS (5 independent parameters, each 1-20)
# =============================================================================
# Each parameter controls how much one spatial property of noise variants
# deviates from the parent GT pattern.
#   1  = minimal deviation (noise very close to parent)
#  10  = moderate deviation
#  20  = extreme deviation (noise barely recognisable as the parent)
#
# All distortions use smooth continuous spatial functions so that RNA and MSI
# grids sampling the same underlying field stay perfectly correlated.
NOISE_SHAPE       = 8   # Sinusoidal coordinate warp — deforms the spatial structure
NOISE_LOCATION    = 8   # Center shift — translates where features appear
NOISE_SCALE       = 8   # Size/frequency change — stretches or compresses features
NOISE_FREQUENCY   = 8   # Additive spatial frequency field — overlays new spatial signal
NOISE_ORIENTATION = 8   # Rotation of the coordinate frame

# Number of noise variants per GT pattern.
# Total noise features = NOISE_PER_PATTERN * len(GT_PATTERNS)
NOISE_PER_PATTERN = 10

# 50 Ground Truth Patterns - organized by category
GT_PATTERNS = [
    # === GRADIENTS (6 patterns) ===
    "Gradient_X", "Gradient_Y", "Gradient_Diagonal_NE", "Gradient_Diagonal_NW",
    "Gradient_Radial_In", "Gradient_Radial_Out",
    # === WAVES & STRIPES (8 patterns) ===
    "Stripes_Vertical", "Stripes_Horizontal", "Stripes_Diagonal_45",
    "Stripes_Diagonal_135", "Waves_Concentric", "Waves_Spiral",
    "Waves_Interference", "Waves_Ripple",
    # === BLOBS & SPOTS (10 patterns) ===
    "Blob_Center", "Blob_TopRight", "Blob_TopLeft", "Blob_BottomRight",
    "Blob_BottomLeft", "Spots_Grid_Dense", "Spots_Grid_Sparse",
    "Spots_Random_Large", "Spots_Triangular", "Spots_Hexagonal",
    # === RINGS & DONUTS (6 patterns) ===
    "Ring_Inner", "Ring_Outer", "Ring_Double", "Ring_Eccentric",
    "Ring_Elliptical", "Ring_Partial",
    # === GEOMETRIC PATTERNS (8 patterns) ===
    "Checkerboard_Fine", "Checkerboard_Coarse", "Quadrant_Alternating",
    "Sectors_4", "Sectors_8", "Triangle_Pattern", "Diamond_Pattern", "Honeycomb",
    # === COMPLEX BIOLOGICAL-LIKE (12 patterns) ===
    "Cortical_Layers", "Hotspot_Cluster", "Edge_Enhancement", "Core_Shell",
    "Branching", "Laminar_Curved", "Mosaic_Irregular", "Gradient_Sigmoid",
    "Bimodal_Distribution", "Punctate_Dense", "Periventricular", "Asymmetric_Lobe",
]

TOTAL_NOISE_FEATURES = NOISE_PER_PATTERN * len(GT_PATTERNS)

os.makedirs(RNA_DIR, exist_ok=True)
os.makedirs(MSI_DIR, exist_ok=True)
np.random.seed(42)

FIELD_WIDTH  = 6000
FIELD_HEIGHT = 6000


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_halfbrain_shape(sample_seed, width=3000, height=3000):
    np.random.seed(sample_seed)
    brain_height = np.random.uniform(4900, 5400)
    hw_ratio = np.random.uniform(1.0, 1.2)
    brain_max_width = brain_height / hw_ratio
    brain_max_width = np.clip(brain_max_width, 4400, 5000)
    max_width_fraction = np.random.uniform(0.5, 0.9)
    rotation = np.random.uniform(-0.05, 0.05)
    offset_x = np.random.uniform(-50, 50)
    offset_y = np.random.uniform(-50, 50)
    left_edge = (width - brain_max_width) / 2 + offset_x
    bottom_edge = (height - brain_height) / 2 + offset_y
    bottom_width = brain_max_width * 0.70
    max_width_y = bottom_edge + brain_height * max_width_fraction
    top_edge = bottom_edge + brain_height

    def is_in_halfbrain(x, y):
        cx = left_edge + brain_max_width * 0.5
        cy = bottom_edge + brain_height * 0.5
        dx = x - cx; dy = y - cy
        x_rot = dx * np.cos(rotation) - dy * np.sin(rotation) + cx
        y_rot = dx * np.sin(rotation) + dy * np.cos(rotation) + cy
        if x_rot < left_edge: return False
        if y_rot < bottom_edge: return False
        if y_rot > top_edge: return False
        y_norm = (y_rot - bottom_edge) / brain_height
        y_norm = np.clip(y_norm, 0, 1)
        if y_norm <= max_width_fraction:
            t = y_norm / max_width_fraction
            right_width = bottom_width + (brain_max_width - bottom_width) * np.sin(t * np.pi / 2)
        else:
            t = (y_norm - max_width_fraction) / (1 - max_width_fraction)
            right_width = brain_max_width * np.sqrt(max(0, 1 - t**2))
        right_edge = left_edge + right_width
        if x_rot > right_edge: return False
        if y_norm > max_width_fraction:
            x_rel = (x_rot - left_edge) / brain_max_width
            x_rel = np.clip(x_rel, 0, 1)
            t = (y_norm - max_width_fraction) / (1 - max_width_fraction)
            if x_rel < 0.05: max_y_norm = 1.0
            else: max_y_norm = max_width_fraction + (1 - max_width_fraction) * np.sqrt(max(0, 1 - x_rel**2))
            if y_norm > max_y_norm: return False
        return True
    return is_in_halfbrain


def generate_pattern_values(coords, pattern_type, width=3000, height=3000,
                            sample_seed=None, add_noise=True,
                            size_scale_range=(0.70, 1.40)):
    x, y = coords[:, 0], coords[:, 1]
    n = len(x)
    val = np.zeros(n)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    cx, cy = x.mean(), y.mean()
    dist_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = dist_center.max() + 1e-8
    dist_norm = dist_center / max_dist
    angle = np.arctan2(y - cy, x - cx)
    pattern_seed = hash(pattern_type) % 2**31

    if sample_seed is not None:
        rng_s = np.random.RandomState(sample_seed ^ pattern_seed)
        dx_offset = rng_s.uniform(-0.15, 0.15) * width
        dy_offset = rng_s.uniform(-0.15, 0.15) * height
        phase_offset = rng_s.uniform(-1.0, 1.0)
        freq_jitter = rng_s.uniform(0.75, 1.25)
        size_scale = rng_s.uniform(size_scale_range[0], size_scale_range[1])
        amp_scale = rng_s.uniform(0.75, 1.05)
        bio_noise_std = rng_s.uniform(0.0, 0.10)
        cx_s = cx + dx_offset; cy_s = cy + dy_offset
        dist_center_s = np.sqrt((x - cx_s)**2 + (y - cy_s)**2)
        angle_s = np.arctan2(y - cy_s, x - cx_s)
    else:
        dx_offset = dy_offset = 0.0
        phase_offset = 0.0; freq_jitter = 1.0; size_scale = 1.0
        amp_scale = 1.0; bio_noise_std = 0.0
        cx_s, cy_s = cx, cy
        dist_center_s = dist_center; angle_s = angle

    if pattern_type == "Gradient_X": val = x_norm + phase_offset * 0.2
    elif pattern_type == "Gradient_Y": val = y_norm + phase_offset * 0.2
    elif pattern_type == "Gradient_Diagonal_NE": val = (x_norm + y_norm) / 2 + phase_offset * 0.15
    elif pattern_type == "Gradient_Diagonal_NW": val = (1 - x_norm + y_norm) / 2 + phase_offset * 0.15
    elif pattern_type == "Gradient_Radial_In": val = 1 - dist_norm + phase_offset * 0.1
    elif pattern_type == "Gradient_Radial_Out": val = dist_norm + phase_offset * 0.1
    elif pattern_type == "Stripes_Vertical":
        base_freq = 80 / freq_jitter
        val = (np.sin(x / base_freq + phase_offset * np.pi) + 1) / 2
    elif pattern_type == "Stripes_Horizontal":
        base_freq = 80 / freq_jitter
        val = (np.sin(y / base_freq + phase_offset * np.pi) + 1) / 2
    elif pattern_type == "Stripes_Diagonal_45":
        base_freq = 100 / freq_jitter
        val = (np.sin((x + y) / base_freq + phase_offset * np.pi) + 1) / 2
    elif pattern_type == "Stripes_Diagonal_135":
        base_freq = 100 / freq_jitter
        val = (np.sin((x - y) / base_freq + phase_offset * np.pi) + 1) / 2
    elif pattern_type == "Waves_Concentric":
        base_freq = 60 / freq_jitter
        val = (np.sin(dist_center_s / base_freq + phase_offset * np.pi) + 1) / 2
    elif pattern_type == "Waves_Spiral":
        spiral = angle_s + dist_center_s / (150 * freq_jitter) + phase_offset
        val = (np.sin(spiral * 3) + 1) / 2
    elif pattern_type == "Waves_Interference":
        d1 = np.sqrt((x - (width*0.3 + dx_offset))**2 + (y - (height*0.5 + dy_offset))**2)
        d2 = np.sqrt((x - (width*0.7 + dx_offset))**2 + (y - (height*0.5 + dy_offset))**2)
        base_freq = 50 / freq_jitter
        val = (np.sin(d1/base_freq + phase_offset) + np.sin(d2/base_freq + phase_offset) + 2) / 4
    elif pattern_type == "Waves_Ripple":
        base_freq = 40 / freq_jitter
        val = np.exp(-dist_norm * 2) * (np.sin(dist_center_s / base_freq + phase_offset * np.pi) + 1) / 2
    elif pattern_type == "Blob_Center":
        sigma = width * 0.2 * size_scale
        val = np.exp(-dist_center_s**2 / (2 * sigma**2))
    elif pattern_type == "Blob_TopRight":
        bx = x.max() * 0.75 + dx_offset; by = y.max() * 0.75 + dy_offset
        sigma = width * 0.15 * size_scale
        val = np.exp(-((x-bx)**2 + (y-by)**2) / (2 * sigma**2))
    elif pattern_type == "Blob_TopLeft":
        bx = x.min() + (x.max()-x.min()) * 0.25 + dx_offset; by = y.max() * 0.75 + dy_offset
        sigma = width * 0.15 * size_scale
        val = np.exp(-((x-bx)**2 + (y-by)**2) / (2 * sigma**2))
    elif pattern_type == "Blob_BottomRight":
        bx = x.max() * 0.75 + dx_offset; by = y.min() + (y.max()-y.min()) * 0.25 + dy_offset
        sigma = width * 0.15 * size_scale
        val = np.exp(-((x-bx)**2 + (y-by)**2) / (2 * sigma**2))
    elif pattern_type == "Blob_BottomLeft":
        bx = x.min() + (x.max()-x.min()) * 0.25 + dx_offset
        by = y.min() + (y.max()-y.min()) * 0.25 + dy_offset
        sigma = width * 0.15 * size_scale
        val = np.exp(-((x-bx)**2 + (y-by)**2) / (2 * sigma**2))
    elif pattern_type == "Spots_Grid_Dense":
        base_freq = 100 / freq_jitter
        raw = np.clip(np.sin(x/base_freq+phase_offset)*np.sin(y/base_freq+phase_offset), 0, 1)
        val = raw ** (1.0 / size_scale)
    elif pattern_type == "Spots_Grid_Sparse":
        base_freq = 200 / freq_jitter
        raw = np.clip(np.sin(x/base_freq+phase_offset)*np.sin(y/base_freq+phase_offset), 0, 1)
        val = raw ** (1.0 / size_scale)
    elif pattern_type == "Spots_Random_Large":
        rng = np.random.RandomState(pattern_seed); sigma = width * 0.1 * size_scale; val = np.zeros(n)
        for _ in range(5):
            bx = rng.uniform(x.min(), x.max()) + dx_offset; by = rng.uniform(y.min(), y.max()) + dy_offset
            val += np.exp(-((x-bx)**2+(y-by)**2) / (2*sigma**2))
        val = val / (val.max() + 1e-8)
    elif pattern_type == "Spots_Triangular":
        spacing = 400 * freq_jitter; spot_sigma = 80 * size_scale; val = np.zeros(n)
        for i in range(10):
            for j in range(10):
                bx = i*spacing+(j%2)*(spacing/2)+dx_offset; by = j*spacing*0.866+dy_offset
                val += np.exp(-((x-bx)**2+(y-by)**2)/(2*spot_sigma**2))
        val = np.clip(val, 0, 1)
    elif pattern_type == "Spots_Hexagonal":
        spacing = 350 * freq_jitter; spot_sigma = 60 * size_scale; val = np.zeros(n)
        for i in range(12):
            for j in range(12):
                bx = i*spacing+(j%2)*(spacing/2)+dx_offset; by = j*spacing*0.866+dy_offset
                val += np.exp(-((x-bx)**2+(y-by)**2)/(2*spot_sigma**2))
        val = np.clip(val, 0, 1)
    elif pattern_type == "Ring_Inner":
        target_r = max_dist * (0.2*size_scale+phase_offset*0.05); ring_width = width*0.04*size_scale
        val = np.exp(-((dist_center_s-target_r)**2)/(2*ring_width**2))
    elif pattern_type == "Ring_Outer":
        target_r = max_dist * (0.7*size_scale+phase_offset*0.05); ring_width = width*0.05*size_scale
        val = np.exp(-((dist_center_s-target_r)**2)/(2*ring_width**2))
    elif pattern_type == "Ring_Double":
        r1 = max_dist*(0.25*size_scale+phase_offset*0.03); r2 = max_dist*(0.55*size_scale+phase_offset*0.03)
        rw = width*0.03*size_scale
        val = np.exp(-((dist_center_s-r1)**2)/(2*rw**2))+np.exp(-((dist_center_s-r2)**2)/(2*rw**2))
        val = val / (val.max() + 1e-8)
    elif pattern_type == "Ring_Eccentric":
        off_cx = cx+width*0.15+dx_offset; off_cy = cy+height*0.1+dy_offset
        dist_off = np.sqrt((x-off_cx)**2+(y-off_cy)**2)
        target_r = max_dist*(0.35*size_scale+phase_offset*0.04); rw = width*0.05*size_scale
        val = np.exp(-((dist_off-target_r)**2)/(2*rw**2))
    elif pattern_type == "Ring_Elliptical":
        a = max_dist*(0.5*size_scale+phase_offset*0.05); b = max_dist*(0.3*size_scale+phase_offset*0.03)
        ed = np.sqrt(((x-cx_s)/a)**2+((y-cy_s)/b)**2); rw = 0.1*size_scale
        val = np.exp(-((ed-1)**2)/(2*rw**2))
    elif pattern_type == "Ring_Partial":
        target_r = max_dist*(0.4*size_scale+phase_offset*0.04); rw = width*0.05*size_scale
        rv = np.exp(-((dist_center_s-target_r)**2)/(2*rw**2))
        ah = (np.pi/2)*size_scale; a_s = -ah+phase_offset*0.3; a_e = ah+phase_offset*0.3
        val = rv * ((angle_s > a_s) & (angle_s < a_e))
    elif pattern_type == "Checkerboard_Fine":
        ss = 200*freq_jitter; xi = ((x+dx_offset)//ss).astype(int); yi = ((y+dy_offset)//ss).astype(int)
        val = ((xi+yi)%2).astype(float)
    elif pattern_type == "Checkerboard_Coarse":
        ss = 500*freq_jitter; xi = ((x+dx_offset)//ss).astype(int); yi = ((y+dy_offset)//ss).astype(int)
        val = ((xi+yi)%2).astype(float)
    elif pattern_type == "Quadrant_Alternating":
        val = ((x>cx_s)&(y>cy_s) | (x<cx_s)&(y<cy_s)).astype(float)
    elif pattern_type == "Sectors_4":
        sector = ((angle_s+np.pi+phase_offset)/(np.pi/2)).astype(int)%4; val = (sector%2).astype(float)
    elif pattern_type == "Sectors_8":
        sector = ((angle_s+np.pi+phase_offset)/(np.pi/4)).astype(int)%8; val = (sector%2).astype(float)
    elif pattern_type == "Triangle_Pattern":
        period = 300*freq_jitter
        val = (np.abs(((x+dx_offset)%period)-period/2)/(period/2)+np.abs(((y+dy_offset)%period)-period/2)/(period/2))/2
    elif pattern_type == "Diamond_Pattern":
        period = 400*freq_jitter
        val = (np.abs(((x+dx_offset)%period)-period/2)+np.abs(((y+dy_offset)%period)-period/2))/period
    elif pattern_type == "Honeycomb":
        spacing = 200*freq_jitter; val = np.zeros(n)
        for i in range(-2, 20):
            for j in range(-2, 20):
                hx = i*spacing+(j%2)*(spacing/2)+dx_offset; hy = j*spacing*0.866+dy_offset
                dist = np.sqrt((x-hx)**2+(y-hy)**2)
                val = np.maximum(val, 1-np.clip(dist/(spacing*0.4*size_scale), 0, 1))
        val = 1-val
    elif pattern_type == "Cortical_Layers":
        n_layers = 5; layer_vals = [0.9, 0.3, 0.7, 0.2, 0.8]
        sd = np.sqrt((x-cx_s)**2+(y-cy_s)**2); sn = sd/(sd.max()+1e-8)
        sc = np.clip(sn/size_scale, 0, 1); li = np.clip((sc*n_layers).astype(int), 0, n_layers-1)
        val = np.array([layer_vals[i] for i in li])
    elif pattern_type == "Hotspot_Cluster":
        rng = np.random.RandomState(pattern_seed); sigma = width*0.05*size_scale; val = np.zeros(n)
        ccx = cx+rng.uniform(-width*0.1, width*0.1)+dx_offset; ccy = cy+rng.uniform(-height*0.1, height*0.1)+dy_offset
        for _ in range(8):
            bx = ccx+rng.normal(0, width*0.08); by = ccy+rng.normal(0, height*0.08)
            val += np.exp(-((x-bx)**2+(y-by)**2)/(2*sigma**2))
        val = val/(val.max()+1e-8)
    elif pattern_type == "Edge_Enhancement":
        sn = dist_center_s/(dist_center_s.max()+1e-8)
        target = np.clip(0.8*size_scale+phase_offset*0.05, 0.3, 1.2); bw = 0.15*size_scale
        val = np.clip(1-np.exp(-((sn-target)**2)/(2*bw**2)), 0, 1)
    elif pattern_type == "Core_Shell":
        cs = width*0.1*size_scale; sr = max_dist*(0.5+phase_offset*0.05)*size_scale; ss = width*0.08*size_scale
        val = 0.7*np.exp(-dist_center_s**2/(2*cs**2))+0.5*np.exp(-((dist_center_s-sr)**2)/(2*ss**2))
        val = val/(val.max()+1e-8)
    elif pattern_type == "Branching":
        nb = 5; ba = 2*np.pi/nb; bw = 0.3*size_scale; ra = angle_s+phase_offset*0.5
        mdb = np.inf*np.ones(n)
        for i in range(nb):
            ta = i*ba; ad2 = np.abs(np.mod(ra-ta+np.pi, 2*np.pi)-np.pi)
            mdb = np.minimum(mdb, ad2)
        val = np.exp(-mdb**2/(2*bw**2))*(1-np.exp(-dist_norm*3))
    elif pattern_type == "Laminar_Curved":
        ca = 200*(1+phase_offset*0.3); cf = 400*freq_jitter; bt = 300*size_scale
        cy2 = (y+dy_offset)-ca*np.sin((x+dx_offset)/cf)
        val = ((cy2//bt).astype(int)%2).astype(float)
    elif pattern_type == "Mosaic_Irregular":
        rng = np.random.RandomState(pattern_seed); ns = max(5, int(20/size_scale))
        sx = rng.uniform(x.min(), x.max(), ns)+dx_offset; sy = rng.uniform(y.min(), y.max(), ns)+dy_offset
        sv = rng.rand(ns); ni = np.zeros(n, dtype=int); md = np.inf*np.ones(n)
        for i in range(ns):
            d = np.sqrt((x-sx[i])**2+(y-sy[i])**2); c = d<md; ni[c] = i; md[c] = d[c]
        val = sv[ni]
    elif pattern_type == "Gradient_Sigmoid":
        shift = 0.5+phase_offset*0.1; steep = 10.0/size_scale
        val = 1/(1+np.exp(-steep*(x_norm-shift)))
    elif pattern_type == "Bimodal_Distribution":
        val = (x<cx_s).astype(float)*0.8+(x>=cx_s).astype(float)*0.3
    elif pattern_type == "Punctate_Dense":
        rng = np.random.RandomState(pattern_seed); ps = 40*size_scale; val = np.zeros(n)
        for _ in range(50):
            px = rng.uniform(x.min(), x.max())+dx_offset; py = rng.uniform(y.min(), y.max())+dy_offset
            val += np.exp(-((x-px)**2+(y-py)**2)/(2*ps**2))
        val = np.clip(val, 0, 1)
    elif pattern_type == "Periventricular":
        vr = 0.15*size_scale+phase_offset*0.02; rr = 0.25*size_scale+phase_offset*0.02; rw = 0.08*size_scale
        sn = dist_center_s/(dist_center_s.max()+1e-8)
        val = np.exp(-((sn-rr)**2)/(2*rw**2))*(sn>=vr)
    elif pattern_type == "Asymmetric_Lobe":
        l1x = cx-width*0.2+dx_offset; l1y = cy+dy_offset; l2x = cx+width*0.15+dx_offset; l2y = cy+height*0.1+dy_offset
        s1 = width*0.15*size_scale; s2 = width*0.10*size_scale
        val = 0.7*np.exp(-((x-l1x)**2+(y-l1y)**2)/(2*s1**2))+0.5*np.exp(-((x-l2x)**2+(y-l2y)**2)/(2*s2**2))
        val = val/(val.max()+1e-8)
    else:
        rng = np.random.RandomState(pattern_seed); val = rng.rand(n)

    val = val * amp_scale
    if add_noise and bio_noise_std > 0:
        rng_noise = np.random.RandomState(sample_seed ^ (pattern_seed + 1))
        val = val + rng_noise.normal(0, bio_noise_std, size=n)
    return np.clip(val, 0, 1)


def generate_noise_from_parent(coords, parent_pattern, variant_idx,
                               width=3000, height=3000,
                               sample_seed=None, size_scale_range=(0.70, 1.40),
                               shape=8, location=8, scale=8,
                               frequency=8, orientation=8):
    """
    Generate a noise feature as a spatially-distorted variant of a parent GT pattern.

    Five independent parameters (each 1-20) control distinct spatial properties:

      shape       — Sinusoidal coordinate warp that deforms the spatial structure.
                    Level 1 ≈ 100 µm warp;  Level 20 ≈ 2000 µm warp.
      location    — Translates the effective pattern center.
                    Level 1 ≈ ±5 % shift;  Level 20 ≈ ±40 % shift.
      scale       — Stretches or compresses (anisotropically) the pattern.
                    Level 1 ≈ ±5 % change;  Level 20 ≈ ±60 % change.
      frequency   — Overlays unique additive + multiplicative spatial fields.
                    Level 1 ≈ 10 % blend;  Level 20 ≈ 60 % blend.
      orientation — Rotates the coordinate frame.
                    Level 1 ≈ ±5°;  Level 20 ≈ ±90°.

    variant_idx (0..NOISE_PER_PATTERN-1) determines which unique set of
    perturbations is applied.  Different variant_idx values produce different
    random draws for each distortion axis.

    CRITICAL INVARIANT: All distortions use smooth continuous spatial functions
    f(x, y). The RNG draws only scalar parameters (never n-dependent arrays).
    This guarantees identical results on RNA and MSI grids.
    """
    n = len(coords)
    x, y = coords[:, 0], coords[:, 1]

    # Convert 1-20 integer controls to 0-1 continuous magnitudes.
    # Floor of 0.03 ensures even level=1 produces a small change.
    d_shape  = max(0.03, (shape - 1) / 19.0)
    d_loc    = max(0.03, (location - 1) / 19.0)
    d_scale  = max(0.03, (scale - 1) / 19.0)
    d_freq   = max(0.03, (frequency - 1) / 19.0)
    d_orient = max(0.03, (orientation - 1) / 19.0)

    # Unique RNG for this parent+variant combination — scalar draws only.
    parent_hash = hash(parent_pattern) % 2**31
    variant_seed = (parent_hash * 1000 + variant_idx * 7 + 42) % 2**31
    rng = np.random.RandomState(variant_seed)

    cx_field = width / 2.0
    cy_field = height / 2.0

    # =====================================================================
    # 1. SHAPE — Multi-frequency sinusoidal coordinate warp
    #    Two warp layers at different spatial frequencies produce complex
    #    non-rigid deformations that vary across the field.
    #    Level 1 ≈ 100 µm;  Level 20 ≈ 2000 µm total displacement.
    # =====================================================================
    warp_base = 100 + d_shape * 1900  # µm of maximum displacement

    # Layer 1: low-frequency broad warp
    w1_amp_x  = rng.uniform(0.4, 0.7) * warp_base
    w1_amp_y  = rng.uniform(0.4, 0.7) * warp_base
    w1_freq_x = rng.uniform(2000, 5000)
    w1_freq_y = rng.uniform(2000, 5000)
    w1_ph_x   = rng.uniform(0, 2 * np.pi)
    w1_ph_y   = rng.uniform(0, 2 * np.pi)

    # Layer 2: higher-frequency local warp
    w2_amp_x  = rng.uniform(0.2, 0.5) * warp_base
    w2_amp_y  = rng.uniform(0.2, 0.5) * warp_base
    w2_freq_x = rng.uniform(800, 2000)
    w2_freq_y = rng.uniform(800, 2000)
    w2_ph_x   = rng.uniform(0, 2 * np.pi)
    w2_ph_y   = rng.uniform(0, 2 * np.pi)

    x_w = (x
           + w1_amp_x * np.sin(y / w1_freq_y + w1_ph_y)
           + w2_amp_x * np.sin(y / w2_freq_y + w2_ph_y))
    y_w = (y
           + w1_amp_y * np.sin(x / w1_freq_x + w1_ph_x)
           + w2_amp_y * np.sin(x / w2_freq_x + w2_ph_x))

    # =====================================================================
    # 2. LOCATION — Center shift / translation
    #    Moves the effective origin of the pattern.
    #    Level 1 ≈ ±5 % of field;  Level 20 ≈ ±40 % of field.
    # =====================================================================
    shift_mag = 0.05 + d_loc * 0.35
    shift_x = rng.uniform(-shift_mag, shift_mag) * width
    shift_y = rng.uniform(-shift_mag, shift_mag) * height

    x_w = x_w + shift_x
    y_w = y_w + shift_y

    # =====================================================================
    # 3. SCALE — Anisotropic coordinate rescaling around field center
    #    X and Y are scaled independently for richer variation.
    #    Level 1 ≈ ±5 % change;  Level 20 ≈ ±60 % change.
    # =====================================================================
    scale_range = 0.05 + d_scale * 0.55
    scale_x = 1.0 + rng.uniform(-scale_range, scale_range)
    scale_y = 1.0 + rng.uniform(-scale_range, scale_range)

    x_w = cx_field + (x_w - cx_field) * scale_x
    y_w = cy_field + (y_w - cy_field) * scale_y

    # =====================================================================
    # 4. ORIENTATION — Rotation around field center
    #    Tilts directional patterns and shifts angular features.
    #    Level 1 ≈ ±5°;  Level 20 ≈ ±90°.
    # =====================================================================
    max_rot = 0.09 + d_orient * 1.48  # radians (≈5° to ≈90°)
    rot_angle = rng.uniform(-max_rot, max_rot)
    cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)
    dx_c = x_w - cx_field
    dy_c = y_w - cy_field
    x_r = cx_field + dx_c * cos_r - dy_c * sin_r
    y_r = cy_field + dx_c * sin_r + dy_c * cos_r

    # =====================================================================
    # Generate parent pattern at the distorted coordinates.
    # The pattern is evaluated at warped positions, producing actual
    # spatial structure changes (not just intensity/color shifts).
    # =====================================================================
    warped_coords = np.column_stack([x_r, y_r])
    val = generate_pattern_values(warped_coords, parent_pattern, width, height,
                                  sample_seed=sample_seed, add_noise=False,
                                  size_scale_range=size_scale_range)

    # =====================================================================
    # 5. FREQUENCY — Additive + multiplicative spatial fields
    #    Two independent sinusoidal fields that alter the spatial frequency
    #    content: one additive (blends in a new pattern) and one
    #    multiplicative (spatially modulates intensity).
    #    Level 1 ≈ 10 % blend;  Level 20 ≈ 60 % blend.
    # =====================================================================
    blend_strength = 0.10 + d_freq * 0.50

    # Additive field
    add_freq  = rng.uniform(300, 1200)
    add_theta = rng.uniform(0, 2 * np.pi)
    add_phase = rng.uniform(0, 2 * np.pi)
    proj = x * np.cos(add_theta) + y * np.sin(add_theta)
    additive = (np.sin(proj / add_freq + add_phase) + 1) / 2

    # Multiplicative field (different angle/frequency)
    mul_freq_x = rng.uniform(500, 2000)
    mul_freq_y = rng.uniform(500, 2000)
    mul_phase  = rng.uniform(0, 2 * np.pi)
    multiplicative = (np.sin(x / mul_freq_x + mul_phase) *
                      np.cos(y / mul_freq_y + mul_phase * 0.7) + 1) / 2

    # Blend: (1-s)*pattern + s/2*additive + s/2*multiplicative*pattern
    s = blend_strength
    val = val * (1 - s) + additive * (s * 0.5) + val * multiplicative * (s * 0.5)

    val = np.clip(val, 0, 1)

    # =====================================================================
    # Per-sample variation (same mechanism as GT patterns)
    # =====================================================================
    if sample_seed is not None:
        perturb_seed = (variant_seed ^ (sample_seed * 1000003)) % 2**31
        rng_p = np.random.RandomState(perturb_seed)
        amp_scale     = rng_p.uniform(0.85, 1.05)
        bio_noise_std = rng_p.uniform(0.0, 0.06)

        val = val * amp_scale
        if bio_noise_std > 0:
            rng_bio = np.random.RandomState((variant_seed ^ sample_seed) + 2)
            val = val + rng_bio.normal(0, bio_noise_std, size=n)

    return np.clip(val, 0, 1)


def get_visium_hex_grid(width, height, spacing=100):
    dy = spacing * np.sqrt(3) / 2
    coords = []; row = 0; y = 0
    while y < height:
        x_offset = (spacing / 2) if row % 2 == 1 else 0
        x = x_offset
        while x < width:
            coords.append([x, y]); x += spacing
        y += dy; row += 1
    return np.array(coords)


def get_cartesian_grid(width, height, spacing):
    x = np.arange(0, width, spacing); y = np.arange(0, height, spacing)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack((xx.ravel(), yy.ravel()))


# =============================================================================
# MAIN GENERATION LOOP
# =============================================================================
if __name__ == "__main__":

    n_gt = len(GT_PATTERNS)
    total_noise = NOISE_PER_PATTERN * n_gt
    n_total_features = n_gt + total_noise

    print("=" * 60)
    print("Generating Ground Truth Dummy Data with 50 PATTERNS")
    print("Each GT pattern produces noise variants with 5 distortion controls")
    print("=" * 60)
    print(f"Groups: {GROUPS}")
    print(f"GT Patterns: {n_gt}")
    print(f"Noise per pattern: {NOISE_PER_PATTERN}")
    print(f"Total noise features: {total_noise}")
    print(f"Total features per sample: {n_gt} GT + {total_noise} Noise = {n_total_features}")
    print(f"Noise distortion controls (1-20):")
    print(f"  Shape:       {NOISE_SHAPE:2d}  (coordinate warp, 100–2000 µm)")
    print(f"  Location:    {NOISE_LOCATION:2d}  (center shift, ±5–40%)")
    print(f"  Scale:       {NOISE_SCALE:2d}  (anisotropic rescale, ±5–60%)")
    print(f"  Frequency:   {NOISE_FREQUENCY:2d}  (additive+multiplicative field, 10–60%)")
    print(f"  Orientation: {NOISE_ORIENTATION:2d}  (rotation, ±5–90°)")
    print(f"Field size: {FIELD_WIDTH}x{FIELD_HEIGHT} um")
    print(f"RNA grid: Visium-like hexagonal, {RNA_SPOT_SPACING}um spacing")
    print(f"MSI grid: Cartesian, {MSI_PIXEL_SIZE}um pixels")
    print(f"Pattern size scale range: {PATTERN_SIZE_SCALE_RANGE[0]:.2f}-{PATTERN_SIZE_SCALE_RANGE[1]:.2f}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    ground_truth_pairs = []

    for group_idx, group in enumerate(GROUPS):
        for sample_idx in range(1, SAMPLES_PER_GROUP + 1):
            sample_id = f"{group}_{sample_idx}"
            seed = hash(sample_id) % 2**32
            print(f"\n  Processing {sample_id} (seed={seed})...")

            tissue_fn = generate_halfbrain_shape(seed, width=FIELD_WIDTH, height=FIELD_HEIGHT)

            # === MSI Data ===
            raw_msi_coords = get_cartesian_grid(FIELD_WIDTH, FIELD_HEIGHT, MSI_PIXEL_SIZE)
            mask_msi = [tissue_fn(xi, yi) for xi, yi in raw_msi_coords]
            msi_coords = raw_msi_coords[mask_msi]

            msi_data = np.zeros((len(msi_coords), n_total_features))
            var_names_msi = []
            col = 0

            for i, pat in enumerate(GT_PATTERNS):
                print(f"    MSI GT {i+1:2d}/{n_gt}: {pat}", end="")
                # GT feature
                msi_data[:, col] = generate_pattern_values(
                    msi_coords, pat, width=FIELD_WIDTH, height=FIELD_HEIGHT,
                    sample_seed=seed, size_scale_range=PATTERN_SIZE_SCALE_RANGE)
                var_names_msi.append(f"MZ_{pat}")
                col += 1

                # Noise variants for this pattern
                for v in range(NOISE_PER_PATTERN):
                    msi_data[:, col] = generate_noise_from_parent(
                        msi_coords, pat, v,
                        width=FIELD_WIDTH, height=FIELD_HEIGHT,
                        sample_seed=seed, size_scale_range=PATTERN_SIZE_SCALE_RANGE,
                        shape=NOISE_SHAPE, location=NOISE_LOCATION,
                        scale=NOISE_SCALE, frequency=NOISE_FREQUENCY,
                        orientation=NOISE_ORIENTATION)
                    var_names_msi.append(f"MZ_Noise_{pat}_{v}")
                    col += 1
                print(f" + {NOISE_PER_PATTERN} noise variants")

            obs_msi = pd.DataFrame(index=[f"spot_{i}" for i in range(len(msi_coords))])
            obs_msi['x_um'] = msi_coords[:, 0]
            obs_msi['y_um'] = msi_coords[:, 1]

            adata_msi = ad.AnnData(X=msi_data, obs=obs_msi)
            adata_msi.var_names = var_names_msi
            adata_msi.write(os.path.join(MSI_DIR, f"halfbrain_{group.lower()}_{sample_idx}_filtered_common.h5ad"))

            # === RNA Data ===
            raw_rna_coords = get_visium_hex_grid(FIELD_WIDTH, FIELD_HEIGHT, spacing=RNA_SPOT_SPACING)
            mask_rna = [tissue_fn(xi, yi) for xi, yi in raw_rna_coords]
            rna_coords = raw_rna_coords[mask_rna]

            rna_data = np.zeros((len(rna_coords), n_total_features))
            var_names_rna = []
            col = 0

            for i, pat in enumerate(GT_PATTERNS):
                print(f"    RNA GT {i+1:2d}/{n_gt}: {pat}", end="")
                # GT feature
                rna_data[:, col] = generate_pattern_values(
                    rna_coords, pat, width=FIELD_WIDTH, height=FIELD_HEIGHT,
                    sample_seed=seed, size_scale_range=PATTERN_SIZE_SCALE_RANGE)
                var_names_rna.append(f"Gene_{pat}")
                if group_idx == 0 and sample_idx == 1:
                    ground_truth_pairs.append((f"Gene_{pat}", f"MZ_{pat}"))
                col += 1

                # Noise variants for this pattern
                for v in range(NOISE_PER_PATTERN):
                    rna_data[:, col] = generate_noise_from_parent(
                        rna_coords, pat, v,
                        width=FIELD_WIDTH, height=FIELD_HEIGHT,
                        sample_seed=seed, size_scale_range=PATTERN_SIZE_SCALE_RANGE,
                        shape=NOISE_SHAPE, location=NOISE_LOCATION,
                        scale=NOISE_SCALE, frequency=NOISE_FREQUENCY,
                        orientation=NOISE_ORIENTATION)
                    var_names_rna.append(f"Gene_Noise_{pat}_{v}")
                    col += 1
                print(f" + {NOISE_PER_PATTERN} noise variants")

            obs_rna = pd.DataFrame(index=[f"spot_{i}" for i in range(len(rna_coords))])
            obs_rna['x_um'] = rna_coords[:, 0]
            obs_rna['y_um'] = rna_coords[:, 1]

            adata_rna = ad.AnnData(X=rna_data, obs=obs_rna)
            adata_rna.var_names = var_names_rna
            adata_rna.write(os.path.join(RNA_DIR, f"{group}_{sample_idx}.h5ad"))

    print("\n" + "=" * 60)
    print("DONE! Data generated.")
    print("=" * 60)
    print(f"\n{n_gt} GROUND TRUTH MATCHES:")
    print("-" * 60)
    for i, (gene, mz) in enumerate(ground_truth_pairs, 1):
        print(f"  {i:2d}. {gene:<30}  <==>  {mz}")
    print("-" * 60)
    print(f"\nTotal features per sample: {n_gt} GT + {total_noise} Noise = {n_total_features}")
    print(f"Noise per pattern: {NOISE_PER_PATTERN}")
    print(f"Noise distortion controls: shape={NOISE_SHAPE}, location={NOISE_LOCATION}, "
          f"scale={NOISE_SCALE}, frequency={NOISE_FREQUENCY}, orientation={NOISE_ORIENTATION}")
    print(f"\nNoise naming: MZ_Noise_<PatternName>_<VariantIdx> / Gene_Noise_<PatternName>_<VariantIdx>")
    print(f"Example: MZ_Noise_Gradient_X_0 is variant 0 of Gradient_X noise")