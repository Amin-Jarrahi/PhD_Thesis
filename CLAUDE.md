# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PhD thesis studying Alzheimer's Disease (AD) progression in mouse brain tissue through integrated analysis of Mass Spectrometry Imaging (MSI) and Spatial Transcriptomics (ST/Visium). The project uses Graph Neural Networks with contrastive learning to align these two modalities across 16 mouse brain samples in a 2x2 factorial design: Young Control (YC), Young AD (YAD), Aged Control (AC), Aged AD (AAD), with 4 replicates each.

## Repository Structure

- **chapter_2/** — MSI data preprocessing pipeline (IMZML parsing, peak detection, normalization, common m/z identification across samples)
- **chapter_3/** — Spatial transcriptomics preprocessing (Visium h5ad loading, sample aggregation, QC, Leiden clustering)
- **chapter_4/** — Cross-modal integration and deep learning (pattern recognition, isotope detection, GNN model development)
  - `code_final/` — Production-ready preprocessing notebooks
  - `dl_code/` and `dl_code_new/` — Deep learning model iterations (GAT, GATv2, Transformers)
  - `code_combined/` — Combined preprocessing pipelines
- **IMZML Tools/** — Raw MSI data files and a Streamlit exploration app

## Architecture

The codebase is entirely Jupyter notebook-driven. Notebooks are numbered to indicate execution order within each chapter (e.g., `1_imzml_reader` → `2_peak_detection` → `3_finding_common`).

### Deep Learning Model (Chapter 4)

```
MSI Input (m/z features) → SpatialGAT → Hidden Embedding (256D)
                                              ↓
                                        ProjectionHead → Contrastive Projection (128D)
                                              ↑
ST Input (gene features) → SpatialGAT → Hidden Embedding (256D)
```

- Contrastive learning with NT-Xent loss aligns MSI and ST embeddings
- Spatial graphs constructed per-sample using physical distance radius (110 µm)
- Stratified batch sampling across experimental groups
- AnnData → PyTorch Geometric conversion via `h5ad_to_PyG.py`

## Key Technologies

- **anndata/scanpy** — Spatial data containers and analysis
- **PyTorch + PyTorch Geometric** — GNN training (GAT/GATv2 architectures)
- **pyimzml** — IMZML mass spectrometry file parsing
- **scipy.spatial.cKDTree** — Spatial graph construction
- **plotly/streamlit** — Interactive visualization
- **TruncatedSVD** — Dimensionality reduction for large feature spaces

## Dependencies

Primary dependencies are listed in `IMZML Tools/requirements.txt`. Core packages: anndata, scanpy, torch, torch_geometric, numpy, pandas, scipy, matplotlib, plotly, streamlit.

## Data Conventions

- **Sample naming**: `YC_1..YC_4, YAD_1..YAD_4, AC_1..AC_4, AAD_1..AAD_4`
- **Data formats**: `.h5ad` (AnnData), `.imzml/.ibd` (raw MSI), `.csv` (metadata/coordinates), `.pt` (PyG graphs)
- **MSI normalization**: TIC (total ion current) normalization
- **Feature reduction**: Typically to 128–200 dimensions

## Important Notes

- Data files are very large (individual h5ad files can be 27–50+ GB). Be cautious with operations that load full datasets into memory.
- The `.gitignore` excludes most data files. Notebooks with embedded outputs can be hundreds of MB.
- No formal test suite or build system exists — validation is done within notebooks.
