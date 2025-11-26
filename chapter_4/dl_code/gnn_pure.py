"""
MSI-ST GNN-Only Model Pipeline
File: MSI_ST_GNN_Pipeline.py

Architecture:
- GAT (Graph Attention Network) for spatial modeling
- No transformers - pure graph-based learning
- Spatial relationships captured through edge features
- Multi-layer GAT with residual connections
"""

import math
import os
from typing import Optional, Tuple, List, Dict
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data as GeometricData, Batch
from torch_geometric.nn import GATConv, global_mean_pool

import anndata as ad
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ========================= UTILITIES =========================

def build_spatial_graph(coords: np.ndarray, k: int = 6, 
                       self_loops: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build kNN spatial graph with proper edge weights.
    Returns: (edge_index, edge_attr)
    """
    N = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=k+1 if not self_loops else k)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    
    if not self_loops:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    
    src_list, dst_list, edge_weights = [], [], []
    for i in range(N):
        for j, neighbor_idx in enumerate(indices[i]):
            src_list.append(i)
            dst_list.append(int(neighbor_idx))
            # Gaussian kernel for edge weights
            dist = distances[i, j]
            edge_weights.append(np.exp(-dist**2 / (2 * np.median(distances)**2)))
    
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)
    
    return edge_index, edge_attr


# ========================= DATA ORGANIZATION =========================

def organize_ad_study_data(data_root: str) -> Dict[str, List[str]]:
    """
    Organize paths for AD study with 16 samples (4 groups × 4 replicates).
    
    Expected directory structure:
        data_root/
        ├── msi/
        │   ├── aged_AD_1.h5ad
        │   ├── aged_AD_2.h5ad
        │   ├── aged_control_1.h5ad
        │   ├── young_AD_1.h5ad
        │   └── ...
        └── st/
            ├── aged_AD_1.h5ad
            ├── aged_AD_2.h5ad
            └── ...
    
    Returns:
        Dictionary with 'msi_paths', 'st_paths', 'sample_ids', 'groups', 'metadata'
    """
    groups = ['aged_AD', 'aged_control', 'young_AD', 'young_control']
    n_replicates = 4
    
    msi_paths = []
    st_paths = []
    sample_ids = []
    sample_groups = []
    
    for group in groups:
        for replicate in range(1, n_replicates + 1):
            sample_id = f"{group}_{replicate}"
            
            msi_path = os.path.join(data_root, 'msi', f"{sample_id}.h5ad")
            st_path = os.path.join(data_root, 'st', f"{sample_id}.h5ad")
            
            # Verify files exist
            if not os.path.exists(msi_path):
                raise FileNotFoundError(f"MSI file not found: {msi_path}")
            if not os.path.exists(st_path):
                raise FileNotFoundError(f"ST file not found: {st_path}")
            
            msi_paths.append(msi_path)
            st_paths.append(st_path)
            sample_ids.append(sample_id)
            sample_groups.append(group)
    
    # Create metadata DataFrame for easy reference
    metadata = {
        'sample_id': sample_ids,
        'group': sample_groups,
        'age': ['aged' if 'aged' in g else 'young' for g in sample_groups],
        'condition': ['AD' if 'AD' in g else 'control' for g in sample_groups],
        'replicate': [i % n_replicates + 1 for i in range(len(sample_ids))]
    }
    
    return {
        'msi_paths': msi_paths,
        'st_paths': st_paths,
        'sample_ids': sample_ids,
        'groups': sample_groups,
        'metadata': metadata
    }


# ========================= STRATIFIED SAMPLING =========================

class StratifiedBatchSampler:
    """
    Ensures balanced sampling across experimental groups during training.
    Prevents model from being biased toward overrepresented groups.
    """
    def __init__(self, sample_groups: List[str], spots_per_sample: List[int],
                 batch_size: int, shuffle: bool = True):
        """
        Args:
            sample_groups: List of group labels for each sample
            spots_per_sample: Number of spots in each sample
            batch_size: Target batch size
            shuffle: Shuffle within groups
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Organize spots by group
        self.group_indices = {}
        offset = 0
        
        for sample_idx, (group, n_spots) in enumerate(zip(sample_groups, spots_per_sample)):
            if group not in self.group_indices:
                self.group_indices[group] = []
            
            # Add global indices for this sample's spots
            sample_indices = list(range(offset, offset + n_spots))
            self.group_indices[group].extend(sample_indices)
            offset += n_spots
        
        self.groups = list(self.group_indices.keys())
        self.n_groups = len(self.groups)
        
    def __iter__(self):
        # Shuffle indices within each group
        if self.shuffle:
            for group in self.groups:
                np.random.shuffle(self.group_indices[group])
        
        # Calculate spots per group per batch (balanced)
        spots_per_group = self.batch_size // self.n_groups
        
        # Create batches with balanced group representation
        group_iterators = {g: iter(self.group_indices[g]) for g in self.groups}
        
        while True:
            batch = []
            groups_exhausted = 0
            
            for group in self.groups:
                group_batch = []
                try:
                    for _ in range(spots_per_group):
                        group_batch.append(next(group_iterators[group]))
                except StopIteration:
                    groups_exhausted += 1
                
                batch.extend(group_batch)
            
            if groups_exhausted == self.n_groups:
                break
            
            if len(batch) >= spots_per_group:  # Ensure minimum batch size
                yield batch
    
    def __len__(self):
        min_group_size = min(len(indices) for indices in self.group_indices.values())
        return (min_group_size * self.n_groups) // self.batch_size


class MultiSampleMSIDataset(Dataset):
    """Dataset for multiple MSI samples with per-sample graph handling."""
    def __init__(self, adata_list: List[ad.AnnData], sample_ids: Optional[List[str]] = None,
                 normalize: str = 'tic', log_transform: bool = True, k_neighbors: int = 6):
        """
        Args:
            adata_list: List of AnnData objects, one per sample
            sample_ids: Optional list of sample identifiers
            normalize: 'tic', 'standard', or None
            log_transform: Apply log1p transformation
            k_neighbors: Number of neighbors for spatial graph
        """
        self.k_neighbors = k_neighbors
        self.samples = []
        
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(adata_list))]
        
        # Process each sample
        for sample_id, adata in zip(sample_ids, adata_list):
            # Extract and normalize data
            X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
            
            if normalize == 'tic':
                tic = X.sum(axis=1, keepdims=True)
                X = X / (tic + 1e-9)
            elif normalize == 'standard':
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            if log_transform:
                X = np.log1p(X)
            
            coords = np.vstack([
                adata.obs['x'].astype(float).values,
                adata.obs['y'].astype(float).values
            ]).T.astype(np.float32)
            
            # Build spatial graph for this sample
            edge_index, edge_attr = build_spatial_graph(coords, k=k_neighbors)
            
            # Store sample info
            n_spots = X.shape[0]
            self.samples.append({
                'sample_id': sample_id,
                'X': X.astype(np.float32),
                'coords': coords,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'n_spots': n_spots,
                'spot_offset': 0  # Will be set in __init__
            })
        
        # Compute cumulative offsets for indexing
        cumsum = 0
        for sample in self.samples:
            sample['spot_offset'] = cumsum
            cumsum += sample['n_spots']
        
        self.total_spots = cumsum
        self.n_features = self.samples[0]['X'].shape[1]
    
    def __len__(self):
        return self.total_spots
    
    def __getitem__(self, idx):
        """Get a single spot from any sample."""
        # Find which sample this index belongs to
        for sample in self.samples:
            if idx < sample['spot_offset'] + sample['n_spots']:
                local_idx = idx - sample['spot_offset']
                return {
                    'x': sample['X'][local_idx],
                    'coords': sample['coords'][local_idx],
                    'sample_id': sample['sample_id'],
                    'local_idx': local_idx,
                    'global_idx': idx
                }
        raise IndexError(f"Index {idx} out of range")
    
    def get_sample(self, sample_id: str) -> Dict:
        """Get all data for a specific sample."""
        for sample in self.samples:
            if sample['sample_id'] == sample_id:
                return sample
        raise ValueError(f"Sample {sample_id} not found")
    
    def get_sample_by_index(self, sample_idx: int) -> Dict:
        """Get all data for a sample by index."""
        return self.samples[sample_idx]
    
    def iter_samples(self):
        """Iterate over all samples."""
        return iter(self.samples)


class MultiSampleSTDataset(Dataset):
    """Dataset for multiple ST samples with per-sample preprocessing and graphs."""
    def __init__(self, adata_list: List[ad.AnnData], sample_ids: Optional[List[str]] = None,
                 n_top_genes: Optional[int] = 2000, n_pca: Optional[int] = 200,
                 normalize: bool = True, k_neighbors: int = 6):
        """
        Args:
            adata_list: List of AnnData objects, one per sample
            sample_ids: Optional list of sample identifiers
            n_top_genes: Number of highly variable genes to select
            n_pca: PCA dimensions (None to skip)
            normalize: Apply normalization
            k_neighbors: Number of neighbors for spatial graph
        """
        self.k_neighbors = k_neighbors
        self.samples = []
        
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(adata_list))]
        
        # First pass: gene selection across all samples
        if n_top_genes:
            print(f"Selecting {n_top_genes} highly variable genes across all samples...")
            combined_adata = ad.concat(adata_list, join='outer', fill_value=0)
            if normalize:
                sc.pp.normalize_total(combined_adata, target_sum=1e4)
                sc.pp.log1p(combined_adata)
            sc.pp.highly_variable_genes(combined_adata, n_top_genes=n_top_genes)
            selected_genes = combined_adata.var_names[combined_adata.var.highly_variable]
            print(f"Selected {len(selected_genes)} genes")
        else:
            selected_genes = None
        
        # Process each sample
        for sample_id, adata in zip(sample_ids, adata_list):
            # Subset to selected genes if applicable
            if selected_genes is not None:
                common_genes = adata.var_names.intersection(selected_genes)
                adata = adata[:, common_genes]
            
            # Normalize
            if normalize:
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
            
            X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
            
            # PCA per sample (or could use shared PCA)
            if n_pca and X.shape[1] > n_pca:
                pca = PCA(n_components=n_pca, random_state=42)
                X = pca.fit_transform(X)
            
            coords = np.vstack([
                adata.obs['x'].astype(float).values,
                adata.obs['y'].astype(float).values
            ]).T.astype(np.float32)
            
            # Build spatial graph
            edge_index, edge_attr = build_spatial_graph(coords, k=k_neighbors)
            
            n_spots = X.shape[0]
            self.samples.append({
                'sample_id': sample_id,
                'X': X.astype(np.float32),
                'coords': coords,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'n_spots': n_spots,
                'spot_offset': 0
            })
        
        # Compute offsets
        cumsum = 0
        for sample in self.samples:
            sample['spot_offset'] = cumsum
            cumsum += sample['n_spots']
        
        self.total_spots = cumsum
        self.n_features = self.samples[0]['X'].shape[1]
    
    def __len__(self):
        return self.total_spots
    
    def __getitem__(self, idx):
        for sample in self.samples:
            if idx < sample['spot_offset'] + sample['n_spots']:
                local_idx = idx - sample['spot_offset']
                return {
                    'x': sample['X'][local_idx],
                    'coords': sample['coords'][local_idx],
                    'sample_id': sample['sample_id'],
                    'local_idx': local_idx,
                    'global_idx': idx
                }
        raise IndexError(f"Index {idx} out of range")
    
    def get_sample(self, sample_id: str) -> Dict:
        for sample in self.samples:
            if sample['sample_id'] == sample_id:
                return sample
        raise ValueError(f"Sample {sample_id} not found")
    
    def get_sample_by_index(self, sample_idx: int) -> Dict:
        return self.samples[sample_idx]
    
    def iter_samples(self):
        return iter(self.samples)


# ========================= GNN MODEL COMPONENTS =========================

class SpatialGAT(nn.Module):
    """
    Multi-layer GAT with edge features, residual connections, and proper normalization.
    Uses attention mechanism to weight spatial neighbors.
    """
    def __init__(self, in_channels: int, hidden: int = 256, out_channels: int = 256,
                 heads: int = 4, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU()
        )
        
        # GAT layers
        for i in range(n_layers):
            if i == 0:
                # First layer
                self.convs.append(GATConv(hidden, hidden // heads, heads=heads, 
                                         concat=True, dropout=dropout, add_self_loops=True))
            elif i == n_layers - 1:
                # Last layer
                self.convs.append(GATConv(hidden, out_channels // heads, heads=heads,
                                         concat=True, dropout=dropout, add_self_loops=True))
            else:
                # Middle layers
                self.convs.append(GATConv(hidden, hidden // heads, heads=heads,
                                         concat=True, dropout=dropout, add_self_loops=True))
            
            self.norms.append(nn.LayerNorm(hidden if i < n_layers - 1 else out_channels))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Residual projections
        self.residual_projs = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden if i == 0 else (hidden if i < n_layers - 1 else hidden)
            out_dim = hidden if i < n_layers - 1 else out_channels
            if in_dim != out_dim:
                self.residual_projs.append(nn.Linear(in_dim, out_dim))
            else:
                self.residual_projs.append(nn.Identity())
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (N, in_channels)
            edge_index: Graph edges (2, num_edges)
        Returns:
            Node embeddings (N, out_channels)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Apply GAT layers with residual connections
        for i, (conv, norm, dropout, res_proj) in enumerate(
            zip(self.convs, self.norms, self.dropouts, self.residual_projs)
        ):
            identity = res_proj(x)
            
            # GAT convolution
            x = conv(x, edge_index)
            x = norm(x)
            
            # Residual connection
            x = x + identity
            
            # Activation and dropout (except last layer)
            if i < self.n_layers - 1:
                x = F.gelu(x)
                x = dropout(x)
        
        return x


class ProjectionHead(nn.Module):
    """Multi-layer projection head for contrastive learning."""
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1, p=2)


class NTXentLoss(nn.Module):
    """Improved NT-Xent loss with numerical stability."""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2: (N, D) L2-normalized embeddings
        """
        batch_size = z1.shape[0]
        
        # Cosine similarity
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature  # (N, N)
        
        # Positive pairs are on diagonal
        positives = torch.diagonal(sim_matrix)
        
        # For each row, denominator is sum over all except self
        nominator = torch.exp(positives)
        denominator = torch.sum(torch.exp(sim_matrix), dim=1)
        
        loss = -torch.log(nominator / (denominator + 1e-8))
        return loss.mean()


# ========================= FULL MODEL =========================

class MSI_ST_GNN_Model(nn.Module):
    """
    Pure GNN architecture for MSI-ST alignment.
    Uses GAT for spatial modeling of both modalities.
    """
    def __init__(self, n_mz_bins: int, n_genes: int, 
                 hidden_dim: int = 256, gnn_layers: int = 3,
                 gat_heads: int = 4, proj_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        
        # MSI encoder: GAT for metabolite spatial patterns
        self.msi_gnn = SpatialGAT(
            in_channels=n_mz_bins,
            hidden=hidden_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            n_layers=gnn_layers,
            dropout=dropout
        )
        
        # ST encoder: GAT for gene expression spatial patterns
        self.st_gnn = SpatialGAT(
            in_channels=n_genes,
            hidden=hidden_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            n_layers=gnn_layers,
            dropout=dropout
        )
        
        # Projection heads for contrastive learning
        self.proj_msi = ProjectionHead(hidden_dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        self.proj_st = ProjectionHead(hidden_dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        
    def encode_msi(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode MSI data through spatial GAT."""
        h = self.msi_gnn(x, edge_index)
        z = self.proj_msi(h)
        return h, z
    
    def encode_st(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode ST data through spatial GAT."""
        h = self.st_gnn(x, edge_index)
        z = self.proj_st(h)
        return h, z
    
    def forward(self, msi_x: torch.Tensor, st_x: torch.Tensor,
                msi_edge_index: torch.Tensor, st_edge_index: torch.Tensor):
        """
        Forward pass through both encoders.
        
        Args:
            msi_x: MSI features (N, n_mz_bins)
            st_x: ST features (N, n_genes)
            msi_edge_index: MSI spatial graph (2, num_edges)
            st_edge_index: ST spatial graph (2, num_edges)
        
        Returns:
            Tuple of (z_msi, z_st, h_msi, h_st)
        """
        h_msi, z_msi = self.encode_msi(msi_x, msi_edge_index)
        h_st, z_st = self.encode_st(st_x, st_edge_index)
        return z_msi, z_st, h_msi, h_st


# ========================= TRAINING =========================

def collate_multi_sample(batch: List[Dict], dataset) -> Dict:
    """
    Collate function that handles spots from multiple samples.
    Creates mini-batches that can span multiple samples.
    """
    # Group by sample_id
    sample_groups = {}
    for item in batch:
        sid = item['sample_id']
        if sid not in sample_groups:
            sample_groups[sid] = []
        sample_groups[sid].append(item)
    
    # Build batch data
    all_x = []
    all_coords = []
    all_edge_indices = []
    all_batch_ids = []
    node_offset = 0
    
    batch_sample_info = []
    
    for batch_idx, (sample_id, items) in enumerate(sample_groups.items()):
        # Get sample data
        sample = dataset.get_sample(sample_id)
        n_nodes_in_batch = len(items)
        
        # Collect node features
        local_indices = [item['local_idx'] for item in items]
        x_batch = sample['X'][local_indices]
        coords_batch = sample['coords'][local_indices]
        
        all_x.append(x_batch)
        all_coords.append(coords_batch)
        all_batch_ids.extend([batch_idx] * n_nodes_in_batch)
        
        # Subgraph extraction: find edges within this subset
        local_indices_set = set(local_indices)
        edge_index = sample['edge_index']
        
        # Filter edges to only those between nodes in batch
        local_to_batch = {local_idx: i for i, local_idx in enumerate(local_indices)}
        
        edge_list = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in local_indices_set and dst in local_indices_set:
                edge_list.append([local_to_batch[src] + node_offset, 
                                 local_to_batch[dst] + node_offset])
        
        if edge_list:
            batch_edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            all_edge_indices.append(batch_edge_index)
        
        batch_sample_info.append({
            'sample_id': sample_id,
            'n_nodes': n_nodes_in_batch,
            'node_offset': node_offset
        })
        
        node_offset += n_nodes_in_batch
    
    # Concatenate everything
    x = torch.from_numpy(np.vstack(all_x))
    coords = torch.from_numpy(np.vstack(all_coords))
    batch_ids = torch.tensor(all_batch_ids, dtype=torch.long)
    
    if all_edge_indices:
        edge_index = torch.cat(all_edge_indices, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return {
        'x': x,
        'coords': coords,
        'edge_index': edge_index,
        'batch': batch_ids,
        'sample_info': batch_sample_info
    }


def train_epoch_multi_sample(model: nn.Module, 
                             msi_dataset: MultiSampleMSIDataset,
                             st_dataset: MultiSampleSTDataset,
                             optimizer: torch.optim.Optimizer,
                             loss_fn: nn.Module,
                             device: torch.device,
                             batch_sampler: Optional[StratifiedBatchSampler] = None,
                             batch_size: int = 512,
                             accumulation_steps: int = 1) -> float:
    """
    Training with stratified sampling across experimental groups.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    # Use stratified sampler if provided, otherwise random sampling
    if batch_sampler is not None:
        batch_iterator = batch_sampler
    else:
        n_spots = min(len(msi_dataset), len(st_dataset))
        indices = torch.randperm(n_spots)
        batch_iterator = [indices[i:min(i + batch_size, n_spots)].tolist() 
                         for i in range(0, n_spots, batch_size)]
    
    optimizer.zero_grad()
    
    for batch_indices in tqdm(batch_iterator, desc="Training"):
        # Get batch data from both modalities
        msi_batch_items = [msi_dataset[idx] for idx in batch_indices]
        st_batch_items = [st_dataset[idx] for idx in batch_indices]
        
        # Collate into proper batch format
        msi_batch = collate_multi_sample(msi_batch_items, msi_dataset)
        st_batch = collate_multi_sample(st_batch_items, st_dataset)
        
        # Move to device
        msi_x = msi_batch['x'].to(device)
        st_x = st_batch['x'].to(device)
        msi_edge_index = msi_batch['edge_index'].to(device)
        st_edge_index = st_batch['edge_index'].to(device)
        
        # Forward pass
        z_msi, z_st, _, _ = model(msi_x, st_x, msi_edge_index, st_edge_index)
        
        # Contrastive loss
        loss = loss_fn(z_msi, z_st)
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights
        if (n_batches + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def validate_multi_sample(model: nn.Module,
                         msi_dataset: MultiSampleMSIDataset,
                         st_dataset: MultiSampleSTDataset,
                         loss_fn: nn.Module,
                         device: torch.device,
                         batch_size: int = 1024) -> Tuple[float, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Validation with per-sample embedding extraction.
    Returns embeddings organized by sample_id.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    # Store embeddings per sample
    msi_embeddings = {}
    st_embeddings = {}
    
    # Process each sample separately for clean embeddings
    for sample_idx in range(len(msi_dataset.samples)):
        msi_sample = msi_dataset.get_sample_by_index(sample_idx)
        st_sample = st_dataset.get_sample_by_index(sample_idx)
        
        sample_id = msi_sample['sample_id']
        print(f"Validating sample: {sample_id}")
        
        # Get full sample data
        msi_x = torch.from_numpy(msi_sample['X']).to(device)
        st_x = torch.from_numpy(st_sample['X']).to(device)
        msi_edge_index = msi_sample['edge_index'].to(device)
        st_edge_index = st_sample['edge_index'].to(device)
        
        # Process in batches if too large
        if msi_x.shape[0] > batch_size:
            msi_z_list, st_z_list = [], []
            
            for i in range(0, msi_x.shape[0], batch_size):
                end_idx = min(i + batch_size, msi_x.shape[0])
                
                # Subgraph for batch (simplified - uses full graph)
                z_msi, z_st, _, _ = model(
                    msi_x[i:end_idx], 
                    st_x[i:end_idx],
                    msi_edge_index, 
                    st_edge_index
                )
                
                msi_z_list.append(z_msi.cpu().numpy())
                st_z_list.append(z_st.cpu().numpy())
                
                # Compute loss on batch
                loss = loss_fn(z_msi, z_st)
                total_loss += loss.item()
                n_batches += 1
            
            msi_embeddings[sample_id] = np.vstack(msi_z_list)
            st_embeddings[sample_id] = np.vstack(st_z_list)
        else:
            # Process entire sample at once
            z_msi, z_st, _, _ = model(msi_x, st_x, msi_edge_index, st_edge_index)
            
            loss = loss_fn(z_msi, z_st)
            total_loss += loss.item()
            n_batches += 1
            
            msi_embeddings[sample_id] = z_msi.cpu().numpy()
            st_embeddings[sample_id] = z_st.cpu().numpy()
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss, msi_embeddings, st_embeddings


# ========================= MAIN PIPELINE =========================

def train_model(msi_adata_paths: List[str], 
                st_adata_paths: List[str],
                sample_ids: Optional[List[str]] = None,
                sample_groups: Optional[List[str]] = None,
                output_dir: str = './results',
                n_epochs: int = 100,
                batch_size: int = 512,
                lr: float = 1e-4,
                use_stratified_sampling: bool = True,
                device: str = 'cuda'):
    """
    Complete training pipeline for multiple samples with experimental design support.
    
    Args:
        msi_adata_paths: List of paths to MSI h5ad files
        st_adata_paths: List of paths to ST h5ad files
        sample_ids: Optional list of sample identifiers
        sample_groups: Optional list of group labels (e.g., 'aged_AD', 'young_control')
        output_dir: Directory to save results
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        use_stratified_sampling: Balance batches across experimental groups
        device: 'cuda' or 'cpu'
    """
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate inputs
    if len(msi_adata_paths) != len(st_adata_paths):
        raise ValueError("Number of MSI and ST samples must match")
    
    n_samples = len(msi_adata_paths)
    if sample_ids is None:
        sample_ids = [f"sample_{i}" for i in range(n_samples)]
    
    if sample_groups is None:
        sample_groups = ['unknown'] * n_samples
        use_stratified_sampling = False
    
    print(f"\n{'='*60}")
    print(f"AD STUDY: Multi-Sample GNN Training Pipeline")
    print(f"{'='*60}")
    print(f"Total samples: {n_samples}")
    
    # Print experimental design
    from collections import Counter
    group_counts = Counter(sample_groups)
    print(f"\nExperimental design:")
    for group, count in sorted(group_counts.items()):
        samples = [sid for sid, g in zip(sample_ids, sample_groups) if g == group]
        print(f"  {group}: {count} samples")
        print(f"    → {', '.join(samples)}")
    
    print(f"\nLoading {n_samples} sample pairs...")
    
    # Load all MSI data
    msi_adata_list = []
    for i, path in enumerate(msi_adata_paths):
        print(f"  MSI {i+1}/{n_samples}: {sample_ids[i]} ({sample_groups[i]})")
        adata = ad.read_h5ad(path)
        msi_adata_list.append(adata)
    
    # Load all ST data
    st_adata_list = []
    for i, path in enumerate(st_adata_paths):
        print(f"  ST  {i+1}/{n_samples}: {sample_ids[i]} ({sample_groups[i]})")
        adata = ad.read_h5ad(path)
        st_adata_list.append(adata)
    
    # Create multi-sample datasets
    print("\n" + "="*60)
    print("Preprocessing MSI data...")
    msi_dataset = MultiSampleMSIDataset(
        msi_adata_list, 
        sample_ids=sample_ids,
        normalize='tic', 
        log_transform=True,
        k_neighbors=6
    )
    
    print("Preprocessing ST data...")
    st_dataset = MultiSampleSTDataset(
        st_adata_list,
        sample_ids=sample_ids,
        n_top_genes=2000,
        n_pca=200,
        normalize=True,
        k_neighbors=6
    )
    
    print(f"\nDataset summary:")
    print(f"  Total MSI spots: {len(msi_dataset):,} across {n_samples} samples")
    print(f"  Total ST spots: {len(st_dataset):,} across {n_samples} samples")
    print(f"  MSI features: {msi_dataset.n_features}")
    print(f"  ST features: {st_dataset.n_features}")
    
    # Print per-sample info
    print("\nPer-sample spot counts:")
    for sample in msi_dataset.samples:
        group = sample_groups[sample_ids.index(sample['sample_id'])]
        print(f"  {sample['sample_id']:20s} ({group:15s}): {sample['n_spots']:5d} spots")
    
    # Create stratified batch sampler if requested
    batch_sampler = None
    if use_stratified_sampling:
        print("\n" + "="*60)
        print("Using stratified batch sampling for balanced training")
        print("="*60)
        
        spots_per_sample = [s['n_spots'] for s in msi_dataset.samples]
        batch_sampler = StratifiedBatchSampler(
            sample_groups=sample_groups,
            spots_per_sample=spots_per_sample,
            batch_size=batch_size,
            shuffle=True
        )
        print(f"Batches per epoch: {len(batch_sampler)}")
    
    # Build model
    print("\n" + "="*60)
    print("Building GNN model...")
    model = MSI_ST_GNN_Model(
        n_mz_bins=msi_dataset.n_features,
        n_genes=st_dataset.n_features,
        hidden_dim=256,
        gnn_layers=3,
        gat_heads=4,
        proj_dim=128,
        dropout=0.1
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    print(f"  Architecture: Pure GAT (Graph Attention Network)")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    loss_fn = NTXentLoss(temperature=0.07)
    
    # Training loop
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*60}")
    print(f"Starting training for {n_epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(n_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch_multi_sample(
            model, msi_dataset, st_dataset, optimizer,
            loss_fn, device, batch_sampler=batch_sampler, 
            batch_size=batch_size, accumulation_steps=1
        )
        train_losses.append(train_loss)
        
        # Validate (per-sample)
        val_loss, msi_embeddings, st_embeddings = validate_multi_sample(
            model, msi_dataset, st_dataset, loss_fn, device
        )
        val_losses.append(val_loss)
        
        scheduler.step()
        
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"  ✓ New best model! (loss: {val_loss:.4f})")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'sample_ids': sample_ids,
                'sample_groups': sample_groups,
            }
            torch.save(checkpoint, os.path.join(output_dir, 'best_model.pt'))
            
            # Save best embeddings per sample
            for sample_id in sample_ids:
                sample_dir = os.path.join(output_dir, 'best_embeddings', sample_id)
                os.makedirs(sample_dir, exist_ok=True)
                
                np.save(os.path.join(sample_dir, 'msi_embeddings.npy'), 
                       msi_embeddings[sample_id])
                np.save(os.path.join(sample_dir, 'st_embeddings.npy'),
                       st_embeddings[sample_id])
        
        # Periodic checkpoints
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }
            torch.save(checkpoint, 
                      os.path.join(output_dir, f'checkpoint_ep{epoch+1}.pt'))
    
    # Final evaluation and save
    print("\n" + "="*60)
    print("Training complete! Generating final embeddings...")
    print("="*60)
    
    _, msi_embeddings, st_embeddings = validate_multi_sample(
        model, msi_dataset, st_dataset, loss_fn, device
    )
    
    # Save embeddings back to AnnData objects
    print("\nSaving annotated data with embeddings...")
    for i, sample_id in enumerate(sample_ids):
        group = sample_groups[i]
        print(f"  {sample_id} ({group})")
        
        # Add embeddings and metadata to AnnData
        msi_adata_list[i].obsm['X_hybrid'] = msi_embeddings[sample_id]
        msi_adata_list[i].obs['group'] = group
        msi_adata_list[i].obs['sample_id'] = sample_id
        
        st_adata_list[i].obsm['X_hybrid'] = st_embeddings[sample_id]
        st_adata_list[i].obs['group'] = group
        st_adata_list[i].obs['sample_id'] = sample_id
        
        # Save augmented AnnData
        sample_dir = os.path.join(output_dir, 'final_embeddings', sample_id)
        os.makedirs(sample_dir, exist_ok=True)
        
        msi_adata_list[i].write_h5ad(
            os.path.join(sample_dir, f'{sample_id}_msi_embeddings.h5ad')
        )
        st_adata_list[i].write_h5ad(
            os.path.join(sample_dir, f'{sample_id}_st_embeddings.h5ad')
        )
    
    # Save training curves and metadata
    import json
    import pandas as pd
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_loss,
        'n_epochs': n_epochs,
        'sample_ids': sample_ids,
        'sample_groups': sample_groups,
        'n_samples': n_samples,
        'experimental_design': dict(group_counts)
    }
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save sample metadata table
    metadata_df = pd.DataFrame({
        'sample_id': sample_ids,
        'group': sample_groups,
        'msi_spots': [s['n_spots'] for s in msi_dataset.samples],
        'st_spots': [s['n_spots'] for s in st_dataset.samples]
    })
    metadata_df.to_csv(os.path.join(output_dir, 'sample_metadata.csv'), index=False)
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"{'='*60}\n")
    
    return model, msi_embeddings, st_embeddings


# ========================= AD STUDY SPECIFIC PIPELINE =========================

def train_ad_study(data_root: str, output_dir: str = './results_ad_study',
                   n_epochs: int = 100, batch_size: int = 512,
                   lr: float = 1e-4, device: str = 'cuda'):
    """
    Convenience function for the AD study with 16 samples.
    
    Args:
        data_root: Root directory containing 'msi/' and 'st/' subdirectories
        output_dir: Where to save results
        n_epochs: Training epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
    """
    
    # Organize the data
    print("Organizing AD study data...")
    data_info = organize_ad_study_data(data_root)
    
    # Train the model
    model, msi_embs, st_embs = train_model(
        msi_adata_paths=data_info['msi_paths'],
        st_adata_paths=data_info['st_paths'],
        sample_ids=data_info['sample_ids'],
        sample_groups=data_info['groups'],
        output_dir=output_dir,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        use_stratified_sampling=True,
        device=device
    )
    
    # Create summary visualizations (optional - requires matplotlib/seaborn)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import json
        
        # Plot training curves
        history = json.load(open(os.path.join(output_dir, 'training_history.json')))
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(history['train_losses'], label='Train Loss', linewidth=2)
        ax.plot(history['val_losses'], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Progress - AD Study (16 samples, GNN)', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300)
        print(f"Saved training curves to {output_dir}/training_curves.png")
        
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    
    return model, msi_embs, st_embs


# ========================= EXAMPLE USAGE =========================

if __name__ == '__main__':
    
    # ===== OPTION 1: Use the AD study convenience function =====
    model, msi_embs, st_embs = train_ad_study(
        data_root='./data',  # Contains msi/ and st/ subdirectories
        output_dir='./results_ad_study_gnn',
        n_epochs=100,
        batch_size=512,
        lr=1e-4,
        device='cuda'
    )
    
    # Access embeddings by sample
    aged_ad_1_msi = msi_embs['aged_AD_1']
    aged_ad_1_st = st_embs['aged_AD_1']
    
    print(f"\nExample embedding shapes:")
    print(f"  aged_AD_1 MSI: {aged_ad_1_msi.shape}")
    print(f"  aged_AD_1 ST:  {aged_ad_1_st.shape}")
    
    # ===== OPTION 2: Manual specification =====
    # If your files are named differently, specify paths manually:
    """
    msi_paths = [
        'data/msi/aged_AD_1.h5ad',
        'data/msi/aged_AD_2.h5ad',
        'data/msi/aged_AD_3.h5ad',
        'data/msi/aged_AD_4.h5ad',
        'data/msi/aged_control_1.h5ad',
        'data/msi/aged_control_2.h5ad',
        'data/msi/aged_control_3.h5ad',
        'data/msi/aged_control_4.h5ad',
        'data/msi/young_AD_1.h5ad',
        'data/msi/young_AD_2.h5ad',
        'data/msi/young_AD_3.h5ad',
        'data/msi/young_AD_4.h5ad',
        'data/msi/young_control_1.h5ad',
        'data/msi/young_control_2.h5ad',
        'data/msi/young_control_3.h5ad',
        'data/msi/young_control_4.h5ad',
    ]
    
    st_paths = [
        'data/st/aged_AD_1.h5ad',
        # ... (same pattern)
    ]
    
    sample_ids = [
        'aged_AD_1', 'aged_AD_2', 'aged_AD_3', 'aged_AD_4',
        'aged_control_1', 'aged_control_2', 'aged_control_3', 'aged_control_4',
        'young_AD_1', 'young_AD_2', 'young_AD_3', 'young_AD_4',
        'young_control_1', 'young_control_2', 'young_control_3', 'young_control_4',
    ]
    
    sample_groups = (
        ['aged_AD'] * 4 + 
        ['aged_control'] * 4 + 
        ['young_AD'] * 4 + 
        ['young_control'] * 4
    )
    
    model, msi_embs, st_embs = train_model(
        msi_adata_paths=msi_paths,
        st_adata_paths=st_paths,
        sample_ids=sample_ids,
        sample_groups=sample_groups,
        output_dir='./results_ad_16samples_gnn',
        n_epochs=100,
        batch_size=512,
        lr=1e-4,
        use_stratified_sampling=True,
        device='cuda'
    )
    """