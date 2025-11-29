import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import Delaunay, distance_matrix
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


# Data configuration
MSI_INPUT_FOLDER = "/home/ajarrah/PhD_Thesis/chapter_2/h5ad_data_processed_4lockmasses_filtered_halfbrain_common/"
MSI_SAMPLE_FILES = [
    "halfbrain_yc_1_filtered_common.h5ad",
    "halfbrain_yc_2_filtered_common.h5ad",
    "halfbrain_yc_3_filtered_common.h5ad",
    "halfbrain_yc_4_filtered_common.h5ad",
    "halfbrain_yad_1_filtered_common.h5ad",
    "halfbrain_yad_2_filtered_common.h5ad",
    "halfbrain_yad_3_filtered_common.h5ad",
    "halfbrain_yad_4_filtered_common.h5ad",
    "halfbrain_ac_1_filtered_common.h5ad",
    "halfbrain_ac_2_filtered_common.h5ad",
    "halfbrain_ac_3_filtered_common.h5ad",
    "halfbrain_ac_4_filtered_common.h5ad",
    "halfbrain_aad_1_filtered_common.h5ad",
    "halfbrain_aad_2_filtered_common.h5ad",
    "halfbrain_aad_3_filtered_common.h5ad",
    "halfbrain_aad_4_filtered_common.h5ad"
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


class SpatialGraphGNN(nn.Module):
    """Graph Neural Network for spatial omics data"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 embedding_dim: int = 64, num_layers: int = 3):
        super(SpatialGraphGNN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * 4))
        
        # Final layer
        self.convs.append(GATConv(hidden_dim * 4, embedding_dim, heads=1, concat=False))
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, batch=None):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        return x


class CrossAnimalPatternMatcher:
    """Match spatial patterns across all animals"""
    
    def __init__(self, output_dir: str = './results', 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.rna_data = {}
        self.msi_data = {}
        self.rna_graphs = {}
        self.msi_graphs = {}
        
        self.groups = ['YC', 'YAD', 'AC', 'AAD']
        
    def load_all_data(self):
        """Load all RNA and MSI data"""
        print("Loading all RNA-seq data...")
        for file, sample_id in zip(RNA_SAMPLE_FILES, RNA_SAMPLE_IDS):
            path = os.path.join(RNA_INPUT_FOLDER, file)
            self.rna_data[sample_id] = sc.read_h5ad(path)
            print(f"  Loaded {sample_id}: {self.rna_data[sample_id].shape}")
        
        print("\nLoading all MSI data...")
        for file, sample_id in zip(MSI_SAMPLE_FILES, MSI_SAMPLE_IDS):
            path = os.path.join(MSI_INPUT_FOLDER, file)
            self.msi_data[sample_id] = sc.read_h5ad(path)
            print(f"  Loaded {sample_id}: {self.msi_data[sample_id].shape}")
    
    def apply_intensity_attenuation(self, adata, sigma: float = 100.0):
        """Apply Gaussian smoothing for intensity attenuation"""
        coords = np.column_stack([adata.obs['x_um'].values, 
                                  adata.obs['y_um'].values])
        
        if hasattr(adata.X, 'toarray'):
            expr = adata.X.toarray()
        else:
            expr = adata.X.copy()
        
        smoothed_expr = np.zeros_like(expr)
        
        for j in range(len(coords)):
            dists = np.sqrt(np.sum((coords - coords[j])**2, axis=1))
            weights = np.exp(-dists**2 / (2 * sigma**2))
            weights /= weights.sum()
            smoothed_expr[j] = expr.T @ weights
        
        return smoothed_expr
    
    def build_spatial_graph(self, adata, sample_id: str, k_neighbors: int = 6) -> Data:
        """Build spatial graph from coordinates"""
        coords = np.column_stack([adata.obs['x_um'].values, 
                                  adata.obs['y_um'].values])
        
        # Build edges using Delaunay triangulation
        try:
            tri = Delaunay(coords)
            edges = set()
            for simplex in tri.simplices:
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        edge = tuple(sorted([simplex[i], simplex[j]]))
                        edges.add(edge)
            edge_list = list(edges)
        except:
            # Fallback to k-NN
            dist_mat = distance_matrix(coords, coords)
            edge_list = []
            for i in range(len(coords)):
                distances = dist_mat[i]
                nearest = np.argsort(distances)[1:k_neighbors+1]
                for j in nearest:
                    edge_list.append([i, j])
        
        # Convert to bidirectional edges
        edge_index = []
        for edge in edge_list:
            edge_index.append([edge[0], edge[1]])
            edge_index.append([edge[1], edge[0]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Node features
        if hasattr(adata, 'X'):
            features = adata.X
            if hasattr(features, 'toarray'):
                features = features.toarray()
        else:
            features = adata.to_df().values
        
        features = torch.tensor(features, dtype=torch.float32)
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        
        data = Data(x=features, edge_index=edge_index, pos=coords_tensor)
        data.sample_id = sample_id
        
        return data
    
    def preprocess_all_data(self, attenuation_sigma: float = 100.0):
        """Preprocess all samples with intensity attenuation"""
        print("\nPreprocessing RNA data with intensity attenuation...")
        for sample_id in RNA_SAMPLE_IDS:
            smoothed = self.apply_intensity_attenuation(
                self.rna_data[sample_id], sigma=attenuation_sigma
            )
            self.rna_data[sample_id].X = smoothed
            print(f"  Processed {sample_id}")
        
        print("\nPreprocessing MSI data with intensity attenuation...")
        for sample_id in MSI_SAMPLE_IDS:
            smoothed = self.apply_intensity_attenuation(
                self.msi_data[sample_id], sigma=attenuation_sigma
            )
            self.msi_data[sample_id].X = smoothed
            print(f"  Processed {sample_id}")
    
    def build_all_graphs(self):
        """Build spatial graphs for all samples"""
        print("\nBuilding RNA spatial graphs...")
        for sample_id in RNA_SAMPLE_IDS:
            self.rna_graphs[sample_id] = self.build_spatial_graph(
                self.rna_data[sample_id], sample_id, k_neighbors=6
            )
            print(f"  {sample_id}: {self.rna_graphs[sample_id].num_nodes} nodes, "
                  f"{self.rna_graphs[sample_id].num_edges} edges")
        
        print("\nBuilding MSI spatial graphs...")
        for sample_id in MSI_SAMPLE_IDS:
            self.msi_graphs[sample_id] = self.build_spatial_graph(
                self.msi_data[sample_id], sample_id, k_neighbors=6
            )
            print(f"  {sample_id}: {self.msi_graphs[sample_id].num_nodes} nodes, "
                  f"{self.msi_graphs[sample_id].num_edges} edges")
    
    def train_gnn_all_samples(self, graphs_dict: Dict[str, Data], input_dim: int,
                              epochs: int = 150, lr: float = 0.001, batch_size: int = 4):
        """Train GNN on all samples together"""
        model = SpatialGraphGNN(input_dim=input_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        
        # Prepare data loader
        graph_list = list(graphs_dict.values())
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            np.random.shuffle(graph_list)
            
            # Process in mini-batches
            for i in range(0, len(graph_list), batch_size):
                batch_graphs = graph_list[i:i+batch_size]
                
                batch_loss = 0
                for data in batch_graphs:
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    
                    # Forward pass
                    embeddings = model(data.x, data.edge_index)
                    
                    # Self-supervised loss: preserve spatial proximity
                    pos = data.pos
                    spatial_dist = torch.cdist(pos, pos)
                    embedding_dist = torch.cdist(embeddings, embeddings)
                    
                    # Normalize
                    spatial_dist = spatial_dist / (spatial_dist.max() + 1e-8)
                    embedding_dist = embedding_dist / (embedding_dist.max() + 1e-8)
                    
                    # Loss
                    loss = F.mse_loss(embedding_dist, spatial_dist)
                    
                    # Add contrastive loss to separate different spatial regions
                    # Sample random pairs
                    n_nodes = embeddings.shape[0]
                    if n_nodes > 100:
                        idx1 = torch.randint(0, n_nodes, (min(100, n_nodes),))
                        idx2 = torch.randint(0, n_nodes, (min(100, n_nodes),))
                        
                        emb_sim = F.cosine_similarity(
                            embeddings[idx1], embeddings[idx2], dim=1
                        )
                        spatial_sim = torch.exp(-spatial_dist[idx1, idx2] / spatial_dist.mean())
                        
                        contrastive_loss = F.mse_loss(emb_sim, spatial_sim)
                        loss = loss + 0.1 * contrastive_loss
                    
                    loss.backward()
                    optimizer.step()
                    batch_loss += loss.item()
                
                total_loss += batch_loss / len(batch_graphs)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/(len(graph_list)//batch_size + 1):.4f}")
        
        return model
    
    def extract_all_embeddings(self, model, graphs_dict: Dict[str, Data]):
        """Extract embeddings for all samples"""
        embeddings_dict = {}
        model.eval()
        
        with torch.no_grad():
            for sample_id, data in graphs_dict.items():
                data = data.to(self.device)
                embeddings = model(data.x, data.edge_index)
                embeddings_dict[sample_id] = embeddings.cpu().numpy()
        
        return embeddings_dict
    
    def find_cross_animal_patterns(self, rna_embeddings: Dict, msi_embeddings: Dict,
                                   top_k: int = 10):
        """Find patterns across all animals"""
        print("\nFinding cross-animal patterns...")
        
        all_matches = {}
        similarity_matrices = {}
        
        # Compare each RNA sample with all MSI samples
        for rna_id in RNA_SAMPLE_IDS:
            all_matches[rna_id] = {}
            similarity_matrices[rna_id] = {}
            
            rna_emb = rna_embeddings[rna_id]
            rna_coords = np.column_stack([
                self.rna_data[rna_id].obs['x_um'].values,
                self.rna_data[rna_id].obs['y_um'].values
            ])
            
            for msi_id in MSI_SAMPLE_IDS:
                msi_emb = msi_embeddings[msi_id]
                msi_coords = np.column_stack([
                    self.msi_data[msi_id].obs['x_um'].values,
                    self.msi_data[msi_id].obs['y_um'].values
                ])
                
                # Compute similarity
                similarity = cosine_similarity(rna_emb, msi_emb)
                similarity_matrices[rna_id][msi_id] = similarity
                
                # Find top matches
                matches = []
                for i in range(len(rna_emb)):
                    top_indices = np.argsort(similarity[i])[-top_k:][::-1]
                    top_scores = similarity[i][top_indices]
                    
                    matches.append({
                        'rna_idx': i,
                        'rna_coord': rna_coords[i],
                        'msi_matches': top_indices,
                        'msi_coords': msi_coords[top_indices],
                        'similarity_scores': top_scores
                    })
                
                all_matches[rna_id][msi_id] = matches
                
                avg_sim = np.mean([m['similarity_scores'].mean() for m in matches])
                print(f"  {rna_id} <-> {msi_id}: avg similarity = {avg_sim:.3f}")
        
        return all_matches, similarity_matrices
    
    def analyze_group_patterns(self, all_matches: Dict, similarity_matrices: Dict):
        """Analyze patterns within and across groups"""
        print("\n" + "="*70)
        print("GROUP-LEVEL PATTERN ANALYSIS")
        print("="*70)
        
        results = []
        
        for rna_id in RNA_SAMPLE_IDS:
            rna_group = rna_id.split('_')[0]
            
            for msi_id in MSI_SAMPLE_IDS:
                msi_group = msi_id.split('_')[0]
                
                matches = all_matches[rna_id][msi_id]
                avg_similarity = np.mean([m['similarity_scores'].mean() for m in matches])
                max_similarity = np.max([m['similarity_scores'].max() for m in matches])
                
                is_same_group = (rna_group == msi_group)
                is_same_animal = (rna_id == msi_id)
                
                results.append({
                    'RNA_Sample': rna_id,
                    'RNA_Group': rna_group,
                    'MSI_Sample': msi_id,
                    'MSI_Group': msi_group,
                    'Same_Group': is_same_group,
                    'Same_Animal': is_same_animal,
                    'Avg_Similarity': avg_similarity,
                    'Max_Similarity': max_similarity
                })
        
        results_df = pd.DataFrame(results)
        
        # Summary statistics
        print("\nWithin-Group vs Cross-Group Comparison:")
        print("-" * 70)
        
        for group in self.groups:
            within_group = results_df[
                (results_df['RNA_Group'] == group) & 
                (results_df['MSI_Group'] == group) &
                (results_df['Same_Animal'] == False)
            ]['Avg_Similarity']
            
            cross_group = results_df[
                (results_df['RNA_Group'] == group) & 
                (results_df['MSI_Group'] != group)
            ]['Avg_Similarity']
            
            print(f"\n{group}:")
            print(f"  Within-group (different animals): {within_group.mean():.4f} ± {within_group.std():.4f}")
            print(f"  Cross-group: {cross_group.mean():.4f} ± {cross_group.std():.4f}")
            print(f"  Difference: {within_group.mean() - cross_group.mean():.4f}")
        
        # Save results
        results_df.to_csv(os.path.join(self.output_dir, 'cross_animal_pattern_results.csv'), 
                         index=False)
        
        return results_df
    
    def visualize_cross_animal_heatmap(self, similarity_matrices: Dict):
        """Create heatmap of cross-animal similarities"""
        print("\nGenerating cross-animal similarity heatmap...")
        
        # Compute average similarity between each pair
        avg_similarities = np.zeros((len(RNA_SAMPLE_IDS), len(MSI_SAMPLE_IDS)))
        
        for i, rna_id in enumerate(RNA_SAMPLE_IDS):
            for j, msi_id in enumerate(MSI_SAMPLE_IDS):
                sim_matrix = similarity_matrices[rna_id][msi_id]
                avg_similarities[i, j] = sim_matrix.mean()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        
        im = ax.imshow(avg_similarities, cmap='RdYlBu_r', aspect='auto')
        
        ax.set_xticks(np.arange(len(MSI_SAMPLE_IDS)))
        ax.set_yticks(np.arange(len(RNA_SAMPLE_IDS)))
        ax.set_xticklabels(MSI_SAMPLE_IDS, rotation=45, ha='right')
        ax.set_yticklabels(RNA_SAMPLE_IDS)
        
        ax.set_xlabel('MSI Samples', fontsize=12, fontweight='bold')
        ax.set_ylabel('RNA Samples', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Animal Pattern Similarity Matrix\n(RNA-seq vs Mass Spectrometry)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add group boundaries
        for i in [3.5, 7.5, 11.5]:
            ax.axhline(i, color='black', linewidth=2)
            ax.axvline(i, color='black', linewidth=2)
        
        # Add values
        for i in range(len(RNA_SAMPLE_IDS)):
            for j in range(len(MSI_SAMPLE_IDS)):
                text = ax.text(j, i, f'{avg_similarities[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Average Similarity')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cross_animal_similarity_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {self.output_dir}/cross_animal_similarity_heatmap.png")
    
    def visualize_best_matches_per_group(self, all_matches: Dict, 
                                        similarity_matrices: Dict):
        """Visualize best matches for each group"""
        print("\nGenerating best match visualizations per group...")
        
        for group in self.groups:
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            axes = axes.flatten()
            
            group_samples = [s for s in RNA_SAMPLE_IDS if s.startswith(group)]
            
            for idx, rna_id in enumerate(group_samples):
                ax = axes[idx]
                
                # Find best matching MSI sample (from same group, different animal)
                best_msi = None
                best_score = -1
                
                for msi_id in MSI_SAMPLE_IDS:
                    if msi_id.startswith(group) and msi_id != rna_id:
                        score = similarity_matrices[rna_id][msi_id].mean()
                        if score > best_score:
                            best_score = score
                            best_msi = msi_id
                
                if best_msi:
                    # Plot RNA spots
                    rna_coords = np.column_stack([
                        self.rna_data[rna_id].obs['x_um'].values,
                        self.rna_data[rna_id].obs['y_um'].values
                    ])
                    
                    # Plot MSI pixels
                    msi_coords = np.column_stack([
                        self.msi_data[best_msi].obs['x_um'].values,
                        self.msi_data[best_msi].obs['y_um'].values
                    ])
                    
                    # Normalize coordinates for visualization
                    rna_norm = (rna_coords - rna_coords.min(axis=0)) / (rna_coords.max(axis=0) - rna_coords.min(axis=0))
                    msi_norm = (msi_coords - msi_coords.min(axis=0)) / (msi_coords.max(axis=0) - msi_coords.min(axis=0))
                    
                    ax.scatter(rna_norm[:, 0], rna_norm[:, 1], 
                             c='blue', alpha=0.4, s=30, label='RNA')
                    ax.scatter(msi_norm[:, 0], msi_norm[:, 1], 
                             c='red', alpha=0.4, s=20, label='MSI')
                    
                    # Draw top matches
                    matches = all_matches[rna_id][best_msi]
                    sorted_matches = sorted(matches, 
                                          key=lambda x: x['similarity_scores'].mean(), 
                                          reverse=True)[:20]
                    
                    for match in sorted_matches:
                        rna_coord_norm = (match['rna_coord'] - rna_coords.min(axis=0)) / (rna_coords.max(axis=0) - rna_coords.min(axis=0))
                        for msi_coord in match['msi_coords'][:2]:
                            msi_coord_norm = (msi_coord - msi_coords.min(axis=0)) / (msi_coords.max(axis=0) - msi_coords.min(axis=0))
                            ax.plot([rna_coord_norm[0], msi_coord_norm[0]], 
                                   [rna_coord_norm[1], msi_coord_norm[1]], 
                                   'g-', alpha=0.2, linewidth=0.5)
                    
                    ax.set_title(f'{rna_id} <-> {best_msi}\nSimilarity: {best_score:.3f}', 
                               fontweight='bold')
                    ax.legend()
                    ax.set_aspect('equal')
            
            plt.suptitle(f'Best Cross-Animal Matches - Group {group}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'best_matches_{group}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved {group} visualization")


def main_cross_animal_analysis():
    """Main analysis pipeline for cross-animal pattern matching"""
    
    print("="*70)
    print("CROSS-ANIMAL SPATIAL PATTERN MATCHING")
    print("RNA-seq (Visium) vs Mass Spectrometry")
    print("="*70)
    
    # Initialize matcher
    matcher = CrossAnimalPatternMatcher(output_dir='./cross_animal_results')
    
    # Load all data
    matcher.load_all_data()
    
    # Preprocess with intensity attenuation
    matcher.preprocess_all_data(attenuation_sigma=100.0)
    
    # Build spatial graphs
    matcher.build_all_graphs()
    
    # Get feature dimensions
    rna_dim = list(matcher.rna_data.values())[0].shape[1]
    msi_dim = list(matcher.msi_data.values())[0].shape[1]
    
    print(f"\nFeature dimensions:")
    print(f"  RNA: {rna_dim} genes")
    print(f"  MSI: {msi_dim} m/z features")
    
    # Train GNN models on all samples
    print("\n" + "="*70)
    print("Training GNN on all RNA samples...")
    print("="*70)
    gnn_rna = matcher.train_gnn_all_samples(
        matcher.rna_graphs, input_dim=rna_dim, epochs=150, lr=0.001, batch_size=4
    )
    
    print("\n" + "="*70)
    print("Training GNN on all MSI samples...")
    print("="*70)
    gnn_msi = matcher.train_gnn_all_samples(
        matcher.msi_graphs, input_dim=msi_dim, epochs=150, lr=0.001, batch_size=4
    )
    
    # Extract embeddings
    print("\nExtracting embeddings for all samples...")
    rna_embeddings = matcher.extract_all_embeddings(gnn_rna, matcher.rna_graphs)
    msi_embeddings = matcher.extract_all_embeddings(gnn_msi, matcher.msi_graphs)
    
    # Find cross-animal patterns
    all_matches, similarity_matrices = matcher.find_cross_animal_patterns(
        rna_embeddings, msi_embeddings, top_k=10
    )
    
    # Analyze group patterns
    results_df = matcher.analyze_group_patterns(all_matches, similarity_matrices)
    
    # Visualizations
    matcher.visualize_cross_animal_heatmap(similarity_matrices)
    matcher.visualize_best_matches_per_group(all_matches, similarity_matrices)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"Results saved to: {matcher.output_dir}/")
    print("  - cross_animal_pattern_results.csv")
    print("  - cross_animal_similarity_heatmap.png")
    print("  - best_matches_[GROUP].png (for each group)")
    
    return matcher, all_matches, similarity_matrices, results_df


if __name__ == "__main__":
    matcher, all_matches, similarity_matrices, results_df = main_cross_animal_analysis()