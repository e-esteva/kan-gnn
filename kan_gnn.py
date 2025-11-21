import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import Delaunay, KDTree
from sklearn.preprocessing import StandardScaler
import pandas as pd


# ============================================================================
# KAN Layer Implementation
# ============================================================================

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer with learnable B-spline basis functions.
    
    This replaces traditional linear + activation with learnable activation functions
    that can adapt to the specific patterns in cell-cell interactions.
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        grid_size: int = 5,
        spline_order: int = 3,
        base_activation: nn.Module = nn.SiLU(),
        grid_range: Tuple[float, float] = (-1, 1)
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Base linear transformation (residual connection)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.base_activation = base_activation
        
        # Spline parameters for learnable activations
        # Each output neuron has its own spline for each input feature
        self.grid_min, self.grid_max = grid_range
        self.register_buffer(
            'grid', 
            torch.linspace(self.grid_min, self.grid_max, grid_size + 2 * spline_order + 1)
        )
        
        # Spline coefficients: [out_features, in_features, num_spline_bases]
        num_bases = grid_size + spline_order
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, num_bases) * 0.1
        )
        
    def b_spline_basis(self, x: torch.Tensor, i: int, k: int) -> torch.Tensor:
        """Compute B-spline basis function recursively (Cox-de Boor formula)"""
        if k == 0:
            return ((x >= self.grid[i]) & (x < self.grid[i + 1])).float()
        
        # Avoid division by zero
        denom1 = self.grid[i + k] - self.grid[i]
        denom2 = self.grid[i + k + 1] - self.grid[i + 1]
        
        term1 = 0
        if denom1 > 1e-8:
            term1 = (x - self.grid[i]) / denom1 * self.b_spline_basis(x, i, k - 1)
        
        term2 = 0
        if denom2 > 1e-8:
            term2 = (self.grid[i + k + 1] - x) / denom2 * self.b_spline_basis(x, i + 1, k - 1)
        
        return term1 + term2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, in_features]
        Returns:
            Output tensor of shape [batch_size, out_features]
        """
        batch_size = x.shape[0]
        
        # Base transformation (residual pathway)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Spline transformation (learnable activation pathway)
        # Clamp input to grid range for numerical stability
        x_clamped = torch.clamp(x, self.grid_min * 0.99, self.grid_max * 0.99)
        
        # Compute all B-spline basis functions efficiently
        num_bases = self.grid_size + self.spline_order
        spline_output = torch.zeros(
            batch_size, self.out_features, device=x.device, dtype=x.dtype
        )
        
        # For each input feature
        for in_idx in range(self.in_features):
            x_feat = x_clamped[:, in_idx].unsqueeze(-1)  # [batch_size, 1]
            
            # Compute basis functions for this feature
            basis_values = []
            for i in range(num_bases):
                basis = self.b_spline_basis(x_feat, i, self.spline_order)
                basis_values.append(basis)
            
            basis_values = torch.cat(basis_values, dim=-1)  # [batch_size, num_bases]
            
            # Weighted combination of basis functions
            # [batch_size, num_bases] @ [num_bases, out_features]
            contribution = basis_values @ self.spline_weight[:, in_idx, :].T
            spline_output += contribution
        
        return base_output + spline_output


# ============================================================================
# Spatial Graph Construction
# ============================================================================

class SpatialGraphBuilder:
    """
    Construct spatial graphs from cell centroid coordinates.
    Supports multiple graph construction strategies.
    """
    
    @staticmethod
    def from_knn(
        centroids: np.ndarray, 
        k: int = 5,
        include_self: bool = False
    ) -> np.ndarray:
        """
        Construct k-nearest neighbors graph.
        
        Args:
            centroids: Cell coordinates [num_cells, 2] or [num_cells, 3]
            k: Number of neighbors
            include_self: Whether to include self-loops
            
        Returns:
            edge_index: [2, num_edges]
        """
        tree = KDTree(centroids)
        distances, indices = tree.query(centroids, k=k+1)  # +1 because query includes self
        
        edge_list = []
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if include_self or i != j:
                    edge_list.append([i, j])
        
        return np.array(edge_list).T
    
    @staticmethod
    def from_radius(
        centroids: np.ndarray,
        radius: float,
        max_neighbors: Optional[int] = None
    ) -> np.ndarray:
        """
        Construct radius-based graph (all neighbors within distance).
        
        Args:
            centroids: Cell coordinates [num_cells, 2] or [num_cells, 3]
            radius: Maximum distance for connectivity
            max_neighbors: Optional limit on neighbors per node
            
        Returns:
            edge_index: [2, num_edges]
        """
        tree = KDTree(centroids)
        edge_list = []
        
        for i, coord in enumerate(centroids):
            neighbors = tree.query_ball_point(coord, radius)
            
            # Remove self and optionally limit neighbors
            neighbors = [n for n in neighbors if n != i]
            if max_neighbors is not None:
                # Keep closest neighbors
                neighbor_dists = np.linalg.norm(centroids[neighbors] - coord, axis=1)
                closest_idx = np.argsort(neighbor_dists)[:max_neighbors]
                neighbors = [neighbors[idx] for idx in closest_idx]
            
            for j in neighbors:
                edge_list.append([i, j])
        
        return np.array(edge_list).T if edge_list else np.array([[], []])
    
    @staticmethod
    def from_delaunay(centroids: np.ndarray) -> np.ndarray:
        """
        Construct Delaunay triangulation graph (biologically motivated).
        
        Args:
            centroids: Cell coordinates [num_cells, 2]
            
        Returns:
            edge_index: [2, num_edges]
        """
        if centroids.shape[1] != 2:
            raise ValueError("Delaunay triangulation requires 2D coordinates")
        
        tri = Delaunay(centroids)
        edge_set = set()
        
        for simplex in tri.simplices:
            # Add all edges of the triangle
            for i in range(3):
                for j in range(i+1, 3):
                    v1, v2 = simplex[i], simplex[j]
                    edge_set.add((min(v1, v2), max(v1, v2)))
        
        edge_list = [[e[0], e[1]] for e in edge_set] + [[e[1], e[0]] for e in edge_set]
        return np.array(edge_list).T
    
    @staticmethod
    def from_neighbor_indices(neighbor_indices: np.ndarray) -> np.ndarray:
        """
        Construct graph from pre-computed neighbor indices.
        
        Args:
            neighbor_indices: [num_cells, num_neighbors] array where each row
                            contains indices of neighbors for that cell
                            
        Returns:
            edge_index: [2, num_edges]
        """
        edge_list = []
        num_cells = neighbor_indices.shape[0]
        
        for i in range(num_cells):
            for j in neighbor_indices[i]:
                if j >= 0:  # -1 can indicate no neighbor
                    edge_list.append([i, int(j)])
        
        return np.array(edge_list).T if edge_list else np.array([[], []])
    
    @staticmethod
    def add_edge_features(
        edge_index: np.ndarray,
        centroids: np.ndarray,
        feature_types: List[str] = ['distance', 'direction']
    ) -> Dict[str, np.ndarray]:
        """
        Compute edge features (useful for edge-conditioned GNNs).
        
        Args:
            edge_index: [2, num_edges]
            centroids: [num_cells, spatial_dim]
            feature_types: List of features to compute
                - 'distance': Euclidean distance
                - 'direction': Normalized direction vector
                - 'relative_position': Unnormalized direction vector
                
        Returns:
            Dictionary of edge features
        """
        src, dst = edge_index[0], edge_index[1]
        src_coords = centroids[src]
        dst_coords = centroids[dst]
        
        edge_features = {}
        
        if 'distance' in feature_types:
            distances = np.linalg.norm(dst_coords - src_coords, axis=1, keepdims=True)
            edge_features['distance'] = distances
        
        if 'direction' in feature_types:
            diff = dst_coords - src_coords
            distances = np.linalg.norm(diff, axis=1, keepdims=True)
            directions = diff / (distances + 1e-8)
            edge_features['direction'] = directions
        
        if 'relative_position' in feature_types:
            edge_features['relative_position'] = dst_coords - src_coords
        
        return edge_features


# ============================================================================
# Data Loading and Integration
# ============================================================================

class CellGraphDataset:
    """
    Load and process cell data into graph format.
    Integrates with Cellpose outputs or custom data formats.
    """
    
    def __init__(
        self,
        normalize_features: bool = True,
        graph_type: str = 'knn',
        graph_params: Optional[Dict] = None
    ):
        """
        Args:
            normalize_features: Whether to standardize biomarker features
            graph_type: One of ['knn', 'radius', 'delaunay', 'precomputed']
            graph_params: Parameters for graph construction
        """
        self.normalize_features = normalize_features
        self.scaler = StandardScaler() if normalize_features else None
        self.graph_type = graph_type
        self.graph_params = graph_params or {}
        self.graph_builder = SpatialGraphBuilder()
        
    def from_cellpose_output(
        self,
        masks: np.ndarray,
        biomarkers: np.ndarray,
        labels: Optional[np.ndarray] = None,
        image_coords: Optional[np.ndarray] = None
    ) -> Data:
        """
        Create graph from Cellpose segmentation masks.
        
        Args:
            masks: Cellpose output masks [H, W] with cell IDs
            biomarkers: Biomarker measurements [num_cells, num_markers]
            labels: Optional cell type labels [num_cells]
            image_coords: Optional image coordinates for spatial context
            
        Returns:
            PyTorch Geometric Data object
        """
        # Extract centroids from masks
        cell_ids = np.unique(masks)[1:]  # Exclude background (0)
        num_cells = len(cell_ids)
        centroids = np.zeros((num_cells, 2))
        
        for idx, cell_id in enumerate(cell_ids):
            cell_mask = masks == cell_id
            y_coords, x_coords = np.where(cell_mask)
            centroids[idx] = [x_coords.mean(), y_coords.mean()]
        
        # Create graph data
        return self.from_arrays(
            centroids=centroids,
            features=biomarkers,
            labels=labels
        )
    
    def from_arrays(
        self,
        centroids: np.ndarray,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        neighbor_indices: Optional[np.ndarray] = None
    ) -> Data:
        """
        Create graph from numpy arrays.
        
        Args:
            centroids: Cell coordinates [num_cells, 2 or 3]
            features: Biomarker measurements [num_cells, num_features]
            labels: Cell type labels [num_cells]
            neighbor_indices: Pre-computed neighbors [num_cells, num_neighbors]
            
        Returns:
            PyTorch Geometric Data object
        """
        num_cells = centroids.shape[0]
        
        # Normalize features
        if self.normalize_features and self.scaler is not None:
            features = self.scaler.fit_transform(features)
        
        # Construct graph
        if self.graph_type == 'precomputed' and neighbor_indices is not None:
            edge_index = self.graph_builder.from_neighbor_indices(neighbor_indices)
        elif self.graph_type == 'knn':
            k = self.graph_params.get('k', 5)
            edge_index = self.graph_builder.from_knn(centroids, k=k)
        elif self.graph_type == 'radius':
            radius = self.graph_params.get('radius', 50.0)
            max_neighbors = self.graph_params.get('max_neighbors', 10)
            edge_index = self.graph_builder.from_radius(
                centroids, radius=radius, max_neighbors=max_neighbors
            )
        elif self.graph_type == 'delaunay':
            edge_index = self.graph_builder.from_delaunay(centroids)
        else:
            raise ValueError(f"Unknown graph_type: {self.graph_type}")
        
        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(features),
            edge_index=torch.LongTensor(edge_index),
            pos=torch.FloatTensor(centroids),
            num_nodes=num_cells
        )
        
        if labels is not None:
            data.y = torch.LongTensor(labels)
        
        return data
    
    def from_dataframe(
        self,
        df: pd.DataFrame,
        centroid_cols: List[str],
        feature_cols: List[str],
        label_col: Optional[str] = None,
        neighbor_cols: Optional[List[str]] = None
    ) -> Data:
        """
        Create graph from pandas DataFrame.
        
        Args:
            df: DataFrame with cell data
            centroid_cols: Column names for coordinates (e.g., ['x', 'y'])
            feature_cols: Column names for biomarkers
            label_col: Column name for cell type labels
            neighbor_cols: Column names for pre-computed neighbors
            
        Returns:
            PyTorch Geometric Data object
        """
        centroids = df[centroid_cols].values
        features = df[feature_cols].values
        labels = df[label_col].values if label_col else None
        neighbor_indices = df[neighbor_cols].values if neighbor_cols else None
        
        return self.from_arrays(centroids, features, labels, neighbor_indices)


# ============================================================================
# KAN-based GNN Layers
# ============================================================================

class KANConv(MessagePassing):
    """
    Graph Convolutional layer using KAN for message transformation.
    
    This replaces the standard MLP in GCN with a KAN, allowing the network
    to learn complex, interpretable functions for aggregating neighbor information.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        grid_size: int = 5,
        aggr: str = 'add',
        **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)
        
        # KAN for transforming concatenated [node_features, neighbor_features]
        self.kan = KANLayer(
            in_features=in_channels * 2,  # Self + neighbor
            out_features=out_channels,
            grid_size=grid_size
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
        """
        # Add self-loops for node's own information
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Normalize by degree
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Start propagating messages
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """
        Construct messages from neighbors.
        
        Args:
            x_i: Central node features [num_edges, in_channels]
            x_j: Neighbor node features [num_edges, in_channels]
            norm: Normalization coefficients [num_edges]
        """
        # Concatenate node's features with neighbor's features
        combined = torch.cat([x_i, x_j], dim=-1)
        
        # Transform using KAN (learns complex interaction functions)
        message = self.kan(combined)
        
        # Normalize
        return norm.view(-1, 1) * message


class KANGATConv(MessagePassing):
    """
    Graph Attention layer using KAN for attention computation and message transformation.
    
    This learns which neighbors are important (attention) and how to combine their
    information (KAN transformation) in a single, interpretable framework.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        grid_size: int = 5,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # KAN for computing attention logits
        self.att_kan = KANLayer(
            in_features=2 * in_channels,
            out_features=heads,
            grid_size=grid_size
        )
        
        # KAN for message transformation
        self.msg_kan = KANLayer(
            in_features=in_channels,
            out_features=heads * out_channels,
            grid_size=grid_size
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        pass
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
        """
        # Transform node features
        x = self.msg_kan(x).view(-1, self.heads, self.out_channels)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x)
        
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                edge_index_i: torch.Tensor, size_i: int) -> torch.Tensor:
        """
        Compute attention-weighted messages.
        """
        # Flatten for attention computation
        x_i_flat = x_i.view(-1, self.heads * self.out_channels)
        x_j_flat = x_j.view(-1, self.heads * self.out_channels)
        
        # Compute attention coefficients using KAN
        combined = torch.cat([x_i_flat, x_j_flat], dim=-1)
        alpha = self.att_kan(combined)  # [num_edges, heads]
        alpha = F.leaky_relu(alpha, 0.2)
        
        # Softmax per node
        alpha = torch.softmax(alpha, dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention to messages
        return x_j * alpha.unsqueeze(-1)


# ============================================================================
# Complete KAN-GNN Model
# ============================================================================

class KAN_GNN(nn.Module):
    """
    Complete Graph Neural Network using KAN layers for cell graph analysis.
    
    This model can predict cell types/states based on:
    1. Cell's own features (morphology, marker expression, etc.)
    2. Neighborhood context (types and states of neighboring cells)
    3. Complex, learnable interaction rules between cells
    """
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        grid_size: int = 5,
        dropout: float = 0.1,
        use_attention: bool = False,
        heads: int = 4
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Input transformation
        self.input_kan = KANLayer(num_node_features, hidden_channels, grid_size=grid_size)
        
        # GNN layers with KAN
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if use_attention:
                conv = KANGATConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    grid_size=grid_size,
                    dropout=dropout
                )
            else:
                conv = KANConv(hidden_channels, hidden_channels, grid_size=grid_size)
            
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # Output layer
        self.output_kan = KANLayer(hidden_channels, num_classes, grid_size=grid_size)
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, num_node_features]
                - edge_index: Graph connectivity [2, num_edges]
        
        Returns:
            Node predictions [num_nodes, num_classes]
        """
        x, edge_index = data.x, data.edge_index
        
        # Input transformation
        x = self.input_kan(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GNN layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_res = x  # Residual connection
            x = conv(x, edge_index)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0:  # Skip first layer residual
                x = x + x_res
        
        # Output
        x = self.output_kan(x)
        
        return x
    
    def interpret_interactions(
        self, 
        layer_idx: int = 0,
        feature_idx: int = 0, 
        output_idx: int = 0,
        feature_name: str = "Feature",
        output_name: str = "Output"
    ) -> None:
        """
        Visualize learned activation functions to interpret cell-cell interactions.
        
        Args:
            layer_idx: Which GNN layer to interpret (0 to num_layers-1)
            feature_idx: Which input feature to visualize
            output_idx: Which output dimension to visualize
            feature_name: Name of the feature for plot label
            output_name: Name of the output for plot label
        """
        # Get the specified conv layer's KAN
        conv = self.convs[layer_idx]
        kan_layer = conv.kan
        
        # Generate input range
        x_range = torch.linspace(
            kan_layer.grid_min, 
            kan_layer.grid_max, 
            200
        )
        
        # Create dummy input with zeros except for the feature of interest
        dummy_input = torch.zeros(200, kan_layer.in_features)
        dummy_input[:, feature_idx] = x_range
        
        # Compute activation
        with torch.no_grad():
            # Only compute for specified feature and output
            activation = []
            for x_val in dummy_input:
                out = kan_layer(x_val.unsqueeze(0))
                activation.append(out[0, output_idx].item())
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_range.numpy(), activation, linewidth=2, color='#2E86AB')
        plt.xlabel(f'{feature_name} Value (Normalized)', fontsize=12)
        plt.ylabel(f'Contribution to {output_name}', fontsize=12)
        plt.title(f'Learned Interaction Function: Layer {layer_idx}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"\nInterpretation Guide:")
        print(f"- Linear region: Proportional response to {feature_name}")
        print(f"- Threshold: Sharp transition indicates decision boundary")
        print(f"- Non-monotonic: Complex interaction (e.g., high AND low activate)")
        print(f"- Saturation: Signal plateaus at extreme values")


# ============================================================================
# Training Utilities
# ============================================================================

def train_epoch(model: nn.Module, data: Data, optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, mask: torch.Tensor) -> float:
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()
    
    out = model(data)
    loss = criterion(out[mask], data.y[mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model: nn.Module, data: Data, mask: torch.Tensor) -> Tuple[float, float]:
    """Evaluate model"""
    model.eval()
    out = model(data)
    pred = out[mask].argmax(dim=1)
    
    correct = (pred == data.y[mask]).sum().item()
    accuracy = correct / mask.sum().item()
    
    loss = F.cross_entropy(out[mask], data.y[mask]).item()
    
    return accuracy, loss


def visualize_graph(data: Data, predictions: Optional[torch.Tensor] = None, 
                   true_labels: Optional[torch.Tensor] = None, max_edges: int = 500):
    """
    Visualize the cell graph with predictions.
    
    Args:
        data: PyG Data object with 'pos' attribute for coordinates
        predictions: Model predictions [num_nodes]
        true_labels: Ground truth labels [num_nodes]
        max_edges: Maximum edges to draw (for visualization clarity)
    """
    fig, axes = plt.subplots(1, 2 if predictions is not None else 1, 
                             figsize=(15, 6) if predictions is not None else (8, 6))
    
    if predictions is None:
        axes = [axes]
    
    pos = data.pos.numpy()
    edge_index = data.edge_index.numpy()
    
    # Plot true labels
    ax = axes[0]
    ax.set_title('True Cell Types', fontsize=14, fontweight='bold')
    
    # Draw edges (sample if too many)
    if edge_index.shape[1] > max_edges:
        edge_sample = np.random.choice(edge_index.shape[1], max_edges, replace=False)
        edge_index_plot = edge_index[:, edge_sample]
    else:
        edge_index_plot = edge_index
    
    for i in range(edge_index_plot.shape[1]):
        src, dst = edge_index_plot[:, i]
        ax.plot([pos[src, 0], pos[dst, 0]], 
               [pos[src, 1], pos[dst, 1]], 
               'gray', alpha=0.2, linewidth=0.5)
    
    # Draw nodes
    if true_labels is not None:
        scatter = ax.scatter(pos[:, 0], pos[:, 1], c=true_labels.numpy(), 
                           cmap='tab10', s=50, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Cell Type')
    else:
        ax.scatter(pos[:, 0], pos[:, 1], s=50, edgecolors='black', linewidth=0.5)
    
    ax.set
