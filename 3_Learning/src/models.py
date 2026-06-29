"""
models.py

PyTorch Geometric models for DOM graph classification.
Supports both node-level and graph-level predictions.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, BatchNorm1d, Dropout, Embedding
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, GlobalAttention


class DOMGCN(torch.nn.Module):
    """
    GCN-based encoder for DOM graphs.
    
    Args:
        num_tags: Number of unique HTML tags for embedding
        tag_embed_dim: Dimension of tag embeddings
        attr_dim: Dimension of attribute features
        text_dim: Dimension of text embeddings
        hidden_dim: Hidden dimension for GCN layers
        num_node_classes: Number of node-level classes (e.g., 2 for binary)
        num_graph_classes: Number of graph-level classes
        num_layers: Number of GCN layers
        dropout: Dropout rate
        pooling: Pooling method ('mean', 'max', 'attention')
    """
    
    def __init__(
        self,
        num_tags: int = 116,
        tag_embed_dim: int = 32,
        attr_dim: int = 134,  # From html_graph_builder attribute features
        text_dim: int = 384,
        hidden_dim: int = 128,
        num_node_classes: int = 2,
        num_graph_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
        pooling: str = "mean",
    ):
        super().__init__()
        
        self.tag_embed_dim = tag_embed_dim
        self.attr_dim = attr_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Tag embedding
        self.tag_embedding = Embedding(num_tags, tag_embed_dim)
        
        # Input projection: combine tag + attributes + text into hidden_dim
        input_dim = tag_embed_dim + attr_dim + text_dim
        self.input_proj = Linear(input_dim, hidden_dim)
        self.input_bn = BatchNorm1d(hidden_dim)
        
        # GCN layers
        self.convs = ModuleList()
        self.bns = ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm1d(hidden_dim))
        
        # Pooling for graph-level readout
        self.pooling = pooling
        if pooling == "attention":
            self.pool_gate = Linear(hidden_dim, 1)
            self.pool = GlobalAttention(self.pool_gate)
        
        # Node-level classifier
        self.node_classifier = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, num_node_classes),
        )
        
        # Graph-level classifier
        graph_input_dim = hidden_dim * 2 if pooling == "meanmax" else hidden_dim
        self.graph_classifier = torch.nn.Sequential(
            Linear(graph_input_dim, hidden_dim),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, num_graph_classes),
        )
    
    def forward(self, x, edge_index, tag_indices, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features [N, attr_dim + text_dim]
            edge_index: Edge indices [2, E]
            tag_indices: Tag indices [N]
            batch: Batch vector for graph-level tasks [N]
        
        Returns:
            node_logits: Node-level predictions [N, num_node_classes]
            graph_logits: Graph-level predictions [num_graphs, num_graph_classes]
        """
        # Embed tags
        tag_embeds = self.tag_embedding(tag_indices)
        
        # Combine tag + attributes + text
        x = torch.cat([tag_embeds, x], dim=-1)
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GCN message passing
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Node-level predictions
        node_logits = self.node_classifier(x)
        
        # Graph-level predictions
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if self.pooling == "mean":
            graph_x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            graph_x = global_max_pool(x, batch)
        elif self.pooling == "meanmax":
            graph_x_mean = global_mean_pool(x, batch)
            graph_x_max = global_max_pool(x, batch)
            graph_x = torch.cat([graph_x_mean, graph_x_max], dim=-1)
        elif self.pooling == "attention":
            graph_x = self.pool(x, batch)
        else:
            graph_x = global_mean_pool(x, batch)
        
        graph_logits = self.graph_classifier(graph_x)
        
        return node_logits, graph_logits
    
    def predict_nodes(self, x, edge_index, tag_indices):
        """Convenience method for node-level inference."""
        node_logits, _ = self.forward(x, edge_index, tag_indices)
        return node_logits.argmax(dim=-1)
    
    def predict_graph(self, x, edge_index, tag_indices, batch=None):
        """Convenience method for graph-level inference."""
        _, graph_logits = self.forward(x, edge_index, tag_indices, batch)
        return graph_logits.argmax(dim=-1)


class DOMAttentionNet(torch.nn.Module):
    """
    GAT-based variant with attention for interpretability.
    Can visualize which DOM elements attend to which others.
    """
    
    def __init__(
        self,
        num_tags: int = 116,
        tag_embed_dim: int = 32,
        attr_dim: int = 134,
        text_dim: int = 384,
        hidden_dim: int = 128,
        num_node_classes: int = 2,
        num_graph_classes: int = 2,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        pooling: str = "mean",
    ):
        super().__init__()
        
        self.tag_embedding = Embedding(num_tags, tag_embed_dim)
        
        input_dim = tag_embed_dim + attr_dim + text_dim
        self.input_proj = Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.convs = ModuleList()
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim * heads
            out_channels = hidden_dim
            concat = i < num_layers - 1  # Concatenate heads for all but last layer
            self.convs.append(
                GATConv(
                    in_channels,
                    out_channels,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                )
            )
        
        self.pooling = pooling
        if pooling == "attention":
            self.pool_gate = Linear(hidden_dim, 1)
            self.pool = GlobalAttention(self.pool_gate)
        
        # Determine final dimension based on last layer's concat behavior
        if num_layers == 1:
            final_dim = hidden_dim
        else:
            # Last layer: concat=False, so output is hidden_dim
            final_dim = hidden_dim
        
        self.node_classifier = torch.nn.Sequential(
            Linear(final_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, num_node_classes),
        )
        
        graph_input_dim = final_dim * 2 if pooling == "meanmax" else final_dim
        self.graph_classifier = torch.nn.Sequential(
            Linear(graph_input_dim, hidden_dim),
            torch.nn.ReLU(),
            Linear(hidden_dim, num_graph_classes),
        )
    
    def forward(self, x, edge_index, tag_indices, batch=None):
        tag_embeds = self.tag_embedding(tag_indices)
        x = torch.cat([tag_embeds, x], dim=-1)
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=0.3, training=self.training)
        
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
        
        node_logits = self.node_classifier(x)
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        if self.pooling == "mean":
            graph_x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            graph_x = global_max_pool(x, batch)
        else:
            graph_x = global_mean_pool(x, batch)
        
        graph_logits = self.graph_classifier(graph_x)
        
        return node_logits, graph_logits
