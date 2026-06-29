"""
models.py

PyTorch Geometric models for DOM graph classification.
Supports both node-level and graph-level predictions.
Multi-label classification for WCAG rule detection.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, BatchNorm1d, Dropout, Embedding
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, GlobalAttention


class DOMAttentionNet(torch.nn.Module):
    """
    GAT-based model for DOM accessibility violation detection.
    
    Multi-task outputs:
    - Binary node violation detection (node_logits)
    - Multi-label rule classification (node_rule_logits)  
    - Graph-level violation detection (graph_logits)
    
    Args:
        num_tags: Number of unique HTML tags for embedding
        tag_embed_dim: Dimension of tag embeddings
        attr_dim: Dimension of attribute features
        text_dim: Dimension of text embeddings
        hidden_dim: Hidden dimension for GAT layers
        num_node_classes: Number of node-level binary classes
        num_graph_classes: Number of graph-level classes
        num_rules: Number of WCAG rules for multi-label classification
        num_layers: Number of GAT layers
        heads: Number of attention heads
        dropout: Dropout rate
        pooling: Pooling method ('mean', 'max', 'attention', 'meanmax')
    """
    
    def __init__(
        self,
        num_tags: int = 116,
        tag_embed_dim: int = 32,
        attr_dim: int = 113,
        text_dim: int = 384,
        hidden_dim: int = 256,
        num_node_classes: int = 2,
        num_graph_classes: int = 2,
        num_rules: int = 46,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.3,
        pooling: str = "mean",
    ):
        super().__init__()
        
        self.tag_embed_dim = tag_embed_dim
        self.attr_dim = attr_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.num_rules = num_rules
        
        # Tag embedding
        self.tag_embedding = Embedding(num_tags, tag_embed_dim)
        
        # Input projection
        input_dim = tag_embed_dim + attr_dim + text_dim
        self.input_proj = Linear(input_dim, hidden_dim)
        self.input_bn = BatchNorm1d(hidden_dim)
        
        # GAT layers with residual connections
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.residual_projs = ModuleList()  # For dimension matching in residuals
        
        for i in range(num_layers):
            in_channels = hidden_dim if i == 0 else hidden_dim
            # Use concat=False for all layers to keep dimensions consistent
            self.convs.append(
                GATConv(
                    in_channels,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,  # Output: hidden_dim
                    dropout=dropout,
                )
            )
            self.bns.append(BatchNorm1d(hidden_dim))
            # Residual projection if dimensions change (not needed here since concat=True always outputs hidden_dim)
            self.residual_projs.append(
                Linear(hidden_dim, hidden_dim) if in_channels != hidden_dim else None
            )
        
        # Pooling for graph-level readout
        self.pooling = pooling
        if pooling == "attention":
            self.pool_gate = Linear(hidden_dim, 1)
            self.pool = GlobalAttention(self.pool_gate)
        
        # Node-level binary classifier (violation / no violation)
        self.node_classifier = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, num_node_classes),
        )
        
        # Node-level multi-label rule classifier (46 rules, independent sigmoids)
        self.node_rule_classifier = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, num_rules),
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
            node_logits: Binary node predictions [N, num_node_classes]
            node_rule_logits: Multi-label rule predictions [N, num_rules]
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
        
        # GAT message passing with residual connections
        for i, (conv, bn, res_proj) in enumerate(zip(self.convs, self.bns, self.residual_projs)):
            residual = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Add residual connection
            if res_proj is not None:
                residual = res_proj(residual)
            x = x + residual  # Residual connection
        
        # Node-level binary predictions
        node_logits = self.node_classifier(x)
        
        # Node-level multi-label rule predictions (raw logits, apply sigmoid during loss)
        node_rule_logits = self.node_rule_classifier(x)
        
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
        
        return node_logits, node_rule_logits, graph_logits
    
    def predict_nodes(self, x, edge_index, tag_indices):
        """Convenience method for binary node-level inference."""
        node_logits, _, _ = self.forward(x, edge_index, tag_indices)
        return node_logits.argmax(dim=-1)
    
    def predict_rules(self, x, edge_index, tag_indices, threshold=0.5):
        """Convenience method for multi-label rule inference."""
        _, node_rule_logits, _ = self.forward(x, edge_index, tag_indices)
        probs = torch.sigmoid(node_rule_logits)
        return (probs > threshold).long()
    
    def predict_graph(self, x, edge_index, tag_indices, batch=None):
        """Convenience method for graph-level inference."""
        _, _, graph_logits = self.forward(x, edge_index, tag_indices, batch)
        return graph_logits.argmax(dim=-1)


# Keep DOMGCN for backward compatibility
class DOMGCN(torch.nn.Module):
    """Legacy GCN model - kept for loading old checkpoints."""
    
    def __init__(self, **kwargs):
        super().__init__()
        # This is a stub - old checkpoints won't load multi-label weights
        raise NotImplementedError("DOMGCN is deprecated. Use DOMAttentionNet instead.")
