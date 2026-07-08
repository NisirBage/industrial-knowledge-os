"""GCN / GraphSAGE / GAT (M9 Component 3): real PyTorch Geometric message-
passing encoders, trained unsupervised via link-prediction negative
sampling — the standard way to get a GNN's node embeddings without labels
(there is no per-node classification target by default). Component 5 (Link
Prediction) and Component 6 (Node Classification) both build directly on
these embeddings, so training them for link prediction first is not a
detour, it's the same objective those components need.

Only this module imports `torch_geometric` — the `NodeEmbeddingModel`
Protocol in base.py never does, so swapping in a different GNN backend
later is a new class here, not an abstraction change.
"""

import time

import torch
import torch.nn.functional as functional
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from app.graph_intelligence.dataset.builder import GraphDatasetBuilder
from app.graph_intelligence.dataset.models import GraphDataset
from app.graph_intelligence.embeddings.models import EmbeddingResult, NodeEmbedding

_DEFAULT_DIMENSIONS = 32
_DEFAULT_HIDDEN = 64
_DEFAULT_EPOCHS = 100
_DEFAULT_LR = 0.01


class _Encoder(torch.nn.Module):
    def __init__(self, conv_cls, in_channels: int, hidden_channels: int, out_channels: int, **conv_kwargs):
        super().__init__()
        self.conv1 = conv_cls(in_channels, hidden_channels, **conv_kwargs)
        self.conv2 = conv_cls(hidden_channels, out_channels, **conv_kwargs)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def _negative_sample(num_nodes: int, num_samples: int, existing_edges: set[tuple[int, int]], generator: torch.Generator) -> torch.Tensor:
    negatives: list[list[int]] = [[], []]
    attempts = 0
    while len(negatives[0]) < num_samples and attempts < num_samples * 20:
        attempts += 1
        source = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
        target = int(torch.randint(0, num_nodes, (1,), generator=generator).item())
        if source == target or (source, target) in existing_edges:
            continue
        negatives[0].append(source)
        negatives[1].append(target)
    return torch.tensor(negatives, dtype=torch.long)


class _GNNEmbeddingBase:
    version = "1.0"
    _conv_cls: type
    _conv_kwargs: dict = {}

    def __init__(
        self,
        dimensions: int = _DEFAULT_DIMENSIONS,
        hidden_dimensions: int = _DEFAULT_HIDDEN,
        epochs: int = _DEFAULT_EPOCHS,
        learning_rate: float = _DEFAULT_LR,
        seed: int = 42,
    ):
        self._dimensions = dimensions
        self._hidden_dimensions = hidden_dimensions
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._seed = seed

    def fit(self, dataset: GraphDataset) -> EmbeddingResult:
        start = time.perf_counter()
        if not dataset.nodes:
            return EmbeddingResult(embeddings=[], model_name=self.name, version=self.version, dimensions=self._dimensions, training_seconds=0.0)

        torch.manual_seed(self._seed)
        generator = torch.Generator().manual_seed(self._seed)
        data = GraphDatasetBuilder.to_pyg_homogeneous(dataset)

        encoder = _Encoder(self._conv_cls, data.x.shape[1], self._hidden_dimensions, self._dimensions, **self._conv_kwargs)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=self._learning_rate)

        num_edges = data.edge_index.shape[1]
        existing_edges = {(int(s), int(t)) for s, t in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())}

        if num_edges > 0:
            encoder.train()
            for _ in range(self._epochs):
                optimizer.zero_grad()
                embeddings = encoder(data.x, data.edge_index)
                positive_scores = (embeddings[data.edge_index[0]] * embeddings[data.edge_index[1]]).sum(dim=-1)
                negative_edge_index = _negative_sample(data.num_nodes, num_edges, existing_edges, generator)
                if negative_edge_index.shape[1] == 0:
                    loss = functional.binary_cross_entropy_with_logits(positive_scores, torch.ones_like(positive_scores))
                else:
                    negative_scores = (embeddings[negative_edge_index[0]] * embeddings[negative_edge_index[1]]).sum(dim=-1)
                    scores = torch.cat([positive_scores, negative_scores])
                    labels = torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)])
                    loss = functional.binary_cross_entropy_with_logits(scores, labels)
                loss.backward()
                optimizer.step()

        encoder.eval()
        with torch.no_grad():
            final_embeddings = encoder(data.x, data.edge_index)

        embeddings = [
            NodeEmbedding(node_id=node.id, vector=final_embeddings[dataset.node_id_to_index[node.id]].tolist())
            for node in dataset.nodes
        ]
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.name,
            version=self.version,
            dimensions=self._dimensions,
            training_seconds=round(time.perf_counter() - start, 4),
        )


class GCNEmbedding(_GNNEmbeddingBase):
    name = "gcn"
    _conv_cls = GCNConv


class GraphSAGEEmbedding(_GNNEmbeddingBase):
    name = "graphsage"
    _conv_cls = SAGEConv


class GATEmbedding(_GNNEmbeddingBase):
    name = "gat"
    _conv_cls = GATConv
