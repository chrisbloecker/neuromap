"""
Checks that batched training computes the same gradient as training on the whole graph, up to rounding, for every method and model.

With dropout 0, one epoch's gradient, the running statistics and the eval loss must match those on the whole graph.
With dropout, batched training draws its masks per node, so batches of 13 nodes must match one batch of all nodes.

Run from the repository's root: .venv/bin/python test/test_batched_gradient.py [cpu|cuda]
"""
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import torch

from torch_geometric.data import Data
from training             import Neuromap, NeuromapD, NOCD, DMoN, DiffPool, MinCut, Ortho


device    = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
tolerance = 1e-4 # largest difference, relative to the largest entry


def make_graph(N: int) -> Data:
    # a directed graph with a hub, with the adjacency matrix as sparse features
    torch.manual_seed(0)
    source = torch.cat([torch.randint(N, (5 * N,)), torch.full((N // 3,), 3)])
    target = torch.cat([torch.randint(N, (5 * N,)), torch.randint(N, (N // 3,))])
    keep   = source != target
    adj    = torch.sparse_coo_tensor(torch.stack([source[keep], target[keep]]), torch.ones(int(keep.sum())), (N, N)).coalesce()
    adj    = torch.sparse_coo_tensor(adj.indices(), torch.ones(adj._nnz()), (N, N)).coalesce()
    return Data(edge_index = adj, x = adj.clone(), num_nodes = N)


def epoch(method, model, data, dropout, batch_size):
    """
    One training epoch's gradient from a fresh model, then the eval pass.
    """
    clusterer = method(device = device, use_model = model, in_channels = data.num_nodes, hidden_channels = 16, out_channels = 5, num_layers = 1 if model == "lin" else 2, dropout = dropout)
    x = clusterer._prepare(data.clone(), batch_size)

    torch.manual_seed(1)
    clusterer._reset()
    clusterer.train()
    clusterer.zero_grad()
    clusterer._gradient(x)

    clusterer.eval()
    with torch.no_grad():
        loss, _ = clusterer.objective(x, clusterer._encode(x))

    gradients = [p.grad.clone() for p in clusterer.parameters() if p.grad is not None]
    buffers   = [b.clone().float() for b in clusterer.model.buffers()]
    return float(loss), gradients, buffers


def difference(a, b):
    # largest difference, relative to the largest entry over all tensors
    # a bias in front of a BatchNorm has gradient 0, so only rounding is left in it
    scale = max([float(v.abs().max()) for v in b] + [1e-30])
    return max([float((u - v).abs().max()) for u, v in zip(a, b)] + [0.0]) / scale


data     = make_graph(90)
failures = []
for method in [Neuromap, NeuromapD, NOCD, DMoN, DiffPool, MinCut, Ortho]:
    for model in ["lin", "mlp", "gcn", "gin", "gat", "sage"]:
        l_full, g_full, b_full = epoch(method, model, data, 0.0, None)
        l_batch, g_batch, b_batch = epoch(method, model, data, 0.0, 13)
        no_dropout = [abs(l_batch - l_full) / abs(l_full), difference(g_batch, g_full), difference(b_batch, b_full)]

        l_one, g_one, b_one = epoch(method, model, data, 0.5, data.num_nodes)
        l_batch, g_batch, b_batch = epoch(method, model, data, 0.5, 13)
        dropout = [abs(l_batch - l_one) / abs(l_one), difference(g_batch, g_one), difference(b_batch, b_one)]

        passed = len(g_full) == len(g_batch) and max(no_dropout + dropout) <= tolerance
        if not passed:
            failures.append(f"{method.__name__} {model}")
        print( f"{method.__name__:9} {model:4}"
             + f" | dropout 0, batched vs whole graph: loss {no_dropout[0]:.1e}, gradient {no_dropout[1]:.1e}, running statistics {no_dropout[2]:.1e}"
             + f" | dropout 0.5, 13 vs {data.num_nodes} per batch: loss {dropout[0]:.1e}, gradient {dropout[1]:.1e}, running statistics {dropout[2]:.1e}"
             + ("" if passed else "  FAILED")
             )

print("all passed" if not failures else f"failed: {', '.join(failures)}")
sys.exit(1 if failures else 0)
