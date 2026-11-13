# Neuromap
Companion repository for the paper **The Map Equation Goes Neural: Mapping Network Flows with Graph Neural Networks** by Christopher Blöcker, Chester Tan, and Ingo Scholtes, published at NeurIPS 2024.
The paper adapts the map equation, an information-theoretic objective function for community detection, as a loss function for detecting communities with (graph) neural networks.

> [!NOTE]
> The code has changed since the paper was published, and results from the current code can differ from those in the paper.
> The code as published with the paper is tagged [`neurips-2024`](https://github.com/chrisbloecker/neuromap/tree/neurips-2024).
> The changes are listed under [Changes since the paper](#changes-since-the-paper).

## How the map equation works
The map equation is an information-theoretic objective function for community detection in complex networks.
The map equation adopts a compression perspective and quantifies how efficiently a random walk on a network can be described based on the given communities.
Those communities that lead to the best compression are considered to describe the organisational structure of the network well.
The map equation's basic principles are described in the paper; more detailed explanations can be found in [Community Detection with the Map Equation and Infomap: Theory and Applications](https://doi.org/10.1145/3779648).

The basic principle goes like this: (1) split the network into communities; (2) assign codewords to the nodes that are unique within each community, for example using Huffman codes; (3) compute how many bits are required per step that a random walker takes on the network -- this is the so-called codelength. The lower the codelength, the better the partition from step (1).
**In practice** no codewords or random walks are used, instead, the map equation computes the codelength analytically.

<img src="img/map-equation-principle.svg" alt="Two different ways to partition the same network into communities. The right-hand variant leads to a lower codelength." width="800"/>




## How Neuromap works
"Neuromap" refers to using the map equation as a loss function in combination with (graph) neural networks to learn (soft) cluster assignments (communities).
The figure below shows the general setup, where the input consists of a graph, and the output is a soft cluster assignment matrix, optimised in an unsupervised fashion by minimising the map equation through gradient descent.

<img src="img/neuromap-architecture.svg" alt="The Neuromap architecture for learning soft cluster assignments minimising the map equation through gradient descent." width="800"/>

The core component is the (two-level) map equation, which can be expressed as follows in Python, using `torch`:

```python
class MapEquationPooling(torch.nn.Module):
    def __init__(self, adj: Tensor, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.F, self.p = mkSmartTeleportationFlow(adj)

        # store the adjacency in CSR format; a product with a COO matrix converts it to CSR on every call
        self.adj = adj.coalesce().to_sparse_csr()

        # this term is constant, so only calculate it once
        self.p_log_p = torch.sum(self.p * torch.nan_to_num(torch.log2(self.p), nan = 0.0))

    def forward(self, x, s):
        # multiply F @ s first; s.T @ F with F in CSR format is 2 to 6 times slower
        C = s.T @ (self.F @ s)

        # flows between modules, summed from the off-diagonal entries of C
        # subtracting the diagonal from the row sums would round a flow of 1e-8 next to a flow near 1 to 0
        C_between = C - torch.diag(torch.diag(C))

        # the row sums of C_between are the flows out of each module, the column sums the flows into it
        q       = torch.sum(C_between)
        q_m     = torch.sum(C_between, dim = 1)
        m_enter = torch.sum(C_between, dim = 0)
        p_m     = q_m + torch.sum(C, dim = 0)

        # clamp the input to log2, so that a flow of 0 gets a finite gradient instead of 0/0
        codelength = torch.sum(q       * torch.log2(q.clamp_min(1e-30))) \
                   - torch.sum(q_m     * torch.log2(q_m.clamp_min(1e-30))) \
                   - torch.sum(m_enter * torch.log2(m_enter.clamp_min(1e-30))) \
                   - self.p_log_p \
                   + torch.sum(p_m     * torch.log2(p_m.clamp_min(1e-30)))

        # multiply x^T @ s first, for the same reason as F @ s above; for x in CSC format, x.t() is in CSR format
        x_pooled   = (x.t() @ s).t()
        # multiply adj @ s first, for the same reason as F @ s above
        adj_pooled = s.T @ (self.adj @ s)

        return x_pooled, adj_pooled, codelength
```

The self-contained `example-usage.py` notebook is intended to provide a minimal example as a starting point.

## Requirements
To run the code in this repository, you will need a couple of libraries.
Since the _exact_ setup varies somewhat depending on the available hardware (GPU available or not?), please refer to the documentation of the respective library.
The code was last tested with PyTorch 2.15 and PyG 2.8.1.
* [PyTorch](https://pytorch.org/)
* [PyG](https://pytorch-geometric.readthedocs.io/en/latest/)
* [PyTorch Geometric Signed Directed](https://pytorch-geometric-signed-directed.readthedocs.io/en/latest/index.html)
* [Open Graph Benchmark (OGB)](https://ogb.stanford.edu/)
* The implementation of [NOCD](https://github.com/shchur/overlapping-community-detection), a baseline that is used in the experiments
* Infomap as a baseline, which is available as the Python package `infomap`
* Some "standard" Python libraries: `matplotlib`, `seaborn`, `scikit-learn`, `networkx`
* Some synthetic LFR networks that are [available here](https://doi.org/10.5281/zenodo.13961218) (generated with code [available here](https://sites.google.com/site/andrealancichinetti/benchmarks))
* The results of our experiments can be found [here](https://doi.org/10.5281/zenodo.13961218)

Or perhaps you are just interested in looking at the plots of the results, which are included in the notebooks.

## Changes since the paper
These changes can make results differ from those in the paper.

* **Neuromap on GPU** ([#1](https://github.com/chrisbloecker/neuromap/pull/1), [#2](https://github.com/chrisbloecker/neuromap/pull/2)): in the published code, `Neuromap.fit` did not pass the device to `MapEquationPooling`, which then computed the flow on the CPU, so `fit` raised an error on a GPU before training. The code used for the paper passed the device; the argument was lost when this repository was put together. On a GPU, the softmax temperature was also not registered as a parameter, so it was not trained. Each trial now also starts with a new temperature, and DMoN with a new pooling layer.
* **Flows of 0** ([#3](https://github.com/chrisbloecker/neuromap/pull/3)): when all nodes fell into one module, the flow between modules rounded to 0, and the gradient of the map equation became NaN. Flows between modules are now summed from the off-diagonal entries of `C`, and the inputs to `log2` are clamped.
* **Nodes without out-links** ([#6](https://github.com/chrisbloecker/neuromap/pull/6)): a random walker on such a node now teleports. Before, its flow was dropped, so the flows summed to less than 1. The LFR networks in the paper have no such nodes.
* **NOCD's number of edges** ([#6](https://github.com/chrisbloecker/neuromap/pull/6)): NOCD got `data.num_edges`, which PyG up to version 2.5 computes as the number of nodes when `data.edge_index` is a sparse adjacency matrix. NOCD now gets the number of stored entries in the adjacency matrix, as in NOCD's own example. From PyG 2.6 on, `data.num_edges` raises an error here.
* **Message direction on directed networks** ([#6](https://github.com/chrisbloecker/neuromap/pull/6)): GCN, GIN, GAT and SAGE got the adjacency matrix with one row per source node, but PyG reads a sparse adjacency matrix with one row per target node. So on directed networks, each node aggregated the features of the nodes it links to. The models now get the transposed adjacency matrix, so that messages flow along the links. Undirected networks are unaffected.
* **DiffPool, MinCut, Ortho and DMoN** ([#6](https://github.com/chrisbloecker/neuromap/pull/6)): `src/pool.py` has copies of PyG's `dense_diff_pool`, `dense_mincut_pool` and `DMoNPooling`, adapted to a sparse adjacency matrix. For the paper, they were copied from PyG 2.3.1. They now follow PyG 2.8.1. PyG has not changed the code of `dense_diff_pool` and `dense_mincut_pool` since 2.3.1, but PyG 2.5.0 fixed DMoN's loss, which before optimised modularity at resolution 1/2, and removed the SELU activation from DMoN's MLP.

Other changes, such as storing sparse matrices in CSR format, make the code faster or clearer without changing results beyond rounding.

## Citation
If you're using Neuromap, please cite
```bibtext
@inproceedings{NEURIPS2024_1f59562c,
 author = {Bl\"{o}cker, Christopher and Tan, Chester and Scholtes, Ingo},
 booktitle = {Advances in Neural Information Processing Systems},
 doi = {10.52202/079017-0554},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {17428--17456},
 publisher = {Curran Associates, Inc.},
 title = {The Map Equation Goes Neural: Mapping Network Flows with Graph Neural Networks},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/1f59562caae05e6aae0ffd1145bea5da-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
