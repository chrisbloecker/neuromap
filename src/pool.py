from __future__ import annotations

from typing import List, Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.nn.models.mlp import MLP


def smart_teleport(A, alpha=0.15, iter=1000):
    # build the transition matrix
    T = torch.nan_to_num(A.T * torch.sum(A, 1).to_dense() ** (-1), nan=0.0).T.to(
        device=A.device
    )

    # distribution according to nodes' in-degrees
    e_v = (torch.sum(A, dim=0) / torch.sum(A)).to_dense().to(device=A.device)

    # calculate the flow distribution with a power iteration
    # p = (1/len(T) * torch.ones(len(T))).to(device = device)
    p = e_v
    for _ in range(iter):
        p = alpha * e_v + (1 - alpha) * p @ T

    # make the flow matrix for minimising the map equation
    F = alpha * A / torch.sum(A) + (1 - alpha) * (p * T.T).T

    return F, p


class NeuromapPooling(torch.nn.Module):
    r"""This criterion computes the map equation codelength for an undirected or directed weighted graph.

    Args:
        A (torch.Tensor): (Unnormalized) Adjacency matrix of the weighted graph.
        epsilon (float, optional): Small epsilon to ensure differentiability of logs.

    """

    def __init__(
        self,
        adj: Tensor,
        epsilon: float = 1e-8,
    ):
        super().__init__()

        self.epsilon = epsilon
        self.F, self.p = smart_teleport(adj)

    def forward(
        self,
        x: Tensor,
        s: Tensor,
        eps: float = 1e-8,
    ):
        s = s + eps

        out_adj = s.T @ self.F @ s

        diag = torch.sparse_coo_tensor(
            indices=[range(len(out_adj)), range(len(out_adj))],
            values=torch.diag(out_adj),
            size=out_adj.shape,
        ).to(device=s.device)

        e1 = torch.sum(out_adj) - torch.trace(out_adj)
        e2 = torch.sum(out_adj - diag, 1)
        e3 = self.p
        e4 = torch.sum(out_adj, 1) + torch.sum(out_adj.T - diag, 1)

        e1 = torch.sum(e1 * torch.nan_to_num(torch.log2(e1), nan=0.0))
        e2 = torch.sum(e2 * torch.nan_to_num(torch.log2(e2), nan=0.0))
        e3 = torch.sum(e3 * torch.nan_to_num(torch.log2(e3), nan=0.0))
        e4 = torch.sum(e4 * torch.nan_to_num(torch.log2(e4), nan=0.0))

        map_equation_loss = e1 - 2 * e2 - e3 + e4

        out = torch.matmul(s.T, x)

        return out, out_adj, map_equation_loss


def diff_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    normalize: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""The differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened adjacency matrix and
    two auxiliary objectives: (1) The link prediction loss

    .. math::
        \mathcal{L}_{LP} = {\| \mathbf{A} -
        \mathrm{softmax}(\mathbf{S}) {\mathrm{softmax}(\mathbf{S})}^{\top}
        \|}_F,

    and (2) the entropy regularization

    .. math::
        \mathcal{L}_E = \frac{1}{N} \sum_{n=1}^N H(\mathbf{S}_n).

    Args:
        x (torch.Tensor): Node feature tensor
            :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
            batch-size :math:`B`, (maximum) number of nodes :math:`N` for
            each graph, and feature dimension :math:`F`.
        adj (torch.Tensor): Adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (torch.Tensor): Assignment tensor
            :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
            with number of clusters :math:`C`.
            The softmax does not have to be applied before-hand, since it is
            executed within this method.
        mask (torch.Tensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        normalize (bool, optional): If set to :obj:`False`, the link
            prediction loss is not divided by :obj:`adj.numel()`.
            (default: :obj:`True`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
        :class:`torch.Tensor`, :class:`torch.Tensor`)
    """
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    out_adj = torch.sparse.mm(adj[0], s[0]).T
    out_adj = torch.sparse.mm(out_adj, s[0])
    out_adj = out_adj[None, :, :]

    # link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = -torch.matmul(s, s.transpose(1, 2)) + adj
    link_loss = torch.norm(link_loss, p=2)
    if normalize is True:
        link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

    return out, out_adj, link_loss, ent_loss


def mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""The MinCut pooling operator from the `"Spectral Clustering in Graph
    Neural Networks for Graph Pooling" <https://arxiv.org/abs/1907.00481>`_
    paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the pooled node feature matrix, the coarsened and symmetrically
    normalized adjacency matrix and two auxiliary objectives: (1) The MinCut
    loss

    .. math::
        \mathcal{L}_c = - \frac{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A}
        \mathbf{S})} {\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D}
        \mathbf{S})}

    where :math:`\mathbf{D}` is the degree matrix, and (2) the orthogonality
    loss

    .. math::
        \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
        \right\|}_F.

    Args:
        x (torch.Tensor): Node feature tensor
            :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
            batch-size :math:`B`, (maximum) number of nodes :math:`N` for
            each graph, and feature dimension :math:`F`.
        adj (torch.Tensor): Adjacency tensor
            :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
        s (torch.Tensor): Assignment tensor
            :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}`
            with number of clusters :math:`C`.
            The softmax does not have to be applied before-hand, since it is
            executed within this method.
        mask (torch.Tensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        temp (float, optional): Temperature parameter for softmax function.
            (default: :obj:`1.0`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
        :class:`torch.Tensor`, :class:`torch.Tensor`)
    """

    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    out_adj = torch.sparse.mm(adj[0], s[0]).T
    out_adj = torch.sparse.mm(out_adj, s[0])
    out_adj = out_adj[None, :, :]

    # MinCut regularization.
    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum("ijk->ij", adj)
    d = _rank3_diag(d_flat)
    # mincut_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_den = _rank3_trace(
        torch.sparse.mm(torch.sparse.mm(d[0], s[0]).T, s[0]).unsqueeze(0)
    )
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
        dim=(-1, -2),
    )
    ortho_loss = torch.mean(ortho_loss)

    EPS = 1e-15

    # Fix and normalize coarsened adjacency matrix.
    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum("ijk->ij", out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return out, out_adj, mincut_loss, ortho_loss


def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum("ijj->i", x)


def _rank3_diag(x: Tensor) -> Tensor:
    # eye = torch.eye(x.size(1)).type_as(x)
    # out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))
    n = x.size(1)
    indices = torch.stack((torch.arange(n), torch.arange(n))).to(x.device)
    out = (
        torch.sparse_coo_tensor(indices, x[0].to_dense(), (n, n))
        .to(x.device)
        .unsqueeze(0)
    )

    return out


EPS = 1e-15


class DMoNPooling(torch.nn.Module):
    r"""The spectral modularity pooling operator from the `"Graph Clustering
    with Graph Neural Networks" <https://arxiv.org/abs/2006.16904>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns the learned cluster assignment matrix, the pooled node feature
    matrix, the coarsened symmetrically normalized adjacency matrix, and three
    auxiliary objectives: (1) The spectral loss

    .. math::
        \mathcal{L}_s = - \frac{1}{2m}
        \cdot{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{B} \mathbf{S})}

    where :math:`\mathbf{B}` is the modularity matrix, (2) the orthogonality
    loss

    .. math::
        \mathcal{L}_o = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_C}{\sqrt{C}}
        \right\|}_F

    where :math:`C` is the number of clusters, and (3) the cluster loss

    .. math::
        \mathcal{L}_c = \frac{\sqrt{C}}{n}
        {\left\|\sum_i\mathbf{C_i}^{\top}\right\|}_F - 1.

    .. note::

        For an example of using :class:`DMoNPooling`, see
        `examples/proteins_dmon_pool.py
        <https://github.com/pyg-team/pytorch_geometric/blob
        /master/examples/proteins_dmon_pool.py>`_.

    Args:
        channels (int or List[int]): Size of each input sample. If given as a
            list, will construct an MLP based on the given feature sizes.
        k (int): The number of clusters.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
    """

    def __init__(self, channels: Union[int, List[int]], k: int, dropout: float = 0.0):
        super().__init__()

        if isinstance(channels, int):
            channels = [channels]

        self.mlp = MLP(channels + [k], act="selu", norm=None)

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.mlp.reset_parameters()

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
                Note that the cluster assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` is
                being created within this method.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`,
            :class:`torch.Tensor`, :class:`torch.Tensor`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        s = self.mlp(x)
        s = F.dropout(s, self.dropout, training=self.training)
        s = torch.softmax(s, dim=-1)

        (batch_size, num_nodes, _), k = x.size(), s.size(-1)

        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        out = F.selu(torch.matmul(s.transpose(1, 2), x))
        # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        out_adj = torch.sparse.mm(adj[0], s[0]).T
        out_adj = torch.sparse.mm(out_adj, s[0])
        out_adj = out_adj[None, :, :]

        # Spectral loss:
        degrees = torch.einsum("ijk->ik", adj).transpose(0, 1)
        m = torch.einsum("ij->", degrees)

        # ca = torch.matmul(s.transpose(1, 2), degrees)
        ca = torch.matmul(s[0].T, degrees).unsqueeze(0)
        cb = torch.matmul(degrees.T, s[0]).unsqueeze(0)

        normalizer = torch.matmul(ca, cb) / 2 / m
        decompose = out_adj - normalizer
        spectral_loss = -_rank3_trace(decompose) / 2 / m
        spectral_loss = torch.mean(spectral_loss)

        # Orthogonality regularization:
        ss = torch.matmul(s.transpose(1, 2), s)
        i_s = torch.eye(k).type_as(ss)
        ortho_loss = torch.norm(
            ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
            dim=(-1, -2),
        )
        ortho_loss = torch.mean(ortho_loss)

        # Cluster loss:
        cluster_loss = (
            torch.norm(torch.einsum("ijk->ij", ss)) / adj.size(1) * torch.norm(i_s) - 1
        )

        # Fix and normalize coarsened adjacency matrix:
        ind = torch.arange(k, device=out_adj.device)
        out_adj[:, ind, ind] = 0
        d = torch.einsum("ijk->ij", out_adj)
        d = torch.sqrt(d)[:, None] + EPS
        out_adj = (out_adj / d) / d.transpose(1, 2)

        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.mlp.in_channels}, "
            f"num_clusters={self.mlp.out_channels})"
        )
