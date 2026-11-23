from __future__ import annotations

from typing import List, Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.nn.models.mlp import MLP

from model import row_slice


def pool_features(s: Tensor, x: Tensor, chunk_size: int) -> Tensor:
    """
    sᵀ x, for x in CPU memory, sparse in CSR format or dense, and s on the GPU.
    x's rows move to the GPU chunk_size at a time.
    The result has no gradient: the pooled features are not part of any loss, and their autograd graph would keep every chunk on the GPU.
    """
    s   = s.detach()
    out = torch.zeros(s.size(1), x.size(1), device = s.device)
    for start in range(0, x.size(0), chunk_size):
        end = min(start + chunk_size, x.size(0))
        out += s[start:end].t() @ row_slice(x, start, end).to(s.device)
    return out


# diff_pool, mincut_pool and DMoNPooling are dense_diff_pool, dense_mincut_pool and DMoNPooling from PyG 2.8.1
# they are changed where adj is a sparse matrix, in CSR format for mincut_pool and DMoNPooling, where x stays in CPU memory in batched training, and in DiffPool's link loss
# they make the identity matrix on the assignments' device instead of on the CPU
# they leave out what no loss in training.py uses and return None in its place:
# the pooled features, the pooled adjacency matrix of mincut_pool and DMoNPooling, and DMoN's orthogonality loss
# mincut_pool and DMoNPooling take the traces in their losses as sums of elementwise products, without the K × K matrices
# each changed line has PyG's version as a comment above it
def diff_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    normalize: bool = True,
    chunk_size: Optional[int] = None,
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
        chunk_size (int, optional): Set in batched training, where
            :obj:`x` stays in CPU memory and can be a CSR matrix, which
            cannot be unsqueezed. Then :obj:`x` is left as it is.
            (default: :obj:`None`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
        :class:`torch.Tensor`, :class:`torch.Tensor`)
    """
    # x = x.unsqueeze(0) if x.dim() == 2 else x
    x = x.unsqueeze(0) if x.dim() == 2 and chunk_size is None else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    # batch_size, num_nodes, _ = x.size()
    batch_size, num_nodes, _ = s.size()

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    # no loss uses the pooled features, so they are not computed
    # out = torch.matmul(s.transpose(1, 2), x)
    # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
    out_adj = s[0].T @ torch.sparse.mm(adj[0], s[0])
    out_adj = out_adj[None, :, :]

    # link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    # link_loss = torch.norm(link_loss, p=2)
    # ‖A − s sᵀ‖² = ‖A‖² − 2 tr(sᵀ A s) + ‖sᵀ s‖², which needs no N × N matrix
    # unsqueeze drops adj's coalesced flag, so read its stored entries with _values; callers store adj coalesced, so the entries are distinct
    link_loss = torch.sqrt(torch.sum(adj._values()**2) - 2 * torch.sum(_rank3_trace(out_adj)) + torch.sum(torch.matmul(s.transpose(1, 2), s)**2))
    if normalize is True:
        link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

    # return out, out_adj, link_loss, ent_loss
    return None, None, link_loss, ent_loss


def mincut_pool(
    x: Tensor,
    adj: Tensor,
    s: Tensor,
    mask: Optional[Tensor] = None,
    temp: float = 1.0,
    chunk_size: Optional[int] = None,
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
        chunk_size (int, optional): Set in batched training, where
            :obj:`x` stays in CPU memory and can be a CSR matrix, which
            cannot be unsqueezed. Then :obj:`x` is left as it is.
            (default: :obj:`None`)

    :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`,
        :class:`torch.Tensor`, :class:`torch.Tensor`)
    """

    # x = x.unsqueeze(0) if x.dim() == 2 else x
    x = x.unsqueeze(0) if x.dim() == 2 and chunk_size is None else x
    # adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    # adj stays a 2D matrix in CSR format, which has no 3D form
    s = s.unsqueeze(0) if s.dim() == 2 else s

    # (batch_size, num_nodes, _), k = x.size(), s.size(-1)
    (batch_size, num_nodes, _), k = s.size(), s.size(-1)

    s = torch.softmax(s / temp if temp != 1.0 else s, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    # no loss uses the pooled features, so they are not computed
    # out = torch.matmul(s.transpose(1, 2), x)
    # the MinCut loss needs only the traces of sᵀ A s and sᵀ D s, which are the sums of s ⊙ (A s) and s ⊙ (D s)
    # so neither K × K matrix is computed
    # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    # MinCut regularization.
    # mincut_num = _rank3_trace(out_adj)
    mincut_num = torch.sum(s[0] * (adj @ s[0]))
    # d_flat = torch.einsum("ijk->ij", adj)
    d_flat = adj @ torch.ones(num_nodes, 1, dtype=s.dtype, device=s.device)
    # d = _rank3_diag(d_flat)
    # mincut_den = _rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_den = torch.sum(s[0] * (d_flat * s[0]))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    # Orthogonality regularization, in orthogonality_loss, which Ortho uses alone.
    ortho_loss = orthogonality_loss(s)

    # no loss uses the coarsened adjacency matrix, so it is not fixed and normalized
    # EPS = 1e-15
    #
    # # Fix and normalize coarsened adjacency matrix.
    # ind = torch.arange(k, device=out_adj.device)
    # out_adj[:, ind, ind] = 0
    # d = torch.einsum("ijk->ij", out_adj)
    # d = torch.sqrt(d)[:, None] + EPS
    # out_adj = (out_adj / d) / d.transpose(1, 2)

    # return out, out_adj, mincut_loss, ortho_loss
    return None, None, mincut_loss, ortho_loss


def _rank3_trace(x: Tensor) -> Tensor:
    return torch.einsum("ijj->i", x)


def orthogonality_loss(s: Tensor) -> Tensor:
    """
    mincut_pool's orthogonality loss, for the assignments s after the softmax, B × N × K.
    Ortho uses it without the rest of mincut_pool.
    """
    k = s.size(-1)

    ss = torch.matmul(s.transpose(1, 2), s)
    # i_s = torch.eye(k).type_as(ss)
    # make the identity on ss's device, instead of on the CPU and copying it over in every call
    i_s = torch.eye(k, dtype=ss.dtype, device=ss.device)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
        dim=(-1, -2),
    )
    ortho_loss = torch.mean(ortho_loss)

    return ortho_loss


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

        self.mlp = MLP(channels + [k], act=None, norm=None)

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
        # adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        # adj stays a 2D matrix in CSR format, which has no 3D form

        s = self.mlp(x)
        s = F.dropout(s, self.dropout, training=self.training)
        s = torch.softmax(s, dim=-1)

        (batch_size, num_nodes, _), C = x.size(), s.size(-1)

        if mask is None:
            mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=x.device)

        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        # x, s = x * mask, s * mask
        s = s * mask

        # no loss uses the pooled features, so they are not computed
        # out = F.selu(torch.matmul(s.transpose(1, 2), x))
        # the spectral loss needs only the trace of the pooled adjacency matrix, which is the sum of s ⊙ (A s)
        # so the K × K matrix is not computed
        # out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # Spectral loss:
        # degrees = torch.einsum("ijk->ij", adj)  # B X N
        degrees = (adj @ torch.ones(num_nodes, 1, dtype=s.dtype, device=s.device)).T  # B X N
        degrees = degrees.unsqueeze(-1) * mask  # B x N x 1
        degrees_t = degrees.transpose(1, 2)  # B x 1 x N

        m = torch.einsum("ijk->i", degrees) / 2  # B
        # m_expand = m.view(-1, 1, 1).expand(-1, C, C)  # B x C x C

        ca = torch.matmul(s.transpose(1, 2), degrees)  # B x C x 1
        cb = torch.matmul(degrees_t, s)  # B x 1 x C

        # the trace of the normalizer is ca · cb / 2m, so the K × K normalizer is not computed either
        # normalizer = torch.matmul(ca, cb) / 2 / m_expand
        # decompose = out_adj - normalizer
        # spectral_loss = -_rank3_trace(decompose) / 2 / m
        spectral_loss = -(torch.sum(s[0] * (adj @ s[0])) - torch.sum(ca[0, :, 0] * cb[0, 0, :]) / 2 / m) / 2 / m
        spectral_loss = spectral_loss.mean()

        # Orthogonality regularization:
        # DMoN leaves the orthogonality loss out of its loss, as the original implementation does, so it is not computed
        # ss = torch.matmul(s.transpose(1, 2), s)
        # i_s = torch.eye(C).type_as(ss)
        # make the identity on s's device, instead of on the CPU and copying it over in every call
        i_s = torch.eye(C, dtype=s.dtype, device=s.device)
        # ortho_loss = torch.norm(
        #     ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s),
        #     dim=(-1, -2),
        # )
        # ortho_loss = ortho_loss.mean()

        # Cluster loss:
        cluster_size = torch.einsum("ijk->ik", s)  # B x C
        cluster_loss = torch.norm(input=cluster_size, dim=1)
        cluster_loss = cluster_loss / mask.sum(dim=1) * torch.norm(i_s) - 1
        cluster_loss = cluster_loss.mean()

        # no loss uses the coarsened adjacency matrix, so it is not fixed and normalized
        # # Fix and normalize coarsened adjacency matrix:
        # ind = torch.arange(C, device=out_adj.device)
        # out_adj[:, ind, ind] = 0
        # d = torch.einsum("ijk->ij", out_adj)
        # d = torch.sqrt(d)[:, None] + EPS
        # out_adj = (out_adj / d) / d.transpose(1, 2)

        # return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss
        return s, None, None, spectral_loss, None, cluster_loss

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.mlp.in_channels}, "
            f"num_clusters={self.mlp.out_channels})"
        )
