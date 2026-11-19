import torch
import torch.nn.functional as F

from typing                           import Optional as Maybe
from torch                            import Tensor
from torch.nn                         import Linear
from torch_geometric.nn               import MLP, GCN, GIN, GAT, GraphSAGE, GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.norm          import BatchNorm
from torch_geometric.utils            import softmax


class BatchState:
    """
    The rows that a model's next forward pass computes in batched training, shared by the model's layers.
    Outside batched training, n_id is None, and the layers work as PyG's.
    """
    def __init__(self) -> None:
        self.n_id      : Maybe[Tensor] = None # global node ids of the rows
        self.num_seeds : int           = 0    # the first num_seeds rows are the batch's nodes, whose outputs are exact
        self.seed      : int           = 0    # seed of the dropout masks, drawn once per epoch
        self.calls     : int           = 0    # dropouts applied so far in this forward pass; each one draws a different mask

    def start(self, n_id: Tensor, num_seeds: int) -> None:
        self.n_id      = n_id
        self.num_seeds = num_seeds
        self.calls     = 0


def _mul32(h: Tensor, c: int) -> Tensor:
    # h * c modulo 2^32, for 0 <= h < 2^32
    # multiply by the 16-bit halves of c, so that the products stay below 2^63, where int64 overflows
    return (h * (c & 0xffff) + (((h * (c >> 16)) & 0xffff) << 16)) & 0xffffffff


def _hash32(h: Tensor) -> Tensor:
    # Chris Wellons's lowbias32 hash of 32-bit values, stored in int64
    h = h ^ (h >> 16)
    h = _mul32(h, 0x7feb352d)
    h = h ^ (h >> 15)
    h = _mul32(h, 0x846ca68b)
    return h ^ (h >> 16)


@torch.compile(dynamic = True)
def _keep(h_row: Tensor, h_col: Tensor, threshold: int) -> Tensor:
    # compiled into one kernel, which does not store the int64 intermediates
    # without compiling, the intermediates take 8 bytes per entry each, and this takes 40 times as long as torch's dropout on the GPU
    return _hash32(h_row[:, None] ^ h_col) >= threshold


def keyed_dropout(x: Tensor, p: float, batch: BatchState, *row_keys: Tensor) -> Tensor:
    """
    Dropout whose mask for x[i, j] depends only on the epoch's seed, the dropout's place in the forward pass, the row keys at i, and j.
    With global node ids as row keys, a node gets the same mask in every batch whose subgraph contains it, and in every pass of the epoch.
    """
    h_row = _hash32(_hash32(torch.tensor(batch.seed, device = x.device)) ^ batch.calls)
    for key in row_keys:
        h_row = _hash32(h_row ^ _hash32(key))
    h_col = _hash32(torch.arange(x.size(1), device = x.device))
    batch.calls += 1

    # keep x[i, j] with probability 1 - p
    # scale in place, so that only one tensor of x's size is allocated
    return x.mul(_keep(h_row, h_col, round(p * 2**32))).div_(1 - p)


class NodeDropout(torch.nn.Module):
    """
    Dropout that keys its masks by global node id and channel in batched training, and is torch's dropout otherwise.
    """
    def __init__(self, p: float, batch: BatchState) -> None:
        super().__init__()

        self.p     = p
        self.batch = batch

    def forward(self, x: Tensor) -> Tensor:
        if self.batch.n_id is None or not self.training:
            return F.dropout(x, p = self.p, training = self.training)
        return keyed_dropout(x, self.p, self.batch, self.batch.n_id)


class StopForward(Exception):
    """
    Ends a forward pass at a NodeBatchNorm, in the passes of batched training that need only the layers up to it.
    """
    def __init__(self, surrogate: Maybe[Tensor] = None) -> None:
        super().__init__()

        self.surrogate = surrogate # backpropagating it gives the gradient through the NodeBatchNorm's statistics


class NodeBatchNorm(torch.nn.Module):
    """
    PyG's BatchNorm that, in batched training, normalises with the mean and variance over all nodes of the graph, as a training pass over the whole graph does.

    Batched training collects the statistics in one pass over all batches per NodeBatchNorm, in the order of the forward pass.
    In such a pass, the first NodeBatchNorm without statistics adds the batch's nodes to them and ends the forward pass.
    After the backward pass of every batch, a pass per NodeBatchNorm, with backprop set, adds the gradient through its statistics.
    These passes go from the last NodeBatchNorm to the first, because a NodeBatchNorm's input depends on the statistics of the ones before it.
    """
    def __init__(self, norm: BatchNorm, batch: BatchState) -> None:
        super().__init__()

        self.norm  = norm
        self.batch = batch
        self.start_epoch()

    def start_epoch(self) -> None:
        self.mean           : Maybe[Tensor] = None  # mean and variance over all nodes in this epoch, once collected
        self.var            : Maybe[Tensor] = None
        self.count          : int           = 0     # nodes collected so far
        self.collected_mean : Tensor | int  = 0     # mean of the collected nodes
        self.collected_m2   : Tensor | int  = 0     # summed squared deviations of the collected nodes from their mean
        self.backprop       : bool          = False

    def finish_statistics(self) -> None:
        """
        Sets the mean and variance from the collected nodes, and moves the running statistics toward them, as a training pass over the whole graph does.
        """
        bn = self.norm.module

        # leaves, which gather the gradient through the statistics in the backward passes
        self.mean = self.collected_mean.float().requires_grad_()
        self.var  = (self.collected_m2 / self.count).float().requires_grad_()

        # BatchNorm moves the running variance toward the unbiased variance
        with torch.no_grad():
            bn.running_mean.mul_(1 - bn.momentum).add_(bn.momentum * self.mean)
            bn.running_var.mul_(1 - bn.momentum).add_(bn.momentum * (self.collected_m2 / (self.count - 1)).float())
            bn.num_batches_tracked += 1

    def forward(self, x: Tensor) -> Tensor:
        if self.batch.n_id is None or not self.training:
            return self.norm(x)

        bn    = self.norm.module
        seeds = x[:self.batch.num_seeds]

        if self.mean is None:
            # combine the batch's mean and squared deviations with those of the earlier batches, as in Chan et al.'s parallel variance algorithm
            # in float64, which keeps the variance from losing digits over many batches
            seeds = seeds.double()
            count = self.count + seeds.size(0)
            mean  = seeds.mean(dim = 0)
            delta = mean - self.collected_mean
            self.collected_m2   = self.collected_m2 + torch.sum((seeds - mean)**2, dim = 0) + delta**2 * self.count * seeds.size(0) / count
            self.collected_mean = self.collected_mean + delta * seeds.size(0) / count
            self.count          = count
            raise StopForward()

        if self.backprop:
            # the mean and the variance are averages over the nodes, so their gradient flows into each node's input
            # the variance's derivative through the mean is zero, so the mean is a constant in the variance's term
            raise StopForward( torch.sum(self.mean.grad * torch.sum(seeds, dim = 0)) / self.count
                             + torch.sum(self.var.grad  * torch.sum((seeds - self.mean.detach())**2, dim = 0)) / self.count
                             )

        return (x - self.mean) / torch.sqrt(self.var + bn.eps) * bn.weight + bn.bias


class KeyedGATConv(GATConv):
    """
    GATConv whose attention dropout, in batched training, keys its masks by the global ids of each link's nodes and by head.
    """
    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, dim_size, edge_index_j):
        if self.batch.n_id is None or not self.training:
            return super().edge_update(alpha_j, alpha_i, edge_attr, index, ptr, dim_size)

        # GATConv.edge_update without link features, with the keyed dropout
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, dim_size)
        return keyed_dropout(alpha, self.dropout, self.batch, self.batch.n_id[index], self.batch.n_id[edge_index_j])


class KeyedGAT(GAT):
    """
    PyG's GAT with KeyedGATConv layers.
    """
    def init_conv(self, in_channels, out_channels, **kwargs):
        # PyG's GAT.init_conv with one attention head, as GATModel uses it
        return KeyedGATConv(in_channels, out_channels, concat = not getattr(self, '_is_conv_to_out', False), dropout = self.dropout.p, **kwargs)


def _use_batch(model: torch.nn.Module, batch: BatchState) -> None:
    """
    Swaps the model's Dropout and BatchNorm layers for NodeDropout and NodeBatchNorm, and gives its KeyedGATConv layers the batch state.
    """
    for name, module in list(model.named_modules()):
        parent, _, child = name.rpartition(".")
        if isinstance(module, torch.nn.Dropout):
            setattr(model.get_submodule(parent), child, NodeDropout(module.p, batch))
        elif isinstance(module, BatchNorm):
            setattr(model.get_submodule(parent), child, NodeBatchNorm(module, batch))
        elif isinstance(module, KeyedGATConv):
            module.batch = batch


class LinearModel(torch.nn.Module):
    def __init__( self
                , in_channels  : int
                , out_channels : int
                , dropout      : float = 0.5
                ) -> None:
        super().__init__()

        self.batch   = BatchState()
        self.lin     = Linear(in_channels, out_channels)
        self.dropout = NodeDropout(dropout, self.batch)

    def forward(self, x):
        return self.dropout(self.lin(x))


class MLPModel(torch.nn.Module):
    def __init__( self
                , in_channels     : int
                , hidden_channels : int
                , num_layers      : int
                , out_channels    : int
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                ):
        super().__init__()

        self.batch = BatchState()

        # forward applies the dropout, so the MLP does not get it
        self.model = MLP( in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , num_layers      = num_layers
                        , out_channels    = out_channels
                        , act             = act
                        , norm            = norm
                        )
        self.dropout = NodeDropout(dropout, self.batch)
        _use_batch(self.model, self.batch)

    def forward(self, x):
        # PyG's MLP.forward with its defaults act_first = False and plain_last = True, and NodeDropout in place of F.dropout
        for lin, norm in zip(self.model.lins[:-1], self.model.norms):
            x = self.dropout(self.model.act(norm(lin(x))))
        return self.model.lins[-1](x)


class GCNModel(torch.nn.Module):
    def __init__( self
                , edge_index
                , in_channels     : int
                , hidden_channels : int
                , num_layers      : int
                , out_channels    : int
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                ):
        super().__init__()

        # normalise the adjacency once, with the whole graph's degrees, as GCNConv would in every forward pass
        # a batch's subgraph lacks links of its outer nodes, so it takes its weights from here
        self.edge_index, _ = gcn_norm(edge_index)
        self.batch         = BatchState()

        self.model = GCN( in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , num_layers      = num_layers
                        , out_channels    = out_channels
                        , act             = act
                        , dropout         = dropout
                        , norm            = norm
                        , normalize       = False
                        , add_self_loops  = False
                        )
        _use_batch(self.model, self.batch)

    def forward(self, x, edge_index = None):
        return self.model(x = x, edge_index = self.edge_index if edge_index is None else edge_index)


class GINModel(torch.nn.Module):
    def __init__( self
                , edge_index
                , in_channels     : int
                , hidden_channels : int
                , num_layers      : int
                , out_channels    : int
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                ):
        super().__init__()

        self.edge_index = edge_index
        self.batch      = BatchState()

        self.model = GIN( in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , num_layers      = num_layers
                        , out_channels    = out_channels
                        , act             = act
                        , norm            = norm
                        , dropout         = dropout
                        )
        _use_batch(self.model, self.batch)

    def forward(self, x, edge_index = None):
        return self.model(x = x, edge_index = self.edge_index if edge_index is None else edge_index)


class GATModel(torch.nn.Module):
    def __init__( self
                , edge_index
                , in_channels     : int
                , hidden_channels : int
                , num_layers      : int
                , out_channels    : int
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                ):
        super().__init__()

        self.edge_index = edge_index
        self.batch      = BatchState()

        self.model = KeyedGAT( in_channels     = in_channels
                             , hidden_channels = hidden_channels
                             , num_layers      = num_layers
                             , out_channels    = out_channels
                             , act             = act
                             , norm            = norm
                             , dropout         = dropout
                             )
        _use_batch(self.model, self.batch)

    def forward(self, x, edge_index = None):
        return self.model(x = x, edge_index = self.edge_index if edge_index is None else edge_index)


class SAGEModel(torch.nn.Module):
    def __init__( self
                , edge_index
                , in_channels     : int
                , hidden_channels : int
                , num_layers      : int
                , out_channels    : int
                , act             : str = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float = 0.5
                ):
        super().__init__()

        self.edge_index = edge_index
        self.batch      = BatchState()

        self.model = GraphSAGE( in_channels     = in_channels
                              , hidden_channels = hidden_channels
                              , num_layers      = num_layers
                              , out_channels    = out_channels
                              , act             = act
                              , norm            = norm
                              , dropout         = dropout
                              )
        _use_batch(self.model, self.batch)

    def forward(self, x, edge_index = None):
        return self.model(x = x, edge_index = self.edge_index if edge_index is None else edge_index)
