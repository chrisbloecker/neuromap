import torch
import nocd

from abc                   import abstractmethod
from model                 import LinearModel as Linear, MLPModel as MLP, GCNModel as GCN, GINModel as GIN, GATModel as GAT, SAGEModel as SAGE
from model                 import NodeBatchNorm, StopForward, keyed_dropout, first_rows
from numpy                 import inf, ones
from scipy.sparse          import csr_matrix
from scipy.sparse.csgraph  import reverse_cuthill_mckee
from time                  import sleep
from typing                import List, Optional as Maybe, Tuple
from torch                 import Tensor
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data  import Data
from torch.nn.functional   import softmax, sigmoid, relu
from torch.nn              import Parameter
from torch.utils.checkpoint import checkpoint
from pool                  import DMoNPooling, mincut_pool, diff_pool, pool_features

import gc


def csr_rows(A: Tensor, rows: Tensor) -> Tensor:
    """
    The given rows of a CSR matrix, in their order, as a CSR matrix.
    torch has no index_select for CSR matrices.
    """
    crow   = A.crow_indices()
    starts = crow[rows]
    counts = crow[rows + 1] - starts
    crow_rows = torch.cat([torch.zeros(1, dtype = crow.dtype, device = crow.device), torch.cumsum(counts, dim = 0)])

    # each entry's position in A is its row's start in A plus its offset within the row
    entries = torch.repeat_interleave(starts - crow_rows[:-1], counts) + torch.arange(int(crow_rows[-1]), device = crow.device)

    return torch.sparse_csr_tensor(crow_rows, A.col_indices()[entries], A.values()[entries], (len(rows), A.size(1)))


def random_projection(adj: Tensor, d: int) -> Tensor:
    """
    Features x = A R for a graph without features, with R an N × d matrix of Gaussian entries with variance 1/d.
    With the adjacency matrix as features, the first layer has N × hidden weights; with x, it has d × hidden.
    By the Johnson–Lindenstrauss lemma, d of order log N / ε² keeps the distances between all pairs of adjacency rows within a factor 1 ± ε, with high probability.
    """
    R = torch.randn(adj.size(1), d, device = adj.device) / d**0.5
    return adj.to_sparse_csr() @ R


# create smart teleportation flow matrix and flow distribution
def mkSmartTeleportationFlow(A, alpha = 0.15, iter = 1000):
    # build the transition matrix
    T = torch.nan_to_num(A.T * (torch.sum(A, 1)**(-1.0)).to_dense(), nan = 0.0).T

    # nodes without out-links have an all-zero row in T, so a walker on them teleports with probability 1
    dangling = torch.sum(A, 1).to_dense() == 0

    # distribution according to nodes' in-degrees
    e_v = (torch.sum(A, dim = 0) / torch.sum(A)).to_dense()

    # calculate the flow distribution with a power iteration
    # p @ T is computed as T^T @ p with T^T in CSR format; a product with a COO matrix converts it to CSR on every call
    # the iteration runs in float64, because in float32 the change per step can stall above the threshold
    # stop when p changes by less than 1e-15 in L1 norm, as Infomap does
    T_T = T.T.coalesce().to(dtype = torch.float64).to_sparse_csr()
    e   = e_v.to(dtype = torch.float64)
    p   = e
    for _ in range(iter):
        p_next = (alpha + (1-alpha) * torch.sum(p[dangling])) * e + (1-alpha) * (T_T @ p)
        change = torch.sum(torch.abs(p_next - p))
        p      = p_next
        if change < 1e-15:
            break
    p = p.to(dtype = A.dtype)

    # make the flow matrix for minimising the map equation
    # teleporting walkers, including those on nodes without out-links, follow a link chosen in proportion to its weight
    # the sum of the two terms stores every entry twice, so coalesce it
    # store F in CSR format; a product with a COO matrix converts it to CSR on every call
    F = ((alpha + (1-alpha) * torch.sum(p[dangling])) * A / torch.sum(A) + (1-alpha) * (p * T.T).T).coalesce().to_sparse_csr()

    return F, p


class MapEquationPooling(torch.nn.Module):
    def __init__(self, adj: Tensor, chunk_size: Maybe[int] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # in batched training, x stays in CPU memory, and x_pooled is computed from chunk_size of its rows at a time
        self.chunk_size = chunk_size

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
        x_pooled   = (x.t() @ s).t() if self.chunk_size is None else pool_features(s, x, self.chunk_size)
        # multiply adj @ s first, for the same reason as F @ s above
        adj_pooled = s.T @ (self.adj @ s)

        return x_pooled, adj_pooled, codelength


class Cluster(torch.nn.Module):
    def __init__( self
                , device          : Maybe[str] = None
                , use_model       : str        = "gin"
                , in_channels     : Maybe[int] = None
                , hidden_channels : Maybe[int] = None
                , out_channels    : Maybe[int] = None
                , num_layers      : int        = 2
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                , *args
                , **kwargs
                ) -> None:
        """
        Initialiser.

        Parameters
        ----------
        use_model : str
            Which model to use.
        
        in_channels : int
            The number of input channels.

        hidden_channels : int
            The number of hidden channels.
        
        out_channels : int
            The number of output channels.
        
        num_layers : int
            The number of layers.
        
        act : str
            The activation function.

        norm : Maybe[str]
            The norm.
        
        dropout : float = 0.0
            The dropout probability.
        """
        super().__init__(*args, **kwargs)

        # set the device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.use_model       = use_model
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels    = out_channels
        self.num_layers      = num_layers
        self.act             = act
        self.norm            = norm
        self.dropout         = dropout
        
        # we only like these models
        if not use_model in ["lin", "mlp", "gcn", "gin", "gat", "sage"]:
            raise Exception(f"Don't know what to do with this model: {self.use_model}")

        self.model = None
    
    def _reset(self):
        """
        Resets the clusterer by creating an fresh model.
        """

        # PyG's conv layers read a sparse edge_index as the transposed adjacency, with one row per target node
        # data.edge_index is A[source, target], so pass its transpose to send messages from source to target
        # store it in CSR format; a product with a COO matrix converts it to CSR on every call
        adj_t = self.data.edge_index.t().coalesce().to_sparse_csr()

        if self.use_model == "lin":
            self.model = Linear( in_channels  = self.in_channels
                               , out_channels = self.out_channels
                               , dropout      = self.dropout
                               )

        elif self.use_model == "mlp":
            self.model = MLP( in_channels     = self.in_channels
                            , hidden_channels = self.hidden_channels
                            , num_layers      = self.num_layers
                            , out_channels    = self.out_channels
                            , act             = self.act
                            , norm            = self.norm
                            , dropout         = self.dropout
                            )

        elif self.use_model == "gcn":
            self.model = GCN( in_channels     = self.in_channels
                            , hidden_channels = self.hidden_channels
                            , num_layers      = self.num_layers
                            , out_channels    = self.out_channels
                            , act             = self.act
                            , norm            = self.norm
                            , dropout         = self.dropout
                            , edge_index      = adj_t
                            )
            
        elif self.use_model == "gin":
            self.model = GIN( in_channels     = self.in_channels
                            , hidden_channels = self.hidden_channels
                            , num_layers      = self.num_layers
                            , out_channels    = self.out_channels
                            , act             = self.act
                            , norm            = self.norm
                            , dropout         = self.dropout
                            , edge_index      = adj_t
                            )
        
        elif self.use_model == "gat":
            self.model = GAT( in_channels     = self.in_channels
                            , hidden_channels = self.hidden_channels
                            , num_layers      = self.num_layers
                            , out_channels    = self.out_channels
                            , act             = self.act
                            , norm            = self.norm
                            , dropout         = self.dropout
                            , edge_index      = adj_t
                            )
        
        elif self.use_model == "sage":
            self.model = SAGE( in_channels     = self.in_channels
                             , hidden_channels = self.hidden_channels
                             , num_layers      = self.num_layers
                             , out_channels    = self.out_channels
                             , act             = self.act
                             , norm            = self.norm
                             , dropout         = self.dropout
                             , edge_index      = adj_t
                             )
        
        self.model = self.model.to(self.device)

    def encode(self, x, adjs = None):
        """
        The model's output, on the whole graph, or in batched training, on a batch's subgraph, with one adjacency matrix per layer.
        """
        if adjs is None:
            return self.model(x)
        return self.model(x, adjs)

    @abstractmethod
    def objective(self, x, z):
        """
        The loss and the assignment from the model's output z for all nodes.
        """
        raise NotImplementedError(f"objective not implemented on {self._get_name()}")

    def forward(self, x):
        return self.objective(x, self.encode(x))

    def extra_regularisation(self) -> float:
        return 0

    def _prepare(self, data: Data, batch_size: Maybe[int]) -> Tensor:
        """
        Stores the graph on the device, and returns the features in the form the model takes them.
        In batched training, the features stay in CPU memory, and _encode_batch moves a batch's rows to the device.
        """
        self.batch_size = batch_size

        if batch_size is None:
            self.data = data.to(self.device)

            x = self.data.x
            if self.use_model in ["gin", "sage"]:
                x = x.to_dense()
            # DiffPool, MinCut and Ortho pool x with dense products, whatever the model
            if self._get_name() in ["DiffPool", "MinCut", "Ortho"]:
                x = x.to_dense()
            # store sparse features in CSR format; a product with a COO matrix converts it to CSR on every call
            if x.is_sparse:
                x = x.coalesce().to_sparse_csr()
            return x

        self.data = data.to(self.device, "edge_index")

        # batches of nodes that are consecutive in reverse Cuthill–McKee order, which puts linked nodes close together
        # then the nodes within a few hops of a batch are fewer, and fewer of them are also near other batches
        # the order changes only an epoch's memory and time; the gradient is the same for any order, up to rounding
        links        = self.data.edge_index.indices().cpu().numpy()
        adj          = csr_matrix((ones(links.shape[1]), (links[0], links[1])), shape = (self.data.num_nodes, self.data.num_nodes))
        order        = torch.as_tensor(reverse_cuthill_mckee(adj, symmetric_mode = False).copy(), dtype = torch.long, device = self.device)
        self.batches = order.split(batch_size)
        # the row of each node in the batches' outputs, put together in order
        self.positions = torch.argsort(order)

        # store sparse features in CSR format, whose rows csr_rows takes
        x = self.data.x.cpu()
        return x.coalesce().to_sparse_csr() if x.is_sparse else x

    def _neighbourhood(self, seeds: Tensor) -> Tuple[Tensor, List[int]]:
        """
        The nodes whose features reach the seeds through the model's layers, ordered by hops from the seeds, one hop per layer.
        Also returns the number of nodes within 0, 1, ..., L hops.
        """
        adj_t    = self.model.edge_index
        seen     = torch.zeros(adj_t.size(0), dtype = torch.bool, device = self.device)
        n_id     = seeds
        frontier = seeds
        counts   = [len(seeds)]
        seen[seeds] = True
        for _ in range(self.num_layers):
            # row v of adj_t holds the nodes that send messages to v
            neighbours     = csr_rows(adj_t, frontier).col_indices()
            frontier       = torch.unique(neighbours[~seen[neighbours]])
            seen[frontier] = True
            n_id           = torch.cat([n_id, frontier])
            counts.append(len(n_id))
        return n_id, counts

    def _subgraph(self, n_id: Tensor, num_rows: int) -> Tensor:
        """
        The rows of the model's adjacency matrix for the first num_rows nodes of n_id, with a column for each node of n_id, in n_id's order.
        The first num_rows nodes are within L - 1 hops of the batch, so all of their neighbours are in n_id.
        """
        rows  = csr_rows(self.model.edge_index, n_id[:num_rows])
        local = torch.full((self.model.edge_index.size(0),), -1, device = self.device)
        local[n_id] = torch.arange(len(n_id), device = self.device)

        row = torch.repeat_interleave(torch.arange(num_rows, device = self.device), rows.crow_indices().diff())
        return torch.sparse_coo_tensor(torch.stack([row, local[rows.col_indices()]]), rows.values(), (num_rows, len(n_id))).coalesce().to_sparse_csr()

    def _encode_batch(self, x: Tensor, seeds: Tensor) -> Tensor:
        """
        The model's output for one batch's nodes, computed on the subgraph of the nodes whose features reach them.
        Layer i of the model's L layers computes the nodes within L - i - 1 hops of the batch, from the nodes within L - i hops.
        Those are the rows that the next layer reads, and the last layer computes the batch's nodes.
        """
        n_id, counts = (seeds, [len(seeds)]) if self.use_model in ["lin", "mlp"] else self._neighbourhood(seeds)
        self.model.batch.start(n_id, len(seeds))

        # move the subgraph's features from CPU memory to the device
        rows  = n_id.cpu()
        x_sub = (csr_rows(x, rows) if x.layout == torch.sparse_csr else x[rows]).to(self.device)
        if self.use_model in ["gin", "sage"]:
            x_sub = x_sub.to_dense()

        if self.use_model in ["lin", "mlp"]:
            return self.encode(x_sub)

        # layer i's adjacency matrix: the first rows of the subgraph's, with a column for each node within L - i hops
        L    = self.num_layers
        adj  = self._subgraph(n_id, counts[L - 1])
        adjs = [first_rows(adj, counts[L - i - 1], counts[L - i]) for i in range(L)]
        return self.encode(x_sub, adjs)

    def _encode(self, x: Tensor) -> Tensor:
        """
        The model's output for all nodes, on the whole graph, or in batched training, batch by batch.
        """
        if self.batch_size is None:
            return self.encode(x)
        return torch.cat([self._encode_batch(x, seeds) for seeds in self.batches])[self.positions]

    def _collect_statistics(self, x: Tensor) -> List[NodeBatchNorm]:
        """
        Collects the mean and variance over all nodes for each NodeBatchNorm, in one pass over the batches each, in the order of the forward pass.
        Returns the NodeBatchNorms in that order.
        """
        norms = [module for module in self.model.modules() if isinstance(module, NodeBatchNorm)]
        for norm in norms:
            norm.start_epoch()

        order = []
        with torch.no_grad():
            while len(order) < len(norms):
                for seeds in self.batches:
                    try:
                        self._encode_batch(x, seeds)
                    except StopForward:
                        pass
                # the first NodeBatchNorm without statistics collected them in this pass
                norm = next(norm for norm in norms if norm.mean is None and norm.count > 0)
                norm.finish_statistics()
                order.append(norm)
        return order

    def _backprop_statistics(self, x: Tensor, norms: List[NodeBatchNorm]) -> None:
        """
        Adds the gradient that flows through each NodeBatchNorm's statistics, in one pass over the batches each, from the last NodeBatchNorm to the first.
        The statistics of a NodeBatchNorm feed the inputs of the ones after it, so their gradient is complete only after those passes.
        """
        for norm in reversed(norms):
            norm.backprop = True
            for seeds in self.batches:
                try:
                    self._encode_batch(x, seeds)
                except StopForward as stop:
                    stop.surrogate.backward()
            norm.backprop = False

    def _gradient(self, x: Tensor) -> None:
        """
        Backpropagates one training epoch's loss into the parameters' gradients.

        In batched training, the loss depends on the model's parameters θ only through the model's output z.
        So dL/dθ = Σ_u (∂z_u/∂θ)ᵀ H_u, with H = ∂L/∂z.
        A pass without gradients gives z, batch by batch, and backpropagating the loss from z gives H.
        A second pass backpropagates H into each batch's output, and the gradients add up to the whole graph's.
        """
        if self.batch_size is None:
            loss, s = self.forward(x = x)
        else:
            # one set of dropout masks per epoch, the same in every pass
            self.model.batch.seed = int(torch.randint(2**32, ()))
            norms = self._collect_statistics(x)

            with torch.no_grad():
                z = self._encode(x)
            z.requires_grad_()
            loss, s = self.objective(x, z)

        loss += self.extra_regularisation()

        if len(torch.nonzero(torch.isnan(s))) > 0:
            raise Exception("Ran into nans.")

        loss.backward()

        if self.batch_size is not None:
            for seeds in self.batches:
                self._encode_batch(x, seeds).backward(z.grad[seeds])
            self._backprop_statistics(x, norms)

    def fit( self
           , data        : Data
           , epochs      : int         = 10000
           , patience    : int         =   100
           , lrs         : List[float] = [10**(-3)]
           , lr_schedule : bool        = False
           , num_trials  : int         =     1
           , max_fails   : int         =     1
           , verbose     : bool        = False
           , batch_size  : Maybe[int]  = None
           ):
        """
        batch_size : Maybe[int]
            Without it, every epoch runs on the whole graph.
            With it, every epoch runs over batches of batch_size nodes, each on the subgraph of the nodes whose features reach it.
            The gradient is the same as on the whole graph, up to rounding, and the features stay in CPU memory.
        """
        x = self._prepare(data, batch_size)
        ls_best : List[float]  = [] # best losses
        ss_best : List[Tensor] = [] # best clusters

        for lr in lrs:
            trial  = 0
            failed = 0
            while trial < num_trials and failed < max_fails:
                try:
                    # reset the model
                    self._reset()

                    l_best  : float  = inf  # best loss
                    s_best  : Tensor = None # best cluster
                    l_ref   : float  = inf  # best loss when patience was last reset
                    l_start : float  = None # loss at epoch 10, where the drop that scales the tolerance starts

                    optimizer = torch.optim.Adam(self.parameters(), lr = lr)
                    if lr_schedule:
                        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer = optimizer, max_lr = lr, epochs = epochs, steps_per_epoch = 1)

                    epoch          = 0
                    no_improvement = 0

                    while epoch < epochs and no_improvement < patience:
                        self.train()
                        optimizer.zero_grad()
                        self._gradient(x)
                        optimizer.step()
                        if lr_schedule:
                            scheduler.step()

                        self.eval()
                        with torch.no_grad():
                            loss, s = self.objective(x, self._encode(x))

                            if verbose:
                                num_communities = len(set(s.argmax(dim = 1).cpu().numpy()))
                                print(f"trial {trial+1:3}/{num_trials}, epoch {epoch:5}, loss = {loss:0.16f}, |M| = {num_communities}")
                            if loss is None:
                                break

                            if loss < l_best:
                                l_best = float(loss)
                                s_best = s

                            # reset patience only when the loss is lower than at the last reset by more than 1e-4 of its drop since epoch 10, and on any decrease before that
                            # measured against that drop, the tolerance does not depend on the units or the offset of a method's loss
                            # the drop starts at epoch 10 because some losses spike in the first epochs, such as NOCD's with a linear model, which starts near 30
                            # a loss that keeps dropping by tiny amounts, such as a codelength while the softmax sharpens, would otherwise reset patience for thousands of epochs
                            if epoch == 10:
                                l_start = float(loss)
                            tolerance = 1e-4 * (l_start - l_ref) if epoch >= 10 else 0
                            if loss < l_ref - tolerance:
                                l_ref          = float(loss)
                                no_improvement = 0
                            else:
                                no_improvement += 1

                            epoch += 1
                    
                    ls_best.append(l_best)
                    ss_best.append(s_best)
                    
                    trial += 1
                except Exception as e:
                    print(e)
                    print(f"{self.use_model} ran into nans... resetting...")
                    failed += 1
                    sleep(1)

                finally:
                    # clean up
                    del self.model
                    gc.collect()
                    with torch.no_grad():
                        torch.cuda.empty_cache()

        return min(list(zip(ls_best, ss_best)), key = lambda p: p[0]) if len(ls_best) > 0 else (None, None)


class Neuromap(Cluster):
    def __init__( self
                , device          : Maybe[str] = None
                , use_model       : str        = "gin"
                , in_channels     : Maybe[int] = None
                , hidden_channels : Maybe[int] = None
                , out_channels    : Maybe[int] = None
                , num_layers      : int        = 2
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                , *args
                , **kwargs
                ) -> None:
        super().__init__( device          = device
                        , use_model       = use_model
                        , in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , out_channels    = out_channels
                        , num_layers      = num_layers
                        , act             = act
                        , norm            = norm
                        , dropout         = dropout
                        , *args
                        , **kwargs
                        )

        # softmax temperature, created directly on the device
        # moving a Parameter to the GPU with .to() returns a plain tensor
        # a plain tensor is not in self.parameters(), so Adam would not train it
        self.t = Parameter(torch.zeros(1, device = self.device))


    def _reset(self):
        super()._reset()
        torch.nn.init.zeros_(self.t)


    def objective(self, x, z):
        s = softmax(z / sigmoid(self.t), dim = 1)

        # the pool gets the features stored in _prepare, so NeuromapD's dropout does not reach x_pooled
        _, _, loss = self.pool(x = self.pool_x, s = s)

        return loss, s


    def _prepare(self, data: Data, batch_size: Maybe[int]) -> Tensor:
        x = super()._prepare(data, batch_size)
        self.pool = MapEquationPooling(adj = self.data.edge_index, chunk_size = batch_size)

        # store sparse features in CSC format for the pool; then x.t() is x^T in CSR format, without a copy
        # in batched training, the pool takes the rows of the features in CPU memory
        if batch_size is None:
            self.pool_x = self.data.x.coalesce().to_sparse_csc() if self.data.x.is_sparse else self.data.x
        else:
            self.pool_x = x

        return x


class NeuromapD(Neuromap):
    def __init__( self
                , device          : Maybe[str] = None
                , use_model       : str        = "gin"
                , in_channels     : Maybe[int] = None
                , hidden_channels : Maybe[int] = None
                , out_channels    : Maybe[int] = None
                , num_layers      : int        = 2
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                , *args
                , **kwargs
                ) -> None:
        super().__init__( device          = device
                        , use_model       = use_model
                        , in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , out_channels    = out_channels
                        , num_layers      = num_layers
                        , act             = act
                        , norm            = norm
                        , dropout         = dropout
                        , *args
                        , **kwargs
                        )
    
    def encode(self, x, adjs = None):
        # dropout does not support sparse tensors, so drop the stored entries of a sparse x
        # dropout leaves zeros at zero, so this is the same as dropout on the dense x
        # in batched training, the masks are keyed by node and feature, so that a node's features get the same mask in every batch
        batch   = self.model.batch
        batched = batch.n_id is not None and self.training
        if x.layout == torch.sparse_csr:
            if batched:
                values = keyed_dropout(x.values()[:, None], self.dropout, batch, torch.repeat_interleave(batch.n_id, x.crow_indices().diff()), x.col_indices())[:, 0]
            else:
                values = torch.nn.functional.dropout(x.values(), p = self.dropout, training = self.training)
            x = torch.sparse_csr_tensor(x.crow_indices(), x.col_indices(), values, x.shape)
        elif batched:
            x = keyed_dropout(x, self.dropout, batch, batch.n_id)
        else:
            x = torch.nn.functional.dropout(x, p = self.dropout, training = self.training)
        return super().encode(x, adjs)


class DMoN(Cluster):
    def __init__( self
                , device          : Maybe[str] = None
                , use_model       : str        = "gin"
                , in_channels     : Maybe[int] = None
                , hidden_channels : Maybe[int] = None
                , out_channels    : Maybe[int] = None
                , num_layers      : int        = 2
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                , *args
                , **kwargs
                ) -> None:
        super().__init__( device          = device
                        , use_model       = use_model
                        , in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , out_channels    = out_channels
                        , num_layers      = num_layers
                        , act             = act
                        , norm            = norm
                        , dropout         = dropout
                        , *args
                        , **kwargs
                        )

    def _reset(self):
        super()._reset()
        self.pool = DMoNPooling(channels = [self.out_channels, self.out_channels], k = self.out_channels, dropout = self.dropout).to(device = self.device)

    def objective(self, x, z):
        s, _, adj, sp, o, c = self.pool(x = z, adj = self.data.edge_index)
        s = s[0]
        # do not use orthogonality loss as per the original implementation
        # https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/dmon.py
        loss = sp + c

        return loss, s


class DiffPool(Cluster):
    def __init__( self
                , device          : Maybe[str] = None
                , use_model       : str        = "gin"
                , in_channels     : Maybe[int] = None
                , hidden_channels : Maybe[int] = None
                , out_channels    : Maybe[int] = None
                , num_layers      : int        = 2
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                , *args
                , **kwargs
                ) -> None:
        super().__init__( device          = device
                        , use_model       = use_model
                        , in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , out_channels    = out_channels
                        , num_layers      = num_layers
                        , act             = act
                        , norm            = norm
                        , dropout         = dropout
                        , *args
                        , **kwargs
                        )
    
    def objective(self, x, z):
        _, adj, l, e = diff_pool(x = x, adj = self.data.edge_index, s = z, chunk_size = self.batch_size)
        loss = l + e

        return loss, softmax(z)


class MinCut(Cluster):
    def __init__( self
                , device          : Maybe[str] = None
                , use_model       : str        = "gin"
                , in_channels     : Maybe[int] = None
                , hidden_channels : Maybe[int] = None
                , out_channels    : Maybe[int] = None
                , num_layers      : int        = 2
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                , *args
                , **kwargs
                ) -> None:
        super().__init__( device          = device
                        , use_model       = use_model
                        , in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , out_channels    = out_channels
                        , num_layers      = num_layers
                        , act             = act
                        , norm            = norm
                        , dropout         = dropout
                        , *args
                        , **kwargs
                        )
    
    def objective(self, x, z):
        _, adj, mc, o = mincut_pool(x = x, adj = self.data.edge_index, s = z, chunk_size = self.batch_size)
        loss = mc + o

        return loss, softmax(z)


class Ortho(Cluster):
    def __init__( self
                , device          : Maybe[str] = None
                , use_model       : str        = "gin"
                , in_channels     : Maybe[int] = None
                , hidden_channels : Maybe[int] = None
                , out_channels    : Maybe[int] = None
                , num_layers      : int        = 2
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                , *args
                , **kwargs
                ) -> None:
        super().__init__( device          = device
                        , use_model       = use_model
                        , in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , out_channels    = out_channels
                        , num_layers      = num_layers
                        , act             = act
                        , norm            = norm
                        , dropout         = dropout
                        , *args
                        , **kwargs
                        )
    
    def objective(self, x, z):
        _, adj, mc, o = mincut_pool(x = x, adj = self.data.edge_index, s = z, chunk_size = self.batch_size)
        loss = o
        s = softmax(z)

        return loss, softmax(s)


class ModelWrapper:
    def __init__(self, model) -> None:
        self.model = model

    def get_weights(self):
        """Return the weight matrices of the model."""
        return [w for n, w in self.model.named_parameters() if "bias" not in n]

    def get_biases(self):
        """Return the bias vectors of the model."""
        return [w for n, w in self.model.named_parameters() if "bias" in n]

class NOCD(Cluster):
    def __init__( self
                , device          : Maybe[str] = None
                , use_model       : str        = "gin"
                , in_channels     : Maybe[int] = None
                , hidden_channels : Maybe[int] = None
                , out_channels    : Maybe[int] = None
                , num_layers      : int        = 2
                , act             : str        = "selu"
                , norm            : Maybe[str] = "batch"
                , dropout         : float      = 0.5
                , *args
                , **kwargs
                ) -> None:
        super().__init__( device          = device
                        , use_model       = use_model
                        , in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , out_channels    = out_channels
                        , num_layers      = num_layers
                        , act             = act
                        , norm            = norm
                        , dropout         = dropout
                        , *args
                        , **kwargs
                        )
    
    def objective(self, x, z):
        z = relu(z)
        loss = self.nocd_decoder.loss_full(z, self.A) if self.batch_size is None else self._loss_in_chunks(z)

        return loss, z

    def _link_sums(self, emb, e1, e2):
        # loss_full's two sums over the links from e1 to e2
        edge_dots = torch.sum(emb[e1] * emb[e2], dim = 1)
        return -torch.sum(torch.log(-torch.expm1(-self.nocd_decoder.eps - edge_dots))), torch.sum(edge_dots)

    def _loss_in_chunks(self, emb):
        """
        BerpoDecoder.loss_full, with its sums over the links taken in chunks of links.
        loss_full multiplies the embeddings of all links' nodes at once, which takes E × K memory.
        checkpoint recomputes each chunk in the backward pass, so only one chunk's products are in memory at a time.
        """
        decoder = self.nocd_decoder

        loss_edges, edge_dots = 0, 0
        for start, end in self.link_chunks:
            loss_chunk, edge_dots_chunk = checkpoint(self._link_sums, emb, self.e1[start:end], self.e2[start:end], use_reentrant = False)
            loss_edges = loss_edges + loss_chunk
            edge_dots  = edge_dots  + edge_dots_chunk

        # the rest as in loss_full, whose neg_scale is 1, because _prepare sets balance_loss
        correction    = torch.sum(emb * emb) + edge_dots
        loss_nonedges = torch.sum(emb @ torch.sum(emb, dim = 0, keepdim = True).t()) - correction
        return (loss_edges / decoder.num_edges + loss_nonedges / decoder.num_nonedges) / 2

    def extra_regularisation(self) -> float:
        weight_decay = 1e-2    # strength of L2 regularization on GNN weights
        return nocd.utils.l2_reg_loss(ModelWrapper(self.model), scale = weight_decay)

    def _prepare(self, data: Data, batch_size: Maybe[int]) -> Tensor:
        x = super()._prepare(data, batch_size)
        self.A = to_scipy_sparse_matrix(edge_index = self.data.edge_index.indices(), edge_attr = self.data.edge_index.values())
        # data.num_edges assumes edge_index has shape [2, num_edges]; here it is the sparse adjacency, so count its stored entries
        self.nocd_decoder = nocd.nn.BerpoDecoder(self.data.num_nodes, self.A.nnz, balance_loss=True)

        if batch_size is not None:
            # chunks of as many links as a batch's nodes have on average
            e1, e2  = self.A.nonzero()
            self.e1 = torch.as_tensor(e1, dtype = torch.long, device = self.device)
            self.e2 = torch.as_tensor(e2, dtype = torch.long, device = self.device)
            step    = -(-len(e1) // len(self.batches))
            self.link_chunks = [(start, min(start + step, len(e1))) for start in range(0, len(e1), step)]

        return x