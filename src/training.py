import torch
import nocd

from abc                   import abstractmethod
from model                 import LinearModel as Linear, MLPModel as MLP, GCNModel as GCN, GINModel as GIN, GATModel as GAT, SAGEModel as SAGE
from numpy                 import inf
from time                  import sleep
from typing                import List, Optional as Maybe
from torch                 import Tensor
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data  import Data
from torch.nn.functional   import softmax, sigmoid, relu
from torch.nn              import Parameter
from pool                  import DMoNPooling, mincut_pool, diff_pool

import gc


# create smart teleportation flow matrix and flow distribution
def mkSmartTeleportationFlow(A, alpha = 0.15, iter = 1000, device : str = "cpu"):
    # build the transition matrix
    T = torch.nan_to_num(A.T * (torch.sum(A, 1)**(-1.0)).to_dense(), nan = 0.0).T.to(device = device)

    # distribution according to nodes' in-degrees
    e_v = (torch.sum(A, dim = 0) / torch.sum(A)).to_dense().to(device = device)

    # calculate the flow distribution with a power iteration
    p = e_v
    for _ in range(iter):
        p = alpha * e_v + (1-alpha) * p @ T
    
    # make the flow matrix for minimising the map equation
    F = alpha * A / torch.sum(A) + (1-alpha) * (p * T.T).T
    
    return F, p


class MapEquationPooling(torch.nn.Module):
    def __init__(self, adj: Tensor, device : str = "cpu", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.adj       = adj
        self.F, self.p = mkSmartTeleportationFlow(self.adj, device = device)

        # this term is constant, so only calculate it once
        self.p_log_p = torch.sum(self.p * torch.nan_to_num(torch.log2(self.p), nan = 0.0))

    def forward(self, x, s):
        C      = s.T @ self.F @ s
        diag_C = torch.diag(C)

        q   = 1.0 - torch.trace(C)
        q_m = torch.sum(C, dim = 1) - diag_C
        m_exit = torch.sum(C, dim = 0) - diag_C
        p_m = q_m + torch.sum(C, dim = 0)


        codelength = torch.sum(q      * torch.nan_to_num(torch.log2(q),      nan = 0.0)) \
                   - torch.sum(q_m    * torch.nan_to_num(torch.log2(q_m),    nan = 0.0)) \
                   - torch.sum(m_exit * torch.nan_to_num(torch.log2(m_exit), nan = 0.0)) \
                   - self.p_log_p \
                   + torch.sum(p_m    * torch.nan_to_num(torch.log2(p_m),    nan = 0.0))

        x_pooled   = torch.matmul(s.T, x)
        adj_pooled = s.T @ self.adj @ s

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
                            , edge_index      = self.data.edge_index
                            )
            
        elif self.use_model == "gin":
            self.model = GIN( in_channels     = self.in_channels
                            , hidden_channels = self.hidden_channels
                            , num_layers      = self.num_layers
                            , out_channels    = self.out_channels
                            , act             = self.act
                            , norm            = self.norm
                            , dropout         = self.dropout
                            , edge_index      = self.data.edge_index
                            )
        
        elif self.use_model == "gat":
            self.model = GAT( in_channels     = self.in_channels
                            , hidden_channels = self.hidden_channels
                            , num_layers      = self.num_layers
                            , out_channels    = self.out_channels
                            , act             = self.act
                            , norm            = self.norm
                            , dropout         = self.dropout
                            , edge_index      = self.data.edge_index
                            )
        
        elif self.use_model == "sage":
            self.model = SAGE( in_channels     = self.in_channels
                             , hidden_channels = self.hidden_channels
                             , num_layers      = self.num_layers
                             , out_channels    = self.out_channels
                             , act             = self.act
                             , norm            = self.norm
                             , dropout         = self.dropout
                             , edge_index      = self.data.edge_index
                             )
        
        self.model = self.model.to(self.device)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError(f"forward not implemented on {self._get_name()}")
    
    def extra_regularisation(self) -> float:
        return 0

    def fit( self
           , data        : Data
           , epochs      : int         = 10000
           , patience    : int         =   100
           , lrs         : List[float] = [10**(-3)]
           , lr_schedule : bool        = False
           , num_trials  : int         =     1
           , max_fails   : int         =     1
           , verbose     : bool        = False
           ):
        """
        """
        self.data = data.to(self.device)
        ls_best : List[float]  = [] # best losses
        ss_best : List[Tensor] = [] # best clusters

        x = self.data.x
        if self.use_model in ["gin", "sage"]:
            x = x.to_dense()
        if self.use_model in ["lin", "mlp", "gcn"] and self._get_name() in ["DiffPool", "MinCut", "Ortho"]:
            x = x.to_dense()

        for lr in lrs:
            trial  = 0
            failed = 0
            while trial < num_trials and failed < max_fails:
                try:
                    # reset the model
                    self._reset()

                    l_best : float  = inf  # best loss
                    s_best : Tensor = None # best cluster

                    optimizer = torch.optim.Adam(self.parameters(), lr = lr)
                    if lr_schedule:
                        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer = optimizer, max_lr = lr, epochs = epochs, steps_per_epoch = 1)

                    epoch          = 0
                    no_improvement = 0

                    while epoch < epochs and no_improvement < patience:
                        self.train()
                        optimizer.zero_grad()

                        loss, s = self.forward(x = x)
                        loss += self.extra_regularisation()

                        if len(torch.nonzero(torch.isnan(s))) > 0:
                            raise Exception("Ran into nans.")

                        loss.backward()
                        optimizer.step()
                        if lr_schedule:
                            scheduler.step()

                        self.eval()
                        with torch.no_grad():
                            loss, s = self.forward(x = x)

                            if verbose:
                                num_communities = len(set(s.argmax(dim = 1).cpu().numpy()))
                                print(f"trial {trial+1:3}/{num_trials}, epoch {epoch:5}, loss = {loss:0.16f}, |M| = {num_communities}")
                            if loss is None:
                                break

                            if loss < l_best:
                                l_best         = float(loss)
                                no_improvement = 0
                                s_best         = s
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

        # softmax temperature
        self.t = Parameter(torch.zeros(1)).to(device = self.device)


    def forward(self, x):
        s = softmax(self.model(x) / sigmoid(self.t), dim = 1)

        if self.training:
            s = s + 1e-8

        _, _, loss = self.pool(x = x, s = s)

        return loss, s


    def fit( self
           , data        : Data
           , epochs      : int         = 10000
           , patience    : int         =   100
           , lrs         : List[float] = [10**(-3)]
           , lr_schedule : bool        = False
           , num_trials  : int         =     1
           , verbose     : bool        = False
           ):
        self.data = data.to(device = self.device)
        self.pool = MapEquationPooling(adj = data.edge_index)
        
        return super().fit(data = data, epochs = epochs, patience = patience, lrs = lrs, lr_schedule = lr_schedule, num_trials = num_trials, verbose = verbose)


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
    
    def forward(self, x):
        x = torch.nn.functional.dropout(x, p = self.dropout, training = self.training)
        return super().forward(x)


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

    def forward(self, x):
        x = self.model(x)
        x, _, adj, sp, o, c = self.pool(x = x, adj = self.data.edge_index)
        x = x[0]
        # do not use orthogonality loss as per the original implementation
        # https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/dmon.py
        loss = sp + c

        return loss, x

    def fit( self
           , data        : Data
           , epochs      : int         = 10000
           , patience    : int         =   100
           , lrs         : List[float] = [10**(-3)]
           , lr_schedule : bool        = False
           , num_trials  : int         =     1
           , verbose     : bool        = False
           ):
        self.pool = DMoNPooling(channels = [self.out_channels, self.out_channels], k = self.out_channels, dropout = self.dropout).to(device = self.device)
        return super().fit(data = data, epochs = epochs, patience = patience, lrs = lrs, lr_schedule = lr_schedule, num_trials = num_trials, verbose = verbose)


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
    
    def forward(self, x):
        s = self.model(x)
        _, adj, l, e = diff_pool(x = x, adj = self.data.edge_index, s = s)
        loss = l + e

        return loss, softmax(s)


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
    
    def forward(self, x):
        s = self.model(x)
        _, adj, mc, o = mincut_pool(x = x, adj = self.data.edge_index, s = s)
        loss = mc + o

        return loss, softmax(s)


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
    
    def forward(self, x):
        s = self.model(x)
        _, adj, mc, o = mincut_pool(x = x, adj = self.data.edge_index, s = s)
        loss = o
        s = softmax(s)

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
    
    def forward(self, x):
        x = relu(self.model(x))
        loss = self.nocd_decoder.loss_full(x, self.A)

        return loss, x

    def extra_regularisation(self) -> float:
        weight_decay = 1e-2    # strength of L2 regularization on GNN weights
        return nocd.utils.l2_reg_loss(ModelWrapper(self.model), scale = weight_decay)

    def fit( self
           , data        : Data
           , epochs      : int         = 10000
           , patience    : int         =   100
           , lrs         : List[float] = [10**(-3)]
           , lr_schedule : bool        = False
           , num_trials  : int         =     1
           , verbose     : bool        = False
           ):
        self.data = data.to(device = self.device)
        self.A = to_scipy_sparse_matrix(edge_index = data.edge_index.indices(), edge_attr = data.edge_index.values())
        self.nocd_decoder = nocd.nn.BerpoDecoder(self.data.num_nodes, self.data.num_edges, balance_loss=True)
        return super().fit(data = data, epochs = epochs, patience = patience, lrs = lrs, lr_schedule = lr_schedule, num_trials = num_trials, verbose = verbose)