import torch
import torch.nn.functional as F

from typing             import Optional as Maybe
from torch.nn           import Linear
from torch_geometric.nn import MLP, GCN, GIN, GAT, GraphSAGE


class LinearModel(torch.nn.Module):
    def __init__( self
                , in_channels  : int
                , out_channels : int
                , dropout      : float = 0.5
                ) -> None:
        super().__init__()

        self.lin     = Linear(in_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x):
        return F.dropout(self.lin(x), training = self.training)


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

        self.model = MLP( in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , num_layers      = num_layers
                        , out_channels    = out_channels
                        , act             = act
                        , dropout         = dropout
                        , norm            = norm
                        )

    def forward(self, x):
        return self.model(x)


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

        self.edge_index = edge_index

        self.model = GCN( in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , num_layers      = num_layers
                        , out_channels    = out_channels
                        , act             = act
                        , dropout         = dropout
                        , norm            = norm
                        )

    def forward(self, x):
        return self.model(x = x, edge_index = self.edge_index)


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

        self.model = GIN( in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , num_layers      = num_layers
                        , out_channels    = out_channels
                        , act             = act
                        , norm            = norm
                        , dropout         = dropout
                        )

    def forward(self, x):
        return self.model(x = x, edge_index = self.edge_index)


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

        self.model = GAT( in_channels     = in_channels
                        , hidden_channels = hidden_channels
                        , num_layers      = num_layers
                        , out_channels    = out_channels
                        , act             = act
                        , norm            = norm
                        , dropout         = dropout
                        )

    def forward(self, x):
        return self.model(x = x, edge_index = self.edge_index)


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

        self.model = GraphSAGE( in_channels     = in_channels
                              , hidden_channels = hidden_channels
                              , num_layers      = num_layers
                              , out_channels    = out_channels
                              , act             = act
                              , norm            = norm
                              , dropout         = dropout
                              )

    def forward(self, x):
        return self.model(x = x, edge_index = self.edge_index)