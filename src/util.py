import networkx as nx
import torch

from infomap              import Infomap
from typing               import Dict, Tuple, List
from pathlib              import Path
from os                   import system
from torch_geometric.data import Data



def read_graph( filename_graph       : str
              , filename_communities : str
              , directed             : bool = False
              , verbose              : bool = False
              ) -> Tuple[nx.Graph, List[int]]:
    """
    Loads an LFR graph and its planted communities.

    Parameters
    ----------
    filename_graph : str
        The graph filename.

    filename_communities : str
        The communities filename.
    
    directed : bool
        Whether the graph should be considered directed.
    
    verbose : bool
        Whether to print some info about the graphs.

    Returns
    -------
    Tuple[nx.Graph, List[int]]
        A pair containing the graph and a list of planted community labels for the nodes.
    """
    # read the network
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    with open(filename_graph, "r") as fh:
        for line in fh:
            u,v = line.split("\t")
            G.add_edge(int(u.strip()), int(v.strip()))

    if verbose:
        print(f"Read graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # read ground truth communities
    ground_truth_assignments = {}
    y_true = []
    with open(filename_communities, "r") as fh:
        for line in fh:
            u,m = line.split("\t")
            ground_truth_assignments[int(u.strip())] = int(m.strip())

    y_true = [ground_truth_assignments[u] for u in sorted(G.nodes)]

    if verbose:
        print(f"Read ground truth assignments with {len(set(y_true))} modules.")

    return G, y_true


def mkBinary( N              : int
            , k              : int
            , maxk           : int
            , mu             : float
            , running_number : int
            , tau1           : float = 2.0
            , tau2           : float = 1.0
            , undirected_LFR : str   = Path("./binary_networks/benchmark")
            , folder         : str   = Path("./data/lfr-binary/")
            ) -> Tuple[nx.Graph, List[int]]:
    """
    Creates or loads a binary LFR network with propoerties as defined by the parameters.

    Parameters
    ----------
    N : int
        The number of nodes.

    k : int
        The average node degree.

    maxk : int
        The maximum node degree.

    mu : float
        The mixing parameter.

    tau1 : float = 2.0
        Power law exponent for the

    tau2 : float = 1.0
        Power law exponent for the

    undirected_LFR : str = Path("./binary_networks/benchmark")
        Path to the binary for creating binary LFR networks.

    folder : str = Path("./data/lfr_binary")
        Path to the Output folder.

    Returns
    -------
    Tuple[nx.Graph, List[int]]
        A pair containing the graph and a list of planted community labels for the nodes.
    """
    folder.mkdir(parents = True, exist_ok = True)

    network_out   = folder.joinpath(f"network-N-{N}-k-{k:.2f}-mu-{mu:.2f}-t1-{tau1:.2f}-t2-{tau2:.2f}-maxk-{maxk}.{running_number:03}.dat")
    community_out = folder.joinpath(f"community-N-{N}-k-{k:.2f}-mu-{mu:.2f}-t1-{tau1:.2f}-t2-{tau2:.2f}-maxk-{maxk}.{running_number:03}.dat")

    # create and name the network and community files if they don't already exist
    if not (network_out.exists() and community_out.exists()):
        system(f"{undirected_LFR} -N {N} -k {k} -mu {mu} -t1 {tau1} -t2 {tau2} -maxk {maxk}")
        system(f"mv network.dat {network_out}")
        system(f"mv community.dat {community_out}")

    G, y_true = read_graph(filename_graph = network_out, filename_communities = community_out, verbose = False, directed = False)
    return G, y_true


def mkDirected( N              : int
              , k              : int
              , maxk           : int
              , mu             : float
              , running_number : int
              , tau1           : float = 2.0
              , tau2           : float = 1.0
              , directed_LFR   : str   = Path("./directed_networks/benchmark")
              , folder         : str   = Path("./data/lfr-directed/")
              ) -> Tuple[nx.DiGraph, List[int]]:
    """
    Creates or loads a directed LFR network with propoerties as defined by the parameters.

    Parameters
    ----------
    N : int
        The number of nodes.

    k : int
        The average node degree.

    maxk : int
        The maximum node degree.

    mu : float
        The mixing parameter.

    tau1 : float = 2.0
        Power law exponent for the

    tau2 : float = 1.0
        Power law exponent for the

    undirected_LFR : str = Path("./binary_networks/benchmark")
        Path to the binary for creating binary LFR networks.

    folder : str = Path("./data/lfr_binary")
        Path to the Output folder.

    Returns
    -------
    Tuple[nx.DiGraph, List[int]]
        A pair containing the graph and a list of planted community labels for the nodes.
    """
    folder.mkdir(parents = True, exist_ok = True)

    network_out   = folder.joinpath(f"network-N-{N}-k-{k:.2f}-mu-{mu:.2f}-t1-{tau1:.2f}-t2-{tau2:.2f}-maxk-{maxk}.{running_number:03}.dat")
    community_out = folder.joinpath(f"community-N-{N}-k-{k:.2f}-mu-{mu:.2f}-t1-{tau1:.2f}-t2-{tau2:.2f}-maxk-{maxk}.{running_number:03}.dat")

    # create and name the network and community files if they don't already exist
    if not (network_out.exists() and community_out.exists()):
        system(f"{directed_LFR} -N {N} -k {k} -mu {mu} -t1 {tau1} -t2 {tau2} -maxk {maxk}")
        system(f"mv network.dat {network_out}")
        system(f"mv community.dat {community_out}")

    G, y_true = read_graph(filename_graph = network_out, filename_communities = community_out, verbose = False, directed = True)
    return G, y_true


def read_graph_overlapping(filename_graph : str, filename_communities : str, directed : bool = False):
    # read the network
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    with open(filename_graph, "r") as fh:
        for line in fh:
            u,v = line.split("\t")
            G.add_edge(int(u.strip()), int(v.strip()))

    # read ground truth communities
    ground_truth_assignments = {}
    with open(filename_communities, "r") as fh:
        for line in fh:
            u,ms = line.split("\t")
            ms   = ms.split(" ")
            ground_truth_assignments[int(u.strip())] = set([int(m.strip()) for m in ms if m != "\n"])

    return G, ground_truth_assignments


def sparse_from_networkx(G : nx.Graph) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    Converts a networkx graph to a sparse tensor.

    Parameters
    ----------
    G : nx.Graph
        The networkx graph, which can be weighted and/or directed.

    Returns
    ------
    Tuple[torch.Tensor, Dict[int, int]]
        A tuple containing the sparse tensor representation of the input graph
        and a dictionary from zero-based IDs to the original node names.
    """

    # always make sure to sort the nodes so they're in the expected order
    the_nodes  = list(sorted(G.nodes))
    node_to_ID = { node:ID for (ID,node) in enumerate(the_nodes) }
    ID_to_node = { ID:node for (ID,node) in enumerate(the_nodes) }

    indices = [[],[]]
    values  = []
    for u in the_nodes:
        for v in sorted(G.neighbors(u)): # again, always sorting...
            weight = 1.0
            data   = G.get_edge_data(u, v)
            if "weight" in data:
                weight = data["weight"]
            indices[0].append(node_to_ID[u])
            indices[1].append(node_to_ID[v])
            values.append(float(weight))
    
    return ( torch.sparse_coo_tensor( indices = indices
                                    , values  = values
                                    , size    = (len(the_nodes), len(the_nodes))
                                    )
           , ID_to_node
           )


def to_dataset(G : nx.Graph, y_true : List[int]) -> Data:
    """
    Takes a networkx graph and a list of community labels for the nodes and
    returns them as a pyg Data representation.

    Parameters
    ----------
    G : nx.Graph
        The networkx graph.

    y_true : List[int]
        List of the nodes' community labels.

    Returns
    -------
    Data
        A Data object where the edge index and node features X are a sparse
        tensor representation of the graph's adjacency matrix and the node
        labels a the nodes' true communities.
    """
    data = Data()
    data.edge_index = sparse_from_networkx(G)[0].coalesce()
    data.x          = sparse_from_networkx(G)[0].coalesce()
    data.y          = torch.Tensor(y_true).long()

    return data


# For convenience, we use Infomap to calculate codelengths for hard partitions
def get_codelength(G, y, directed = False) -> float:
    """
    """
    im = Infomap(silent = True, two_level = True, no_infomap = True, recorded_teleportation = directed)
    im.add_networkx_graph(G)
    im.initial_partition = dict(enumerate(y, start = 1))
    im.run()
    return im.codelength


def get_hard_clusters(S):
    hard = S.argmax(dim = 1).cpu().numpy()
    y    = []
    cluster_to_ID = dict((cluster,ix) for ix,cluster in enumerate(set(hard)))
    for cluster in hard:
        y.append(cluster_to_ID[cluster])
    return y


def load_modules(filename):
    assignments = {}

    with open(filename, "r") as fh:
        for line in fh:
            u,m = line.split("\t")
            assignments[int(u.strip())] = int(m.strip())

    return assignments