import numpy             as np
import networkx          as nx
import seaborn           as sb
import matplotlib.pyplot as plt
import torch

def plot_overlapping(G, S, eps = 1e-3, pos = None, figsize = (5,5), node_scale = 1.0, node_palette = sb.color_palette("colorblind"), link_palette = sb.color_palette("pastel")):
    if pos is None:
        pos = nx.kamada_kawai_layout(G)
        pos = { k:(x,-y) for k,(x,y) in pos.items() }

    A = torch.tensor(nx.adjacency_matrix(G, nodelist = sorted(G.nodes)).todense(), dtype=torch.float)
    p = torch.sum(A, dim = 1) / torch.sum(A)

    existing_modules = []

    for module_ix, total_assignments in enumerate(torch.sum(S,0).numpy()):
        if total_assignments > eps:
            existing_modules.append(module_ix)

    S_reduced = S[:,existing_modules]

    fig, ax = plt.subplots(1, 1, figsize = figsize)

    modules = dict()

    for ix,node in enumerate(sorted(G.nodes())):
        assignment = np.array([s if s > eps else 0 for s in S_reduced[ix].numpy()])
        modules[node] = assignment / sum(assignment)

    edgelist     = []
    edge_colours = []
    edge_widths  = []
    for (u,v) in G.edges:
        edgelist.append((u,v))
        edge_colours.append([link_palette[m % len(link_palette)] for m,s in enumerate(modules[u]) if s > 0][0] if all(modules[u] == modules[v]) else "grey")
        d = G.get_edge_data(u,v)
        edge_widths.append(d["weight"] if "weight" in d else 1)
        
    nx.draw_networkx_edges( G = G
                          , pos = pos
                          , nodelist = sorted(G.nodes)
                          , edgelist = edgelist
                          , width = edge_widths
                          , edge_color = edge_colours
                          , ax = ax
                          , min_source_margin = 1
                          , min_target_margin = 1
                          , arrows = True
                          , connectionstyle = "arc3,rad=0.1"
                          )

    for ix,node in enumerate(sorted(G.nodes())):
        assignment = modules[node]
        ax.pie( [s for s in assignment if s > 0] # flows
              , colors = [node_palette[m % len(node_palette)] for m,s in enumerate(assignment) if s > 0]
              , center = pos[node]
              , radius = 0.25 * np.sqrt(float(p[ix])) * node_scale
              , startangle = 0 # startangle
              , wedgeprops = { "linewidth": 1, "edgecolor": "white" }
              )
    plt.autoscale()

    plt.tight_layout()


def plot_S(S, G):
    fig, ax = plt.subplots(1,1,figsize=(6,5))
    sb.heatmap(S, cmap = sb.color_palette("viridis", as_cmap=True), ax = ax)
    ax.set_xlabel("module")
    ax.set_ylabel("node")
    plt.show()