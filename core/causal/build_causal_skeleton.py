from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
import networkx as nx
import sys
import os

from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def build_causal_skeleton(df):
    
    pc_G = build_PC(df)
    fci_G = build_FCI(df)
    ges_G = build_GES(df)
    lingam_G = build_LiNGAM(df)

    union_G = nx.compose(pc_G, fci_G)
    union_G = nx.compose(union_G, ges_G)
    union_G = nx.compose(union_G, lingam_G)

    

    return union_G

def build_PC(df):
    # print("PC")
    data_array = df.values
    CG = pc(data_array, node_names=df.columns.tolist(), show_progress=False)
    G = CG.G
    undirected_graph = nx.Graph()
    for edge in G.get_graph_edges():
        u = edge.get_node1().get_name()
        v = edge.get_node2().get_name()
        undirected_graph.add_edge(u, v)
    return undirected_graph

def build_FCI(df):
    data_array = df.values
    # print("FCI")
    with suppress_stdout():
        G, _ = fci(data_array, verbose = False, show_progress = False)
    undirected_graph = nx.Graph()
    for edge in G.get_graph_edges():
        u = edge.get_node1().get_name()
        v = edge.get_node2().get_name()
        undirected_graph.add_edge(u, v)
    mapping = {"X{}".format(i+1): name for i, name in enumerate(df.columns.tolist())}
    undirected_graph = nx.relabel_nodes(undirected_graph, mapping)
    return undirected_graph

def build_GES(df):
    # print("GES")
    data_array = df.values
    with suppress_stdout():
        Record = ges(data_array)
    G = Record['G']
    undirected_graph = nx.Graph()
    for edge in G.get_graph_edges():
        u = edge.get_node1().get_name()
        v = edge.get_node2().get_name()
        undirected_graph.add_edge(u, v)
    mapping = {"X{}".format(i+1): name for i, name in enumerate(df.columns.tolist())}
    undirected_graph = nx.relabel_nodes(undirected_graph, mapping)
    return undirected_graph

def build_LiNGAM(df):
    # print("lin")
    data_array = df.values
    model = lingam.ICALiNGAM(42, 700)
    model.fit(data_array)
    undirected_graph = nx.convert_matrix.from_numpy_array(model.adjacency_matrix_)
    mapping = {i: name for i, name in enumerate(df.columns.tolist())}
    undirected_graph = nx.relabel_nodes(undirected_graph, mapping)
    return undirected_graph
