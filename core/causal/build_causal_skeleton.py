from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
import networkx as nx
import sys
import os

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
    original_stdout = sys.stdout
    sys.stdout = sys.stderr = open(os.devnull, 'w')
    G, _ = fci(data_array, verbose = False, show_progress = False)
    sys.stdout = original_stdout
    undirected_graph = nx.Graph()
    for edge in G.get_graph_edges():
        u = edge.get_node1().get_name()
        v = edge.get_node2().get_name()
        undirected_graph.add_edge(u, v)
    mapping = {"X{}".format(i+1): name for i, name in enumerate(df.columns.tolist())}
    undirected_graph = nx.relabel_nodes(undirected_graph, mapping)
    return undirected_graph

def build_GES(df):
    data_array = df.values
    original_stdout = sys.stdout
    sys.stdout = sys.stderr = open(os.devnull, 'w')
    Record = ges(data_array)
    sys.stdout = original_stdout
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
    data_array = df.values
    model = lingam.ICALiNGAM(42, 700)
    model.fit(data_array)
    undirected_graph = nx.convert_matrix.from_numpy_array(model.adjacency_matrix_)
    mapping = {i: name for i, name in enumerate(df.columns.tolist())}
    undirected_graph = nx.relabel_nodes(undirected_graph, mapping)
    return undirected_graph
