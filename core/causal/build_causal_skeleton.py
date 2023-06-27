from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.FCMBased import lingam
from causallearn.utils.GraphUtils import GraphUtils
import networkx as nx
import numpy as np

def build_causal_skeleton(df):

    pc_G = build_PC(df)

    fci_G = build_FCI(df)
    ges_G = build_GES(df)
    lingam_G = build_LiNGAM(df)

    

    

    union_G = nx.union(pc_G, fci_G,)

    union_G = nx.union(union_G, ges_G)

    union_G = nx.union(union_G, lingam_G)
    

    return union_G.to_undirected()

def build_PC(df):

    # Convert pandas DataFrame to numpy array
    data_array = df.values

    # Get the graph
    CG = pc(data_array)

    CG.draw_pydot_graph(labels = df.columns.tolist())

    G = CG.G

    undirected_graph = nx.Graph()

    for edge in G.get_graph_edges():

        u = edge.get_node1() 
        v = edge.get_node2()

        # Step 4: Add undirected edge (u, v) and (v, u)
        undirected_graph.add_edge(u, v)

    # Convert GeneralGraph to NetworkX graph
    return undirected_graph

def build_FCI(df):

    # Convert pandas DataFrame to numpy array
    data_array = df.values

    # Get the graph
    G, _ = fci(data_array)

    undirected_graph = nx.Graph()

    for edge in G.get_graph_edges():

        u = edge.get_node1() 
        v = edge.get_node2()

        # Step 4: Add undirected edge (u, v) and (v, u)
        undirected_graph.add_edge(u, v)

    # Convert GeneralGraph to NetworkX graph
    return undirected_graph

def build_GES(df):

    data_array = df.values
    Record = ges(data_array)

    G = Record['G']

    undirected_graph = nx.Graph()

    

    for edge in G.get_graph_edges():
        u = edge.get_node1() 
        v = edge.get_node2()

        # Step 4: Add undirected edge (u, v) and (v, u)
        undirected_graph.add_edge(u, v)

    # Convert GeneralGraph to NetworkX graph
    return undirected_graph

def build_LiNGAM(df):
    data_array = df.values
    model = lingam.ICALiNGAM(42, 700)
    model.fit(data_array)

    # Convert numpy matrix to NetworkX graph
    return nx.from_numpy_matrix(model.adjacency_matrix_)
