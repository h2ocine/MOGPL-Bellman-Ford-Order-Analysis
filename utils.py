from Graph import Graph
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
import networkx as nx

def display_bellman_ford_results(dist, converged_in):
    """
        Affiche les résultats de Bellman ford
    """
    print("Résultats de l'algorithme de Bellman-Ford :")
    print("-------------------------------------------")
    print("Distances les plus courtes :")
    for vertex, distance in dist.items():
        print(f"Vers le sommet {vertex} : {distance}")
    print("-------------------------------------------")
    print(f"L'algorithme a convergé en {converged_in} itérations.")

def get_best_src_node(G):
    """
        Renvoie un sommet source à partir duquel on peut atteindre au moins la moitié des sommets du graphe G.
    """
    src_node = None
    for node in G.vertices:
        arr = np.array(G.edges)                             
        indices = np.where(arr[:, 0] == node)[0]
        resultats = arr[indices]

        if len(resultats) >= 3:
            src_node = node
    
    if not src_node:
        print("Erreur  get_best_src_node ")
    
    return src_node

def get_random_order(G):
    """
        Renvoie une liste contenant l'ordre aléatoire des sommets du graphe G.
    """
    vertices_copy = copy.deepcopy(G.vertices)
    random.shuffle(vertices_copy)
    return vertices_copy


def generate_random_graph(nb_vertices : int, nb_edges : int, bounds : int):
    """
        Génère aléatoirement un graphe avec nb_vertices noeud, nb_edges arcs et des poids bornée par bounds.
        Renvoie le graphe génèré
    """
    vertices = list(range(1, nb_vertices + 1))
    aretes = []
    random_graph = Graph(vertices)

    for _ in range(nb_edges):
        while(1):
            u = random.choice(vertices) 
            v = random.choice(vertices)

            if ( (u, v, _ not in random_graph.edges) and u != v):
                break

        random_graph.add_edge(u,v,1)
    
    return Graph.generate_weighted_graph(random_graph, bounds)


def order_edges(G:'Graph', order:list):
    """
        Ordonne la liste des arêtes en fonction de l'ordre spécifié des sommets.
        Renvoie une nouvelle liste d'arêtes ordonnée.
    """
    new_edges = []
    for node in order: 
        arr = np.array(G.edges)                             
        indices = np.where(arr[:, 0] == node)[0]
        resultats = arr[indices]
        [new_edges.append(edge) for edge in list(resultats)]

    return new_edges

def relaxation(G, src):
    """
        Effectue la relaxation des arêtes pour calculer les distances les plus courtes à partir du sommet src dans le graphe G.
        Renvoie un tuple contenant les distances les plus courtes, les itérations dans lesquelles chaque sommet a convergé et les chemins les plus courts.
    """
    # Initialisation
    dist = {node: float('inf') for node in G.vertices}           # initialisation du dictionnaire pour les distances
    dist[src] = 0
    paths = {node: [] for node in G.vertices}                    # initialisation du dictionnaire pour les chemins 
    paths[src] = [src]
    converged_in = 0
            
    for i in range(len(G.vertices) - 1):
        dist_copy = dist.copy()
        for u, v, w in G.edges:                                   
            if dist[u] != float('inf') and dist[u] + w < dist[v]:   # condition pour la mise à jours des chemins
                dist[v] = dist[u] + w                               # Mise à jour des distances
                paths[v] = paths[u] + [v]                           # Mise à jour des paths
        if dist  == dist_copy:                                      # détection de convergence
            converged_in  = i
            break
    return (dist, converged_in, paths)

def has_negative_cycle(new_edges:list, dist:dict):
    """
        Vérifie s'il y a des cycles de poids négatif dans le graphe représenté par les arêtes et les distances données.
        Renvoie True s'il y a des cycles de poids négatif et False sinon.
    """
    for u, v, w in new_edges:   
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return True
        
    return False

def correct_negative_cycles(graph: 'Graph', dist: dict, paths: dict):
    """
        Modifie le graphe pour supprimer les circuits négatifs en convertissant les poids négatifs de ses arcs en positifs.

        Paramètres :
            - graph : Le graphe à modifié.
            - dist : Un dictionnaire contenant les distances les plus courtes calculées à partir de la relaxation.
            - paths : Un dictionnaire contenant les chemins les plus courts calculés à partir de la relaxation.
    """
    contains_negative_cycle = get_negative_cycle(paths)

    while contains_negative_cycle:
        # On récupère le cycle négatif
        negative_cycles = get_negative_cycle(paths)
        # On convertit les poids négatifs en positif
        for cycle in negative_cycles:
            for node in cycle:
                for u,v,w in graph.edges:
                    if (u == node and w < 0 ):
                        graph.edges.remove((u, v, w))
                        graph.edges.append((u, v, abs(w)))

        _, _, paths = relaxation(graph, src=0)
        contains_negative_cycle = get_negative_cycle(paths)


def get_negative_cycle( paths : dict) -> list:
    """
        Trouve un cycle de poids négatif dans les chemins donnés.
        Renvoie une liste de cycles de poids négatif.
    """
    already_seen = []
    index_cycle = []
    cycles = []
    for i, path in enumerate(paths.values()):
        already_seen.append([])
        for node in path:
            if node in already_seen[i]:
                index_cycle.append(i)
                break
            already_seen[i].append(node)
    for i in index_cycle:
        cycles.append(already_seen[i])
    return cycles

def generate_level_graph(nb_level, bound):
        """
            Cette fonction crée un graph avec des niveau
        """
        level_graph = Graph([i for i in range(nb_level * 4)])

        edges = [((i*4)+j, k, 0) for i in range(nb_level-1) for j in range(4) for k in [(i*4)+4+k for k in range(4)]]

        [level_graph.add_edge(u, v, w) for u, v, w in edges]
        level_graph = Graph.generate_weighted_graph(level_graph, bound)

        return level_graph


#-----------------------------
#        QUESTION 09 :       
#-----------------------------
def get_greedyfas_stat(initial_graph, nb_graph):
    """
    Calcule les statistiques pour l'algorithme greedyfas.

    Args:
        initial_graph (Graph): Le graphe initial.
        nb_graph (int): Le nombre de graphes à générer.

    Returns:
        float: Le pourcentage de convergence amélioré par rapport à un ordre aléatoire.
        int : Le nombre d'itération pour converger avec l'ordre aléatoire
        int : Le nombre d'itération pour converger avec l'ordre glouton.
    """
    bound = 10
    src = get_best_src_node(initial_graph)  # Le meilleur noeud source à considérer

    G_greedy_order = []
    G_greedy_order_bf = []
    union_shortest_paths = []

    for i in range(nb_graph):
        G_greedy_order.append(Graph.generate_weighted_graph(initial_graph, bound))
        G_greedy_order_bf.append(G_greedy_order[i].bellman_ford(src))

    H = Graph.generate_weighted_graph(initial_graph, bound)
    H_bf = H.bellman_ford(src)


    for graph_bf in G_greedy_order_bf:
        _, _, path = graph_bf
        union_shortest_paths.append(path)
        path = []

    _, _, path = H_bf
    union_shortest_paths.append(path)

    T = Graph.union_shortest_paths(union_shortest_paths, initial_graph.vertices)
    greedy_order = T.gloutonFas()

    src_T = get_best_src_node(initial_graph)
    _, converged_in_ordered, _ = H.bellman_ford(src_T, greedy_order)

    # Générer un ordre aléatoire avec la fonction get_random_order()
    random_ordre = get_random_order(H)
    _, converged_in_glouton, _ = H.bellman_ford(src, random_ordre)
    converged_in_ordered += 1
    converged_in_glouton += 1

    stat = 100 * ((converged_in_glouton - converged_in_ordered) / converged_in_glouton)
    return stat, converged_in_ordered, converged_in_glouton


#-----------------------------
#        QUESTION 10 :       
#-----------------------------
import random

def generate_level_graph(nb_level, nb_vertices_per_level, bound):
    """
        Cette fonction crée un graphe avec des niveaux 
        :param nb_level: nombre de niveaux
        :param nb_vertices_per_level: nombre de sommets par niveau
        :param bound: borne pour les poids des arcs
        :return: Le graphe généré 
    """
    level_graph = Graph([i for i in range(nb_level * nb_vertices_per_level)])

    edges = [((i * nb_vertices_per_level) + j, k, 0) for i in range(nb_level - 1) for j in range(nb_vertices_per_level) for k in [(i * nb_vertices_per_level) + nb_vertices_per_level + k for k in range(nb_vertices_per_level)]]

    [level_graph.add_edge(u, v, w) for u, v, w in edges]
    level_graph = Graph.generate_weighted_graph(level_graph, bound)

    return level_graph