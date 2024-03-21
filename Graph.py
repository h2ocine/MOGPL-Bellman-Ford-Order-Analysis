import numpy as np
import copy
import random
import networkx as nx
import matplotlib.pyplot as plt
import utils

class Graph:
    def __init__(self, vertices):
        self.vertices = vertices # Liste de sommets du graphe
        self.edges = [] # Liste des arrêtes du graphe 

    #---------------------------------------
    #        FONCTIONS DE CLASS :       
    #---------------------------------------

    def add_edge(self, u, v, w):
        """
            Ajouter une arrête (u,v) de poid w
        """
        self.edges.append((u, v, w))
    
    def get_paths(self):
        """
            Retourne les paths.
        """
        return self.paths
    
    def get_node_successors(self, node):
        """
            Retourne les successeurs du sommet spécifié dans le graphe.
        """
        succ = []
        for edge in self.edges:
            if edge[0] == node:
                succ.append(edge[1])
        return
    
    def get_node_predecessors(self, node):
        """
            Retourne les predecessors du sommet spécifié dans le graphe.
        """
        pred = []
        for edge in self.edges:
            if edge[1] == node:
                pred.append(edge[0])
        return
    
    def get_nb_successor(self, node):
        """
            Retourne le nombre de successeurs du sommet spécifié dans le graphe.
        """
        count = 0
        for edge in self.edges:
            if edge[0] == node:
                count += 1
        return count
    
    def get_nb_predecessor(self, node):
        """
            Retourne le nombre de prédécesseurs du sommet spécifié dans le graphe.
        """
        count = 0
        for edge in self.edges:
            if edge[1] == node:
                count += 1
        return count

    def remove_vertex(self, vertex):
        """
            Supprime le sommet spécifié du graphe et toutes les arêtes associées.
        """
        list.remove(self.vertices, vertex)
        new_graph = []
        for edge in self.edges:
            if edge[0] != vertex and edge[1] != vertex:
                new_graph.append(edge)
        self.edges = new_graph


    #-----------------------------
    #        QUESTION 01 :       
    #-----------------------------
    
    def bellman_ford(self, src, order = None):
        """
            Implémente l'algorithme de Bellman-Ford pour trouver les plus courts chemins à partir d'un sommet source dans un graphe.

            Paramètres :
                - src : Le sommet source à partir duquel trouver les plus courts chemins.
                - order : L'ordre dans lequel les arêtes doivent être traitées. (facultatif)

            Retour :
                - dist : Un dictionnaire contenant les distances les plus courtes depuis le sommet source vers chaque sommet du graphe.
                - converged_in : l'itérations où l'algorithme converge.
                - paths : Un dictionnaire contenant les chemins les plus courts depuis le sommet source vers chaque sommet du graphe.
        """
        # Trier la liste des arcs
        new_edges = self.edges
        if(order):                                                  #gestion de l'odre
            new_edges = utils.order_edges(self,order)               #ordonne la liste des arcs en fonction des sommets données ordonnées (tout les arcs avec u en premier)
            
            self.edges = new_edges                   

        (dist, converged_in, paths) = utils.relaxation(self, src)

        # Vérification de la présence de cycles de poids négatif
        if utils.has_negative_cycle(new_edges, dist):     
            return None

        return (dist, converged_in, paths) 

    #-----------------------------
    #        QUESTION 02 :       
    #-----------------------------

    def gloutonFas(self):
        """
            Effectue l'ordonnancement glouton de graphes acycliques
        """
        graph = copy.deepcopy(self) # Copie du graphe

        for edge in self.edges:
            u, v, w = edge
            graph.add_edge(u, v, w)

        s1 = []  
        s2 = []  

        while graph.edges:
            # Recherche des sommets sources (sans prédécesseur)
            while any(graph.get_nb_predecessor(node) == 0 for node in graph.vertices):
                u = next(node for node in graph.vertices if graph.get_nb_predecessor(node) == 0)                # Récupération du sommet source
                s1.append(u)                                                                                    # Ajout du sommet source à la fin de s1
                graph.remove_vertex(u)                                                                          # Suppression du sommet u dans le graphe

            # Recherche des sommets puits (sans successeur)
            while any(graph.get_nb_successor(node) == 0 for node in graph.vertices):
                v = next(node for node in graph.vertices if graph.get_nb_successor(node) == 0)                  # Récupération du sommet puit
                s2.insert(0, v)                                                                                 # Ajout du sommet sans successeur au début de s2
                graph.remove_vertex(v)                                                                          # Suppression du sommet v du graphe

            # Sélection du sommet avec la plus grande différence d+(u) - d-(u)
            if(graph.vertices):
                u = max(graph.vertices, key=lambda node: graph.get_nb_successor(node) - graph.get_nb_predecessor(node)) # Récupération du sommet
                s1.append(u)                                                                                           # Ajout du sommet u à s1
                graph.remove_vertex(u)                                                                                           # Suppression du sommet u du graphe

        return s1 + s2

    #-----------------------------
    #        QUESTION 03 :       
    #-----------------------------

    @staticmethod
    def generate_weighted_graph(self, bound):
        """
            Génère un graphe pondéré à partir d'un graphe initial en assignant des poids aléatoires parmi les entiers de l'intervalle [-bound,+bound] aux arêtes.
        """
        have_neg_cycle = True
        while(have_neg_cycle):
            new_graph = Graph(self.vertices)
            for u, v, w in self.edges:
                weight = random.randint(-bound, bound)
                new_graph.add_edge(u, v, weight)
            
            source = utils.get_best_src_node(self)
            dist,_,paths = utils.relaxation(new_graph,source)
            utils.correct_negative_cycles(new_graph,dist,paths)

            dist,_,paths = utils.relaxation(new_graph,source)
            have_neg_cycle = utils.has_negative_cycle(new_graph.edges, dist)
        
        return new_graph

    
    #-----------------------------
    #        QUESTION 04 :       
    #-----------------------------
            
    @staticmethod
    def union_shortest_paths(graphs_paths: list, vertices: list):
        """
            Crée un nouveau graphe qui représente l'union des chemins les plus courts à partir d'une liste de graphes. 
        """
        union_graph = Graph(vertices)
        for graph_paths in graphs_paths:                                    # Parcours des graphes
            for path in graph_paths.values():                               # Parcours des chemins                                           
                for i in range(len(path) - 1):                              # Parcours des sommets dans un chemin
                    if (path[i],path[i+1],1) not in union_graph.edges:      # Eviter la répétition
                        union_graph.add_edge(path[i], path[i + 1], 1)       # Ajoute l'arête si elle n'existe pas déjà dans l'union_graph

        return union_graph
    

    #----------------------------------------
    #        FONCTION D'AFFICHAGE :       
    #----------------------------------------
    def draw_graph(self, graph_name):
        """
        Affiche le graphe en utilisant la bibliothèque NetworkX.

        :param graph_name: Le nom du graphe à afficher.
        """
        # Création du graphe NetworkX
        nx_graph = nx.DiGraph()

        # Ajout des sommets
        nx_graph.add_nodes_from(self.vertices)

        # Ajout des arcs avec des poids
        for source, destination, weight in self.edges:
            nx_graph.add_edge(source, destination, weight=weight)

        # Ajustement de l'espacement entre les nœuds (disposition circulaire)
        pos = nx.circular_layout(nx_graph)

        # Récupération des poids des arcs
        edge_labels = nx.get_edge_attributes(nx_graph, 'weight')

        # Affichage du graphe
        plt.figure(figsize=(10, 8))

        # Ajustement de la taille des nœuds
        nx.draw_networkx(nx_graph, pos, with_labels=True, node_size=1000, node_color='lightgray', font_size=15)

        # Affichage des étiquettes des arcs
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=15, rotate=False)

        plt.title(graph_name)  # Ajout du nom du graphe
        plt.axis('off')
        plt.show()

        return None

    @staticmethod
    def draw_graphs(graphs_list, graphs_names):
        """
        Affiche plusieurs graphes avec leurs noms.

        :param graphs_list: Une liste de graphes à afficher.
        :param graphs_names: Une liste de noms correspondant aux graphes.
        """
        num_graphs = len(graphs_list)
        fig, axes = plt.subplots(1, num_graphs, figsize=(10*num_graphs, 8))

        for i, graph in enumerate(graphs_list):
            nx_graph = nx.DiGraph()
            nx_graph.add_nodes_from(graph.vertices)
            for source, destination, weight in graph.edges:
                nx_graph.add_edge(source, destination, weight=weight)
            pos = nx.circular_layout(nx_graph)
            edge_labels = nx.get_edge_attributes(nx_graph, 'weight')

            axes[i].set_title(graphs_names[i])
            nx.draw_networkx(nx_graph, pos, with_labels=True, node_size=1000, node_color='lightgray', font_size=15, ax=axes[i])
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_size=15, rotate=False, ax=axes[i])
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()