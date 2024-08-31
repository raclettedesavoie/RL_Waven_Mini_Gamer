import networkx as nx
import matplotlib.pyplot as plt

def draw_mcts_tree(root):
    # Crée un graphe vide
    G = nx.DiGraph()
    
    # Fonction récursive pour ajouter les nœuds et les arêtes
    def add_edges(node):
        for child in node.children:
            G.add_edge(node.id, child.id)
            add_edges(child)
    
    add_edges(root)
    
    print(f"Nombre d'arêtes dans le graphe : {G.number_of_edges()}")
    # Dessine l'arbre
    pos = nx.spring_layout(G, seed=42)  # Pour une disposition agréable
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    plt.show()