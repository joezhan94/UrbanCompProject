import osmnx as ox
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image

regions = {'amsterdam': (52.36443115222168, 4.885260921630859, 52.37551284777832, 4.901809078369141), 'london': (51.49654515222168, -0.026412578369140623, 51.50762684777832, 0), 'utrecht': (52.09445915222168, 4.991725921630859, 52.10554084777832, 5.008274078369141), 'copenhagen': (55.69445915222168, 12.54172592163086, 55.70554084777832, 12.558274078369141), 'strasbourg': (48.59445915222168, 7.691725921630859, 48.60554084777832, 7.70827407836914), 'paris': (48.89445915222168, 2.3417259216308595, 48.90554084777832, 2.3582740783691407)}

for region in regions:
    G_road = ox.graph_from_bbox((regions[region][1], regions[region][0], regions[region][3], regions[region][2]), network_type='drive')
    G_bike = ox.graph_from_bbox((regions[region][1], regions[region][0], regions[region][3], regions[region][2]), network_type='bike')
    # Make figure same size as image
    # fig, ax = plt.subplots(figsize=(2880/300, 2880/300), dpi=300)
    fig, ax = ox.plot_graph(G_bike, show=False, close=False, node_size=0, edge_color='b', edge_linewidth=5, bgcolor='#000')
    fig, ax = ox.plot_graph(G_road, show=False, close=False, node_size=0, edge_color='r', edge_linewidth=6, bgcolor='#000', ax=ax)
    plt.axis('off')
    SIZE = 10
    size = 576*(SIZE+1), 576*(SIZE+1)
    fig.set_size_inches(size[0]/300, size[1]/300)
    plt.gca().set_position([0, 0, 1, 1])
    plt.gca().set_xlim([regions[region][1], regions[region][3]])
    plt.gca().set_ylim([regions[region][0], regions[region][2]])
    plt.savefig(f"data/masks/{region}.png", pad_inches=0, dpi=300)
    plt.close()
    # Resize to 2880x2880
    
    print(f"Saved {region}.png")