# MEAM 6240, UPenn

from Node import *
from Edge import *
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.colors import Normalize

class Graph:
  def __init__(self, filename = None):
    """ Constructor """
    self.Nv = 0
    self.V = []
    self.E = []
    self.root = None
    self.text_labels = []
    
    # for plotting
    self.animatedt = 100 # milliseconds
    self.fig = plt.figure()
    self.ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    self.ax.set_aspect('equal', 'box')
    self.pts, = self.ax.plot([], [], 'bo')
    self.anim = None

    # Change plot to scatter for color control
    self.seed_scatter = self.ax.scatter([], [], s=80, c='green', edgecolors='black', marker='o')
    self.node_scatter = self.ax.scatter([], [], s=80, cmap='coolwarm', edgecolors='black', marker='o')
    self.negative_norm = Normalize(vmin=-10, vmax=0)  # Adjust based on your expected gradient range
    self.norm = Normalize(vmin=0, vmax=10)

    
    # for reading in graphs if they come from a file
    if not(filename is None):
      # read the graph from a file
      with open(filename) as f:
        # nodes
        line = f.readline()
        self.Nv = int(line);
        for inode in range(self.Nv):
          self.addNode(Node(inode))

        # edges      
        line = f.readline()
        while line:
          data = line.split()
        
          in_nbr = int(data[0])
          out_nbr = int(data[1])
          cost = float(data[2])
        
          self.addEdge(in_nbr, out_nbr, cost)
        
          line = f.readline()
      
      f.close()
    
  def __str__(self):
    """ Printing """
    return "Graph: %d nodes, %d edges" % (self.Nv, len(self.E))
    
  ################################################
  #
  # Modify the graph
  #
  ################################################

  def addNode(self, n):
    """ Add a node to the graph """
    self.V.append(n)
    self.Nv += 1
    
  def addEdge(self, i, o, c):
    """ Add an edge between two nodes """
    e = Edge(i, o, c)
    self.V[i].addOutgoing(e)
    self.V[o].addIncoming(e)
    self.E.append(e)
    
  ################################################
  #
  # Start and Stop computations
  #
  ################################################

  def gatherNodeLocationsAndColors(self):
    """ Collect state information and color info from all nodes """
    x = []
    y = []
    colors = []

    gradients = [v.gradient for v in self.V if not v.is_seed]
    if gradients:
      self.norm.vmin = min(gradients)
      self.norm.vmax = max(gradients)

    for v in self.V:
      x.append(v.state[0])
      y.append(v.state[1])
      if v.is_seed:
        colors.append('green')
      else:
        # Use normalized gradient mapped to colormap
        grad_norm = self.norm(v.gradient)
        rgba = cm.coolwarm(grad_norm)
        colors.append(rgba)

    return x, y, colors

  def run(self):
    """ Run the alg on all of the nodes """
    # Start running the threads
    for i in range(self.Nv):
      self.V[i].start()

  def stop(self):
    """ Send a stop signal """
    # Send a stop signal
    for i in range(self.Nv):
      self.V[i].terminate()
    # Wait until all the nodes are done
    for i in range(self.Nv):
      self.V[i].join()
      
  ################################################
  #
  # Animation helpers
  #
  ################################################

  def gatherNodeData(self):
    seed_x, seed_y = [], []
    node_x, node_y, node_colors = [], [], []

    finite_gradients = [v.gradient for v in self.V if not v.is_seed and v.gradient != float('inf')]
    finite_negative_gradients = [g for g in finite_gradients if g < 0]
    finite_positive_gradients = [g for g in finite_gradients if g >= 0]

    # Normalize ranges separately
    if finite_positive_gradients:
        self.norm.vmin = min(finite_positive_gradients)
        self.norm.vmax = max(finite_positive_gradients)
    else:
        self.norm.vmin, self.norm.vmax = 0, 1

    if finite_negative_gradients:
        self.negative_norm.vmin = min(finite_negative_gradients)
        self.negative_norm.vmax = max(finite_negative_gradients)
    else:
        self.negative_norm.vmin, self.negative_norm.vmax = -1, 0

    for v in self.V:
        if v.is_seed:
            seed_x.append(v.state[0])
            seed_y.append(v.state[1])
        else:
            node_x.append(v.state[0])
            node_y.append(v.state[1])

            g = v.gradient
            if g == float('inf'):
                color = 'white'
            elif g < 0:
                color = cm.Purples(self.negative_norm(g))  # New colormap for negative
            else:
                color = cm.coolwarm(self.norm(g))

            node_colors.append(color)

    return seed_x, seed_y, node_x, node_y, node_colors

      
  def setupAnimation(self):
    self.anim = animation.FuncAnimation(
    self.fig,
    self.animate,
    interval=self.animatedt,
    blit=True,
    cache_frame_data=False)
    plt.show()
    
  def animate(self, i):
    sx, sy, nx, ny, nc = self.gatherNodeData()

    self.seed_scatter.set_offsets(np.c_[sx, sy])
    self.node_scatter.set_offsets(np.c_[nx, ny])
    self.node_scatter.set_facecolor(nc)

    # Clear old text labels
    for label in self.text_labels:
        label.remove()
    self.text_labels.clear()

    # Add new text labels for each node
    for v in self.V:
        x, y = v.state[0], v.state[1]
        label = self.ax.text(x, y, str(v.uid), color='black', ha='center', va='center', fontsize=8, weight='bold')
        self.text_labels.append(label)

    return [self.seed_scatter, self.node_scatter] + self.text_labels
