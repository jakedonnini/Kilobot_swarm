# MEAM 6240, UPenn
# Homework 1

from Node import *
from Edge import *
from Graph import *
from PIL import Image

import numpy as np

def load_binary_shape(image_path, threshold=128):
  """
  Convert a black-and-white bitmap image to a binary numpy array.
  Pixels < threshold become 1 (filled), otherwise 0.
  The image is flipped vertically to match bottom-left anchoring.
  """
  img = Image.open(image_path).convert('L')  # grayscale
  arr = np.array(img)
  binary = (arr < threshold).astype(int)
  binary = np.flipud(binary)  # anchor bottom-left
  np.savetxt("binaries", binary, fmt='%d')
  return binary

def generateRandomGraph(N, filename):
  binary = load_binary_shape(filename)
  G = Graph()

  # Define 4 seed nodes in a clump
  seed_positions = [
    np.array([0.0, 0.0, 0.0]),
    np.array([0.05, 0.1, 0.0]),
    np.array([-0.05, 0.1, 0.0]),
    np.array([0.0, 0.2, 0.0])
  ]

  for i, pos in enumerate(seed_positions):
    seed = Node(i, binary, is_seed=True)
    seed.setState(pos)
    G.addNode(seed)

  # Grid dimensions
  # num_rows = int(np.floor(np.sqrt(N)))
  # num_cols = int(np.ceil(N / num_rows))
  num_rows = 2  # fixed number of rows for better packing
  num_cols = int(np.ceil(N / num_rows))
  spacing = 0.1  # adjust spacing here for denser or looser packing

  start_x = 0.0  # starting x offset to the right of seeds
  start_y = -0.1  # starting y offset below the seeds
  
  for idx in range(N):
    uid = idx + len(seed_positions)
    row = idx // num_cols
    col = idx % num_cols

    x = start_x + col * spacing
    y = start_y - row * spacing
    n = Node(uid, binary)
    n.setState(np.array([x, y, 0.0]))

    G.addNode(n)

    new_node_index = G.Nv - 1  # this node’s index in V

    for other in G.V[:-1]:  # all previous nodes
      other_index = G.V.index(other)
      G.addEdge(other_index, new_node_index, 0)
      G.addEdge(new_node_index, other_index, 0)

  return G



### MAIN
if __name__ == '__main__':
  filename = "Black_square_10.png"

  # generate a random graph with 10 nodes
  G = generateRandomGraph(50, filename)
  
  # print out the graph descriptor
  print(G)
  for inode in G.V:
    print(inode)

  print("========== Starting now ==========")
  print("Close the figure to stop the simulation")
  G.run()             # start threads in nodes
  G.setupAnimation()  # set up plotting, halt after 1 s
  print("Sending stop signal.....")
  G.stop()            # send stop signal
  print("========== Terminated ==========")