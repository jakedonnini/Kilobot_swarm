# MEAM 6240, UPenn
# Final Project

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
  binary = np.flipud(binary)  # flip vertically to match bottom-left anchoring
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
    G.finished_nodes.append(seed)  # seeds are always active/finished

  # Grid dimensions
  batch_size = 40  # number of nodes to activate at once
  batch_size = min(batch_size, N)  # ensure we don't exceed available nodes
  # num_rows = int(np.floor(np.sqrt(N)))
  # num_cols = int(np.ceil(N / num_rows))
  num_rows = 3  # fixed number of rows for better packing
  num_cols = int(np.ceil(batch_size/ num_rows))
  spacing = 0.1  # adjust spacing here for denser or looser packing

  start_x = 0.0  # starting x offset to the right of seeds
  start_y = -0.1  # starting y offset below the seeds
  
  all_nodes = []
  for i in range(N // batch_size):
    for idx in range(batch_size):
      uid = idx + len(seed_positions) + i * batch_size
      row = idx // num_cols
      col = idx % num_cols

      x = start_x + col * spacing
      y = start_y - row * spacing
      n = Node(uid, binary)
      n.setState(np.array([x, y, 0.0]))
      all_nodes.append(n)
      G.addNode(n)

  # Create edges for ALL nodes (seeds + normal)
  for i, node_i in enumerate(G.V):
    for j, node_j in enumerate(G.V):
      if i != j:
        G.addEdge(i, j, 0)

  # Batch control AFTER the loop
  G.inactive_nodes = all_nodes[batch_size:]
  G.active_nodes = all_nodes[:batch_size]

  # Start all threads ONCE
  for node in G.V:
    node.start()

  for node in G.V:
    if node.is_seed:
      node.is_active = True

  return G



### MAIN
if __name__ == '__main__':
  start = time.time()
  # Black_star_40
  # Black_square_10.png
  # filename = "Black_square_10.png"
  # filename = "K_20.png"
  filename = "wrench_10.png"
  # filename = "circle_15.png"
  # filename = "P_20.png"
  # filename = "Black_star_20.png"

  # generate a random graph with 10 nodes
  G = generateRandomGraph(120, filename)
  
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
  print("Time elapsed: ", time.time() - start)