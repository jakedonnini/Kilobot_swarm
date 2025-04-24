# MEAM 6240, UPenn

from threading import Thread
from queue import Empty
import numpy as np
import time
import math
import random

class Node(Thread):
    def __init__(self, uid, binary, is_seed=False, r_sense=0.15):
        super().__init__()
        self.uid = uid
        self.out_nbr = []
        self.in_nbr = []
        self.state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.done = False
        self.nominaldt = 0.05
        self.start_time = None
        self.moving = False  # True if currently moving

        self.is_seed = is_seed
        # firt seed has gradient 0, others have gradient 1
        if is_seed and self.uid == 0:
          self.gradient = 0
        elif is_seed:
          self.gradient = 1
        else:
          self.gradient = float('inf')
        self.r_sense = r_sense
        self.r_sense_initial = r_sense  # Store initial sensing radius for later use
        self.visible_neighbors = {}  # uid: (pos, grad, dist)
        self.first_time_goal = True

        self.can_move = False
        self.edge_node = False

        self.density_threshold = 3  # edge if fewer than this many neighbors
        self.move_speed = 0.02

        self.reached_goal = False
        self.reached_same_grad_in_goal = False

        self.last_neighbors = set()
        self.last_neighbor_change_time = None
        self.neighbor_change_delay = 1  # seconds

        # goal state
        self.binary = np.array(binary)
        self.cell_size = 0.1

        flip_binary = np.flipud(binary)

        # Find all positions where binary == 1
        rows, cols = np.where(flip_binary == 1)

        # Step 1: Find the most bottom row (max row index)
        bottom_row = max(rows)

        # Step 2: Among the bottom-most rows, find the left-most column
        candidate_cols = cols[rows == bottom_row]
        left_col = min(candidate_cols)

        # Set anchor based on bottom-left most 1
        self.anchor = np.array([left_col, bottom_row]) * self.cell_size
        # self.anchor = np.zeros(2)
        self.goal_indices = set(zip(rows, cols))
        # print(f"Anchor set to bottom-left 1 at bitmap grid position ({bottom_row}, {left_col}) -> anchor: {self.anchor}")



    def pos_to_grid(self, pos):
      """Convert continuous position to grid coordinates."""
      rel_pos = (self.anchor - pos) / self.cell_size
      # if self.uid == 47:
      #   print(f"Node {self.uid} pos_to_grid: {pos} -> rel_pos: {rel_pos}, anchor: {self.anchor}, cell_size: {self.cell_size}")
      i, j = int(round(rel_pos[1])), int(round(rel_pos[0]))
      return (i, j)
    
    def is_inside_goal(self, pos):
      return self.pos_to_grid(pos) in self.goal_indices

    def is_about_to_exit(self, pos, proposed_pos):
      return self.is_inside_goal(pos) and not self.is_inside_goal(proposed_pos)

    def get_goal_gradient_value(self, pos):
      return 0 if self.is_inside_goal(pos) else float('inf')

    def setState(self, state):
      self.state = state

    def addOutgoing(self, e):
      """ Add an edge for outgoing messages """
      self.out_nbr.append(e)
    
    def addIncoming(self, e):
      """ Add an edge for incoming messages """
      self.in_nbr.append(e)

    def terminate(self):
      """ stop sim """
      self.done = True

    def __str__(self):
      """ Printing """
      return "Node %d has %d in_nbr and %d out_nbr Gradent %f" % (self.uid, len(self.in_nbr), len(self.out_nbr), self.gradient)

    def run(self):
      """ Send messages, Retrieve message, Transition """
      while (not self.done):
        start = time.time()

        if self.start_time is None:
          self.start_time = start

        self.send()
        self.transition()
        
        # Wait 2 seconds before running systemdynamics
        if time.time() - self.start_time >= 2:
          self.systemdynamics()
        
        end = time.time()
        time.sleep(max(self.nominaldt - (end-start), 0))

    def send(self):
        # if self.moving:
        #   return  # Don't broadcast while moving
        
        for out_edge in self.out_nbr:
            out_edge.put((self.uid, self.state[:2], self.gradient, len(self.visible_neighbors), self.reached_goal))

    def transition(self):
      self.visible_neighbors.clear()
      for in_edge in self.in_nbr:
        try:
          sender_uid, sender_pos, sender_grad, num_neighbors, neighbor_reached_goal = in_edge.get_nowait()
          dist = np.linalg.norm(sender_pos - self.state[:2])
          # print(f"Node {self.uid} received from {sender_uid}: pos={sender_pos}, grad={sender_grad}, dist={dist}")
          if dist <= self.r_sense:
            self.visible_neighbors[sender_uid] = (sender_pos, sender_grad, dist, num_neighbors, neighbor_reached_goal, sender_uid)
        except Empty:
          continue

      current_neighbors = set(self.visible_neighbors.keys())
      if current_neighbors != self.last_neighbors:
        self.last_neighbors = current_neighbors
        self.last_neighbor_change_time = time.time()

      # Update gradient
      if not self.is_seed:
        min_grad = min(
          [grad for _, grad, _, _, _, _ in self.visible_neighbors.values()],
          default=float('inf')
        )
        self.gradient = min_grad + 1

      # Handle nodes that reached the goal:
      if self.reached_goal:
        self.r_sense = self.r_sense_initial
        # print(f"Node {self.uid} reached the goal. gradent: {self.gradient} {[(id, grad) for _, grad, _, _, _, id in self.visible_neighbors.values()]}")
        return

      larger_grad = False
      same_grad = False
      max_grad = 0
      min_num_neighbors = 100  # Arbitrary large number
      lowest_y_value = float('inf')  # Initialize to a very large value
      grads_of_neighbors_in_goal = []
      for pos, grad, _, num_neighbors, goal, _ in self.visible_neighbors.values():
        # if neighbor in goal
        if goal:
          grads_of_neighbors_in_goal.append(grad)
        max_grad = max(max_grad, grad)
        min_num_neighbors = min(min_num_neighbors, num_neighbors)

        if grad == self.gradient:
          lowest_y_value = min(lowest_y_value, pos[1])  # Track the lowest y value

      # if there is a larger gradient, this node can't move
      if max_grad > self.gradient:
        larger_grad = True
      elif max_grad == self.gradient:
        # Prioritize the node with the lowest y value
        # print(f"Node {self.uid} min num {min_num_neighbors} {len(self.visible_neighbors)}, same_grad: {same_grad}, larger_max: {self.gradient}, {max_grad} lowest_y_value: {lowest_y_value}, state[1]: {self.state[1]}")
        if min_num_neighbors >= len(self.visible_neighbors):
          if lowest_y_value >= self.state[1]:
            # print(f"\nPOP\nNode {self.uid} min num {min_num_neighbors} {len(self.visible_neighbors)}, same_grad: {same_grad}, larger_max: {self.gradient}, {max_grad} lowest_y_value: {lowest_y_value}, state[1]: {self.state[1]}")
            same_grad = False
        else:
          same_grad = True      

      if len(self.visible_neighbors) == 0:
        self.r_sense *= 1.1  # increase sensing radius if no neighbors are detected
      else:
        self.r_sense = self.r_sense_initial

      # second stop condition: if this node is in the goal and has the same gradient as a neighbor in the goal, stop moving
      if self.is_inside_goal(self.state[:2]) and self.gradient in grads_of_neighbors_in_goal:
        self.reached_same_grad_in_goal = True
        return

      # if self.uid == 19 or self.uid == 36 or self.uid == 53:
      #   print(f"Node {self.uid} larger_grad: {larger_grad}, same_grad: {same_grad}, num: {len(self.visible_neighbors)}, {min_num_neighbors} gradient: {self.gradient}, moving: {self.moving} y: {self.state[1]}, {lowest_y_value}, {lowest_y_value >= self.state[1]}")
                    
      self.can_move = not larger_grad and not same_grad

    def systemdynamics(self):
      current_time = time.time()
      if (self.last_neighbor_change_time is not None 
          and not self.moving 
          and (current_time - self.last_neighbor_change_time) < self.neighbor_change_delay):
        return  # Wait before moving due to neighbor list change

      if (not self.can_move or self.is_seed) and not self.moving:
        return  # only move if lowest gradient AND on edge
      
      if self.reached_goal:
        self.moving = False
        return

      self.moving = True

      # Compute average vector to neighbors
      if not self.visible_neighbors:
        return

      pos = self.state[:2]
      desired_spacing = 0.12
      travel_dist = 0.05
      radial_force = np.zeros(2)
      spacing_dir_sum = np.zeros(2)  # Used to compute tangent

      # Step 1: spacing correction and direction averaging
      for nbr_pos, _, dist, _, _, _ in self.visible_neighbors.values():
        vec = nbr_pos - pos
        if dist > 1e-6:
          error = dist - travel_dist
          radial_force += (vec / dist) * error
          spacing_dir_sum += vec

      # Normalize spacing direction
      if np.linalg.norm(spacing_dir_sum) > 1e-6:
        spacing_dir_sum /= np.linalg.norm(spacing_dir_sum)

      # Step 2: get centroid to define consistent clockwise tangent
      positions = np.array([p for p, _, _, _, _, _ in self.visible_neighbors.values()])
      centroid = np.mean(positions, axis=0)
      to_center = centroid - pos
      if np.linalg.norm(to_center) > 1e-6:
        to_center /= np.linalg.norm(to_center)

      tangent = np.array([-to_center[1], to_center[0]])  # clockwise
      tangent /= np.linalg.norm(tangent) + 1e-6

      # Step 3: weighted combination of tangent + radial spacing control
      move_vec = 0.7 * tangent + 0.3 * radial_force
      move_vec /= np.linalg.norm(move_vec) + 1e-6

      proposed_pos = pos + self.move_speed * move_vec

      # Before trying to move, check if node is about to leave goal
      if (self.is_about_to_exit(pos, proposed_pos) or self.reached_same_grad_in_goal) and self.first_time_goal:
        self.reached_goal = True
        self.moving = False
        self.first_time_goal = False
        self.gradient = float('inf')
        return

      # Step 4: collision-aware movement with angular sweeping
      def rotate_vector(v, angle_rad):
        """Rotate a 2D vector by given angle in radians."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

      # Try original move vector first
      angles_to_try = [0, -5, -10, -20, -30, -45, 5]  # degrees
      moved = False

      for angle_deg in angles_to_try:
        angle_rad = np.radians(angle_deg)
        adjusted_vec = rotate_vector(move_vec, angle_rad)
        adjusted_vec /= np.linalg.norm(adjusted_vec) + 1e-6

        proposed_pos = pos + self.move_speed * adjusted_vec

        collision = False
        for nbr_pos, _, _, _, _, _ in self.visible_neighbors.values():
          if np.linalg.norm(proposed_pos - nbr_pos) < 0.9 * desired_spacing:
            collision = True
            break

        if not collision:
          self.state[:2] = proposed_pos
          moved = True
          break  # stop on first safe direction

      # print(f"Node {self.uid} moved to {self.state[:2]} with angle {angle_deg} degrees.")

      if not moved:
        # Step 5: Fallback – try to move away from centroid
        push_vec = pos - centroid
        if np.linalg.norm(push_vec) > 1e-6:
            push_vec /= np.linalg.norm(push_vec)
        else:
            push_vec = np.array([1.0, 0.0])  # arbitrary fallback

        proposed_pos = pos + self.move_speed * push_vec

        collision = False
        for nbr_pos, _, _, _, _, _ in self.visible_neighbors.values():
            if np.linalg.norm(proposed_pos - nbr_pos) < 0.9 * desired_spacing:
                collision = True
                break

        if not collision:
            # print(f"Node {self.uid} using fallback push-out to avoid blockage.")
            self.state[:2] = proposed_pos
        else:
            # print(f"Node {self.uid} completely blocked — staying still.")
            self.moving = False
