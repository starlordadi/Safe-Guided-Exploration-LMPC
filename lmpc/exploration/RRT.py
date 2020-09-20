import numpy as np
import casadi as ca
from copy import deepcopy
from lmpc.envs.double_integrator import DoubleIntegrator
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import math
import pickle
import os.path as osp

count = 0

class Node():
	"""RRT nodes"""
	def __init__(self, state=None):
		self.state_arr = []
		self.state = state
		self.prev_action = []
		self.cost = 0
		self.cost_arr = []
		self.path_to_node = []
		self.action = None
		self.cost_to_go = None
		self.parent = None

	def _is_goal(self, goal, d_thresh=5.0):
		return (np.linalg.norm(self.state[:2] - goal[:2]) <= d_thresh)

class RRT(Node):
	"""Class for Rapid Random Tree methods"""
	def __init__(self, env, mode, n_states, steer_horizon, rewire=False):
		super(RRT, self).__init__()
		self.env = env
		self.mode = mode
		self.node_arr = []
		self.solution = []
		self.steer_horizon = steer_horizon		
		self.n_states = n_states
		self.rewire = rewire

	def sample_random(self, goal):
		# randomly sample a point in the observation space
		global count
		rand_node = Node()
		state = self.env.observation_space.sample() #TODO: check feasibility of state
		# state[2] = state[3] = 0
		if count % 5 == 0:
			rand_node.state = goal
		else:
			rand_node.state = state

		if rand_node._is_goal(goal=goal, d_thresh=5.0):
			rand_node.state = goal
		elif (self.env.obstacle(rand_node.state)):
			rand_node = self.sample_random(goal=goal)

		count = count + 1
		return rand_node

	def find_nearest_node(self, sample_node):
		dist = [self.get_dist(sample_node, node) for node in self.node_arr]
		near_node_ind = dist.index(min(dist))
		return near_node_ind
	
	def get_dist(self, node1, node2):
		A = self.env.Ad
		B = self.env.Bd
		Q = self.env.Q
		R = self.env.R
		P = la.solve_discrete_are(A, B, Q, R)
		dist = (node1.state - node2.state).T @ P @ (node1.state - node2.state) 
		return np.round(dist, decimals=3)	

	def steer(self, from_node, to_node, goal_node):
		if self.mode == 'mpc':
			# steer tree to desired node (using 1. MPC 2. LMPC 3. Feedback policy)
			horizon = 5
			temp_node = Node(self.node_arr[from_node].state)
			dist = self.get_dist(temp_node, to_node)
			n = self.env.observation_space.shape[0]
			d = self.env.action_space.shape[0]
			# while(dist >= 0.1):
			# define variables
			x = ca.SX.sym('x', (horizon+1) * n)
			u = ca.SX.sym('u', horizon * d)
			# define cost
			cost = 0
			for i in range(horizon):
				cost += ca.norm_2(x[(i+1)*n:(i+2)*n] - to_node.state)
			# define constraints
			current_state = temp_node.state
			constraints = []
			for j in range(horizon):
				next_state = (self.env.A @ (current_state)) + (self.env.B @ (u[j*d:(j+1)*d]))
				constraints = ca.vertcat(constraints, x[(j+1)*n + 0] == next_state[0])
				constraints = ca.vertcat(constraints, x[(j+1)*n + 1] == next_state[1])
				constraints = ca.vertcat(constraints, x[(j+1)*n + 2] == next_state[2])
				constraints = ca.vertcat(constraints, x[(j+1)*n + 3] == next_state[3])
			lbg = [0] * (horizon) * n
			ubg = [0] * (horizon) * n
			# solve
			opts = {'verbose':False, 'ipopt.print_level':0, 'print_time':0}
			nlp = {'x':ca.vertcat(x,u), 'f':cost, 'g':constraints}
			solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
			lbx = current_state.tolist() + [-100]*n*horizon + [-1]*d*horizon
			ubx = current_state.tolist() + [100]*n*horizon + [1]*d*horizon

			sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
			sol_val = np.array(sol['x'])
			u_star = sol_val[(horizon+1)*n:(horizon+1)*n+d]

			new_node = Node()
			new_node.state = self.env._next_state(current_state, u_star.reshape(-1))
			new_node.parent = self.node_arr[from_node]

		if self.mode == 'lqr':
			n = self.env.n
			d = self.env.d
			
			# get lqr gains
			Q=self.env.Q
			R=self.env.R
			# Q = np.eye(n)
			# R = np.eye(d)
			A=self.env.Ad
			B=self.env.Bd
			P = la.solve_discrete_are(A, B, Q, R)
			Ks = -np.linalg.inv(R + B.T.dot(P).dot(B)).dot(B.T).dot(P).dot(A)

			# obtain optimal control action
			start_state = self.node_arr[from_node].state
			goal_state = to_node.state
			
			# obtain new node
			new_node = Node()
			if len(self.node_arr[from_node].state_arr) == 0:
				self.node_arr[from_node].state_arr.append(start_state)
				self.node_arr[from_node].cost_to_go = (start_state - goal_node).T @ P @ (start_state - goal_node)

			for i in range(self.steer_horizon):
				state_list = []
				u_star = np.dot(Ks, (start_state - goal_state))
				u_star = np.clip(u_star,-1.0,1.0)
				self.node_arr[from_node].action = u_star
				for j in range(self.n_states):
					state_list.append(self.env.f(start_state, u_star))
				start_state = sum(state_list)/len(state_list)
				if np.linalg.norm(start_state[:2] - goal_node[:2]) <= 1.0:
					new_node.path_to_node.append(start_state)
					new_node.prev_action.append(u_star)
					new_node.cost_arr.append(np.linalg.norm(start_state[:2] - goal_node[:2]))
					break
				else:	
					new_node.path_to_node.append(start_state)
					new_node.prev_action.append(u_star)
					new_node.cost_arr.append(np.linalg.norm(start_state[:2] - goal_node[:2]))

			new_node.state_arr = state_list.copy()
			new_node.state = start_state
			new_node.parent = self.node_arr[from_node]
			new_node.cost_to_go = (new_node.state - goal_node).T @ P @ (new_node.state - goal_node)
			# new_node.cost = self.node_arr[from_node].cost + new_node.cost_to_go
			self.node_arr[from_node].cost = (self.node_arr[from_node].state - goal_node).T @ Q @ (self.node_arr[from_node].state - goal_node) + \
							(u_star.T @ R @ u_star)
			check = self.check_collision(node=new_node)
			if check:
				return None

		if self.mode == 'cartesian':
			n = self.env.observation_space.shape[0]
			d = self.env.action_space.shape[0]			
			dist = 5.0
			# get cartesian coordinates
			start = self.node_arr[from_node].state[::2]
			end = to_node.state[::2]

			# get the slope
			alpha =  math.atan2(end[-1] - start[-1],end[-2] - start[-2])

			# get the new point coordinates
			mat = np.array([math.cos(alpha), math.sin(alpha)])
			new_point = start + dist * mat

			# make new node		
			new_node = Node()
			new_node.state = np.ones(n)
			new_node.state[::2] = new_point
			new_node.path_to_node.append(new_node.state)
			new_node.state_arr.append(new_node.state)
			new_node.parent = self.node_arr[from_node]
			new_node.cost = self.node_arr[from_node].cost + 1
			# new_node.cost_to_go = (new_node.state - goal_node).T @ P @ (new_node.state - goal_node)

			# check collision of new point
			check = self.check_collision(node=new_node)
			if check:
				return None

		return new_node		

	def check_collision(self, node):
		check = np.max([self.env.obstacle(state) for state in node.path_to_node])
		# check = self.env.obstacle(node.state)
		return check

	def build_tree(self, start_state, goal_state):
		# initialize tree with starting node
		start_node = Node(start_state)
		goal_node = Node(goal_state)
		self.node_arr.append(start_node)
		# while goal not reached or fixed number of nodes
		while not self.node_arr[-1]._is_goal(goal_node.state):
			# randomly sample point in the observation space
			rnd_sample = self.sample_random(goal=goal_node.state)
			# find nearest point in the tree
			nearest_node_index = self.find_nearest_node(rnd_sample)
			nearest_node = self.node_arr[nearest_node_index]
			# plan to the sampled node and add each node to the tree
			new_node = self.steer(from_node=nearest_node_index, to_node=rnd_sample, goal_node=goal_node.state)
			if new_node == None:
				continue
			# print('orig', new_node.parent.state)
			if self.rewire:
				self.rewire_tree(new_node, goal_node=goal_node.state)
			self.node_arr.append(new_node)
			plt.cla()
			plt.xlim([-75, 25])
			plt.ylim([-30, 70])
			self.env.plot_env()
			for node in self.node_arr:
				if node.parent == None:
					continue
				x = [node.path_to_node[i][0] for i in range(len(node.path_to_node))]
				x.insert(0, node.parent.state[0])
				y = [node.path_to_node[i][1] for i in range(len(node.path_to_node))]
				y.insert(0, node.parent.state[1])
				plt.plot(x,y, color='blue', marker='.')
				plt.plot(rnd_sample.state[0], rnd_sample.state[1], 'ro')
			plt.pause(0.01)

	def plot_obstacle(self):
		for obst in self.env.obstacle.obs:
			x = [obst.boundsx[0], obst.boundsx[0], obst.boundsx[1], obst.boundsx[1], obst.boundsx[0]]
			y = [obst.boundsy[0], obst.boundsy[1], obst.boundsy[1], obst.boundsy[0], obst.boundsy[0]]
			plt.plot(x,y, color='black')

	def get_final_traj(self):
		obs, act, costs, cost_sum = [], [], [], 0
		x = []
		y = []
		state_list = []
		current_node = self.node_arr[-1]
		
		while not current_node.parent == None:
			l = len(current_node.path_to_node)
			for j in range(l):
				# obs.insert(0, current_node.path_to_node[l-j-1])
				obs.insert(0, current_node.state)
				# act.insert(0, current_node.prev_action[l-j-1])
				act.insert(0, current_node.action)
				costs.insert(0, current_node.cost)
				cost_sum = cost_sum + current_node.cost_arr[l-j-1]
			x.insert(0, [current_node.path_to_node[i][0] for i in range(len(current_node.path_to_node))])
			y.insert(0, [current_node.path_to_node[i][1] for i in range(len(current_node.path_to_node))])
			# points = [[state[0], state[2]] for state in current_node.state_arr]
			# points = np.array(points)
			# hull = ConvexHull(points)
			# for simplex in hull.simplices:
			# 	plt.plot(points[simplex,0], points[simplex,1], 'r-')
			current_node = current_node.parent

		obs.insert(0, current_node.state)
		values = np.cumsum(np.array(costs)[::-1])[::-1]
		values = np.append(values, 0)
		stabilizable_obs = current_node.state
		data_dict = {
			"obs": (obs),
            "act": (act),
            "cost_sum": cost_sum,
            "costs": costs,
            "value": values,
            "iter_cost" : values[0]
		}

		# x = [state for sublist in x for state in sublist]
		# y = [state for sublist in y for state in sublist]
		# plt.plot(x,y, color='blue', marker='o')
		# self.plot_obstacle()
		# plt.show()
		return data_dict

	def find_near_nodes(self, node):
		thresh = 1.0
		near_indices = []
		for i in range(len(self.node_arr)):
			dist = self.get_dist(self.node_arr[i], node)
			if dist <= thresh:
				near_indices.append(i)
		return near_indices

	def rewire_tree(self, node, goal_node):
		# find neighbouring nodes
		neighbours = self.find_near_nodes(node)
		# choose the best node to join the last node
		for i in range(len(neighbours)):
			if self.node_arr[neighbours[i]].cost + 1 < node.cost:
				node = self.steer(neighbours[i], node, goal_node)


if __name__ == '__main__':
	env = DoubleIntegrator()
	tree = RRT(env=env, mode='lqr', n_states=1, steer_horizon=1, rewire=False)
	start_config = Node(state = np.array([-50,0,0,0]))
	goal_config = Node(state = np.array([0,0,0,0]))
	data = []
	for i in range(5):
		tree = RRT(env=env, mode='lqr', n_states=1, steer_horizon=1, rewire=False)
		tree.build_tree(start_state=np.array([-10,-10,0,0]), goal_state=np.array([0,0,0,0]))
		data_dict = tree.get_final_traj()
		# print(data_dict["value"])
		data.append(data_dict)
	file = open("demos_2.p", 'wb')
	pickle.dump(data, file)
	file.close()
