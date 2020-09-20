import numpy as np
from math import *
import matplotlib.pyplot as plt
from gym.spaces import Box

# define constants
HORIZON = 5
DEL_T = 1.0
NOISE_SCALE = 0.0
START_STATE = [-50, 0, 0, 0]
START_STATE_3d = [-50, 0, 0, 0, 0, 0]
GOAL_STATE = [0, 0, 0, 0]
GOAL_STATE_3d = [0, 0, 0, 0, 0, 0]

# define class
class MultiObs(object):
	"""docstring for MultiObs"""
	def __init__(self, obs_list):
		super(MultiObs, self).__init__()
		self.obstacle = []
		for obs in obs_list:
			self.obstacle.append(Obstacle(obs))

	def __call__(self, x, y):
		check = np.max(obs(x,y) for obs in self.obstacle)
		return check
		

class Obstacle(object):
	"""docstring for Obstacle"""
	def __init__(self, param):
		super(Obstacle, self).__init__()
		self.xs = param[0]
		self.ys = param[1]
		self.a = param[2]
		self.b = param[3]

	def __call__(self, state):
		check = (((state[0]-self.xs)/self.a)**2 + ((state[1]-self.ys)/self.b)**2 <= 1.5)
		return check

class Obstacle3d(object):
	"""docstring for Obstacle"""
	def __init__(self, param):
		super(Obstacle, self).__init__()
		self.xs = param[0]
		self.ys = param[1]
		self.zs = param[2]
		self.a = param[3]
		self.b = param[4]
		self.c = param[5]

	def __call__(self, x, y, z):
		check = (((x-self.xs)/self.a)**2 + ((y-self.ys)/self.b)**2 + ((z-self.zs)/self.c)**2<= 1)
		return check
		

class DoubleIntegrator(object):
	"""docstring for DoubleIntegrator"""
	def __init__(self):
		super(DoubleIntegrator, self).__init__()
		self.n = 4
		self.d = 2
		self.horizon = HORIZON
		self.state = START_STATE
		self.obstacle = Obstacle([-25.0, 10.0, 10.0, 20.0])

		self.observation_space = Box(np.array([-1, -0.75, -0.1, -0.1]) * np.float(50), np.array([0.2, 0.75, 0.1, 0.1]) * np.float(50))
		self.action_space = Box(-np.ones(self.n), np.ones(self.n))

		self.A = np.array([[0, 0, 1, 0],
						   [0, 0, 0, 1],
						   [0, 0, 0, 0],
						   [0, 0, 0, 0]])
		self.B = np.array([[0, 0],
						   [0, 0],
						   [1, 0],
						   [0, 1]])
		self.del_t = DEL_T
		self.Ad = (self.A * self.del_t) + np.eye(self.n)
		self.Bd = (self.B * self.del_t)

		# self.Q = 10e-3*np.eye(self.n)
		self.Q = np.eye(self.n) @ np.diag([0.1, 0.1, 1.0, 1.0])
		self.R = 1*np.eye(self.d)		

		self.start = np.array(START_STATE)
		self.goal = np.array(GOAL_STATE)
		self.exploration_targets = None

	def reset(self):
		self.state = np.array(START_STATE)
		self.goal = np.array(GOAL_STATE)

	def step(self, u):
		obs = self.Ad @ self.state + self.Bd @ u + NOISE_SCALE * np.random.rand(self.n)
		self.state = obs

	def f(self, x, u):
		return self.Ad @ x + self.Bd @ u

	def get_demo(self):
		start = self.start
		goal = self.goal
		mid_goal = (start + goal)/2.0
		mid_goal[1] = self.obstacle.ys + self.obstacle.b + 18
		time_steps = 40	
		angle = atan2(mid_goal[1]-start[1], mid_goal[0]-start[0])
		obs, act, costs, cost_sum = [start], [], [], 0
		data_dict = {}
		for i in range(time_steps):
			if i == 0:
				ac = np.array([cos(angle), sin(angle)])
			elif i == 1:
				ac = np.array([cos(angle), sin(angle)])
			elif i == 2:
				ac = np.array([cos(angle), sin(angle)])
			elif i == time_steps/2.0 - 3:
				ac = np.array([-cos(angle), -sin(angle)])
			elif i == time_steps/2.0 - 2:
				ac = np.array([-cos(angle), -sin(angle)])
			elif i == time_steps/2.0 - 1:
				ac = np.array([-cos(angle), -sin(angle)])
			elif i == time_steps/2.0 + 0:
				ac = np.array([cos(angle), -sin(angle)])
			elif i == time_steps/2.0 + 1:
				ac = np.array([cos(angle), -sin(angle)])
			elif i == time_steps/2.0 + 2:
				ac = np.array([cos(angle), -sin(angle)])			
			elif i == time_steps-1:
				ac = np.array([-cos(angle), sin(angle)])
			elif i ==  time_steps-2:
				ac = np.array([-cos(angle), sin(angle)])
			elif i ==  time_steps-3:
				ac = np.array([-cos(angle), sin(angle)])
			else:
				ac = np.array([0, 0])

			act.append(ac)
			costs.append((self.state-goal).T @ self.Q @ (self.state-goal) + ac.T @ self.R @ ac)
			# costs.append(1)
			self.step(ac)
			obs.append(self.state)
		costs.append((self.state-goal).T @ self.Q @ (self.state-goal) + ac.T @ self.R @ ac)
		obs.append(goal)
		act.append(np.zeros(self.d))
		costs.append(0)
		# costs.append(0)

		data_dict["obs"] = obs
		data_dict["act"] = act
		data_dict["costs"] = costs
		data_dict["value"] = np.cumsum(np.array(costs)[::-1])[::-1]
		data_dict["iter_cost"] = data_dict["value"][0]

		return data_dict


	def plot_env(self):
		theta = np.linspace(0, 2*pi, 100)
		x = self.obstacle.xs + self.obstacle.a * np.cos(theta)
		y = self.obstacle.ys + self.obstacle.b * np.sin(theta)
		plt.xlim((-75, 25))
		plt.ylim((-30, 70))
		plt.plot(x,y, color='black')

			
class DoubleIntegrator3d(object):
	"""docstring for DoubleIntegrator"""
	def __init__(self):
		super(DoubleIntegrator3d, self).__init__()
		self.n = 6
		self.d = 3
		self.horizon = HORIZON
		self.state = START_STATE_3d
		# self.obstacle = Obstacle([-25.0, 0.0, 6.0, 8.0])

		self.A = np.array([[0, 0, 0, 1, 0, 0],
						   [0, 0, 0, 0, 1, 0],
						   [0, 0, 0, 0, 0, 1],
						   [0, 0, 0, 0, 0, 0],
						   [0, 0, 0, 0, 0, 0],
						   [0, 0, 0, 0, 0, 0]])
		self.B = np.array([[0, 0, 0],
						   [0, 0, 0],
						   [0, 0, 0],
						   [1, 0, 0],
						   [0, 1, 0]
						   [0, 0, 1]])
		self.del_t = DEL_T
		self.Ad = (self.A * self.del_t) + np.eye(self.n)
		self.Bd = (self.B * self.del_t)

		# self.Q = 10e-3*np.eye(self.n)
		self.Q = np.eye(self.n) @ np.diag([0.01, 0.01, 0.1, 0.1])
		self.R = np.eye(self.d)		

		self.start = np.array(START_STATE_3d)
		self.goal = np.array(GOAL_STATE_3d)
		self.exploration_targets = None

	def reset(self):
		self.state = np.array(START_STATE)
		self.goal = np.array(GOAL_STATE)

	def step(self, u):
		obs = self.Ad @ self.state + self.Bd @ u + NOISE_SCALE * np.random.rand(self.n)
		self.state = obs

	def f(self, x, u):
		return self.Ad @ x + self.Bd @ u

	def get_demo(self):
		start = self.start
		goal = self.goal
		mid_goal = (start + goal)/2.0
		mid_goal[1] = self.obstacle.ys + self.obstacle.b + 20
		time_steps = 28
		angle = atan2(mid_goal[1]-start[1], mid_goal[0]-start[0])
		obs, act, costs, cost_sum = [start], [], [], 0
		data_dict = {}
		for i in range(time_steps):
			if i == 0:
				ac = np.array([cos(angle), sin(angle)])
			elif i == 1:
				ac = np.array([cos(angle), sin(angle)])
			elif i == 2:
				ac = np.array([cos(angle), sin(angle)])
			elif i == time_steps/2.0 - 2:
				ac = np.array([-cos(angle), -sin(angle)])
			elif i == time_steps/2.0 - 1:
				ac = np.array([-cos(angle), -sin(angle)])
			elif i == time_steps/2.0 - 0:
				ac = np.array([-cos(angle), -sin(angle)])
			elif i == time_steps/2.0 + 1:
				ac = np.array([cos(angle), -sin(angle)])
			elif i == time_steps/2.0 + 2:
				ac = np.array([cos(angle), -sin(angle)])
			elif i == time_steps/2.0 + 3:
				ac = np.array([cos(angle), -sin(angle)])			
			elif i == time_steps-1:
				ac = np.array([-cos(angle), sin(angle)])
			elif i ==  time_steps-2:
				ac = np.array([-cos(angle), sin(angle)])
			elif i ==  time_steps-3:
				ac = np.array([-cos(angle), sin(angle)])
			else:
				ac = np.array([0, 0])

			act.append(ac)
			costs.append((self.state-goal).T @ self.Q @ (self.state-goal) + ac.T @ self.R @ ac)
			# costs.append(1)
			self.step(ac)
			obs.append(self.state)
		costs.append((self.state-goal).T @ self.Q @ (self.state-goal) + ac.T @ self.R @ ac)
		# costs.append(0)

		data_dict["obs"] = obs
		data_dict["act"] = act
		data_dict["costs"] = costs
		data_dict["value"] = np.cumsum(np.array(costs)[::-1])[::-1]
		data_dict["iter_cost"] = data_dict["value"][0]

		return data_dict


	def plot_env(self):
		theta = np.linspace(0, 2*pi, 100)
		x = self.obstacle.xs + self.obstacle.a * np.cos(theta)
		y = self.obstacle.ys + self.obstacle.b * np.sin(theta)
		plt.plot(x,y, color='black')
