import numpy as np
import matplotlib.pyplot as plt
from lmpc.controller.safe_set import SafeSet
from lmpc.controller.ftocp import FTOCP

class Explorer(object):
	"""docstring for Explorer"""
	def __init__(self, env):
		super(Explorer, self).__init__()
		self.env = env
		self.globalSS = SafeSet()
		self.ftocp = FTOCP(env=env)
		self.l = 5

	def move_to_target(self):
		timestep = 0
		reachable = True
		# while reachable in the safe set
		self.ftocp.lam = np.array([])
		while reachable:
			x_prev = self.env.state
			# get safe set and value
			ss_local, val_local = self.get_local_ss(timestep)
			if len(ss_local) == 0:
				reachable = False
				continue
			# solve ftocp
			# x0 = np.array(self.globalSS.ss[iter_val][:self.ftocp.horizon+1]).flatten()
			x0 = np.zeros((self.ftocp.horizon + 1)*self.env.n)
			# u0 = np.array(self.globalSS.actionSS[iter_val][:self.ftocp.horizon]).flatten()
			u0 = np.zeros((self.ftocp.horizon)*self.env.d)
			xGuess = np.concatenate((x0, u0), axis=0)
			sol = self.ftocp.solve_problem(ss_local, val_local, xGuess, use_guess=True, mode='exp')
			# plot
			plt.cla()
			x = sol["pred_horizon"][:, 0]
			y = sol["pred_horizon"][:, 1]
			x_ss = [arr[0] for arr in ss_local]
			y_ss = [arr[1] for arr in ss_local]
			plt.xlim((-75, 25))
			plt.ylim((-30, 70))
			self.env.plot_env()
			self.globalSS.plot_ss()
			plt.plot(x, y, marker='.', color='blue')
			plt.plot(x_ss, y_ss, 'bs')
			plt.pause(0.01)
			# apply action to env
			u = sol["u"].reshape(-1)
			self.env.step(u)
			timestep = timestep + 1
			if np.linalg.norm(x_prev-self.env.state) <= 10e-3:
				reachable = False
				print("reached target")

	def return_to_goal(self):
		data = {}
		obs, act, costs, cost_sum = [self.env.state], [], [], 0
		timestep = 0
		while np.linalg.norm(self.env.state[:2] - self.env.goal[:2]) >= 1:
			# ss, val = self.get_safe_set()
			ss, val = self.get_local_ss_goal(timestep)
			# x0 = np.zeros((self.ftocp.horizon + 1)*self.env.n)
			# u0 = np.zeros((self.ftocp.horizon)*self.env.d)
			# xGuess = np.concatenate((x0, u0), axis=0)

			sol = self.ftocp.solve_problem(ss, val, use_guess=True)
			# plot
			plt.cla()
			x = sol["pred_horizon"][:, 0]
			y = sol["pred_horizon"][:, 1]
			x_ss = [arr[0] for arr in ss]
			y_ss = [arr[1] for arr in ss]
			plt.xlim((-75, 25))
			plt.ylim((-30, 70))
			self.env.plot_env()
			self.globalSS.plot_ss()
			plt.plot(x, y, marker='.', color='blue')
			plt.plot(x_ss, y_ss, 'bs')
			plt.pause(0.01)

			u = sol["u"].reshape(-1)
			self.env.step(u)

			obs.append(self.env.state)
			act.append(u)
			costs.append(sol["costs"])
			cost_sum = cost_sum + sol["costs"]
			timestep = timestep + 1
		costs.append(0)

		data["obs"] = obs
		data["act"] = act
		data["costs"] = costs
		data["value"] = np.cumsum(np.array(costs)[::-1])[::-1]
		data["iter_cost"] = data["value"][0]

		# print('2nd traj:', data["value"])

		return data

	def get_safe_set(self):
		state_list = [arr for iteration in self.globalSS.ss for arr in iteration]
		val_list = [val for iteration in self.globalSS.valueSS for val in iteration]

		return state_list, val_list

	def get_local_ss_goal(self, timestep):
		state_list = []
		val_list = []
		opt = len(self.globalSS.ss[-1]) + self.ftocp.horizon
		t_star = opt - timestep
		for i in range(len(self.globalSS.ss)):
			Ti = len(self.globalSS.ss[i])
			indices_reverse = -(1+np.arange(max(0,min(t_star-self.ftocp.horizon-self.l,Ti-1)), max(0,min(t_star-self.ftocp.horizon,Ti-1))))
			# indices_reverse = np.arange(-max(0,min(t_star-self.ftocp.horizon,Ti)), -max(0,min(t_star-self.ftocp.horizon-self.l,Ti))).tolist()
			if len(indices_reverse)==0:
				indices_reverse = np.append(indices_reverse, -1)
			for j in indices_reverse:
				state_list.append(self.globalSS.ss[i][j])
				val_list.append(self.globalSS.valueSS[i][j])

		return state_list, val_list

	def get_local_ss(self, timestep):
		state_list = []
		val_list = []
		def_state = self.globalSS.ss[-1][0]
		def_val = self.globalSS.valueSS[-1][0]
		for i in range(len(self.globalSS.ss)):
			indices = -np.arange(min(timestep+1, len(self.globalSS.ss[i])), min(timestep+1+self.l, len(self.globalSS.ss[i])))
			for j in indices:
				state_list.append(self.globalSS.ss[i][j])
				val_list.append(self.globalSS.valueSS[i][j])

		if len(state_list) == 0:
			state_list.append(def_state)
			val_list.append(def_val)

		return state_list, val_list

			

