import numpy as np
from lmpc.controller.safe_set import SafeSet
from lmpc.controller.ftocp import FTOCP
import matplotlib.pyplot as plt

class LMPC(object):
	"""docstring for LMPC"""
	def __init__(self, env, P=100, l=15):
		super(LMPC, self).__init__()
		self.env = env
		self.ftocp = FTOCP(env=env)
		self.globalSS = SafeSet()
		self.iterSS = None
		self.P = P
		self.l = l
		self.optimal = None
		self.ss_list = [self.globalSS]
		self.slack = 0

	def solve(self, iter_val):
		# get parameters
		start = np.array(self.env.start)
		goal = np.array(self.env.goal)
		n = self.env.n
		d = self.env.d
		timestep = 0
		goal_threshold = 10.0
		data_cl = {}
		obs, act, costs, cost_sum = [start], [], [], 0
		# get initial guess
		self.ftocp.xGuess = None
		self.ftocp.lam = np.array([])
		x0 = np.array(self.globalSS.ss[iter_val][:self.ftocp.horizon+1]).flatten()
		# x0 = np.zeros((self.ftocp.horizon + 1)*self.env.n)
		# u0 = np.array(self.globalSS.actionSS[iter_val][:self.ftocp.horizon]).flatten()
		u0 = np.zeros((self.ftocp.horizon)*self.env.d)
		xGuess = np.concatenate((x0, u0), axis=0)
		# while not reached goal
		while (np.linalg.norm(self.env.state[:] - goal[:]) > goal_threshold):
			sol_list = []
			check = True
			for i in range(len(self.ss_list)):
				self.iterSS = self.ss_list[i]
				localSS, localVal = self.sample_local_ss(iter_val, timestep, cost_sum)
				sol = self.ftocp.solve_problem(localSS, localVal, xGuess,  use_guess=True)
				if sol["sol_status"] == False:
					check = False
				sol_dict = {
				"sol":sol,
				"ss": localSS,
				"val": localVal
				}
				sol_list.append(sol_dict)

			# if check:
			# 	print('original:', sol_list[0]["sol"]["term_cost"])
			# 	print('new:', sol_list[1]["sol"]["term_cost"])

			sol, localss, localval = self.get_best_solution(sol_list)
			if sol == None:
				self.slack = min(self.slack + 1, self.ftocp.horizon)
				continue
			plt.cla()
			x = sol["pred_horizon"][:, 0]
			y = sol["pred_horizon"][:, 1]
			ss_x = [arr[0] for arr in localss]
			ss_y = [arr[1] for arr in localss]
			plt.xlim((-75, 25))
			plt.ylim((-30, 70))
			self.env.plot_env()
			self.globalSS.plot_ss()
			plt.plot(x, y, marker='.', color='blue')
			plt.plot(ss_x, ss_y, 'bs')
			plt.pause(0.01)

			u = sol["u"].reshape(-1)
			self.env.step(u)
			timestep = timestep+1

			obs.append(self.env.state)
			act.append(u)
			costs.append(sol["costs"])
			cost_sum = cost_sum + sol["costs"]
		costs.append(0)

		data_cl["obs"] = obs
		data_cl["act"] = act
		data_cl["costs"] = costs
		data_cl["value"] = np.cumsum(np.array(costs)[::-1])[::-1]
		data_cl["iter_cost"] = data_cl["value"][0]

		# print('1st traj:', data_cl["value"])

		return data_cl

		# 	get local safe set
		# 	formulate problem
		# 	get solution
		# 	apply control
		# 	timestep ++
		# 	store closed loop data

	def sample_local_ss(self, iter_val, timestep, cost_sum):
		state_list = []
		val_list = []
		minVal = self.optimal - cost_sum
		n_iter = min(self.P, len(self.iterSS.ss)) 

		for i in range(1, n_iter+1):
			t=0
			Ti = len(self.iterSS.ss[-i])
			Tj = self.optimal
			t_star = Tj - timestep
			delta = max(0, min(timestep+Ti-Tj, Ti))
			lim = min(delta + self.ftocp.horizon + self.l, Ti)
			indices = np.arange(delta+self.ftocp.horizon-self.slack, lim).tolist()
			indices_reverse = np.arange(-max(0,min(t_star-self.ftocp.horizon+self.slack,Ti)), -max(0,min(t_star-self.ftocp.horizon,Ti)-self.l)).tolist()
			# print(indices_reverse)
			for k in indices_reverse:
				state_list.append(self.iterSS.ss[-i][k])
				val_list.append(self.iterSS.valueSS[-i][k])

		if len(state_list) == 0:
			state_list.append(self.env.goal)
			val_list.append(0)

		return state_list, val_list

	def get_best_solution(self, sol_list):
		opt_term_cost = 10e9
		sol, ss, val = None, None, None 
		for i in range(len(sol_list)):
			soln = sol_list[i]
			# print(soln["sol"]["sol_status"])
			if soln["sol"]["sol_status"] == False:
				continue
			elif soln["sol"]["term_cost"] < opt_term_cost:
				sol = soln["sol"]
				ss = soln["ss"]
				val = soln["val"]
				opt_term_cost = sol["term_cost"]

		return sol, ss, val





