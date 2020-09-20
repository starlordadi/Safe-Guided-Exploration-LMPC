import numpy as np
import casadi as ca

class FTOCP(object):
	"""docstring for FTOCP"""
	def __init__(self, env, horizon=6):
		super(FTOCP, self).__init__()
		self.env = env
		self.horizon = horizon
		self.Q = env.Q
		self.R = env.R
		self.openLoop = None
		self.xGuess = None
		self.lam = np.array([])
		self.len_ss = None

	def solve_problem(self, ss, val, xGuess=None, use_guess = False, mode='task'):
		# define variables
		x = ca.SX.sym('x', self.env.n*(self.horizon + 1))
		u = ca.SX.sym('u', self.env.d*self.horizon)
		lam = ca.SX.sym('lam', len(ss))
		slack = ca.SX.sym('slack', int(self.env.n))
		slack_obst = ca.SX.sym('slack_obst', self.horizon)

		# get constraints
		constraints, lbg, ubg = self.get_constraints(x=x, u=u, slack=slack, slack_obst=slack_obst, lam=lam, safe_set=ss, val_list=val)

		# get cost
		cost = self.get_cost(x=x, u=u, slack=slack, slack_obst=slack_obst, lam=lam, safe_set=ss, val_list=val, mode=mode)

		# create nlp problem
		opts = {'verbose':False, 'ipopt.print_level':0, 'print_time':0}
		nlp = {'x':ca.vertcat(x,u,lam,slack,slack_obst), 'f':cost, 'g':constraints}
		solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
		
		# solve with box constraints
		lbx = self.env.state.reshape(-1).tolist() + [-1000]*(self.horizon * self.env.n) + [-1]*(self.horizon * self.env.d) + [0]*(lam.shape[0]) + \
			  [-1000]*int(self.env.n) + [1]*(self.horizon)
		ubx = self.env.state.reshape(-1).tolist() + [1000]*(self.horizon * self.env.n) + [1]*(self.horizon * self.env.d) + [1]*(lam.shape[0]) + \
			  [1000]*int(self.env.n) + [1000]*(self.horizon)

		# create guess
		if use_guess and self.lam.flatten().shape[0] == 0:
			x0 = np.concatenate((xGuess, np.zeros(len(ss) + int(self.env.n) + self.horizon)))
			sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0.tolist())
		elif use_guess:
			x0 = np.concatenate((self.xGuess, np.zeros(len(ss) + int(self.env.n) + self.horizon)))
			sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0.tolist())
		else:
			sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
		
		sol_x = np.array(sol['x'])
		pred_horizon = sol_x[:self.env.n * (self.horizon + 1)].reshape([self.horizon + 1, self.env.n])
		u = sol_x[self.env.n * (self.horizon + 1):self.env.n * (self.horizon + 1) + self.env.d]
		u_horizon = sol_x[self.env.n * (self.horizon + 1):self.env.n * (self.horizon + 1) + self.horizon*self.env.d]
		self.lam = sol_x[self.env.n*(self.horizon+1) + self.env.d*(self.horizon): self.env.n*(self.horizon+1) + self.env.d*(self.horizon) + len(ss)]
		cost = (self.env.state-self.env.goal).T @ self.Q @ (self.env.state-self.env.goal) + u.T @ self.R @ u
		# cost = 1.0
		term_cost = self.lam.T @ np.array(val).reshape([-1,1])
		cost = np.asscalar(cost)
		term_cost = np.asscalar(term_cost)

		# self.xGuess = np.concatenate((pred_horizon[0:, :].flatten(), u_horizon.flatten()))#, self.lam.flatten(), np.zeros(int(self.env.n)+self.horizon-1)))
		self.xGuess = np.concatenate((pred_horizon[0:, :].flatten(), np.zeros(self.env.d*self.horizon)))

		sol_dict = {
		"pred_horizon": pred_horizon,
		"u": u,
		"costs": cost,
		"term_cost": term_cost,
		"sol_status": solver.stats()["success"]
		}

		# if not solver.stats()["success"]:
		# print(sol["g"])
		# print(np.array(ss).T @ lam)

		return sol_dict


	def get_constraints(self, x, u, slack, slack_obst, lam, safe_set, val_list):
		constraints = []
		n = self.env.n
		d = self.env.d
		N = self.horizon
		#dynamics constraints
		q0 = self.env.state
		for i in range(N):
			q_next = self.env.f(q0,u[d*i:d*(i+1)])
			for j in range(n):
				constraints = ca.vertcat(constraints, x[n*(i+1)+j] - q_next[j,0])
			q0 = q_next

		#obstacle constraints
		for i in range(1,N+1):
			constraints = ca.vertcat(constraints, (x[n*(i)+0]-self.env.obstacle.xs)**2*(self.env.obstacle.b)**2 \
				+ (x[n*(i)+1]-self.env.obstacle.ys)**2*(self.env.obstacle.a)**2 - (self.env.obstacle.a*self.env.obstacle.b)**2*slack_obst[i-1]*1.0)

		#terminal constraint
		state_list = np.array(safe_set)
		xf = state_list.T @ lam
		for j in range(int(n)):		
			constraints = ca.vertcat(constraints, (x[n*(N)+j]) - xf[j])# + slack[j])

		#lam constraint
		constraints = ca.vertcat(constraints, np.ones(lam.shape[0]).reshape([1,lam.shape[0]]) @ lam - 1)

		#lbg and ubg
		lbg = [0]*(n*N) + [0]*(N) + [0]*int(n) + [0]
		ubg = [0]*(n*N) + [0]*(N) + [0]*int(n) + [0]

		return constraints, lbg, ubg

	def get_cost(self, x, u, slack, slack_obst, lam, safe_set, val_list, mode):
		n = self.env.n
		d = self.env.d
		N = self.horizon
		goal = self.env.goal
		cost_mode = 'quad'
		#terminal constraint cost
		# cost = 10e15*ca.dot(slack, slack)
		cost = 0
		#stage cost
		if cost_mode == 'quad':
			for i in range(N):
				cost = cost + (x[n*i:n*(i+1)]-goal).T @ self.Q @ (x[n*i:n*(i+1)]-goal)
				cost = cost + u[d*i:d*(i+1)].T @ self.R @ u[d*i:d*(i+1)]
		if cost_mode == 'unit':
			for i in range(N):
				cost = cost + 1.0

		#terminal cost
		term_val = lam.T @ np.array(val_list).reshape([-1,1])
		if mode == 'task':
			cost = cost + term_val

		return cost
	