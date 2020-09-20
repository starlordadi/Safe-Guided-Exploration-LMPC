import numpy as np
import matplotlib.pyplot as plt
from math import *
from lmpc.controller.lmpc import LMPC
from lmpc.envs.double_integrator import DoubleIntegrator
from lmpc.exploration.explorer import Explorer
from lmpc.exploration.RRT import RRT

def start():
	# create environment
	env = DoubleIntegrator()
	task_goal = env.goal
	# create LMPC and exploration object
	lmpc_obj = LMPC(env)
	exp_obj = Explorer(env)
	rrt = RRT(env=env, mode='lqr', n_states=1, steer_horizon=1, rewire=False)
	# get demonstration for task safe set and exploration safe set
	# rrt.build_tree(start_state=env.start,goal_state=env.goal)
	# data_1 = rrt.get_final_traj()
	data = env.get_demo()
	data_exp = {
	"obs":[env.goal],
	"act":[np.zeros(env.d)],
	"costs":[0],
	"value":[0],
	"iter_cost":0
	}
	print("completed demonstration")
	# initialize safe set
	opt = lmpc_obj.globalSS.update(data)
	lmpc_obj.optimal = opt
	_ = exp_obj.globalSS.update(data_exp)
	lmpc_obj.ss_list.append(exp_obj.globalSS)
	# while iter < max_iter
	max_iter = 16
	lmpc_iter = 0
	theta = 0
	# run lmpc
	prev_target = task_goal
	while lmpc_iter < max_iter:
		env.reset()
		data_cl = lmpc_obj.solve(iter_val = lmpc_iter)
		print("completed {:d} iteration".format(lmpc_iter+1))
		# print(data_cl["value"])
		opt = lmpc_obj.globalSS.update(data_cl)
		# update global safe set from closed loop data
		lmpc_obj.optimal = opt
		lmpc_iter = lmpc_iter+1
		if lmpc_iter >= max_iter:
			break
		# start exploration
		print("start exploration")
		# provide target
		if lmpc_iter <= 8:
			# target_state = np.array([-10*(lmpc_iter), -10*(lmpc_iter), 0, 0])
			target_state = np.array([-25, -20, 0, 0])
		else:
			# target_state = np.array([-10*(lmpc_iter), -10*(5-(lmpc_iter)), 0, 0])
			target_state = np.array([-50, 0, 0, 0])
		# target_state = np.array([-50, 0, 0, 0])
		# get RRT demo from target to goal
		# rrt = RRT(env=env, mode='lqr', n_states=1, steer_horizon=1, rewire=False)
		# rrt.build_tree(start_state=target_state,goal_state=task_goal)
		# data_rrt = rrt.get_final_traj()
		# print('got RRT demo')
		# update exploration safe set with demo
		# exp_obj.globalSS.update(data_rrt)
		# run exploration
		env.goal = target_state
		exp_obj.move_to_target()
		# return to goal
		env.goal = task_goal
		data_exp_cl = exp_obj.return_to_goal()
		# update exploration safe set
		_ = exp_obj.globalSS.update(data_exp_cl)
		# _ = exp_obj.globalSS.ss.pop(-2)
		# _ = exp_obj.globalSS.valueSS.pop(-2)
		prev_target = target_state

		print("exploration completed")

	plt.cla()
	env.plot_env()
	lmpc_obj.globalSS.plot_value_map()
	exp_obj.globalSS.max_value = lmpc_obj.globalSS.max_value = max(lmpc_obj.globalSS.max_value, exp_obj.globalSS.max_value)
	exp_obj.globalSS.plot_value_map()
	plt.show()

if __name__ == '__main__':
	start()