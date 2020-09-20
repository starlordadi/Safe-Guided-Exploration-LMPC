import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class SafeSet(object):
	"""docstring for SafeSet"""
	def __init__(self):
		super(SafeSet, self).__init__()
		self.ss = []
		self.valueSS = []
		self.actionSS = []
		self.iter_costs = None
		self.max_vel = 0
		self.max_value = 0

	def update(self, data):
		self.ss.append(data["obs"])
		self.valueSS.append(data["value"])
		self.actionSS.append(data["act"])
		val_list = [iteration[0] for iteration in self.valueSS]
		time_list = [len(iteration) for iteration in self.valueSS]
		self.iter_costs = val_list
		self.max_vel = max(self.max_vel, max([np.sqrt(arr[2]**2 + arr[3]**2) for arr in data["obs"]]))
		if len(self.ss) != 1:
			self.max_value = max(self.max_value, max(data["value"]))
		return min(time_list)

	def plot_ss(self):
		for it in range(len(self.ss)):
			x = [arr[0] for arr in self.ss[it]]
			y = [arr[1] for arr in self.ss[it]]
			plt.plot(x, y, 'rs')

	def plot_velocity_map(self):
		for it in range(len(self.ss)):
			vel_arr = [np.sqrt(arr[2]**2 + arr[3]**2) for arr in self.ss[it]]/self.max_vel
			colors = cm.magma(vel_arr)
			for j in range(len(vel_arr)):
				plt.plot(self.ss[it][j][0], self.ss[it][j][1], marker='s', color=colors[j])
			# plt.colorbar()

	def plot_value_map(self):
		for it in range(len(self.valueSS)):
			val_arr = self.valueSS[it]/self.max_value
			colors = cm.viridis(val_arr)
			for j in range(len(val_arr)):
				plt.plot(self.ss[it][j][0], self.ss[it][j][1], marker='s', color=colors[j])
