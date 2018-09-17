import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(9)

class SOMNetwork:
	def __init__(self, layers_dim,cyclic = False, mp = False):
		super(SOMNetwork, self).__init__()
		self.layers = len(layers_dim)-1
		self.learning_rate = 0.01
		self.W = self.generate_weight(layers_dim)
		self.step_size = 0.2
		self.cyclic = cyclic
		self.mp = mp
		if(cyclic):
			self.get_neighbourhood = self.get_neighbourhood_cyclic
		elif(mp):
			self.get_neighbourhood = self.get_neighbourhood_mp
		else:
			self.get_neighbourhood = self.get_neighbourhood

	def generate_weight(self,layers_dim):
		W = np.random.uniform(0,1,layers_dim).T
		return W

	def get_neighbourhood(self,i, size = 50):
		neighbourhood = np.array(list(range(i-size, i))+list(range(i, i+size+1)))
		if(neighbourhood[0] < 0):
			neighbourhood = neighbourhood[np.where(neighbourhood >= 0)[0]]
		if(neighbourhood[len(neighbourhood)-1] > len(self.W)-1):
		 	neighbourhood = neighbourhood[np.where(neighbourhood < len(self.W))[0]]

		return neighbourhood

	def get_neighbourhood_cyclic(self, i, size = 2):
		neighbourhood = np.array(list(range(i-size, i))+list(range(i, i+size+1)))
		neighbourhood = neighbourhood % len(self.W)
		return neighbourhood

	def get_neighbourhood_mp(self, idx, size = 2):
		count = 0
		neighbourhood = np.zeros((10,10), dtype='object')
		rowNumber = 0
		columnNumber = 0
		radius = size
		for i in range(10):
			for j in range(10):
				neighbourhood[i][j] = (count, self.W[count])
				if(idx == count):
					rowNumber = i + 1
					columnNumber = j + 1
				count += 1

		temp = np.zeros((10,10), dtype = 'object')
		for i in range(rowNumber-1-radius,rowNumber+radius):
			for j in range(columnNumber-1-radius, columnNumber+radius):
				if(i >= 0 and i < len(neighbourhood) and j >= 0 and j < len(neighbourhood[0])):
					temp[i][j] = neighbourhood[i][j]


		neighbs = []
		for i in range(len(temp)):
			for j in range(len(temp[0])):
				if(temp[i][j] != 0):
					neighbs.append(temp[i][j][0])

		return neighbs	 	


	def train(self, X, epochs):
		for epoch in range(epochs):
			for i,x in enumerate(X):
				winner = 0
				winner_dist = 999999999
				for j, w in enumerate(self.W):
					temp_distance = np.subtract(x,w)
					distance = np.matmul(temp_distance.T,temp_distance)

					if(distance < winner_dist):
						winner = j
						winner_dist = distance
				
				if(self.cyclic != True and self.mp != True):
					neighbourhood = self.get_neighbourhood(winner, epochs-epoch)
				else:
					treshhold = epochs / 3
					size = 3
					if(epoch > treshhold and epoch < 2*treshhold):
						size = 2
					elif(epoch > 2*treshhold):
						size = 0

					neighbourhood = self.get_neighbourhood(winner, size)
					
				for neighbour in neighbourhood:
					self.W[neighbour] += self.step_size*np.subtract(x, self.W[neighbour])

	def predict(self, X):
		winners = np.zeros(len(X), dtype='int')
		for i,x in enumerate(X):
			winner = 0
			winner_dist = 999999999
			for j, w in enumerate(self.W):
				temp_distance = np.subtract(x,w)
				distance = np.matmul(temp_distance.T,temp_distance)

				if(distance < winner_dist):
					winner = j
					winner_dist = distance

			winners[i] = winner

		return winners
	
