import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(9)

class SOMNetwork:
	def __init__(self, layers_dim, epochs=1000, radius=2):
		super(SOMNetwork, self).__init__()
		self.learning_rate = 0.01
		self.W = self.generate_weight(layers_dim)
		self.epochs = epochs
		self.radius = radius
		self.radius_decrease_factor = 500

	def generate_weight(self,layers_dim):
		"""Initialize the SOM map
			Parameters
			----------
			layers_dim : array
				contains the dimensions of the map
				[nrRows, nrCols, inputDim]

			Returns
			-------
			W : nrROws x nrCols x inputDim array
				with uniformly sampled values between 0 and 1
		"""
		nr_rows = layers_dim[0]
		nr_cols = layers_dim[1]
		input_dim = layers_dim[2]
		W = np.random.uniform(0,1,(nr_rows, nr_cols, input_dim))
		return W

	def get_neighbourhood(self, winner):
		""" Get the neigbourhood of winning node within
		specified radius

		Parameters
		----------
		winner : array 
			coordinates of winning node
			[x, y]
		radius : integer
			radius of neighbourhood

		Returns
		-------
		neighbourhood : array
			array with coordinates of all neighbours
			to the winning node
		"""
		nr_rows = self.W.shape[0]
		nr_cols = self.W.shape[1]

		row_span = np.arange(winner[0] - self.radius, winner[0] + self.radius + 1)
		col_span = np.arange(winner[1] - self.radius, winner[1] + self.radius + 1)

		neighbourhood = []
		for i in range((2*self.radius) + 1):
			for j in range((2*self.radius) + 1):
				if((row_span[i] > (nr_rows - 1)) or (row_span[i] < 0) \
					or (col_span[j] > (nr_cols - 1)) or (col_span[j] < 0)):
					continue
				else: 
					neighbourhood.append([row_span[i], col_span[j]])

		return neighbourhood 

	def get_winner(self, x):
		""" Get the winning node for an input

		Parameters
		----------
		x : 1 x m array
			one input data point
		epochs : integer
			total number of epochs
		epoch : integer
			the current epoch
		
		Returns
		-------
		winner : array
			array containing coordinates of the winning node
		radius : integer
			the radius of the neighbourhood
		"""

		winner = []
		winner_dist = 999999999
		for i in range(len(self.W)):
			for j in range(len(self.W[0])):
				temp_distance = np.subtract(x,self.W[i][j])
				distance = np.matmul(temp_distance.T,temp_distance)

				if(distance < winner_dist):
					winner = [i, j]
					winner_dist = distance

		return winner


	def train(self, X):
		""" Update the SOM map to move the nodes towards
		the input data

		Parameters
		----------
		X : n x m array
			the input data set
		epochs : integer
			number of epochs used for training
		"""
		for epoch in range(self.epochs):
			for x in X:
				winner = self.get_winner(x)
				neighbourhood = self.get_neighbourhood(winner)
					
				for n in neighbourhood:
					self.W[n[0]][n[1]] += self.learning_rate*np.subtract(x, self.W[n[0]][n[1]])

			self.decay_learning_rate(epoch)
			self.decay_radius(epoch)

	def predict(self, X):
		""" Predict the winning node for each data point after
		training is done (for visualization purposes)

		Parameters
		----------
		X : n x m array
			the input data set

		Returns
		-------
		winners : n x 3 array
			studentID + the coordinates of the winning node Array([studentID, x, y])
			for each input 
		"""
		winners = np.zeros((len(X), 3), dtype='int')
		for i, x in enumerate(X):
			winner = 0
			winner_dist = 999999999
			for j in range(len(self.W)):
				for k in range(len(self.W[0])): 
					temp_distance = np.subtract(x, self.W[j][k])
					distance = np.matmul(temp_distance.T,temp_distance)

					if(distance < winner_dist):
						winner = [i, k, j]
						winner_dist = distance

			winners[i] = winner

		return winners

	def decay_learning_rate(self, epoch):
		""" Decay the learning rate so that the SOM
			settles for a solution

			Parameters:
			-----------
			epochs : integer
				used in the exp function so that the decay factor decreases
				as the number of iterations (epochs) increases
		"""
		self.learning_rate = self.learning_rate * np.exp(-epoch/self.epochs)

	def decay_radius(self, epoch):
		"""Decays the neighbourhood radius to
		decrease the affected area once the map
		gets closer to a solution

		Parameters:
			-----------
			epochs : integer
				used in the exp function so that the decay factor decreases
				as the number of iterations (epochs) increases
		"""
		radius_float = self.radius * np.exp(-epoch/self.radius_decrease_factor)
		self.radius = int(round(radius_float))
	
