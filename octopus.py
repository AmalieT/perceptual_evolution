import numpy as np
import random
import os, sys

class World():

	def __init__(self, width, height, n_crabs):

		self.width = width
		self.height = height

		#crab, empty, wall
		self.n_types = 3

		self.grid = np.zeros((width, height))
		#grid boundaries are walls
		self.grid[0,:] = 2
		self.grid[:,0] = 2
		self.grid[-1,:] = 2
		self.grid[:,-1] = 2

		self.n_crabs = n_crabs

	def rebirth(self):
		#When the world begins anew, it must redistribute its crabs
		#Generate crabs and distribute them
		non_wall_squares = np.zeros(((self.width-2)*(self.height-2)))

		non_wall_squares[:self.n_crabs] = 1
		#shuffle the crabs!
		np.random.shuffle(non_wall_squares)
		non_wall_squares = np.reshape(non_wall_squares, (self.width-2, self.height-2))

		self.grid[1:-1,1:-1] = non_wall_squares

		return self

	def eat_crab(self, x, y):
		#If there's a crab on the square, eat it and return True
		if self.grid[x,y] == 1:
			self.grid[x,y] = 0
			return True
		else:
			#Otherwies return False
			return False

	def get_perceptual_neighbourhood(self, x, y):
		perceptual_neighbourhood = np.array([self.grid[x,y], self.grid[x+1,y], self.grid[x-1,y], self.grid[x,y+1], self.grid[x,y-1]])

		return perceptual_neighbourhood

	def is_legal_move(self, x, y, move):
		move_pos = np.array([x,y]) + move
		if self.grid[move_pos[0], move_pos[1]] != 2:
			return True
		else:
			return False

class Octopus():

	def __init__(self, world, permissible_steps=200, n_lives=100):

		#the octopus that owns the world
		self.world = world

		self.permissible_steps = permissible_steps
		self.lives = n_lives
		self.total_reward = 0

		#Random starting position
		self.x = np.random.randint(1,self.world.width-1)
		self.y = np.random.randint(1,self.world.height-1)

		self.perceptual_neighbourhood = self.world.get_perceptual_neighbourhood(self.x, self.y)

		self.directions = {
			"north": [0,1],
			"south": [0,-1],
			"east": [1,0],
			"west": [-1,0]
		}

		moves_dict = {i: lambda m=m: self.move(m) for i,m in enumerate(self.directions.keys())}
		for m in moves_dict.values():
			m()
		n_moves = len(moves_dict.keys())

		self.actions_dict = {
			**moves_dict,
			n_moves: self.move_randomly,
			n_moves+1: self.pause,
			n_moves+2: self.eat_crab,
		}

		moves_rewards_dict = {i: {True: 0, False: -5} for i,m in enumerate(self.directions.keys())}

		self.rewards_dict = {
			**moves_rewards_dict,
			n_moves: {True: 0, False: -5},
			n_moves+1: {True: 0, False: 0},
			n_moves+2: {True: 10, False: -1},
		}


		self.perceptual_size = self.world.n_types**len(self.perceptual_neighbourhood)
		self.decision_kernel = np.random.randint(0,len(self.actions_dict.keys()), self.perceptual_size)

	def samsara(self):
		while self.lives > 0:
			self.remaining_steps = self.permissible_steps
			self.reward = 0
			self.world.rebirth()

			while self.remaining_steps > 0:
				self.next_action()
			self.total_reward += self.reward
			self.lives -= 1

		return self.total_reward
	def next_action(self):

		decision_int = int(sum([d*self.world.n_types**i for i,d in enumerate(self.perceptual_neighbourhood)]))


		action = self.decision_kernel[decision_int]

		self.take_action(action)

		return self

	def take_action(self, action):
		is_success = self.actions_dict[action]()
		reward = self.rewards_dict[action][is_success]

		self.remaining_steps -= 1
		self.reward += reward

		return self

	def move(self, direction):
		move = self.directions[direction]

		if self.world.is_legal_move(self.x, self.y, move):
			self.x += move[0]
			self.y += move[1]

			self.perceptual_neighbourhood = self.world.get_perceptual_neighbourhood(self.x, self.y)
			return True
		else:
			return False

	def move_randomly(self):
		direction = random.choice(list(self.directions.keys()))

		return self.move(direction)

	def pause(self):
		return True

	def eat_crab(self):
		result = self.world.eat_crab(self.x, self.y)
		self.perceptual_neighbourhood = self.world.get_perceptual_neighbourhood(self.x, self.y)

		return result
