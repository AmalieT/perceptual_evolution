import numpy as np
import random
import os, sys
from scipy.special import softmax

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

		self.n_moves = len(moves_dict.keys())

		self.actions_dict = {
			**moves_dict,
			self.n_moves: self.move_randomly,
			self.n_moves+1: self.pause,
			self.n_moves+2: self.eat_crab,
		}

		self.n_actions = len(self.actions_dict.keys())

		moves_rewards_dict = {i: {True: 0, False: -5} for i,m in enumerate(self.directions.keys())}

		self.rewards_dict = {
			**moves_rewards_dict,
			self.n_moves: {True: 0, False: -5},
			self.n_moves+1: {True: 0, False: 0},
			self.n_moves+2: {True: 10, False: -1},
		}


		self.perceptual_size = self.world.n_types**len(self.perceptual_neighbourhood)
		self.decision_kernel = self.random_gene()

	def samsara(self):
		while self.lives > 0:
			#The world is born anew and the octopus tries again
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

	def random_gene(self):
		random_gene = np.random.randint(0, self.n_actions, self.perceptual_size)

		return random_gene

class MolluskEvolver():

	def __init__(self, n_octopodes, n_generations):

		self.n_octopodes=n_octopodes
		self.n_generations=n_generations
		
		self.permissible_steps = 50
		self.octopus_lives = 50

		self.world = World(12, 12, 50)


		self.octopode_ensemble = [self.create_octopus() for _ in range(self.n_octopodes)]
		self.octopus_gene_length = len(self.octopode_ensemble[0].decision_kernel)
		self.octopus_action_space = self.octopode_ensemble[0].n_actions
		self.mutation_chance = 0.01

	def create_octopus(self):
		octopus = Octopus(self.world, permissible_steps=self.permissible_steps, n_lives=self.octopus_lives)

		return octopus

	def evolve_octopodes(self):
		fitnesses = []
		while self.n_generations > 0:
			fitness = self.next_generation()
			print(self.n_generations)
			print(np.mean(fitness))
			print(np.max(fitness))
			print("\n")
			fitnesses.append(fitness)
			self.n_generations -= 1

		return fitnesses


	def next_generation(self):
		#Octopodes live, die, and are reborn within the confines of a single list comprehension
		raw_fitness = np.array([o.samsara() for o in self.octopode_ensemble])

		norm_fitness = softmax(raw_fitness)

		self.octopode_ensemble = self.panmixia(norm_fitness)
		return raw_fitness/self.octopus_lives

	def panmixia(self, octopus_fitness):
		next_generation_octopodes = []
		while len(next_generation_octopodes) < self.n_octopodes:
			octopus_sub, octopus_domme = np.random.choice(self.octopode_ensemble, size=2, replace=False, p=octopus_fitness)
			child_A, child_B = self.breed_octopodes(octopus_sub, octopus_domme)
			next_generation_octopodes.append(child_A)
			next_generation_octopodes.append(child_B)
		
		return next_generation_octopodes

	def breed_octopodes(self, octopus_sub, octopus_domme):
		split_spot = np.random.randint(0,self.octopus_gene_length)

		def merge_genes(child_oct, oct_A, oct_B):
			child_oct.decision_kernel[:split_spot] = oct_A.decision_kernel[:split_spot]
			child_oct.decision_kernel[split_spot:] = oct_B.decision_kernel[split_spot:]

		child_A = self.create_octopus()
		merge_genes(child_A, octopus_sub, octopus_domme)
		self.mutate(child_A)

		child_B = self.create_octopus()
		merge_genes(child_B, octopus_domme, octopus_sub)
		self.mutate(child_B)

		return child_A, child_B

	def mutate(self, octopus):
		mutation_mask = np.random.rand(self.octopus_gene_length) < self.mutation_chance
		mutated_gene = octopus.random_gene()

		octopus.decision_kernel = np.where(mutation_mask, mutated_gene, octopus.decision_kernel)

		return octopus