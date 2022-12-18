import numpy as np
import random
import os, sys
from scipy.special import softmax
from time import sleep

class World():

	def __init__(self, width, height, n_crabs, crab_max=10):

		self.width = width
		self.height = height

		#wall, green, red
		self.n_types = 3

		self.crab_max = crab_max

		self.grid = np.zeros((width, height))

		#grid boundaries are walls
		self.grid[0,:] = -1
		self.grid[:,0] = -1
		self.grid[-1,:] = -1
		self.grid[:,-1] = -1

		self.n_crabs = n_crabs

	def rebirth(self):
		#When the world begins anew, it must redistribute its crabs
		#Generate crabs and distribute them
		non_wall_squares = np.zeros(((self.width-2)*(self.height-2)))

		non_wall_squares[:self.n_crabs] = 1
		#shuffle the crabs!
		np.random.shuffle(non_wall_squares)
		non_wall_squares = np.reshape(non_wall_squares, (self.width-2, self.height-2))

		crab_quantity_grid = np.random.randint(1, self.crab_max+1, (self.width-2, self.height-2))

		non_wall_squares = np.multiply(non_wall_squares, crab_quantity_grid)

		self.grid[1:-1,1:-1] = non_wall_squares

		return self

	def eat_crab(self, x, y):
		crabs = self.grid[x,y]
		self.grid[x,y] = 0
		return crabs

	def get_perceptual_neighbourhood(self, x, y):
		perceptual_neighbourhood = np.array([self.grid[x,y], self.grid[x+1,y], self.grid[x-1,y], self.grid[x,y+1], self.grid[x,y-1]])

		return perceptual_neighbourhood

	def is_legal_move(self, x, y, move):
		move_pos = np.array([x,y]) + move
		if self.grid[move_pos[0], move_pos[1]] != -1:
			return True
		else:
			return False

class Octopus():

	def __init__(self, world, permissible_steps=200, n_lives=100, max_crabs=10):

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
			self.n_moves+1: self.move_randomly,
			self.n_moves+2: self.eat_crab,
		}

		self.n_actions = len(self.actions_dict.keys())

		moves_rewards_dict = {i: {True: 0, False: -5} for i,m in enumerate(self.directions.keys())}

		self.rewards_dict = {
			**moves_rewards_dict,
			self.n_moves: {True: 0, False: -5},
			self.n_moves+1: {True: 0, False: -5},
			self.n_moves+2: {
				0:-1,
				1:1,
				2:3,
				3:6,
				4:9,
				5:10,
				6:9,
				7:6,
				8:3,
				9:1,
				10:0,
			},
		}

		self.max_crabs = max_crabs
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

	def illustrated_life(self):
		self.remaining_steps = self.permissible_steps
		self.reward = 0
		self.world.rebirth()

		def mapping(square):
			if square == 0:
				return "🌊 "
			if square > 0:
				return "🦀 "
			if square == -1:
				return "🧱 🧱"

		perception_dict = self.build_perception_dict()
		def perceptual_mapping(square):
			if square == -1:
				return "🧱 🧱"

			p = perception_dict[square]
			if p == 0:
				return "💚 "
			if p == 1:
				return "❤️  "

		while self.remaining_steps > 0:
			print("Total Score: {}\n".format(self.reward))
			for i in range(self.world.height):
				line_str = ""
				for j in range(self.world.width):
					if i == self.x and j == self.y:
						line_str += "🐙 "
					else:
						line_str += mapping(self.world.grid[i,j])
				line_str += "\t"
				for j in range(self.world.width):
					if i == self.x and j == self.y:
						line_str += "🐙 "
					else:
						line_str += perceptual_mapping(self.world.grid[i,j])

				# line_str += "\n"
				print(line_str)
			self.next_action()
			sleep(0.5)
			os.system('clear')
		return self

	def build_perception_dict(self):
		perception_dict = {i:self.decision_kernel[-i-1] for i in range(0,self.max_crabs+1)}
		perception_dict[-1] = -1

		return perception_dict

	def perceive(self):
		perception_dict = self.build_perception_dict()

		perceived_neighbourhood = [perception_dict[d] for d in self.perceptual_neighbourhood]
		return perceived_neighbourhood

	def next_action(self):

		perceived_neighbourhood = self.perceive()
		decision_int = int(sum([(d+1)*self.world.n_types**i for i,d in enumerate(perceived_neighbourhood)]))


		action = self.decision_kernel[decision_int]

		self.take_action(action)

		return self

	def take_action(self, action):
		result = self.actions_dict[action]()
		reward = self.rewards_dict[action][result]

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
		crabs_eaten = self.world.eat_crab(self.x, self.y)
		self.perceptual_neighbourhood = self.world.get_perceptual_neighbourhood(self.x, self.y)

		return crabs_eaten

	def random_gene(self):
		random_gene_action = np.random.randint(0, self.n_actions, self.perceptual_size)
		random_gene_perceive = np.random.randint(0,2, self.max_crabs+1)

		random_gene = np.concatenate([random_gene_action, random_gene_perceive])

		return random_gene

class MolluskEvolver():

	def __init__(self, n_octopodes, n_generations):

		self.n_octopodes=n_octopodes
		self.n_generations=n_generations
		
		self.permissible_steps = 50
		self.octopus_lives = 10

		self.world = World(12, 12, 50)


		self.octopode_ensemble = [self.create_octopus() for _ in range(self.n_octopodes)]
		self.octopus_gene_length = len(self.octopode_ensemble[0].decision_kernel)
		self.octopus_action_space = self.octopode_ensemble[0].n_actions
		self.mutation_chance = 0.005

	def create_octopus(self):
		octopus = Octopus(self.world, permissible_steps=self.permissible_steps, n_lives=self.octopus_lives)

		return octopus

	def evolve_octopodes(self):
		fitnesses = []
		while self.n_generations > 0:
			terminal = self.n_generations == 1
			fitness = self.next_generation(terminal=terminal)
			if self.n_generations%10 == 0:
				print(self.n_generations)
				print(np.mean(fitness))
				print(np.max(fitness))
				print(np.std(fitness))
				print("\n")
			fitnesses.append(fitness)
			self.n_generations -= 1

		return fitnesses


	def next_generation(self, terminal=False):
		#Octopodes live, die, and are reborn within the confines of a single list comprehension
		raw_fitness = np.array([o.samsara() for o in self.octopode_ensemble])

		norm_fitness = raw_fitness - min(raw_fitness)

		pop_mean = np.mean(norm_fitness)
		pop_std = np.std(norm_fitness)
		def sigma_scaling(f):
			scaled =  1 + (f - pop_mean)/(2*pop_std)
			if scaled < 0:
				scaled = 0.1
			return scaled
		norm_fitness = np.array([sigma_scaling(x) for x in norm_fitness])

		norm_fitness = norm_fitness/sum(norm_fitness)

		if not terminal:
			self.octopode_ensemble = self.panmixia(norm_fitness)
		else:
			print("terminal")
		return raw_fitness/self.octopus_lives

	def panmixia(self, octopus_fitness):
		octo_parents = np.random.choice(self.octopode_ensemble, size=self.n_octopodes, replace=True, p=octopus_fitness)
		next_generation_octopodes = [self.breed_octopodes(octo_parents[i], octo_parents[i+1]) for i in range(0, self.n_octopodes, 2)]
		next_generation_octopodes = [c for x in next_generation_octopodes for c in x]
		
		return next_generation_octopodes

	def breed_octopodes(self, octopus_sub, octopus_domme):
		split_A = np.random.randint(0,self.octopus_gene_length)
		split_B = np.random.randint(0,self.octopus_gene_length)

		def merge_genes(child_oct, oct_A, oct_B):
			child_oct.decision_kernel = oct_A.decision_kernel
			child_oct.decision_kernel[min(split_A, split_B):max(split_A, split_B)] = oct_B.decision_kernel[min(split_A, split_B):max(split_A, split_B)]

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