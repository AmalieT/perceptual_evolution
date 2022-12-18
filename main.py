from octopus import MolluskEvolver
import numpy as np

# new_world = World(12,12,50)
# print(new_world.get_perceptual_neighbourhood(1,1))
# print(new_world.eat_crab(1,1))
# print(new_world.eat_crab(1,2))
# print(new_world.eat_crab(2,1))
# print(new_world.get_perceptual_neighbourhood(1,1))
# print(new_world.grid)
# octopus = Octopus(new_world)
# print(octopus.actions_dict)
# print(octopus.samsara())
# print(new_world.grid)


me = MolluskEvolver(n_octopodes=200, n_generations=1000)
# octopus = me.octopode_ensemble[0]
# octopus.illustrated_life()

fitnesses = me.evolve_octopodes()
avg_fitnesses = [np.mean(f) for f in fitnesses]
best = np.argmax(fitnesses[-1])
print(avg_fitnesses)

octomax = me.octopode_ensemble[best]
octomax.illustrated_life()
print(avg_fitnesses)