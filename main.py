from octopus import World, Octopus

new_world = World(12,12,50)
# print(new_world.get_perceptual_neighbourhood(1,1))
# print(new_world.eat_crab(1,1))
# print(new_world.eat_crab(1,2))
# print(new_world.eat_crab(2,1))
# print(new_world.get_perceptual_neighbourhood(1,1))
# print(new_world.grid)
octopus = Octopus(new_world)
# print(octopus.actions_dict)

print(octopus.samsara())
print(new_world.grid)
