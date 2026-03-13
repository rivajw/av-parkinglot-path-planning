import carla

client = carla.Client("localhost",2000)
client.set_timeout(10)

world = client.load_world("Town05")
print(world.get_map().name)

world = client.get_world()

spectator = world.get_spectator()

transform = spectator.get_transform()
print(transform)
