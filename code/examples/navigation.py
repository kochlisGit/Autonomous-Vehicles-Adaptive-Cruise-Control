import glob
import os
import sys

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    sys.path.append('../../carla')
except IndexError:
    pass

import carla
from agent.navigation_assistant import NavigationAssistant
import time
import math


def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return math.ceil( 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) )


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprint_library = world.get_blueprint_library()

settings = world.get_settings()
settings.fixed_delta_seconds = None
settings.synchronous_mode = True
world.apply_settings(settings)

spawn_points = world.get_map().get_spawn_points()

start = spawn_points[156]
start.location.y += 10
dest = spawn_points[130]
'''
start = spawn_points[127]
start.location.y -= 10
dest = spawn_points[167]
dest.location.x += 20
'''

vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
vehicle = world.spawn_actor(vehicle_bp, start)

world.tick()
agent = NavigationAssistant(vehicle)
agent.set_destination(dest.location)

steps = 0
target_speed = 0
start = time.time()

while not agent.done():
    target_speed += 3
    if target_speed > 65:
        target_speed = 65
    agent.set_speed(target_speed)

    for i in range(7):
        world.tick()
        control = agent.run_step(False)
        vehicle.apply_control(control)

    if steps % 30 == 0:
        print( get_speed(vehicle) )
    steps += 1


end = time.time()

print('Total steps =', steps)
print('Total time =', end-start, 'seconds')

settings = world.get_settings()
settings.fixed_delta_seconds = None
settings.synchronous_mode = False
world.apply_settings(settings)
