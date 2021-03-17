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
import time

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprint_library = world.get_blueprint_library()

vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_points = world.get_map().get_spawn_points()
num_of_spawn_points = len(spawn_points)

print('Num of spawn points =', num_of_spawn_points)

for i in range(115, 130):
    print(i)
    vehicle = world.spawn_actor( vehicle_bp, spawn_points[i] )
    time.sleep(2)
    vehicle.destroy()
