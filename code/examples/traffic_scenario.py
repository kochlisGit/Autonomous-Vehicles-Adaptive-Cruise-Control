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
from simulation.traffic_manager import TrafficManager
import time

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

settings = world.get_settings()
settings.fixed_delta_seconds = None
settings.synchronous_mode = True
world.apply_settings(settings)

traffic_manager = TrafficManager(client, world, True)
traffic_manager.spawn_vehicles()

start_time = time.time()
waiting_time = 5
while time.time() - start_time < waiting_time:
    world.tick()

traffic_manager.destroy_vehicles()

traffic_manager.spawn_vehicles()
traffic_manager.disable_lane_changing()
traffic_manager.randomize_vehicle_speed_limits()

start_time = time.time()
waiting_time = 30

while time.time() - start_time < waiting_time:
    world.tick()

traffic_manager.destroy_vehicles()

settings = world.get_settings()
settings.fixed_delta_seconds = None
settings.synchronous_mode = False
world.apply_settings(settings)
