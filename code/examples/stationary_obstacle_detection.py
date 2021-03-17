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
from agent.stationary_obstacle_assistant import StationaryObstacleAssistant
import time

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()
blueprint_library = world.get_blueprint_library()

settings = world.get_settings()
settings.fixed_delta_seconds = None
settings.synchronous_mode = True
world.apply_settings(settings)

spawn_points = world.get_map().get_spawn_points()
start_point = spawn_points[0]
dest_point = spawn_points[100]

start_point.location.y -= 30
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
vehicle = world.spawn_actor(vehicle_bp, start_point)

start_point.location.y += 60
start_point.location.x += 0.5
vehicle2 = world.spawn_actor(vehicle_bp, start_point)
npc_vehicle_list = [vehicle2]

world.tick()
agent = NavigationAssistant(vehicle)
agent.set_speed(40)
agent.set_destination(dest_point.location)

obstacle_detector = StationaryObstacleAssistant(world.get_map(), vehicle, npc_vehicle_list)

start_time = time.time()
simulation_time = 10

while time.time() - start_time < simulation_time:
    world.tick()
    hazard_detected = obstacle_detector.detect_stationary_vehicle_ahead()
    control = agent.run_step(hazard_detected)
    vehicle.apply_control(control)

start_time = time.time()
simulation_time = 10

control2 = carla.VehicleControl()
control2.steer = 0.0
control2.throttle = 1.0
control2.brake = 0.0
control2.hand_brake = False
vehicle2.apply_control(control2)

while time.time() - start_time < simulation_time:
    world.tick()
    hazard_detected = obstacle_detector.detect_stationary_vehicle_ahead()
    control = agent.run_step(hazard_detected)
    vehicle.apply_control(control)

vehicle.destroy()
for npc_vehicle in npc_vehicle_list:
    npc_vehicle.destroy()

settings = world.get_settings()
settings.fixed_delta_seconds = None
settings.synchronous_mode = False
world.apply_settings(settings)
