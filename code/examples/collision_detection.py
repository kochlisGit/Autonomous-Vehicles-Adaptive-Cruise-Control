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

end = False

def print_collision(data):
    global end
    print(data.other_actor.type_id)
    print(data.other_actor.semantic_tags)
    print(10 in data.other_actor.semantic_tags)
    end = True

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

collision_detector_bp = blueprint_library.find('sensor.other.collision')
collision_detector = world.spawn_actor(collision_detector_bp, carla.Transform(), attach_to=vehicle)
collision_detector.listen(
    lambda sensor_data: print_collision(sensor_data)
)

start_point.location.y += 60
start_point.location.x += 0.5
vehicle2 = world.spawn_actor(vehicle_bp, start_point)
npc_vehicle_list = [vehicle2]

world.tick()
agent = NavigationAssistant(vehicle)
agent.set_speed(40)
agent.set_destination(dest_point.location)

while not end:
    world.tick()
    control = agent.run_step(False)
    vehicle.apply_control(control)

vehicle.destroy()
for npc_vehicle in npc_vehicle_list:
    npc_vehicle.destroy()

settings = world.get_settings()
settings.fixed_delta_seconds = None
settings.synchronous_mode = False
world.apply_settings(settings)
