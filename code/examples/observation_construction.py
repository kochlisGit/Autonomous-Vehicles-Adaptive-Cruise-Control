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
from simulation.traffic_manager import TrafficManager
from simulation import image_data_utils
from queue import Queue
import numpy as np
import cv2
import time

image_height = 512
image_width = 512
fov = 10


def sensor_callback(sensor_queue, sensor_name, sensor_data):
    if sensor_name == 'segmentation':
        sensor_data.convert(carla.ColorConverter.CityScapesPalette)
        data = np.reshape( sensor_data.raw_data, (image_height, image_width, 4) )
    elif sensor_name == 'depth':
        sensor_data.convert(carla.ColorConverter.LogarithmicDepth)
        data = np.reshape( sensor_data.raw_data, (image_height, image_width, 4) )
    else:
        data = None

    sensor_queue.put( (sensor_name, data), True, 1.0 )


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

settings = world.get_settings()
settings.fixed_delta_seconds = None
settings.synchronous_mode = True
world.apply_settings(settings)

queue = Queue()

vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

seg_cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
seg_cam_bp.set_attribute('image_size_x', f'{image_width}')
seg_cam_bp.set_attribute('image_size_y', f'{image_height}')
seg_cam_bp.set_attribute('fov', f'{fov}')

depth_cam_bp = blueprint_library.find('sensor.camera.depth')
depth_cam_bp.set_attribute('image_size_x', f'{image_width}')
depth_cam_bp.set_attribute('image_size_y', f'{image_height}')
depth_cam_bp.set_attribute('fov', f'{fov}')

spawn_points = world.get_map().get_spawn_points()
start = spawn_points[156]
start.location.y += 10
dest = spawn_points[130]

vehicle = world.spawn_actor(vehicle_bp, start)
camera_spawn_point = carla.Transform(
    carla.Location(x=vehicle.bounding_box.extent.x, z=vehicle.bounding_box.extent.z/2)
)
seg_cam = world.spawn_actor(seg_cam_bp, camera_spawn_point, attach_to=vehicle)
seg_cam.listen( lambda sensor_data: sensor_callback(queue, 'segmentation', sensor_data) )

depth_cam = world.spawn_actor(depth_cam_bp, camera_spawn_point, attach_to=vehicle)
depth_cam.listen( lambda sensor_data: sensor_callback(queue, 'depth', sensor_data) )

traffic_manager = TrafficManager(client, world, True)
traffic_manager.spawn_vehicles()
traffic_manager.disable_lane_changing()
traffic_manager.randomize_vehicle_speed_limits()

world.tick()

navigator = NavigationAssistant(vehicle)
navigator.set_speed(40)
navigator.set_destination(dest.location)

start_time = time.time()
waiting_time = 10
steps = 0

while time.time() - start_time < waiting_time:
    world.tick()
    steps += 1
    control = navigator.run_step(False)
    vehicle.apply_control(control)

print('Steps so far =', steps)

print('Starting collection...')

start = time.time()
for i in range(3):
    camera_data = {'segmentation': None, 'depth': None}
    while camera_data['segmentation'] is None or camera_data['depth'] is None:
        world.tick()
        control = navigator.run_step(False)
        vehicle.apply_control(control)
        while not queue.empty():
            name, data = queue.get(True, 1.0)
            camera_data[name] = data

    seg_data = camera_data['segmentation']
    dep_data = camera_data['depth']

    vehicle_indices = image_data_utils.get_vehicle_indices(seg_data)
    observation = image_data_utils.get_normalized_depth_of_vehicles(dep_data, vehicle_indices)

    cv2.imshow('segmentation' + str(i), seg_data)
    cv2.imshow('depth' + str(i), dep_data)
    cv2.imshow('observation' + str(i), observation)
    print(image_data_utils.get_front_vehicle_distance(observation))

end = time.time()
print('Observation needed', end-start , 'seconds to construct')

cv2.waitKey()
cv2.destroyAllWindows()

start = time.time()
for i in range(5):
    world.tick()
    control = navigator.run_step(False)
    vehicle.apply_control(control)
end = time.time()

print('Running 10 steps for average', end-start, 'seconds.')

traffic_manager.destroy_vehicles()
navigator = None

settings = world.get_settings()
settings.fixed_delta_seconds = None
settings.synchronous_mode = False
world.apply_settings(settings)
