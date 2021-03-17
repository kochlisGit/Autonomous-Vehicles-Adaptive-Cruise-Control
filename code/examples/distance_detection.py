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
from simulation import image_data_utils
from queue import Queue
import time
import numpy as np
import cv2


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


image_height = 512
image_width = 512
fov = 10

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

spawn_point = spawn_points[0]
dest_point = spawn_points[100]

spawn_point.location.y -= 30
spawn_point.location.x += 2
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

camera_spawn_point = carla.Transform(
    carla.Location(x=vehicle.bounding_box.extent.x, z=vehicle.bounding_box.extent.z/2)
)
seg_cam = world.spawn_actor(seg_cam_bp, camera_spawn_point, attach_to=vehicle)
seg_cam.listen( lambda sensor_data: sensor_callback(queue, 'segmentation', sensor_data) )

depth_cam = world.spawn_actor(depth_cam_bp, camera_spawn_point, attach_to=vehicle)
depth_cam.listen( lambda sensor_data: sensor_callback(queue, 'depth', sensor_data) )

spawn_point.location.y += 12
spawn_point.location.x -= 0.3
vehicle2 = world.spawn_actor(vehicle_bp, spawn_point)

obstacle_detector = StationaryObstacleAssistant( world.get_map(), vehicle, [vehicle2] )

world.tick()
queue = Queue()
world.tick()

camera_data = {'segmentation': None, 'depth': None}
for i in range(2):
    name, data = queue.get(True, 1.0)
    camera_data[name] = data

seg_data = camera_data['segmentation']
dep_data = camera_data['depth']

vehicle_indices = image_data_utils.get_vehicle_indices(seg_data)
observation = image_data_utils.get_normalized_depth_of_vehicles(dep_data, vehicle_indices)

cv2.imshow('segmentation', seg_data)
cv2.imshow('depth', dep_data)
cv2.imshow('observation', observation)

print(image_data_utils.get_front_vehicle_distance(observation))
print( obstacle_detector.detect_stationary_vehicle_ahead() )

cv2.waitKey()
cv2.destroyAllWindows()

world.tick()
agent = NavigationAssistant(vehicle)
agent.set_speed(40)
agent.set_destination(dest_point.location)

waiting_time = 10
start = time.time()

hazard_detected = False
while time.time() - start < waiting_time:
    world.tick()
    hazard_detected = obstacle_detector.detect_stationary_vehicle_ahead()
    control = agent.run_step(hazard_detected)
    vehicle.apply_control(control)

vehicle2.destroy()
vehicle.destroy()
depth_cam.destroy()
seg_cam.destroy()

settings = world.get_settings()
settings.fixed_delta_seconds = None
settings.synchronous_mode = False
world.apply_settings(settings)