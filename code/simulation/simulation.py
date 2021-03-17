import carla
from agent.navigation_assistant import NavigationAssistant
from agent.stationary_obstacle_assistant import StationaryObstacleAssistant
from simulation.traffic_manager import TrafficManager
from simulation import image_data_utils
from enum import Enum
from queue import Queue
import numpy as np
import cv2
import os

VEHICLE_TAG = 10


class SensorType(Enum):
    SEGMENTATION_CAMERA = 1
    DEPTH_CAMERA = 2
    COLLISION_DETECTOR = 3


class AgentState(Enum):
    DANGER = 0
    STABILITY = 1
    ACCELERATION_LOW = 2
    ACCELERATION_MEDIUM = 3
    ACCELERATION_HIGH = 4

class Simulator:
    def __init__(self, log_dir=None):
        # Defining simulator variables.
        self._client = None
        self._world = None
        self._traffic_manager = None
        self._log_dir = log_dir

        # Defining environment blueprints.
        self._vehicle_bp = None
        self._segmentation_camera_bp = None
        self._depth_camera_bp = None
        self._collision_detector_bp = None

        # Defining agent actors.
        self._vehicle = None
        self._navigation_assistant = None
        self._stationary_obstacle_assistant = None
        self._sensor_list = []

        # Defining sensor attributes.
        self.image_height = 64
        self.image_width = 64
        self.num_of_frames = 3
        self._fov = 10

        # Defining sensor data placeholders.
        self._sensor_data_queue = Queue()
        self._front_vehicle_distance = 0.0
        self._collision_detected = None
        self._target_speed = 0
        self._step = 0

        # Defining training properties.
        self.vehicle_min_speed = 0
        self.vehicle_max_speed = 65
        self._vehicle_acceleration_increment = 1.5
        self._max_episode_steps = 1000
        self._waiting_steps = 4
        self._collision_penalty_reward = -200
        self._speed_reward_factor = 0.5
        self._safety_reward_factor = 0.5
        self._safety_distance_threshold = 0.13

    def get_speed(self):
        return self._target_speed

    def get_step(self):
        return self._step

    # Connects the client with the simulator's retrieves the world.
    def _connect(self):
        # Connecting the client to simulator's server.
        self._client = carla.Client(host='localhost', port=2000)
        self._client.set_timeout(seconds=20.0)

        # Retrieving the environment.
        self._world = self._client.get_world()

        # Initializing the traffic manager.
        self._traffic_manager = TrafficManager(client=self._client, world=self._world)

    # Adjusts the synchronization and the delta time (execution step time) of the simulator.
    def _sync(self, sync, delta_time):
        settings = self._world.get_settings()
        settings.fixed_delta_seconds = delta_time
        settings.synchronous_mode = sync
        self._world.apply_settings(settings)

    # Retrieves the blueprints of the vehicle and the sensors.
    def _retrieve_agent_blueprints(self):
        # Retrieving the blueprint library.
        blueprint_library = self._world.get_blueprint_library()

        # Building the blueprint of the vehicle.
        self._vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        self._vehicle_bp.set_attribute('color', '255,0,0')

        # Building the blueprint of the segmentation camera.
        self._segmentation_camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        self._segmentation_camera_bp.set_attribute('image_size_x', f'{self.image_width}')
        self._segmentation_camera_bp.set_attribute('image_size_y', f'{self.image_height}')
        self._segmentation_camera_bp.set_attribute('fov', f'{self._fov}')

        # Building the blueprint of the depth camera.
        self._depth_camera_bp = blueprint_library.find('sensor.camera.depth')
        self._depth_camera_bp.set_attribute('image_size_x', f'{self.image_width}')
        self._depth_camera_bp.set_attribute('image_size_y', f'{self.image_height}')
        self._depth_camera_bp.set_attribute('fov', f'{self._fov}')

        # Building the blueprint of the collision detector.
        self._collision_detector_bp = blueprint_library.find('sensor.other.collision')

    # Initializes the environment and starts the simulation.
    def start(self):
        self._connect()
        self._sync(sync=True, delta_time=None)
        self._retrieve_agent_blueprints()

        if self._log_dir is not None:
            os.mkdir(self._log_dir)
            print('Logging Directory has been created.')

    # Stores pre-processes sensor data and stores them in a Thread-Safe Queue.
    def _sensor_callback(self, sensor_name, sensor_data):
        if sensor_name == SensorType.SEGMENTATION_CAMERA:
            sensor_data.convert(carla.ColorConverter.CityScapesPalette)
            data = np.reshape(sensor_data.raw_data, newshape=(self.image_height, self.image_width, 4))
        elif sensor_name == SensorType.DEPTH_CAMERA:
            sensor_data.convert(carla.ColorConverter.LogarithmicDepth)
            data = np.reshape(sensor_data.raw_data, newshape=(self.image_height, self.image_width, 4))
        elif sensor_name == SensorType.COLLISION_DETECTOR:
            data = sensor_data.other_actor.semantic_tags
        else:
            data = None
        self._sensor_data_queue.put( (sensor_name, data), True, 1.0 )

    # Spawns the agent's vehicle, sensors, navigation controller.
    def _spawn_agent(self):
        # Retrieving world's spawn points.
        world_map = self._world.get_map()
        spawn_point_list = world_map.get_spawn_points()

        vehicle_spawn_point = spawn_point_list[156]
        vehicle_spawn_point.location.y += 10
        vehicle_dest_point = spawn_point_list[130]
        '''
        vehicle_spawn_point = spawn_point_list[127]
        vehicle_spawn_point.location.y -= 10
        vehicle_dest_point = spawn_point_list[167]
        vehicle_dest_point.location.x += 20
        '''

        # Spawning the vehicle.
        self._vehicle = self._world.try_spawn_actor(self._vehicle_bp, vehicle_spawn_point)

        # Spawning the sensors.
        camera_spawn_point = carla.Transform(
            carla.Location(x=self._vehicle.bounding_box.extent.x, z=self._vehicle.bounding_box.extent.z / 2)
        )

        segmentation_camera = self._world.spawn_actor(
            self._segmentation_camera_bp,
            camera_spawn_point,
            attach_to=self._vehicle)
        segmentation_camera.listen(lambda sensor_data:
                                   self._sensor_callback(SensorType.SEGMENTATION_CAMERA, sensor_data))
        self._sensor_list.append(segmentation_camera)

        depth_camera = self._world.spawn_actor(
            self._depth_camera_bp,
            camera_spawn_point,
            attach_to=self._vehicle
        )
        depth_camera.listen(
            lambda sensor_data:
            self._sensor_callback(SensorType.DEPTH_CAMERA, sensor_data)
        )
        self._sensor_list.append(depth_camera)

        collision_detector = self._world.spawn_actor(
            self._collision_detector_bp,
            carla.Transform(),
            attach_to=self._vehicle
        )
        collision_detector.listen(
            lambda sensor_data: self._sensor_callback(SensorType.COLLISION_DETECTOR, sensor_data)
        )
        self._sensor_list.append(collision_detector)

        # Spawning the navigation agent.
        self._world.tick()
        self._navigation_assistant = NavigationAssistant(self._vehicle)
        self._navigation_assistant.set_destination(vehicle_dest_point.location)

        # Enabling the stationary obstacle assistant.
        npc_vehicle_list = self._traffic_manager.get_vehicle_list()
        self._stationary_obstacle_assistant = StationaryObstacleAssistant(world_map, self._vehicle, npc_vehicle_list)

    # Generates an observation of the environment from sensor data.
    def _get_observation_frame(self, hazard_detected):
        sensor_data_dict = {
            SensorType.SEGMENTATION_CAMERA: None,
            SensorType.DEPTH_CAMERA: None,
            SensorType.COLLISION_DETECTOR: None
        }

        # Waits until valid data from all sensors are gathered.
        while sensor_data_dict[SensorType.SEGMENTATION_CAMERA] is None or \
                sensor_data_dict[SensorType.DEPTH_CAMERA] is None:

            self._world.tick()
            control = self._navigation_assistant.run_step(hazard_detected)
            self._vehicle.apply_control(control)

            while not self._sensor_data_queue.empty():
                sensor_type, sensor_data = self._sensor_data_queue.get(True, 1.0)
                sensor_data_dict[sensor_type] = sensor_data

        # Checks for collisions.
        collision = sensor_data_dict[SensorType.COLLISION_DETECTOR]

        # Generates the observation.
        segmentation_frame = sensor_data_dict[SensorType.SEGMENTATION_CAMERA]
        depth_frame = sensor_data_dict[SensorType.DEPTH_CAMERA]

        obstacle_indices = image_data_utils.get_vehicle_indices(segmentation_frame)
        observation = image_data_utils.get_normalized_depth_of_vehicles(depth_frame, obstacle_indices)

        if self._log_dir:
            cv2.imwrite(self._log_dir + 'segmentation_' + str(self._step) + '.png', segmentation_frame)
            cv2.imwrite(self._log_dir + 'depth_' + str(self._step) + '.png', depth_frame)

        return observation, collision

    # Builds and returns a stack of observation frames.
    def _get_observation(self, hazard_detected):
        observation = []
        for i in range(self.num_of_frames):
            obs_frame, collision = self._get_observation_frame(hazard_detected)
            observation.append(obs_frame)

            if collision:
                self._collision_detected = collision
                for j in range(i + 1, self.num_of_frames):
                    observation.append(obs_frame)
                break

        # Computing distance from the front vehicle.
        self._front_vehicle_distance = image_data_utils.get_front_vehicle_distance(observation[self.num_of_frames - 1])

        if self._log_dir:
            for i in range(self.num_of_frames):
                observation_image = observation[i] * 255
                cv2.imwrite(self._log_dir + 'observation_' + str(self._step) + '_' + str(i) + '.png', observation_image)

        return np.float32(observation)

    # Resets the environment.
    def reset(self):
        # Destroying all previous actors.
        self._destroy_agent()
        self._traffic_manager.destroy_vehicles()

        # Re-spawning the NCPs.
        self._traffic_manager.spawn_vehicles()
        self._traffic_manager.disable_lane_changing()
        self._traffic_manager.randomize_vehicle_speed_limits()

        # Re-spawning the agent.
        self._spawn_agent()

        # Resetting placeholders.
        self._sensor_data_queue = Queue()
        self._front_vehicle_distance = 0.0
        self._collision_detected = None
        self._target_speed = 0
        self._step = 0

        # Constructing an observation.
        observation = self._get_observation(hazard_detected=False)

        return observation

    # Executes a step in the environment.
    def step(self, state):
        # Checking whether there are stationary vehicles ahead.
        stationary_obstacle_detected = self._stationary_obstacle_assistant.detect_stationary_vehicle_ahead()
        while stationary_obstacle_detected:
            self._world.tick()
            self._target_speed = 0
            control = self._navigation_assistant.run_step(hazard_detected=True)
            self._vehicle.apply_control(control)
            stationary_obstacle_detected = self._stationary_obstacle_assistant.detect_stationary_vehicle_ahead()

        hazard_detected = state == AgentState.DANGER
        if state == AgentState.DANGER:
            self._target_speed = 0
        elif state == AgentState.STABILITY:
            self._target_speed += 0
        else:
            if state == AgentState.ACCELERATION_LOW:
                self._target_speed += self._vehicle_acceleration_increment
            elif state == AgentState.ACCELERATION_MEDIUM:
                self._target_speed += (2*self._vehicle_acceleration_increment)
            elif state == AgentState.ACCELERATION_HIGH:
                self._target_speed += (3*self._vehicle_acceleration_increment)

            if self._target_speed >= self.vehicle_max_speed:
                self._target_speed = self.vehicle_max_speed

        # Executing a step. Waits some time before taking next action.
        self._navigation_assistant.set_speed(self._target_speed)

        for i in range(self._waiting_steps):
            self._world.tick()
            control = self._navigation_assistant.run_step(hazard_detected)
            self._vehicle.apply_control(control)
            if self._navigation_assistant.done():
                break

        # Generating an observation from the environment.
        observation = self._get_observation(hazard_detected)
        self._step += 1

        # Computing rewards.
        if self._collision_detected:
            # Checking if collision occurred with other vehicle or it's server's fault.
            if VEHICLE_TAG in self._collision_detected:
                print('Vehicle collision detected..!')
                reward = self._collision_penalty_reward
            else:
                print('Computation ERROR detected in simulation..! Resetting environment.')
                reward = 0
            done = True
        elif self._navigation_assistant.done():
            print('Episode ended successfully, no Collision detected..! Steps =', self._step)
            reward = (self._max_episode_steps - self._step) * 0.4
            done = True
        else:
            done = self._step > self._max_episode_steps
            if done:
                print('Episode step limit exceeded..!')

            if self._target_speed == 0:
                reward = - self._front_vehicle_distance
            else:
                speed_reward = self._speed_reward_factor * (self._target_speed / self.vehicle_max_speed)
                safety_reward = self._safety_reward_factor * (1 - self._front_vehicle_distance)

                if self._front_vehicle_distance > self._safety_distance_threshold:
                    reward = speed_reward + safety_reward
                else:
                    reward = -speed_reward - safety_reward

        if self._log_dir is not None:
            print('\nStep:', self._step,
                  '\nSpeed:', self._target_speed,
                  '\nFront Vehicle Distance:', self._front_vehicle_distance,
                  '\nAction:', state,
                  '\nReward:', reward)

        return observation, reward, done

    # Destroys the agent's vehicle, sensors and the navigation agent.
    def _destroy_agent(self):
        for sensor in self._sensor_list:
            sensor.destroy()
        self._sensor_list = []
        self._navigation_assistant = None
        self._stationary_obstacle_assistant = None
        if self._vehicle is not None:
            self._vehicle = None

    # Terminates the connection and restores the original settings.
    def close(self):
        self._traffic_manager.destroy_vehicles()
        self._destroy_agent()
        self._sync(sync=False, delta_time=None)
