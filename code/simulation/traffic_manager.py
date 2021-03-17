import carla
import random

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor
DestroyActor = carla.command.DestroyActor


class TrafficManager:
    def __init__(self, client, world, sync=True):
        # Storing simulation variables.
        self._client = client
        self._world = world
        self._sync = sync

        # Defining traffic manager.
        self._traffic_manager = None

        # Defining traffic placeholders.
        self._vehicle_blueprint_list = []
        self._vehicle_spawn_points = []
        self._vehicle_list = []

        # Defining vehicle spawn positions.
        self._vehicle_spawn_points_indices = [
            3, 4, 5, 6, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 29, 32, 33, 34, 35, 43,
            45, 69, 83, 87, 88, 89, 107, 108, 115, 116, 117, 119, 125, 147, 148, 154, 155, 161, 168,
            169, 170, 172, 179, 180, 185, 189, 190, 191, 194, 195, 196, 197, 199, 201, 202, 203, 205, 207, 216,
            222, 223, 224, 225, 226, 227, 246, 259, 250, 253
        ]
        self._num_of_vehicles = len(self._vehicle_spawn_points_indices)
        '''
        self._vehicle_spawn_points_indices = [
            2, 3, 4, 5, 6, 19, 20, 21, 22, 23, 24, 45, 49, 50, 51, 52, 70, 71, 72, 73,
            74, 75, 83, 84, 87, 88, 89, 94, 95, 96, 97, 155, 116, 117, 118, 128, 129, 130, 131, 132,
            133, 134, 135, 137, 138, 139, 140, 141, 142, 143, 144, 147, 148, 185, 186, 189, 190, 199, 205, 216,
            222, 223, 224, 225, 226, 227
        ]
        self._num_of_vehicles = len(self._vehicle_spawn_points_indices)
        '''

        # Defining available speed limits of vehicles.
        self._vehicle_speed_limits = [10, -15, -35]

        self._build_traffic_manager()
        self._build_random_blueprints()
        self._build_spawn_points()

    # Returns the list of all spawned vehicle actors.
    def get_vehicle_list(self):
        return self._vehicle_list

    # Sets the traffic manager of the simulator.
    def _build_traffic_manager(self):
        self._traffic_manager = self._client.get_trafficmanager(port=8000)
        self._traffic_manager.set_synchronous_mode(True)
        self._traffic_manager.set_global_distance_to_leading_vehicle(5.0)

    # Builds random blueprints for the NPC vehicles (with 4 wheels only).
    def _build_random_blueprints(self):
        blueprint_library = self._world.get_blueprint_library()
        all_vehicle_blueprints = blueprint_library.filter('vehicle.*.*')
        self._vehicle_blueprint_list = [
            blueprint for blueprint in all_vehicle_blueprints if int( blueprint.get_attribute('number_of_wheels') ) == 4
        ]

        # Randomizing color of vehicles.
        for blueprint in self._vehicle_blueprint_list:
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)

    # Retrieves pre-defined spawn points from the map.
    # It also adds additional spawn points that will cause traffic.
    def _build_spawn_points(self):
        spawn_points = self._world.get_map().get_spawn_points()
        self._vehicle_spawn_points = [spawn_points[ind] for ind in self._vehicle_spawn_points_indices]

        sp = spawn_points[169]
        x = sp.location.x
        y = sp.location.y
        z = sp.location.z
        rotation = sp.rotation

        for i in range(6):
            x += 6
            self._vehicle_spawn_points.append( carla.Transform(carla.Location(x, y, z), rotation) )
            self._num_of_vehicles += 1

    # Spawns random vehicles in random pre-defined positions.
    def spawn_vehicles(self):
        # Choosing random blueprints.
        random_vehicle_blueprints = random.choices(population=self._vehicle_blueprint_list, k=self._num_of_vehicles)
        batch = []

        # Spawns vehicles.
        for blueprint, spawn_point in zip(random_vehicle_blueprints, self._vehicle_spawn_points):
            batch.append(SpawnActor(blueprint, spawn_point)
                         .then( SetAutopilot( FutureActor, True, self._traffic_manager.get_port() ) ) )

        command_responses = self._client.apply_batch_sync(batch, self._sync)

        vehicle_ids = [response.actor_id for response in command_responses]

        self._vehicle_list = self._world.get_actors(vehicle_ids)

        # Waiting world to retrieve every vehicle in the simulation.
        if self._sync:
            self._world.tick()
        else:
            self._world.wait_for_tick()

    def destroy_vehicles(self):
        if self._vehicle_list:
            self._client.apply_batch( [carla.command.DestroyActor(vehicle) for vehicle in self._vehicle_list] )

    def disable_lane_changing(self):
        for vehicle in self._vehicle_list:
            self._traffic_manager.auto_lane_change(vehicle, False)
        self._world.tick()

    def randomize_vehicle_speed_limits(self):
        for vehicle in self._vehicle_list:
            speed_limit = random.choice(self._vehicle_speed_limits)
            self._traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_limit)
        self._world.tick()