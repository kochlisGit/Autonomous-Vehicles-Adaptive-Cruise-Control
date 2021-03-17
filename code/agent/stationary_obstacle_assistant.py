import carla
from agents.tools.misc import is_within_distance_ahead


class StationaryObstacleAssistant:
    def __init__(self, world_map, vehicle, npc_vehicle_list, proximity_vehicle_threshold=13):
        self._vehicle = vehicle
        self._world_map = world_map
        self._npc_vehicle_list = npc_vehicle_list
        self._proximity_vehicle_threshold = proximity_vehicle_threshold

    # Detects stationary vehicles ahead of the agent's vehicle.
    def detect_stationary_vehicle_ahead(self):
        # Retrieving location of the agent.
        vehicle_location = self._vehicle.get_location()
        vehicle_waypoint = self._world_map.get_waypoint(vehicle_location)

        for npc_vehicle in self._npc_vehicle_list:

            # Checking if the npc vehicle is stationary.
            npc_vehicle_control = npc_vehicle.get_control()
            if npc_vehicle_control.throttle == 0.0:

                # Retrieving the location of the candidate front vehicle.
                npc_vehicle_location = npc_vehicle.get_location()
                npc_vehicle_waypoint = self._world_map.get_waypoint(npc_vehicle_location)

                # Checking if the npc vehicle is located on the lane with the agent's vehicle.
                if npc_vehicle_waypoint.road_id == vehicle_waypoint.road_id and \
                        npc_vehicle_waypoint.lane_id == vehicle_waypoint.lane_id:

                    # Checking if the npc vehicle is close to the agent's vehicle.
                    if is_within_distance_ahead(npc_vehicle.get_transform(),
                                                self._vehicle.get_transform(),
                                                self._proximity_vehicle_threshold):
                        return True
        return False
