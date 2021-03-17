import carla
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


class NavigationAssistant:
    def __init__(self, vehicle):
        self._vehicle = vehicle
        self._map = vehicle.get_world().get_map()

        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.4,
            'K_I': 0,
            'dt': 1.0/20.0}
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={
                'target_speed' : 0,
                'lateral_control_dict' :args_lateral_dict
            }
        )
        self._hop_resolution = 2.0
        self._grp = self._initialize_global_route_planner()

    # Initializes the global route planner.
    def _initialize_global_route_planner(self):
        dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        return grp

    # Sets vehicle's speed.
    def set_speed(self, target_speed):
        self._local_planner.set_speed(target_speed)

    # Sets vehicle's destination & computes the optimal route towards the destination.
    def set_destination(self, location):
        start_waypoint = self._map.get_waypoint( self._vehicle.get_location() )
        end_waypoint = self._map.get_waypoint(location)

        # Computes the optimal route for the starting location.
        route_trace = self._grp.trace_route(start_waypoint.transform.location, end_waypoint.transform.location)
        self._local_planner.set_global_plan(route_trace)

    # Executes one step of navigation.
    def run_step(self, hazard_detected, debug=False):
        if hazard_detected:
            control = self._emergency_stop()
        else:
            control = self._local_planner.run_step(debug=debug)

        return control

    # Checks whether the vehicle reached its destination.
    def done(self):
        return self._local_planner.done()

    # Returns a control that forces the vehicle to stop.
    @staticmethod
    def _emergency_stop():
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 1.0
        control.hand_brake = False

        return control
