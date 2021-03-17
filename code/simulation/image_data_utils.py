import numpy as np

VEHICLE_SEGMENTATION_COLOR = (142, 0, 0, 255)


# Get indices of vehicles.
def get_vehicle_indices(segmentation_image):
    return np.where( np.all(segmentation_image == VEHICLE_SEGMENTATION_COLOR, axis=-1) )


# Get normalized depth values of vehicle pixels.
def get_normalized_depth_of_vehicles(depth_image, vehicle_indices):
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    observation = np.full(shape=(height, width, 1), fill_value=1, dtype=np.float32)

    for y, x in zip( vehicle_indices[0], vehicle_indices[1] ):
        observation[y][x][0] = depth_image[y][x][0] / 255

    return observation


# Compute distance of the front vehicle.
def get_front_vehicle_distance(observation):
    width = observation.shape[1]
    xMin = 3 * width // 8
    xMax = 5 * width // 8

    return np.min( observation[:, xMin:xMax] )