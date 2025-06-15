"""Autonomous driving with CNN-based lane following.

This script uses a trained convolutional neural network to predict steering
angles for autonomous lane following in CARLA.
"""
import argparse
import cv2
import numpy as np
import carla

from .config import (
    DEFAULT_CARLA_HOST,
    DEFAULT_CARLA_PORT,
    CARLA_TIMEOUT_SECONDS,
    VEHICLE_BLUEPRINT_FILTER,
    TOWN05_GOOD_ROAD_IDS,
    DEFAULT_CAMERA_CONFIG,
)
from .carla_utils import (
    create_client,
    setup_synchronous_mode,
    spawn_vehicle_on_road,
    destroy_all_vehicles,
    destroy_all_sensors,
)
from .lane_predictor import (
    LanePredictor,
    SpeedController,
    VehicleMonitor,
    OverlayRenderer,
)


MODEL_PATH = "./model/lane_model"


def setup_camera(world: carla.World, vehicle: carla.Vehicle, camera_config) -> tuple:
    """Setup and attach RGB camera to vehicle.

    Args:
        world: CARLA world instance.
        vehicle: Vehicle to attach camera to.
        camera_config: Camera configuration.

    Returns:
        Tuple of (camera actor, camera_data dict, image_width, image_height).
    """
    camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(camera_config.image_size_x))
    camera_bp.set_attribute("image_size_y", str(camera_config.image_size_y))

    camera_init_trans = carla.Transform(
        carla.Location(z=camera_config.pos_z, x=camera_config.pos_x)
    )
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    def camera_callback(image, data_dict):
        data_dict["image"] = np.reshape(
            np.copy(image.raw_data), (image.height, image.width, 4)
        )

    image_width = camera_bp.get_attribute("image_size_x").as_int()
    image_height = camera_bp.get_attribute("image_size_y").as_int()

    camera_data = {"image": np.zeros((image_height, image_width, 4))}
    camera.listen(lambda image: camera_callback(image, camera_data))

    return camera, camera_data


def run_autonomous_loop(
    world: carla.World,
    vehicle: carla.Vehicle,
    camera_data: dict,
    predictor: LanePredictor,
    speed_controller: SpeedController,
    monitor: VehicleMonitor,
    renderer: OverlayRenderer,
) -> None:
    """Main autonomous driving loop.

    Args:
        world: CARLA world instance.
        vehicle: Vehicle to control.
        camera_data: Dictionary containing camera image data.
        predictor: Lane predictor instance.
        speed_controller: Speed controller instance.
        monitor: Vehicle monitor instance.
        renderer: Overlay renderer instance.
    """
    cv2.namedWindow("RGB Camera", cv2.WINDOW_AUTOSIZE)

    # Get initial image
    image = camera_data["image"]
    predicted_angle = predictor.predict_angle(image)
    initial_image = renderer.render_angle(image.copy(), predicted_angle)
    cv2.imshow("RGB Camera", initial_image)

    running = True
    while running:
        # CARLA Tick
        world.tick()

        # Check for quit key
        if cv2.waitKey(1) == ord("q"):
            running = False
            break

        # Get latest camera image
        image = camera_data["image"]

        # Predict steering angle
        predicted_angle = predictor.predict_angle(image)

        # Get current speed
        speed = monitor.get_speed_kph(vehicle)

        # Render overlays
        display_image = renderer.render_angle(image.copy(), predicted_angle)
        display_image = renderer.render_speed(display_image, speed)

        # Calculate and apply control
        throttle = speed_controller.calculate_throttle(speed)
        vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=-predicted_angle)
        )

        # Update display
        cv2.imshow("RGB Camera", display_image)

    # Cleanup
    cv2.destroyAllWindows()


def main(args: argparse.Namespace) -> None:
    """Entry point for autonomous driving script.

    Args:
        args: Command-line arguments.
    """
    # Connect to CARLA and setup world
    client = create_client(args.host, args.port, CARLA_TIMEOUT_SECONDS)

    if args.town:
        client.load_world(args.town)

    world = client.get_world()
    original_settings = setup_synchronous_mode(world, client)

    try:
        # Spawn vehicle on preferred road
        vehicle = spawn_vehicle_on_road(
            world,
            road_ids=TOWN05_GOOD_ROAD_IDS,
            filter_pattern=VEHICLE_BLUEPRINT_FILTER,
            autopilot=False,
        )

        # Setup camera
        camera, camera_data = setup_camera(world, vehicle, DEFAULT_CAMERA_CONFIG)

        # Initialize prediction and control components
        predictor = LanePredictor(args.model)
        speed_controller = SpeedController()
        monitor = VehicleMonitor()
        renderer = OverlayRenderer()

        # Get initial prediction
        image = camera_data["image"]
        predicted_angle = predictor.predict_angle(image)

        # Run autonomous driving loop
        run_autonomous_loop(
            world,
            vehicle,
            camera_data,
            predictor,
            speed_controller,
            monitor,
            renderer,
        )

    finally:
        # Cleanup resources
        cv2.destroyAllWindows()

        if camera:
            camera.stop()

        destroy_all_sensors(world)
        destroy_all_vehicles(world)

        world.apply_settings(original_settings)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    argparser = argparse.ArgumentParser(
        description="Autonomous driving with CNN lane following"
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default=DEFAULT_CARLA_HOST,
        help=f"IP of the host server (default: {DEFAULT_CARLA_HOST})",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=DEFAULT_CARLA_PORT,
        type=int,
        help=f"TCP port to listen to (default: {DEFAULT_CARLA_PORT})",
    )
    argparser.add_argument(
        "--model",
        metavar="PATH",
        default=MODEL_PATH,
        help=f"Path to trained model (default: {MODEL_PATH})",
    )
    argparser.add_argument(
        "--town",
        metavar="NAME",
        default=None,
        help="CARLA town/map to load (default: current map)",
    )

    return argparser.parse_args()


if __name__ == "__main__":
    main(parse_args())
