"""Basic CARLA environment with multi-sensor grid display.

This script sets up a vehicle in CARLA with multiple sensors (4 RGB cameras,
LiDAR, and Semantic LiDAR) and displays their data in a grid layout.
"""
import argparse
import random
import pygame
import carla

from .config import (
    DEFAULT_CARLA_HOST,
    DEFAULT_CARLA_PORT,
    CARLA_TIMEOUT_SECONDS,
    TRAFFIC_MANAGER_PORT,
    VEHICLE_BLUEPRINT_FILTER,
    DEFAULT_SIMULATION_SETTINGS,
    DEFAULT_WEATHER,
    DEFAULT_GRID_SIZE,
)
from .carla_utils import (
    create_client,
    setup_synchronous_mode,
    set_weather,
    spawn_vehicle,
    destroy_actors,
    restore_world_settings,
)
from .sensor_manager import (
    DisplayManager,
    CustomTimer,
    get_default_sensor_configs,
    spawn_sensors_from_configs,
)


def run_simulation(args: argparse.Namespace, client: carla.Client) -> None:
    """Main simulation loop.

    Args:
        args: Command-line arguments.
        client: CARLA client instance.
    """
    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()

    try:
        # Get the world and original settings
        world = client.get_world()
        original_settings = world.get_settings()

        # Setup synchronous mode if requested
        if args.sync:
            setup_synchronous_mode(world, client)

        # Set weather conditions
        set_weather(world)

        # Spawn vehicle with random spawn point
        vehicle = spawn_vehicle(
            world,
            filter_pattern=VEHICLE_BLUEPRINT_FILTER,
            autopilot=False,
        )
        vehicle_list.append(vehicle)

        # Setup display manager for sensor grid
        display_manager = DisplayManager(
            grid_size=DEFAULT_GRID_SIZE,
            window_size=[args.width, args.height],
        )

        # Spawn all configured sensors
        spawn_sensors_from_configs(world, display_manager, vehicle, get_default_sensor_configs())

        # Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        while True:
            # CARLA Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        call_exit = True
                        break

            if call_exit:
                break

    finally:
        # Cleanup
        if display_manager:
            display_manager.destroy()

        if vehicle_list:
            destroy_actors(vehicle_list)

        if original_settings:
            restore_world_settings(world, original_settings)


def main() -> None:
    """Entry point for the script."""
    argparser = argparse.ArgumentParser(description="Grid of sensors on vehicle")
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
        "--sync", action="store_true", help="Synchronous mode execution"
    )
    argparser.add_argument(
        "--async", dest="sync", action="store_false", help="Asynchronous mode execution"
    )
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="1280x720",
        help="window resolution (default: 1280x720)",
    )

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split("x")]

    try:
        client = create_client(args.host, args.port, CARLA_TIMEOUT_SECONDS)
        run_simulation(args, client)

    except KeyboardInterrupt:
        print("\nKeyboard Interrupt or Cancelled by user")


if __name__ == "__main__":
    main()
