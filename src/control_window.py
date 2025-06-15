"""Manual vehicle control using Pygame.

This script provides a Pygame-based manual control interface for a vehicle
that has already been spawned in CARLA.
"""
import argparse
import pygame
import carla

from .config import (
    DEFAULT_CARLA_HOST,
    DEFAULT_CARLA_PORT,
    CARLA_TIMEOUT_SECONDS,
    VEHICLE_BLUEPRINT_FILTER,
    THROTTLE_INCREMENT,
    BRAKE_INCREMENT,
    STEER_INCREMENT,
    MANUAL_CONTROL_FPS,
)
from .carla_utils import (
    create_client,
    find_vehicle_by_pattern,
)


class ManualController:
    """Handles manual vehicle control via keyboard input."""

    def __init__(self, vehicle: carla.Vehicle, world: carla.World) -> None:
        """Initialize the manual controller.

        Args:
            vehicle: Vehicle to control.
            world: CARLA world instance.
        """
        self.vehicle = vehicle
        self.world = world
        self.control = carla.VehicleControl()

    def update_control_from_keys(self, keys: pygame.key.ScancodeWrapper) -> None:
        """Update vehicle control based on pressed keys.

        Args:
            keys: Pygame key state from pygame.key.get_pressed().
        """
        # Throttle control (W/↑)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.control.throttle = min(self.control.throttle + THROTTLE_INCREMENT, 1.0)
        else:
            self.control.throttle = 0.0

        # Brake control (S/↓)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.control.brake = min(self.control.brake + BRAKE_INCREMENT, 1.0)
        else:
            self.control.brake = 0.0

        # Steering control (A/← and D/→)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.control.steer = max(self.control.steer - STEER_INCREMENT, -1.0)
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.control.steer = min(self.control.steer + STEER_INCREMENT, 1.0)
        else:
            self.control.steer = 0.0

        # Handbrake (Space)
        self.control.hand_brake = keys[pygame.K_SPACE]

    def apply_control(self) -> None:
        """Apply the current control state to the vehicle."""
        self.vehicle.apply_control(self.control)


def run_control_loop(
    vehicle: carla.Vehicle,
    world: carla.World,
    window_size: tuple = (512, 256),
) -> None:
    """Run the manual control loop.

    Args:
        vehicle: Vehicle to control.
        world: CARLA world instance.
        window_size: Pygame window (width, height).
    """
    pygame.init()
    pygame.display.set_caption("Pygame CARLA manual control window")
    screen = pygame.display.set_mode(window_size)

    controller = ManualController(vehicle, world)
    clock = pygame.time.Clock()
    done = False

    while not done:
        # Update control based on keyboard input
        keys = pygame.key.get_pressed()
        controller.update_control_from_keys(keys)
        controller.apply_control()

        # Advance simulation
        world.tick()

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Update display
        pygame.display.flip()
        pygame.display.update()
        clock.tick(MANUAL_CONTROL_FPS)


def main(args: argparse.Namespace) -> None:
    """Entry point for manual control script.

    Args:
        args: Command-line arguments.

    Raises:
        RuntimeError: If no vehicle is found matching the filter pattern.
    """
    # Connect to CARLA
    client = create_client(args.host, args.port, CARLA_TIMEOUT_SECONDS)
    world = client.get_world()
    world.wait_for_tick()

    # Find existing vehicle
    vehicle = find_vehicle_by_pattern(world, args.vehicle_filter)
    if vehicle is None:
        raise RuntimeError(
            f"No vehicle found matching pattern: {args.vehicle_filter}. "
            "Ensure a vehicle is spawned in CARLA before running this script."
        )

    print(f"Controlling vehicle: {vehicle.type_id}")

    # Run control loop
    run_control_loop(vehicle, world, (args.width, args.height))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    argparser = argparse.ArgumentParser(
        description="Manual vehicle control via Pygame"
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
        "--vehicle-filter",
        metavar="PATTERN",
        default=VEHICLE_BLUEPRINT_FILTER,
        help=f"Vehicle blueprint filter pattern (default: {VEHICLE_BLUEPRINT_FILTER})",
    )
    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="512x256",
        help="window resolution (default: 512x256)",
    )

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split("x")]
    return args


if __name__ == "__main__":
    main(parse_args())
