import carla
import time
from typing import List, Optional, Tuple

from .config import (
    DEFAULT_CARLA_HOST,
    DEFAULT_CARLA_PORT,
    CARLA_TIMEOUT_SECONDS,
    TRAFFIC_MANAGER_PORT,
    VEHICLE_BLUEPRINT_FILTER,
    DEFAULT_SIMULATION_SETTINGS,
    DEFAULT_WEATHER,
    TOWN05_GOOD_ROAD_IDS,
    SPAWN_DELAY_SECONDS,
)


class CarlaConnectionError(Exception):
    pass


class VehicleSpawnError(Exception):
    pass


def create_client(
    host: str = DEFAULT_CARLA_HOST,
    port: int = DEFAULT_CARLA_PORT,
    timeout: float = CARLA_TIMEOUT_SECONDS,
) -> carla.Client:
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        _ = client.get_world()
        return client
    except Exception as e:
        raise CarlaConnectionError(f"Failed to connect to CARLA at {host}:{port}: {e}") from e


def setup_synchronous_mode(
    world: carla.World,
    client: carla.Client,
    settings: Optional[carla.WorldSettings] = None,
    traffic_manager_port: int = TRAFFIC_MANAGER_PORT,
) -> carla.WorldSettings:
    original_settings = world.get_settings()

    traffic_manager = client.get_trafficmanager(traffic_manager_port)
    traffic_manager.set_synchronous_mode(True)

    if settings is None:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DEFAULT_SIMULATION_SETTINGS.fixed_delta_seconds

    world.apply_settings(settings)

    return original_settings


def set_weather(world: carla.World, weather: Optional[carla.WeatherParameters] = None) -> None:
    if weather is None:
        weather = carla.WeatherParameters(
            cloudiness=DEFAULT_WEATHER.cloudiness,
            precipitation=DEFAULT_WEATHER.precipitation,
            precipitation_deposits=DEFAULT_WEATHER.precipitation_deposits,
            wind_intensity=DEFAULT_WEATHER.wind_intensity,
            sun_azimuth_angle=DEFAULT_WEATHER.sun_azimuth_angle,
            sun_altitude_angle=DEFAULT_WEATHER.sun_altitude_angle,
            fog_density=DEFAULT_WEATHER.fog_density,
            wetness=DEFAULT_WEATHER.wetness,
        )
    world.set_weather(weather)


def get_vehicle_blueprint(world: carla.World, filter_pattern: str = VEHICLE_BLUEPRINT_FILTER) -> carla.ActorBlueprint:
    blueprint_library = world.get_blueprint_library()
    blueprints = blueprint_library.filter(filter_pattern)

    if not blueprints:
        raise VehicleSpawnError(f"No vehicle blueprints found matching pattern: {filter_pattern}")

    return blueprints[0]


def find_spawn_points_by_road_id(
    world: carla.World, road_ids: List[int], lane_type: carla.LaneType = carla.LaneType.Driving
) -> List[carla.Transform]:
    spawn_points = world.get_map().get_spawn_points()
    good_spawn_points = []

    for point in spawn_points:
        waypoint = world.get_map().get_waypoint(
            point.location, project_to_road=True, lane_type=lane_type
        )
        if waypoint.road_id in road_ids:
            good_spawn_points.append(point)

    return good_spawn_points


def spawn_vehicle(
    world: carla.World,
    blueprint: Optional[carla.ActorBlueprint] = None,
    spawn_point: Optional[carla.Transform] = None,
    filter_pattern: str = VEHICLE_BLUEPRINT_FILTER,
    autopilot: bool = False,
) -> carla.Vehicle:
    if blueprint is None:
        blueprint = get_vehicle_blueprint(world, filter_pattern)

    if spawn_point is None:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise VehicleSpawnError("No spawn points available in the map")
        import random
        spawn_point = random.choice(spawn_points)

    vehicle = world.try_spawn_actor(blueprint, spawn_point)
    if vehicle is None:
        raise VehicleSpawnError(f"Failed to spawn vehicle at {spawn_point.location}")

    vehicle.set_autopilot(autopilot)
    return vehicle


def spawn_vehicle_on_road(
    world: carla.World,
    road_ids: List[int],
    blueprint: Optional[carla.ActorBlueprint] = None,
    filter_pattern: str = VEHICLE_BLUEPRINT_FILTER,
    autopilot: bool = False,
) -> carla.Vehicle:
    if blueprint is None:
        blueprint = get_vehicle_blueprint(world, filter_pattern)

    spawn_points = find_spawn_points_by_road_id(world, road_ids)
    if not spawn_points:
        raise VehicleSpawnError(f"No spawn points found on road IDs: {road_ids}")

    import random
    spawn_point = random.choice(spawn_points)

    vehicle = world.try_spawn_actor(blueprint, spawn_point)
    if vehicle is None:
        raise VehicleSpawnError(f"Failed to spawn vehicle at {spawn_point.location}")

    vehicle.set_autopilot(autopilot)
    time.sleep(SPAWN_DELAY_SECONDS)
    return vehicle


def find_vehicle_by_pattern(world: carla.World, pattern: str = VEHICLE_BLUEPRINT_FILTER) -> Optional[carla.Vehicle]:
    actors = world.get_actors().filter(pattern)
    for actor in actors:
        if isinstance(actor, carla.Vehicle):
            return actor
    return None


def destroy_actor(actor: carla.Actor) -> None:
    if actor is not None and actor.is_alive:
        actor.destroy()


def destroy_actors(actors: List[carla.Actor]) -> None:
    for actor in actors:
        destroy_actor(actor)


def destroy_all_vehicles(world: carla.World) -> None:
    for actor in world.get_actors().filter("*vehicle*"):
        destroy_actor(actor)


def destroy_all_sensors(world: carla.World) -> None:
    for actor in world.get_actors().filter("*sensor*"):
        destroy_actor(actor)


def restore_world_settings(world: carla.World, settings: carla.WorldSettings) -> None:
    world.apply_settings(settings)
