from dataclasses import dataclass
from typing import Dict, List


DEFAULT_CARLA_HOST = "127.0.0.1"
DEFAULT_CARLA_PORT = 2000
CARLA_TIMEOUT_SECONDS = 5.0
TRAFFIC_MANAGER_PORT = 8000

VEHICLE_BLUEPRINT_FILTER = "*model3*"


@dataclass
class SimulationSettings:
    synchronous_mode: bool = True
    fixed_delta_seconds: float = 0.05
    traffic_manager_synchronous: bool = True


DEFAULT_SIMULATION_SETTINGS = SimulationSettings()


@dataclass
class WeatherConfig:
    cloudiness: float = 0.0
    precipitation: float = 22.0
    precipitation_deposits: float = 0.0
    wind_intensity: float = 0.0
    sun_azimuth_angle: float = 34.0
    sun_altitude_angle: float = 10.0
    fog_density: float = 0.0
    wetness: float = 1.5


DEFAULT_WEATHER = WeatherConfig()

CAMERA_HEIGHT = 2.4


@dataclass
class CameraConfig:
    image_size_x: int = 512
    image_size_y: int = 256
    pos_z: float = 1.6
    pos_x: float = 0.9


DEFAULT_CAMERA_CONFIG = CameraConfig()

LIDAR_CHANNELS = 64
LIDAR_RANGE = 100
LIDAR_POINTS_PER_SECOND = 250000
LIDAR_ROTATION_FREQUENCY = 20
SEMANTIC_LIDAR_POINTS_PER_SECOND = 100000
LIDAR_RANGE_MULTIPLIER = 2.0


@dataclass
class DisplayConfig:
    grid_size: List[int]
    window_width: int
    window_height: int


DEFAULT_GRID_SIZE = [2, 3]
DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720


@dataclass
class ModelConfig:
    preferred_speed_kph: int = 60
    speed_threshold_kph: int = 5
    yaw_adjustment_degrees: float = 35.0
    max_steer_angle_degrees: float = 35.0
    image_height: int = 360
    image_width: int = 640
    height_crop_portion: float = 0.4
    width_crop_portion: float = 0.5


DEFAULT_MODEL_CONFIG = ModelConfig()

CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150
NORMALIZATION_FACTOR = 255.0
MPS_TO_KPH_MULTIPLIER = 3.6


@dataclass
class TextDisplayConfig:
    font: int = 0
    position_speed: tuple = (30, 30)
    position_angle: tuple = (30, 50)
    font_scale: float = 0.5
    color: tuple = (255, 255, 255)
    thickness: int = 1


DEFAULT_TEXT_DISPLAY = TextDisplayConfig()

THROTTLE_INCREMENT = 0.05
BRAKE_INCREMENT = 0.2
STEER_INCREMENT = 0.05
MANUAL_CONTROL_FPS = 60

SPAWN_DELAY_SECONDS = 5
TOWN05_GOOD_ROAD_IDS = [37]

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)

SENSOR_TYPE_RGB_CAMERA = "RGBCamera"
SENSOR_TYPE_LIDAR = "LiDAR"
SENSOR_TYPE_SEMANTIC_LIDAR = "SemanticLiDAR"
SENSOR_TYPE_RADAR = "Radar"
