import time
import numpy as np
import pygame
import carla
from dataclasses import dataclass
from typing import Optional, Dict, List

from .config import (
    CAMERA_HEIGHT,
    LIDAR_CHANNELS,
    LIDAR_RANGE,
    LIDAR_POINTS_PER_SECOND,
    LIDAR_ROTATION_FREQUENCY,
    SEMANTIC_LIDAR_POINTS_PER_SECOND,
    LIDAR_RANGE_MULTIPLIER,
    COLOR_WHITE,
)


@dataclass
class SensorConfig:
    sensor_type: str
    transform: carla.Transform
    sensor_options: Dict[str, str]
    display_pos: List[int]


class CustomTimer:
    def __init__(self) -> None:
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self) -> float:
        return self.timer()


class DisplayManager:
    def __init__(self, grid_size: List[int], window_size: List[int]) -> None:
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            window_size, pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list: List["SensorManager"] = []

    def get_window_size(self) -> List[int]:
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self) -> List[int]:
        return [
            int(self.window_size[0] / self.grid_size[1]),
            int(self.window_size[1] / self.grid_size[0]),
        ]

    def get_display_offset(self, grid_pos: List[int]) -> List[int]:
        disp_size = self.get_display_size()
        return [int(grid_pos[1] * disp_size[0]), int(grid_pos[0] * disp_size[1])]

    def add_sensor(self, sensor: "SensorManager") -> None:
        self.sensor_list.append(sensor)

    def get_sensor_list(self) -> List["SensorManager"]:
        return self.sensor_list

    def render(self) -> None:
        if not self.render_enabled():
            return

        for sensor in self.sensor_list:
            sensor.render()

        pygame.display.flip()

    def destroy(self) -> None:
        for sensor in self.sensor_list:
            sensor.destroy()

    def render_enabled(self) -> bool:
        return self.display is not None


class SensorManager:
    def __init__(
        self,
        world: carla.World,
        display_man: DisplayManager,
        sensor_type: str,
        transform: carla.Transform,
        attached: carla.Actor,
        sensor_options: Dict[str, str],
        display_pos: List[int],
    ) -> None:
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self._init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

    def _init_sensor(
        self,
        sensor_type: str,
        transform: carla.Transform,
        attached: carla.Actor,
        sensor_options: Dict[str, str],
    ) -> Optional[carla.Actor]:
        if sensor_type == "RGBCamera":
            return self._init_rgb_camera(transform, attached, sensor_options)
        elif sensor_type == "LiDAR":
            return self._init_lidar(transform, attached, sensor_options)
        elif sensor_type == "SemanticLiDAR":
            return self._init_semantic_lidar(transform, attached, sensor_options)
        elif sensor_type == "Radar":
            return self._init_radar(transform, attached, sensor_options)
        else:
            return None

    def _init_rgb_camera(
        self,
        transform: carla.Transform,
        attached: carla.Actor,
        sensor_options: Dict[str, str],
    ) -> carla.Actor:
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        disp_size = self.display_man.get_display_size()
        scalar = 1
        disp_size = [256, 256] * scalar
        camera_bp.set_attribute("image_size_x", str(disp_size[0]))
        camera_bp.set_attribute("image_size_y", str(disp_size[1]))
        for key in sensor_options:
            camera_bp.set_attribute(key, sensor_options[key])

        camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
        camera.listen(self._save_rgb_image)
        return camera

    def _init_lidar(
        self,
        transform: carla.Transform,
        attached: carla.Actor,
        sensor_options: Dict[str, str],
    ) -> carla.Actor:
        lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "100")
        lidar_bp.set_attribute(
            "dropoff_general_rate",
            lidar_bp.get_attribute("dropoff_general_rate").recommended_values[0],
        )
        lidar_bp.set_attribute(
            "dropoff_intensity_limit",
            lidar_bp.get_attribute("dropoff_intensity_limit").recommended_values[0],
        )
        lidar_bp.set_attribute(
            "dropoff_zero_intensity",
            lidar_bp.get_attribute("dropoff_zero_intensity").recommended_values[0],
        )

        for key in sensor_options:
            lidar_bp.set_attribute(key, sensor_options[key])

        lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)
        lidar.listen(self._save_lidar_image)
        return lidar

    def _init_semantic_lidar(
        self,
        transform: carla.Transform,
        attached: carla.Actor,
        sensor_options: Dict[str, str],
    ) -> carla.Actor:
        lidar_bp = self.world.get_blueprint_library().find(
            "sensor.lidar.ray_cast_semantic"
        )
        lidar_bp.set_attribute("range", "100")

        for key in sensor_options:
            lidar_bp.set_attribute(key, sensor_options[key])

        lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)
        lidar.listen(self._save_semanticlidar_image)
        return lidar

    def _init_radar(
        self,
        transform: carla.Transform,
        attached: carla.Actor,
        sensor_options: Dict[str, str],
    ) -> carla.Actor:
        radar_bp = self.world.get_blueprint_library().find("sensor.other.radar")
        for key in sensor_options:
            radar_bp.set_attribute(key, sensor_options[key])

        radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
        radar.listen(self._save_radar_image)
        return radar

    def get_sensor(self) -> Optional[carla.Actor]:
        return self.sensor

    def _save_rgb_image(self, image: carla.Image) -> None:
        self.t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        self._update_timing_stats()

    def _process_lidar_data(self, points_per_channel: bytes, channels: int) -> None:
        disp_size = self.display_man.get_display_size()
        lidar_range = LIDAR_RANGE_MULTIPLIER * float(self.sensor_options["range"])

        points = np.frombuffer(points_per_channel, dtype=np.dtype("f4"))
        points = np.reshape(points, (int(points.shape[0] / channels), channels))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = COLOR_WHITE

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

    def _update_timing_stats(self) -> None:
        t_end = self.timer.time()
        self.time_processing += t_end - self.t_start
        self.tics_processing += 1

    def _save_lidar_image(self, image: carla.LidarMeasurement) -> None:
        self.t_start = self.timer.time()
        self._process_lidar_data(image.raw_data, 4)
        self._update_timing_stats()

    def _save_semanticlidar_image(self, image: carla.LidarMeasurement) -> None:
        self.t_start = self.timer.time()
        self._process_lidar_data(image.raw_data, 6)
        self._update_timing_stats()

    def _save_radar_image(self, radar_data: carla.RadarMeasurement) -> None:
        self.t_start = self.timer.time()
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype("f4"))
        points = np.reshape(points, (len(radar_data), 4))
        self._update_timing_stats()

    def render(self) -> None:
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self) -> None:
        if self.sensor is not None:
            self.sensor.destroy()


def get_default_sensor_configs() -> List[SensorConfig]:
    return [
        SensorConfig(
            sensor_type="RGBCamera",
            transform=carla.Transform(
                carla.Location(x=0, z=CAMERA_HEIGHT), carla.Rotation(yaw=-90)
            ),
            sensor_options={},
            display_pos=[0, 0],
        ),
        SensorConfig(
            sensor_type="RGBCamera",
            transform=carla.Transform(
                carla.Location(x=0, z=CAMERA_HEIGHT), carla.Rotation(yaw=0)
            ),
            sensor_options={},
            display_pos=[0, 1],
        ),
        SensorConfig(
            sensor_type="RGBCamera",
            transform=carla.Transform(
                carla.Location(x=0, z=CAMERA_HEIGHT), carla.Rotation(yaw=90)
            ),
            sensor_options={},
            display_pos=[0, 2],
        ),
        SensorConfig(
            sensor_type="RGBCamera",
            transform=carla.Transform(
                carla.Location(x=0, z=CAMERA_HEIGHT), carla.Rotation(yaw=180)
            ),
            sensor_options={},
            display_pos=[1, 1],
        ),
        SensorConfig(
            sensor_type="LiDAR",
            transform=carla.Transform(carla.Location(x=0, z=CAMERA_HEIGHT)),
            sensor_options={
                "channels": str(LIDAR_CHANNELS),
                "range": str(LIDAR_RANGE),
                "points_per_second": str(LIDAR_POINTS_PER_SECOND),
                "rotation_frequency": str(LIDAR_ROTATION_FREQUENCY),
            },
            display_pos=[1, 0],
        ),
        SensorConfig(
            sensor_type="SemanticLiDAR",
            transform=carla.Transform(carla.Location(x=0, z=CAMERA_HEIGHT)),
            sensor_options={
                "channels": str(LIDAR_CHANNELS),
                "range": str(LIDAR_RANGE),
                "points_per_second": str(SEMANTIC_LIDAR_POINTS_PER_SECOND),
                "rotation_frequency": str(LIDAR_ROTATION_FREQUENCY),
            },
            display_pos=[1, 2],
        ),
    ]


def spawn_sensors_from_configs(
    world: carla.World,
    display_manager: DisplayManager,
    vehicle: carla.Vehicle,
    configs: List[SensorConfig],
) -> None:
    for config in configs:
        SensorManager(
            world,
            display_manager,
            config.sensor_type,
            config.transform,
            vehicle,
            config.sensor_options,
            display_pos=config.display_pos,
        )
