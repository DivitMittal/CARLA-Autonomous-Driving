import math
import cv2
import numpy as np
import carla
from pathlib import Path
from typing import Tuple, Optional

from keras.models import load_model

from .config import (
    DEFAULT_MODEL_CONFIG,
    CANNY_THRESHOLD_LOW,
    CANNY_THRESHOLD_HIGH,
    NORMALIZATION_FACTOR,
    MPS_TO_KPH_MULTIPLIER,
    DEFAULT_TEXT_DISPLAY,
)


class ImagePreprocessor:
    def __init__(self, config: Optional["ModelConfig"] = None) -> None:
        self.config = config or DEFAULT_MODEL_CONFIG
        self._calculate_crop_dimensions()

    def _calculate_crop_dimensions(self) -> None:
        self.height_from = int(self.config.image_height * (1 - self.config.height_crop_portion))
        self.width_from = int((self.config.image_width - self.config.image_width * self.config.width_crop_portion) / 2)
        self.width_to = self.width_from + int(self.config.width_crop_portion * self.config.image_width)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = np.float32(image)
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        gray_image = cv2.resize(gray_image, (self.config.image_width, self.config.image_height))

        gray_image = gray_image[self.height_from:, self.width_from:self.width_to]
        gray_image = gray_image.astype(np.uint8)

        canny = cv2.Canny(gray_image, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

        canny = canny / NORMALIZATION_FACTOR

        canny = canny[:, :, np.newaxis]
        canny = np.expand_dims(canny, axis=0)

        return canny


class SpeedController:
    def __init__(
        self,
        preferred_speed_kph: float = DEFAULT_MODEL_CONFIG.preferred_speed_kph,
        speed_threshold_kph: float = DEFAULT_MODEL_CONFIG.speed_threshold_kph,
    ) -> None:
        self.preferred_speed_kph = preferred_speed_kph
        self.speed_threshold_kph = speed_threshold_kph

    def calculate_throttle(self, current_speed_kph: float) -> float:
        if current_speed_kph >= self.preferred_speed_kph:
            return 0.0
        elif current_speed_kph < self.preferred_speed_kph - self.speed_threshold_kph:
            return 0.8
        else:
            return 0.3


class LanePredictor:
    def __init__(
        self,
        model_path: str,
        config: Optional["ModelConfig"] = None,
    ) -> None:
        self.config = config or DEFAULT_MODEL_CONFIG
        self.preprocessor = ImagePreprocessor(self.config)

        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            model_path_obj = Path(__file__).parent.parent / model_path

        self.model = load_model(str(model_path_obj), compile=False)
        self.model.compile()

    def predict_angle(self, image: np.ndarray) -> float:
        preprocessed = self.preprocessor.preprocess(image)
        angle = self.model(preprocessed, training=False)

        return angle.numpy()[0][0] * self.config.yaw_adjustment_degrees / self.config.max_steer_angle_degrees


class VehicleMonitor:
    @staticmethod
    def get_speed_kph(vehicle: carla.Vehicle) -> float:
        velocity = vehicle.get_velocity()
        speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return round(MPS_TO_KPH_MULTIPLIER * speed_mps, 0)

    @staticmethod
    def get_acceleration_mps2(vehicle: carla.Vehicle) -> float:
        acceleration = vehicle.get_acceleration()
        return round(math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2), 1)


class OverlayRenderer:
    def __init__(self, config: Optional["TextDisplayConfig"] = None) -> None:
        self.config = config or DEFAULT_TEXT_DISPLAY
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def render_speed(self, image: np.ndarray, speed_kph: float) -> np.ndarray:
        position = self.config.position_speed
        return cv2.putText(
            image,
            f"Speed: {int(speed_kph)}",
            position,
            self.font,
            self.config.font_scale,
            self.config.color,
            self.config.thickness,
            cv2.LINE_AA,
        )

    def render_angle(self, image: np.ndarray, angle_degrees: float) -> np.ndarray:
        position = self.config.position_angle
        return cv2.putText(
            image,
            f"Predicted angle in lane: {int(angle_degrees * 90)}",
            position,
            self.font,
            self.config.font_scale,
            self.config.color,
            self.config.thickness,
            cv2.LINE_AA,
        )
