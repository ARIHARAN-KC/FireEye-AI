import pytest
import cv2
from src.fire_detector import Detector
from src.notification_service import NotificationService
from src.config import Config


@pytest.fixture
def fire_detector():
    return Detector(Config.MODEL_PATH)


@pytest.fixture
def notification_service():
    config = Config()
    return NotificationService(config)


@pytest.fixture
def sample_frame():
    return cv2.imread('data/test_image.png')


def test_full_detection_flow(fire_detector, notification_service, sample_frame):
    """Test complete detection and notification flow"""
    processed_frame, detection = fire_detector.process_frame(sample_frame)
    if detection:
        result = notification_service.send_alert(processed_frame, detection)
        assert result is True
