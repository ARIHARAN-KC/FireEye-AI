import pytest
import cv2
import numpy as np
from pathlib import Path
from src.config import Config, setup_logging
from src.fire_detector import Detector
from src.notification_service import NotificationService


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment and logging"""
    setup_logging()
    Config.validate()
    yield


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing"""
    return np.zeros((500, 500, 3), dtype=np.uint8)


@pytest.fixture
def fire_detector():
    """Create FireDetector instance"""
    return Detector(Config.MODEL_PATH)


@pytest.fixture
def notification_service():
    """Create NotificationService instance"""
    return NotificationService(Config)
