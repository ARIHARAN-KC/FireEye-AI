import pytest
import cv2
import numpy as np
from src.fire_detector import Detector
from src.config import Config

@pytest.fixture
def fire_detector():
    return Detector(Config.MODEL_PATH)


@pytest.fixture
def sample_frame():
    return cv2.imread('data/test_image.png')


def test_fire_detector_initialization(fire_detector):
    """Test FireDetector initialization"""
    assert fire_detector.model is not None
    assert fire_detector.target_height == 640


def test_resize_frame(fire_detector, sample_frame):
    """Test frame resizing"""
    resized = fire_detector.resize_frame(sample_frame)
    assert resized.shape[0] == 640
    assert resized.shape[2] == 3


def test_process_frame(fire_detector, sample_frame):
    """Test frame processing"""
    processed_frame, detection = fire_detector.process_frame(sample_frame)
    assert isinstance(processed_frame, np.ndarray)
    assert isinstance(detection, (str, type(None)))
