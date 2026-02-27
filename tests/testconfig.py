import pytest
import os
from src.config import Config, setup_logging


def test_config_validation():
    """Test configuration validation"""
    assert Config.RECEIVER_WHATSAPP_NUMBER is not None
    assert Config.IMGUR_CLIENT_ID is not None


def test_directories_exist():
    """Test required directories exist"""
    assert Config.MODEL_PATH.parent.exists()
    assert Config.DETECTED_FIRES_DIR.exists()


def test_logging_setup():
    """Test logging configuration"""
    setup_logging()
    assert os.path.exists('logs/fire_detection.log')
