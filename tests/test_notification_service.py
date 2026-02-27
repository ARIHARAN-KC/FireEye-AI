import pytest
import cv2
from src.notification_service import NotificationService
from pathlib import Path
from src.config import Config


@pytest.fixture
def notification_service():
    config = Config()
    service = NotificationService(config)
    yield service
    # Ensure proper cleanup after each test
    service.cleanup()


@pytest.fixture
def sample_frame():
    return cv2.imread('data/test_image.png')


def test_notification_service_init(notification_service):
    """Test NotificationService initialization"""
    assert notification_service.executor is not None
    assert notification_service.loop is not None
    assert not notification_service.loop.is_closed()


def test_save_frame(notification_service, sample_frame):
    """Test frame saving functionality"""
    path = notification_service.save_frame(sample_frame)
    assert path.exists()
    path.unlink()  # Cleanup


def test_imgur_upload(notification_service, sample_frame):
    """Test Imgur upload functionality"""
    path = notification_service.save_frame(sample_frame)
    url = notification_service.upload_image(path)
    assert url is not None and url.startswith('http')
    path.unlink()  # Cleanup


def test_whatsapp_alert(notification_service, sample_frame):
    """Test WhatsApp alert sending"""
    result = notification_service.send_alert(sample_frame, '---TESTS---')
    assert result is True
