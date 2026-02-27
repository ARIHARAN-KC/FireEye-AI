"""
Fire Detection System
A real-time fire detection and notification system using computer vision.
"""


from .config import Config, setup_logging
from .fire_detector import Detector
from .notification_service import NotificationService

__all__ = [
    'Config',
    'setup_logging',
    'Detector',
    'NotificationService',
]
