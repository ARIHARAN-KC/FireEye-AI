import cv2
import logging
from pathlib import Path
from src.config import Config, setup_logging
from src.notification_service import NotificationService


def send_existing_fire_image():
    """Utility to send most recent fire detection image via WhatsApp"""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        Config.validate()
        notification_service = NotificationService(Config)

        # Get the most recent fire detection image
        fire_images = list(
            Config.DETECTED_FIRES_DIR.glob('fire_detected_*.jpg'))
        if not fire_images:
            logger.error("No fire detection images found!")
            return False

        # Sort by modification time and get the most recent
        latest_image = max(fire_images, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found latest fire image: {latest_image}")

        # Read the image
        frame = cv2.imread(str(latest_image))
        if frame is None:
            logger.error(f"Failed to read image: {latest_image}")
            return False

        # Send the alert
        success = notification_service.send_whatsapp_alert(frame)
        if success:
            logger.info("Successfully sent existing fire image!")
        else:
            logger.error("Failed to send fire image!")

        return success

    except Exception as e:
        logger.error(f"Error in send_existing_fire_image: {e}")
        return False


if __name__ == "__main__":
    send_existing_fire_image()
