import cv2
import logging
import sys
from pathlib import Path
from config import Config, setup_logging
from fire_detector import Detector
from notification_service import NotificationService
import time


def main():
    # Initialize logging and configuration
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Fire Detection System")

    try:
        # Validate configuration
        # Config.validate()
        logger.debug("Configuration validation successful")

        # Initialize services
        notification_service = NotificationService(Config)
        logger.info("Initialized notification services")

        # System self-test
        # if not notification_service.send_test_message():
        #     logger.critical("System self-test failed. Shutting down.")
        #     sys.exit(1)
        # logger.info("System self-test passed")

        # Initialize detection components
        detector = Detector(Config.MODEL_PATH, iou_threshold=0.20)
        logger.info(f"Loaded detection model: {Config.MODEL_PATH.name}")

        # Video processing setup
        cap = cv2.VideoCapture(str(Config.VIDEO_SOURCE))
        # cap = cv2.VideoCapture(0) # for webcam
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {Config.VIDEO_SOURCE}")
            sys.exit(1)
        logger.info(f"Processing video source: {Config.VIDEO_SOURCE}")

        # State management
        alert_cooldown = Config.ALERT_COOLDOWN  # Seconds between alerts
        last_alert_time = 0

        next_detection_to_report = "any"  # "Fire" or "Smoke"
        # Main processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video processing completed")
                break

            # Detection pipeline
            processed_frame, detection = detector.process_frame(frame)

            # Alert logic with cooldown
            if detection:
                current_time = time.time()
                if (next_detection_to_report == "any" or detection == next_detection_to_report) \
                        and (current_time - last_alert_time) > alert_cooldown:
                    logger.warning(f"{detection} Detected! Queueing alert")
                    notification_service.send_alert(processed_frame, detection)
                    last_alert_time = current_time
                    next_detection_to_report = "Smoke" if detection == "Fire" else "Fire"

            # Display output
            cv2.imshow("Fire Detection System", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User initiated shutdown")
                break

    except Exception as e:
        logger.critical(f"Critical system failure: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup resources
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        logger.info("System shutdown complete")


if __name__ == "__main__":
    main()
