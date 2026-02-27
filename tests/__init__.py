import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Constants
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
TEST_DATA_DIR = TEST_DIR / 'test_data'
SAMPLE_VIDEO_PATH = TEST_DATA_DIR / 'sample.mp4'
SAMPLE_IMAGE_PATH = TEST_DATA_DIR / 'sample_fire.jpg'

# Create test data directory if it doesn't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
