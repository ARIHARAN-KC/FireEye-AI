import os
import shutil
import numpy as np
import logging
from tests import TEST_DATA_DIR

class YOLODatasetPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.splits = ['train', 'valid', 'test']

        # Configure logging
        log_filename = f'logs/dataset_preprocessing.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def is_valid_bbox(self, bbox_coords):
        """Validate YOLO bbox coordinates"""
        return (len(bbox_coords) == 4 and
                all(0 <= coord <= 1 for coord in bbox_coords))

    def process_labels(self):
        """Robust label processing with strict validation"""
        total_processed_files = 0
        total_converted_labels = 0

        for split in self.splits:
            labels_dir = os.path.join(self.dataset_path, split, 'labels')

            if not os.path.exists(labels_dir):
                self.logger.warning(
                    f"Labels directory not found for split {split}")
                continue

            for label_file in os.listdir(labels_dir):
                label_path = os.path.join(labels_dir, label_file)

                with open(label_path, 'r') as f:
                    lines = f.readlines()

                cleaned_lines = []
                processed_count = 0

                for line in lines:
                    parts = line.strip().split()

                    # Validate basic label structure
                    if len(parts) < 5:
                        self.logger.warning(f"Invalid label in {label_file}: {line.strip()}")
                        processed_count+=1
                        continue

                    try:
                        class_idx = int(parts[0])
                        coords = list(map(float, parts[1:]))

                        # Standard YOLO bbox
                        if self.is_valid_bbox(coords):
                            cleaned_lines.append(line)
                            continue

                        # Polygon conversion attempt
                        if len(coords) >= 6 and len(coords) % 2 == 0:
                            x_coords = coords[0::2]
                            y_coords = coords[1::2]

                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)

                            width = round(x_max - x_min, 5)
                            height = round(y_max - y_min, 5)

                            center_x = round(x_min + width / 2, 5)
                            center_y = round(y_min + height / 2, 5)

                            new_bbox = [center_x, center_y, width, height]

                            if self.is_valid_bbox(new_bbox):
                                new_line = f"{class_idx} {new_bbox[0]} {
                                    new_bbox[1]} {new_bbox[2]} {new_bbox[3]}\n"
                                cleaned_lines.append(new_line)
                                processed_count += 1
                            else:
                                self.logger.warning(
                                    f"Invalid converted bbox in {label_file}")
                        else:
                            self.logger.warning(f"Unprocessable label in {
                                                label_file}: {line.strip()}")

                    except (ValueError, IndexError) as e:
                        self.logger.error(f"Error processing label in {
                                          label_file}: {line.strip()} - {e}")

                # Only write if changes were detected
                if len(set(cleaned_lines)) != len(cleaned_lines):
                    self.logger.warning(f"Duplicates detected in {label_file}: {
                                        len(lines)-len(set(cleaned_lines))} duplicates")
                    processed_count += (len(cleaned_lines) - len(set(cleaned_lines)))
                    cleaned_lines = list(set(cleaned_lines))

                if processed_count:
                    with open(label_path, 'w') as f:
                        f.writelines(cleaned_lines)
                    self.logger.info(f"Processed {label_file}: {processed_count} labels")
                    total_processed_files += 1
                    total_converted_labels += processed_count

        self.logger.info(f"Total processed files: {total_processed_files}")
        self.logger.info(f"Total converted labels: {total_converted_labels}")

    def validate_dataset(self):
        """Comprehensive dataset validation"""
        is_valid = True
        validation_report = {
            'missing_directories': [],
            'missing_labels': {},
            'duplicate_labels': []
        }

        for split in self.splits:
            images_dir = os.path.join(self.dataset_path, split, 'images')
            labels_dir = os.path.join(self.dataset_path, split, 'labels')

            if not (os.path.exists(images_dir) and os.path.exists(labels_dir)):
                self.logger.warning(f"Missing directories in {split}")
                validation_report['missing_directories'].append(split)
                is_valid = False
                continue

            image_files = set(os.listdir(images_dir))
            label_files = set(os.listdir(labels_dir))

            # Check matching images and labels
            expected_labels = {
                f.rsplit('.', 1)[0] + '.txt' for f in image_files}
            missing_labels = expected_labels - label_files

            if missing_labels:
                self.logger.warning(f"Missing labels in {
                                    split}: {missing_labels}")
                validation_report['missing_labels'][split] = missing_labels
                is_valid = False

            # Duplicate label validation
            for label_file in label_files:
                label_path = os.path.join(labels_dir, label_file)
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                # Check for duplicate labels within the file
                unique_labels = set(lines)
                if len(unique_labels) != len(lines):
                    self.logger.warning(
                        f"Duplicate labels found in {label_file}")
                    validation_report['duplicate_labels'].append(label_file)
                    is_valid = False

        return is_valid, validation_report

    def backup_dataset(self):
        """Create a backup of the entire dataset"""
        backup_dir = os.path.join(
            self.dataset_path, 'backup')
        shutil.copytree(self.dataset_path, backup_dir, dirs_exist_ok=True)
        self.logger.info(f"Dataset backed up to {backup_dir}")
        return backup_dir

    def preprocess(self):
        """Run full preprocessing pipeline"""
        self.logger.info("Starting dataset preprocessing...")

        # Backup dataset
        backup_path = self.backup_dataset()

        # Initial validation
        self.logger.info("Performing initial dataset validation...")
        initial_valid, initial_report = self.validate_dataset()
        self.logger.info(f"Initial dataset validation: {
                         'Valid' if initial_valid else 'Invalid'}")

        # Process labels
        self.logger.info("Processing labels...")
        self.process_labels()

        # Final validation
        self.logger.info("Performing final dataset validation...")
        final_valid, final_report = self.validate_dataset()

        # Generate comprehensive report
        self.generate_preprocessing_report(
            backup_path,
            initial_valid,
            initial_report,
            final_valid,
            final_report
        )

        return final_valid


    def generate_preprocessing_report(self, backup_path, initial_valid, initial_report, final_valid, final_report):
        """Generate a comprehensive preprocessing report"""
        self.logger.info("=" * 50)
        self.logger.info("YOLO DATASET PREPROCESSING REPORT")
        self.logger.info("=" * 50)

        self.logger.info(f"Dataset Path: {self.dataset_path}")
        self.logger.info(f"Backup Path: {backup_path}")

        self.logger.info("Initial Validation:")
        self.logger.info(f"Overall Status: {
                        'Valid' if initial_valid else 'Invalid'}")
        self.logger.info(f"Missing Directories: {
                        initial_report['missing_directories']}")
        self.logger.info(f"Missing Labels: {initial_report['missing_labels']}")
        self.logger.info(f"Duplicate Labels: {initial_report['duplicate_labels']}")

        self.logger.info("Final Validation:")
        self.logger.info(f"Overall Status: {
                        'Valid' if final_valid else 'Invalid'}")
        self.logger.info(f"Missing Directories: {
                        final_report['missing_directories']}")
        self.logger.info(f"Missing Labels: {final_report['missing_labels']}")
        self.logger.info(f"Duplicate Labels: {final_report['duplicate_labels']}")


def main():
    dataset_path = str(TEST_DATA_DIR)
    try:
        preprocessor = YOLODatasetPreprocessor(dataset_path)
        is_valid = preprocessor.preprocess()

        if is_valid:
            print("Dataset preprocessing completed successfully!")
        else:
            print(f"Dataset preprocessing completed with some issues. Check the logs for details at 'datasetpreprocessing_*.log'.")

    except Exception as e:
        logging.error(f"Preprocessing Failed: {e}")


if __name__ == '__main__':
    main()


#correct way to run the code is python -m tests.clean_labels