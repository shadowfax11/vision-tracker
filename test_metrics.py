import os 
import sys

submodule_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'ultralytics')
if submodule_dir not in sys.path: sys.path.insert(0, submodule_dir)
from ultralytics import YOLO

import argparse

parser = argparse.ArgumentParser(description="Test YOLO pose estimation model training.")
parser.add_argument('--dataset_yaml_path', type=str, default='dataset.yaml', help='Path to the dataset YAML file. Default is "dataset.yaml".')
parser.add_argument('--model_weights', type=str, required=True, help='Path to the weights file.')
args = parser.parse_args()

model = YOLO(args.model_weights)

metrics = model.val(data=args.dataset_yaml_path, 
                    split='val')

# --- Metrics Output ---
# The console will print detailed metrics:
# Box Metrics (mAP50-95(B), mAP50(B), etc.) - For the object detection part
# Pose Metrics (mAP50-95(P), mAP50(P), etc.) - For the keypoint estimation part (based on OKS)
print("\n--- Validation Metrics ---")
# metrics object contains detailed results if needed
print(f"Box Precision: {metrics.box.p}")
print(f"Box Recall: {metrics.box.r}")
print(f"Box mAP50: {metrics.box.map50}")
print(f"Box mAP50-95: {metrics.box.map}")
print(f"Pose Precision: {metrics.pose.p}")
print(f"Pose Recall: {metrics.pose.r}")
print(f"Pose mAP50: {metrics.pose.map50}")
print(f"Pose mAP50-95: {metrics.pose.map}")
