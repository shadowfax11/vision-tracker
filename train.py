import os 
import sys

submodule_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'ultralytics')
if submodule_dir not in sys.path: sys.path.insert(0, submodule_dir)
from ultralytics import YOLO

import yaml 
import numpy as np 
import argparse
from dataclasses import dataclass, field 
import matplotlib; matplotlib.use('Agg')    # use non-interactive backend

parser = argparse.ArgumentParser(description="Train/Fine-tune a YOLO pose estimation model on a custom dataset.")
parser.add_argument('--dataset_yaml_path', type=str, default='dataset.yaml', help='Path to the dataset YAML file. Default is "dataset.yaml".')
parser.add_argument('--model_yaml_path', type=str, default='yolov8n-pose.yaml', help='Path to the model YAML file. Default is "yolov8n-pose.yaml".')
parser.add_argument('--model_weights', type=str, default='yolov8n-pose.pt', help='Path to the weights file. Default is "yolov8n-pose.pt".')
parser.add_argument('--train_from_scratch', action='store_true', help='Flag to train the model from scratch. If set, the weights will be ignored.')
parser.add_argument('--training_project_name', type=str, default='runs/train', help='Name of the training project. Default is "runs/train".')

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train. Default is 100.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training. Default is 16.')
parser.add_argument('--img_size', type=int, default=640, help='Image size for training. Default is 640.')
args = parser.parse_args()


@dataclass(frozen=True)
class DatasetConfig: 
    IMAGE_SIZE:     int = args.img_size 
    BATCH_SIZE:     int = args.batch_size
    CLOSE_MOSAIC:   int = 10
    MOSAIC:         float = 0.4 
    FLIPLR:         float = 0.0
    FLIPUD:         float = 0.0

with open(args.dataset_yaml_path, 'r') as f:
    data_dict = yaml.safe_load(f)

@dataclass(frozen=True)
class TrainingConfig: 
    DATASET_YAML:   str = args.dataset_yaml_path
    MODEL_YAML:     str = args.model_yaml_path
    WEIGHTS:        str = args.model_weights
    TRAIN_FROM_SCRATCH: bool = args.train_from_scratch
    EPOCHS:         int = args.epochs
    KPT_SHAPE:      tuple = (data_dict['kpt_shape'][0], data_dict['kpt_shape'][1])
    PROJECT:        str = args.training_project_name
    NAME:           str = f"{args.model_yaml_path.split('.')[0]}_{EPOCHS}_epochs"
    CLASSES_DICT:   dict = field(default_factory = lambda:data_dict['names'])
    FLIP_IDXES:     list = field(default_factory=lambda:data_dict.get('flip_idx',[]))

train_config = TrainingConfig()
data_config = DatasetConfig()

if args.train_from_scratch: 
    print('Training {} from scratch...'.format(train_config.MODEL_YAML.split('.')[0]))
    model = YOLO(train_config.MODEL_YAML)
else:
    print('Loading {} with weights {}...'.format(train_config.MODEL_YAML.split('.')[0], train_config.WEIGHTS))
    model = YOLO(train_config.MODEL_YAML).load(train_config.WEIGHTS)

model.train(
    data = train_config.DATASET_YAML,   # path to dataset.yaml
    epochs = train_config.EPOCHS,       # number of epochs to train
    batch = data_config.BATCH_SIZE,     # batch size
    imgsz = data_config.IMAGE_SIZE,     # image size
    project = train_config.PROJECT,     # project name
    name = train_config.NAME,           # name of the run
    close_mosaic = data_config.CLOSE_MOSAIC, # close mosaic
    mosaic = data_config.MOSAIC,       # mosaic augmentation
    flipud = data_config.FLIPUD,       # flip up down
    fliplr = data_config.FLIPLR,       # flip left right
)