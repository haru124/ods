from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# Cityscapes 19-class label map (training IDs)
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

# Cityscapes label ID → training ID mapping
# (Cityscapes has 34 raw label IDs; only 19 are used for training)
LABEL_ID_TO_TRAIN_ID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255,
    5: 255, 6: 255,
    7: 0,   # road
    8: 1,   # sidewalk
    9: 255, 10: 255,
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: 255, 15: 255, 16: 255,
    17: 5,  # pole
    18: 255,
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10, # sky
    24: 11, # person
    25: 12, # rider
    26: 13, # car
    27: 14, # truck
    28: 15, # bus
    29: 255,
    30: 255,
    31: 16, # train
    32: 17, # motorcycle
    33: 18, # bicycle
}

# Detection: 8 categories (subset used for bounding box tasks)
DETECTION_CLASSES = [
    "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle",
]

# gtFine suffixes
SEG_SUFFIX = "_gtFine_labelIds.png"
POLYGON_SUFFIX = "_gtFine_polygons.json"
COLOR_SUFFIX = "_gtFine_color.png"