import os
from pathlib import Path
import logging

# logging config
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'ods'

list_of_files = [
    # github actions (CI/CD placeholder)
    ".github/workflows/.gitkeep",

    # source package
    f"src/{project_name}/__init__.py",

    # components — one file per pipeline stage
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/model_builder.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",

    # config — reads config.yaml and params.yaml
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    # entity — dataclasses for config return types
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",

    # constants — paths to config files
    f"src/{project_name}/constants/__init__.py",

    # pipeline — ties components together in order
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_data_validation.py",
    f"src/{project_name}/pipeline/stage_03_model_builder.py",
    f"src/{project_name}/pipeline/stage_04_model_trainer.py",
    f"src/{project_name}/pipeline/stage_05_model_evaluation.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",

    # utils — common helper functions
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",

    # models — modular architecture
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/models/backbone.py",
    f"src/{project_name}/models/neck.py",
    f"src/{project_name}/models/detection_head.py",
    f"src/{project_name}/models/segmentation_head.py",
    f"src/{project_name}/models/model.py",

    # losses
    f"src/{project_name}/losses/__init__.py",
    f"src/{project_name}/losses/losses.py",

    # datasets
    f"src/{project_name}/datasets/__init__.py",
    f"src/{project_name}/datasets/cityscapes_dataset.py",
    f"src/{project_name}/datasets/transforms.py",

    # evaluation
    f"src/{project_name}/evaluation/__init__.py",
    f"src/{project_name}/evaluation/metrics.py",

    # inference
    f"src/{project_name}/inference/__init__.py",
    f"src/{project_name}/inference/pytorch_infer.py",
    f"src/{project_name}/inference/onnx_infer.py",

    # config files
    "config/config.yaml",
    "params.yaml",

    # data folders — gitignored, just create placeholders
    "data/cityscapes/.gitkeep",
    "data/foggy_cityscapes/.gitkeep",
    "data/idd/.gitkeep",

    # research notebooks — experiment here first
    "research/01_EDA.ipynb",
    "research/02_data_ingestion.ipynb",
    "research/03_model_building.ipynb",
    "research/04_model_training.ipynb",
    "research/05_model_evaluation.ipynb",

    # experiment tracking results
    "experiments/comparison.md",
    "experiments/onnx_benchmark.md",

    # app and entry points
    "main.py",
    "app.py",
    "setup.py",
    "requirements.txt",
    "Dockerfile",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filename} — skipping")
