import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")


project_name = "Scoliosis-Xray-Classification"

list_of_files = [
    ".github/workflows/.gitkeep",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/utils/__init__.py",
    "src/pipeline/__init__.py",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py"
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} is already exists")
