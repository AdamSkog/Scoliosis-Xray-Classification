import logging
import os
import warnings
from pathlib import Path

from src.pipeline.pipeline import pipeline
from src.utils.common import read_yaml

# Set environment variable to disable oneDNN custom operations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Read parameters
params = read_yaml(Path("params.yaml"))
log_file = os.path.join(params["log_dir"], "pipeline.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

if __name__ == "__main__":
    pipeline()
