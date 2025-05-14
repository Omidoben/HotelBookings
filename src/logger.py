import logging
import os
from datetime import datetime

# create logs directory
LOG_DIR="logs"
os.makedirs(LOG_DIR, exist_ok=True)     # ensures that a directory named logs exists in the same location where your script is run. exist_ok=True prevents errors if the directory is already there.

LOG_FILE=os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")      # Generates a unique log file name. Creates a log file name based on the current timestamp

# Configure the logger
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
