import logging
import os
from datetime import datetime


current_time = datetime.now()
log_file_folder = current_time.strftime("%Y_%m_%d")
log_file = f"{current_time.strftime('%H')}.log"

logs_path = os.path.join(os.getcwd(), "logs", log_file_folder)
os.makedirs(logs_path, exist_ok=True)

log_file_path = os.path.join(logs_path, log_file)

logging.basicConfig(
    filename=log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
