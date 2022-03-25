import os
import logging


ROOT_DIR: str = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

processed_path: str = os.path.join(ROOT_DIR, 'data/processed')
raw_path: str = os.path.join(ROOT_DIR, 'data/raw')

# define logger configuration for data package:
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_fmt)
