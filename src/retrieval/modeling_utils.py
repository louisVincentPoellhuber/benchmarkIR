from accelerate import Accelerator

import os
import logging

import dotenv
dotenv.load_dotenv()


JOBID = os.getenv("SLURM_JOB_ID")
if JOBID == None: JOBID = "local"
logging.basicConfig( 
    encoding="utf-8", 
    filename=f"slurm-{JOBID}.log", 
    filemode="a", 
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level = logging.INFO
    )

MAIN_PROCESS = Accelerator().is_main_process
STORAGE_DIR = os.getenv("STORAGE_DIR")

def log_message(message, level=logging.WARNING):
    if MAIN_PROCESS:
        print(message)
        logging.log(msg=message, level=level)
