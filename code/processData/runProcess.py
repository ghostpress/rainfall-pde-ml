import os
import processFunctions as F
import config

os.system("config.py")

TOP = config.data_dir
OUT = config.dest_dir
N_FILES, N_FOLDERS = config.files, config.folders

assert N_FILES > 0 & N_FOLDERS > 0, "Check config files"




