import os
import processFunctions as F

args = F.read_config_file()

TOP = args[0]
OUT = args[1]
N_FILES, N_FOLDERS = int(args[2]), int(args[3])

assert (N_FILES > 0) & (N_FOLDERS > 0), "Check config files"

#print(F.read_config_file())




