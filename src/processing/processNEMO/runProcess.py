import os
import fnmatch
import processFunctions as F

args = F.read_config_file()

TOP = args[0]
OUT = args[1]
N_FILES, N_FOLDERS = int(args[2]), int(args[3])

assert (N_FILES > 0) & (N_FOLDERS > 0), "Check config files."

file_count = 0
files_to_process = []

print("Retrieving files under %s" % TOP)

for root, dirnames, filenames in os.walk(TOP):
    for filename in fnmatch.filter(filenames, "*.nc"):
        files_to_process.append(os.path.join(root, filename))

assert len(files_to_process) == N_FILES, "Something went wrong when retrieving files."

test_commands = ["cdo", "-sellonlatbox,-18,-10,34,56", "-sellevel,0.494024992", "-selname,thetao"]

for file in files_to_process[:3]:
    code = F.process_file(file, test_commands, "test_sst", OUT)

    if(code == 0):
        file_count += 1
    else:
        print("Something went wrong. Check the error message from CDO and try again.")
    

print("Processed {:,} / {:,} files.".format(file_count, N_FILES))


