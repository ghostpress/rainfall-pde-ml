# --------- CONFIGURATION FILE FOR DATA PROCESSING ----------

import os

data_dir = ""
dest_dir = ""

print("Configuring data processing environment.")

if os.path.exists("config.txt"):

    # Read the relevant variables from that file
    with open("config.txt", "r") as f:

        contents = f.readlines()
        data_dir += contents[0].strip("\n")
        dest_dir += contents[1].strip("\n")

else:
    # This is the first time the processing code has been run - need to ask for and save relevant variables only once

    # First, the data directory:
    data_dir += input("Enter top data directory, eg. './data/' that holds data files to be processed: ")

    # Next, the destination directory:
    dest_dir += input("Enter the output directory, eg. './data/out/' to hold the processed data files when done: ")

    files,folders = 0,0
    for _, dirnames, filenames in os.walk(data_dir):
        files += len(filenames)
        folders += len(dirnames)

    # print("{:,} files, {:,} folders".format(files, folders))    
    
    # Create the config file

    with open("config.txt", "w") as f:
        f.write(data_dir + "\n")
        f.write(dest_dir + "\n")

    print("Configuration complete for processing {:,} files, {:,} folders in %s".format(files, folders) % data_dir)
    
        

